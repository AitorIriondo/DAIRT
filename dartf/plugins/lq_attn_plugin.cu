// TensorRT 10 IPluginV3 plugin "LQAttention": fused windowed multi-head attention with token-major I/O.
//   inputs : q, k, v  fp16 [..., H*D] token-major (any leading dims; tokens grouped in windows of N along the flattened rows)
//   output : o        fp16 [..., H*D] token-major
//   attrs  : window (N), heads (H), scale (float)
// Kernel: lq_attn_kernel.cuh (flash-attention-2 structure), BR = 96 for N=576, 192 for N=5184 (both divide; fallback 64).
// nvcc -O3 -std=c++17 -arch=sm_87 --use_fast_math -Xcompiler -fPIC -shared -I /usr/include/aarch64-linux-gnu lq_attn_plugin.cu -o lq_attn.so -lnvinfer
#include "lq_attn_kernel.cuh"
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
using namespace nvinfer1;

static const char* kAttnName = "LQAttention"; static const char* kAttnVersion = "1"; static const char* kAttnNs = "";

class LQAttnPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    int window = 576, heads = 16, hd = 64; float scale = 0.125f;
    LQAttnPlugin(int n, int h, float s, int d = 64) : window(n), heads(h), hd(d), scale(s) {}
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
        if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (t == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    }
    IPluginV3* clone() noexcept override { return new LQAttnPlugin(window, heads, scale, hd); }
    char const* getPluginName() const noexcept override { return kAttnName; }
    char const* getPluginVersion() const noexcept override { return kAttnVersion; }
    char const* getPluginNamespace() const noexcept override { return kAttnNs; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override {
        return io[pos].desc.format == TensorFormat::kLINEAR && io[pos].desc.type == DataType::kHALF;
    }
    int32_t getOutputDataTypes(DataType* out, int32_t, DataType const*, int32_t) const noexcept override { out[0] = DataType::kHALF; return 0; }
    int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t, IExprBuilder&) noexcept override { out[0] = in[0]; return 0; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override { return 0; }
    int32_t getValidTactics(int32_t*, int32_t) noexcept override { return 0; }
    int32_t getNbTactics() noexcept override { return 0; }
    char const* getTimingCacheID() noexcept override { return nullptr; }
    int32_t getFormatCombinationLimit() noexcept override { return 1; }
    char const* getMetadataString() noexcept override { return nullptr; }
    int32_t setTactic(int32_t) noexcept override { return 0; }
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override { return 0; }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        fields_.clear(); fields_.emplace_back("window", &window, PluginFieldType::kINT32, 1); fields_.emplace_back("heads", &heads, PluginFieldType::kINT32, 1); fields_.emplace_back("scale", &scale, PluginFieldType::kFLOAT32, 1); fields_.emplace_back("hd", &hd, PluginFieldType::kINT32, 1);
        fc_.nbFields = (int32_t)fields_.size(); fc_.fields = fields_.data(); return &fc_;
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const*, void const* const* inputs, void* const* outputs, void*, cudaStream_t st) noexcept override {
        int64_t M = 1; for (int i = 0; i < in[0].dims.nbDims - 1; i++) M *= in[0].dims.d[i];
        const __half* q = (const __half*)inputs[0]; const __half* k = (const __half*)inputs[1]; const __half* v = (const __half*)inputs[2]; __half* o = (__half*)outputs[0];
        cudaError_t e;
        if (hd == 32) e = lqattn::launch_pp<64, false, 32>(q, k, v, o, (int)M, window, heads, scale, st);   // pipelined kernel (K11): BR=64 divides 576 and 5184
        else e = lqattn::launch_pp<64, false, 64>(q, k, v, o, (int)M, window, heads, scale, st);
        return e == cudaSuccess ? 0 : 1;
    }
private: std::vector<PluginField> fields_; PluginFieldCollection fc_{};
};
class LQAttnCreator : public IPluginCreatorV3One {
public:
    LQAttnCreator() { fields_ = {PluginField("window", nullptr, PluginFieldType::kINT32, 1), PluginField("heads", nullptr, PluginFieldType::kINT32, 1), PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1), PluginField("hd", nullptr, PluginFieldType::kINT32, 1)}; fc_.nbFields = 4; fc_.fields = fields_.data(); }
    char const* getPluginName() const noexcept override { return kAttnName; }
    char const* getPluginVersion() const noexcept override { return kAttnVersion; }
    char const* getPluginNamespace() const noexcept override { return kAttnNs; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        int d = 64, n = 576, h = 16; float s = 0.125f;
        for (int i = 0; i < fc->nbFields; i++) { std::string nm = fc->fields[i].name; if (nm == "window") n = *(const int32_t*)fc->fields[i].data; else if (nm == "heads") h = *(const int32_t*)fc->fields[i].data; else if (nm == "scale") s = *(const float*)fc->fields[i].data; else if (nm == "hd") d = *(const int32_t*)fc->fields[i].data; }
        return new LQAttnPlugin(n, h, s, d);
    }
private: std::vector<PluginField> fields_; PluginFieldCollection fc_{};
};
static LQAttnCreator gAttnCreator;
extern "C" IPluginCreatorInterface* lq_attn_creator() { return &gAttnCreator; }
