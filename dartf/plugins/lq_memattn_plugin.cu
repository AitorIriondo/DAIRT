// TensorRT 10 IPluginV3 plugin "LQMemAttn": single-head D=256 attention with per-batch valid key counts (SAM 3 tracker memory attention).
//   inputs : q fp16 [Bq, Qn, 256] (Bq = 1 broadcasts), k fp16 [B, Kmax, 256], v fp16 [B, Kmax, 256], nvalid int32 [B]
//   output : o fp16 [B, Qn, 256]
//   attrs  : scale (float), cfg (int, -1 = pick by device)
#include "lq_memattn_kernel.cuh"
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <vector>
#include <string>
using namespace nvinfer1;
static const char* kMemName = "LQMemAttn"; static const char* kMemVersion = "1"; static const char* kMemNs = "";
class LQMemAttnPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    float scale = 0.0625f; int cfg = -1;
    LQMemAttnPlugin(float s, int c) : scale(s), cfg(c) {}
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
        if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (t == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    }
    IPluginV3* clone() noexcept override { return new LQMemAttnPlugin(scale, cfg); }
    char const* getPluginName() const noexcept override { return kMemName; }
    char const* getPluginVersion() const noexcept override { return kMemVersion; }
    char const* getPluginNamespace() const noexcept override { return kMemNs; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override {
        if (io[pos].desc.format != TensorFormat::kLINEAR) return false;
        return pos == 3 ? io[pos].desc.type == DataType::kINT32 : io[pos].desc.type == DataType::kHALF;
    }
    int32_t getOutputDataTypes(DataType* out, int32_t, DataType const*, int32_t) const noexcept override { out[0] = DataType::kHALF; return 0; }
    int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t, IExprBuilder&) noexcept override {
        out[0].nbDims = 3; out[0].d[0] = in[1].d[0]; out[0].d[1] = in[0].d[1]; out[0].d[2] = in[0].d[2]; return 0;
    }
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
        fields_.clear(); fields_.emplace_back("scale", &scale, PluginFieldType::kFLOAT32, 1); fields_.emplace_back("cfg", &cfg, PluginFieldType::kINT32, 1);
        fc_.nbFields = (int32_t)fields_.size(); fc_.fields = fields_.data(); return &fc_;
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const*, void const* const* inputs, void* const* outputs, void*, cudaStream_t st) noexcept override {
        const int Bq = (int)in[0].dims.d[0], Qn = (int)in[0].dims.d[1], B = (int)in[1].dims.d[0], Kmax = (int)in[1].dims.d[1];
        if (in[0].dims.d[2] != 256 || in[1].dims.d[2] != 256) return 1;
        cudaError_t e = lqmem::launch_auto((const __half*)inputs[0], (const __half*)inputs[1], (const __half*)inputs[2], (__half*)outputs[0], (const int*)inputs[3], B, Bq, Qn, Kmax, scale, st, cfg);
        return e == cudaSuccess ? 0 : 1;
    }
private: std::vector<PluginField> fields_; PluginFieldCollection fc_{};
};
class LQMemAttnCreator : public IPluginCreatorV3One {
public:
    LQMemAttnCreator() { fields_ = {PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1), PluginField("cfg", nullptr, PluginFieldType::kINT32, 1)}; fc_.nbFields = 2; fc_.fields = fields_.data(); }
    char const* getPluginName() const noexcept override { return kMemName; }
    char const* getPluginVersion() const noexcept override { return kMemVersion; }
    char const* getPluginNamespace() const noexcept override { return kMemNs; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        float s = 0.0625f; int c = -1;
        for (int i = 0; i < fc->nbFields; i++) { std::string nm = fc->fields[i].name; if (nm == "scale") s = *(const float*)fc->fields[i].data; else if (nm == "cfg") c = *(const int32_t*)fc->fields[i].data; }
        return new LQMemAttnPlugin(s, c);
    }
private: std::vector<PluginField> fields_; PluginFieldCollection fc_{};
};
static LQMemAttnCreator gMemAttnCreator;
extern "C" IPluginCreatorInterface* lq_memattn_creator() { return &gMemAttnCreator; }
