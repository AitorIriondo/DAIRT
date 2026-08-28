// Grounding (DETR-style) encoder-layer plugins for SAM3's grounding engine, fp16 (stage 1: fusion/layout only).
//   LQEncSelfAttn(x, pos) -> x' :  n = LN(x); qk = (n+pos)·[Wq|Wk] + [bq|bk];  v = n·Wv + bv;  a = attn(q, k, v; H=8, d=32, N tokens);
//                                   x' = x + a·Wo + bo            (x, pos, x' fp16 [N, 1, C=256], token-major since B = 1)
//   LQEncFfn(x) -> x' :            x' = x + relu(LN(x)·W1 + b1)·W2 + b2
// GEMMs via cuBLASLt fp16 (bias / relu-bias epilogues, residual through beta=1 C); attention via lq_attn_kernel (D=32).
#include "lq_attn_kernel.cuh"
#include <cublasLt.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
using namespace nvinfer1;

namespace lqg {
// LayerNorm (affine) fp16 -> fp16, optional + pos into a second output; one warp per row, C in {256, 1024}
template <int C>
__global__ void ln_kernel(const __half* __restrict__ x, const __half* __restrict__ pos, __half* __restrict__ n_out, __half* __restrict__ npos_out, const float* __restrict__ g, const float* __restrict__ b, float eps, int M) {
    constexpr int PER = C / 32; int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31; if (row >= M) return;
    const __half* xr = x + (size_t)row * C; float v[PER]; float s = 0.f;
    #pragma unroll
    for (int i = 0; i < PER / 2; i++) { float2 f = __half22float2(*reinterpret_cast<const __half2*>(xr + (lane + 32 * i) * 2)); v[2 * i] = f.x; v[2 * i + 1] = f.y; s += f.x + f.y; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    float mean = s / C, ss = 0.f;
    #pragma unroll
    for (int i = 0; i < PER; i++) { float d = v[i] - mean; ss += d * d; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    float rstd = rsqrtf(ss / C + eps);
    #pragma unroll
    for (int i = 0; i < PER / 2; i++) {
        int c = (lane + 32 * i) * 2; float a0 = (v[2 * i] - mean) * rstd * g[c] + b[c], a1 = (v[2 * i + 1] - mean) * rstd * g[c + 1] + b[c + 1];
        __half2 h = __floats2half2_rn(a0, a1); *reinterpret_cast<__half2*>(n_out + (size_t)row * C + c) = h;
        if (npos_out) { float2 p = __half22float2(*reinterpret_cast<const __half2*>(pos + (size_t)row * C + c)); *reinterpret_cast<__half2*>(npos_out + (size_t)row * C + c) = __floats2half2_rn(a0 + p.x, a1 + p.y); }
    }
}
// Y[M,N] (fp16, row-major, ldY) = X[M,K] (ldX) · W[K,N] (row-major) + bias[N] (+ relu) (+ C[M,N] residual, ldC)
struct Lt {
    cublasLtHandle_t h = nullptr; void* ws = nullptr; size_t wsz = 32 << 20;
    void init() { if (!h) { cublasLtCreate(&h); cudaMalloc(&ws, wsz); } }
    bool gemm(int M, int N, int K, const __half* X, int ldX, const __half* W, const __half* bias, bool relu, const __half* C, int ldC, __half* Y, int ldY, cudaStream_t st) {
        init(); cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t lA, lB, lC, lD; cublasLtMatmulPreference_t pref;
        cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t tn = CUBLAS_OP_N; cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tn, sizeof tn); cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof tn);
        cublasLtEpilogue_t ep = relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS; cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof ep);
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias);
        // column-major view: Y^T[N,M] = W^T[N,K] · X^T[K,M]
        cublasLtMatrixLayoutCreate(&lA, CUDA_R_16F, N, K, N); cublasLtMatrixLayoutCreate(&lB, CUDA_R_16F, K, M, ldX); cublasLtMatrixLayoutCreate(&lC, CUDA_R_16F, N, M, C ? ldC : ldY); cublasLtMatrixLayoutCreate(&lD, CUDA_R_16F, N, M, ldY);
        cublasLtMatmulPreferenceCreate(&pref); cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz);
        cublasLtMatmulHeuristicResult_t heur; int nh = 0; cublasLtMatmulAlgoGetHeuristic(h, op, lA, lB, lC, lD, pref, 1, &heur, &nh);
        float alpha = 1.f, beta = C ? 1.f : 0.f; cublasStatus_t s = CUBLAS_STATUS_NOT_SUPPORTED;
        if (nh > 0) s = cublasLtMatmul(h, op, &alpha, W, lA, X, lB, &beta, C ? C : Y, lC, Y, lD, &heur.algo, ws, wsz, st);
        cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(lA); cublasLtMatrixLayoutDestroy(lB); cublasLtMatrixLayoutDestroy(lC); cublasLtMatrixLayoutDestroy(lD); cublasLtMatmulDescDestroy(op);
        return s == CUBLAS_STATUS_SUCCESS;
    }
};
struct HW { std::vector<__half> w, b; __half* dW = nullptr; __half* dB = nullptr; void up() { cudaMalloc(&dW, w.size() * 2); cudaMemcpy(dW, w.data(), w.size() * 2, cudaMemcpyHostToDevice); cudaMalloc(&dB, b.size() * 2); cudaMemcpy(dB, b.data(), b.size() * 2, cudaMemcpyHostToDevice); } void free() { if (dW) cudaFree(dW); if (dB) cudaFree(dB); dW = dB = nullptr; } };
static std::vector<__half> to_half(const float* p, int n) { std::vector<__half> v(n); for (int i = 0; i < n; i++) v[i] = __float2half(p[i]); return v; }
}  // namespace lqg

#define LQG_COMMON(NIN_SHAPE)                                                                                                    \
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {                                        \
        if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);                                     \
        if (t == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);                                 \
        return static_cast<IPluginV3OneCore*>(this); }                                                                           \
    int32_t getNbOutputs() const noexcept override { return 1; }                                                                \
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; } \
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override { return io[pos].desc.format == TensorFormat::kLINEAR && io[pos].desc.type == DataType::kHALF; } \
    int32_t getOutputDataTypes(DataType* out, int32_t, DataType const*, int32_t) const noexcept override { out[0] = DataType::kHALF; return 0; } \
    int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t, IExprBuilder&) noexcept override { out[0] = in[NIN_SHAPE]; return 0; } \
    int32_t getValidTactics(int32_t*, int32_t) noexcept override { return 0; }                                                  \
    int32_t getNbTactics() noexcept override { return 0; }                                                                       \
    char const* getTimingCacheID() noexcept override { return nullptr; }                                                         \
    int32_t getFormatCombinationLimit() noexcept override { return 1; }                                                          \
    char const* getMetadataString() noexcept override { return nullptr; }                                                        \
    int32_t setTactic(int32_t) noexcept override { return 0; }                                                                   \
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override { return 0; }    \
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }                                     \
    char const* getPluginVersion() const noexcept override { return "1"; }                                                        \
    char const* getPluginNamespace() const noexcept override { return ""; }

// ------------------------------------------------------------ LQEncSelfAttn -----------------------------------------------------------
struct SaParams { int C = 256, H = 8, N = 5184; float eps = 1e-5f, scale = 0.176776695f; std::vector<float> g, b, wqk, bqk, wv, bv, wo, bo; };
class LQEncSaPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    SaParams p; lqg::HW qk, v, o; float* dG = nullptr; float* dB = nullptr; lqg::Lt lt;
    explicit LQEncSaPlugin(SaParams pp) : p(std::move(pp)) {}
    ~LQEncSaPlugin() override { qk.free(); v.free(); o.free(); if (dG) cudaFree(dG); if (dB) cudaFree(dB); }
    LQG_COMMON(0)
    IPluginV3* clone() noexcept override { return new LQEncSaPlugin(p); }
    char const* getPluginName() const noexcept override { return "LQEncSelfAttn"; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
        int64_t M = 1; for (int i = 0; i < in[0].desc.dims.nbDims - 1; i++) M *= in[0].desc.dims.d[i] > 0 ? in[0].desc.dims.d[i] : 1;
        return (size_t)M * p.C * 2 * 5 + (1 << 20);   // n, n+pos, qk (2C), v, attn out
    }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        f_.clear(); f_.emplace_back("C", &p.C, PluginFieldType::kINT32, 1); f_.emplace_back("H", &p.H, PluginFieldType::kINT32, 1); f_.emplace_back("N", &p.N, PluginFieldType::kINT32, 1);
        f_.emplace_back("eps", &p.eps, PluginFieldType::kFLOAT32, 1); f_.emplace_back("scale", &p.scale, PluginFieldType::kFLOAT32, 1);
        const char* nm[8] = {"g", "b", "wqk", "bqk", "wv", "bv", "wo", "bo"}; std::vector<float>* vv[8] = {&p.g, &p.b, &p.wqk, &p.bqk, &p.wv, &p.bv, &p.wo, &p.bo};
        for (int i = 0; i < 8; i++) f_.emplace_back(nm[i], vv[i]->data(), PluginFieldType::kFLOAT32, (int32_t)vv[i]->size());
        fc_.nbFields = (int32_t)f_.size(); fc_.fields = f_.data(); return &fc_;
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const*, void const* const* inputs, void* const* outputs, void* ws, cudaStream_t st) noexcept override {
        int64_t M = 1; for (int i = 0; i < in[0].dims.nbDims - 1; i++) M *= in[0].dims.d[i];
        if (!qk.dW) { qk.w = lqg::to_half(p.wqk.data(), (int)p.wqk.size()); qk.b = lqg::to_half(p.bqk.data(), (int)p.bqk.size()); qk.up(); v.w = lqg::to_half(p.wv.data(), (int)p.wv.size()); v.b = lqg::to_half(p.bv.data(), (int)p.bv.size()); v.up(); o.w = lqg::to_half(p.wo.data(), (int)p.wo.size()); o.b = lqg::to_half(p.bo.data(), (int)p.bo.size()); o.up();
            cudaMalloc(&dG, p.C * 4); cudaMemcpy(dG, p.g.data(), p.C * 4, cudaMemcpyHostToDevice); cudaMalloc(&dB, p.C * 4); cudaMemcpy(dB, p.b.data(), p.C * 4, cudaMemcpyHostToDevice); }
        const __half* x = (const __half*)inputs[0]; const __half* pos = (const __half*)inputs[1]; __half* y = (__half*)outputs[0];
        __half* n = (__half*)ws; __half* npos = n + (size_t)M * p.C; __half* qkb = npos + (size_t)M * p.C; __half* vb = qkb + (size_t)M * 2 * p.C; __half* ab = vb + (size_t)M * p.C;
        if (p.C == 256) lqg::ln_kernel<256><<<(unsigned)((M + 7) / 8), 256, 0, st>>>(x, pos, n, npos, dG, dB, p.eps, (int)M); else return 1;
        if (!lt.gemm((int)M, 2 * p.C, p.C, npos, p.C, qk.dW, qk.dB, false, nullptr, 0, qkb, 2 * p.C, st)) return 1;
        if (!lt.gemm((int)M, p.C, p.C, n, p.C, v.dW, v.dB, false, nullptr, 0, vb, p.C, st)) return 1;
        cudaError_t e = lqattn::launch_pp<64, false, 32>(qkb, qkb + p.C, vb, ab, (int)M, p.N, p.H, p.scale, st, 0.f, 2 * p.C, p.C, p.C);
        if (e != cudaSuccess) return 1;
        return lt.gemm((int)M, p.C, p.C, ab, p.C, o.dW, o.dB, false, x, p.C, y, p.C, st) ? 0 : 1;
    }
private: std::vector<PluginField> f_; PluginFieldCollection fc_{};
};
class LQEncSaCreator : public IPluginCreatorV3One {
public:
    LQEncSaCreator() { const char* nm[13] = {"C", "H", "N", "eps", "scale", "g", "b", "wqk", "bqk", "wv", "bv", "wo", "bo"}; for (int i = 0; i < 13; i++) f_.emplace_back(nm[i], nullptr, i < 3 ? PluginFieldType::kINT32 : PluginFieldType::kFLOAT32, 0); fc_.nbFields = 13; fc_.fields = f_.data(); }
    char const* getPluginName() const noexcept override { return "LQEncSelfAttn"; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        SaParams p;
        for (int i = 0; i < fc->nbFields; i++) { const PluginField& f = fc->fields[i]; std::string n = f.name; const float* fd = (const float*)f.data;
            if (n == "C") p.C = *(const int32_t*)f.data; else if (n == "H") p.H = *(const int32_t*)f.data; else if (n == "N") p.N = *(const int32_t*)f.data; else if (n == "eps") p.eps = *fd; else if (n == "scale") p.scale = *fd;
            else { std::vector<float>* tgt = n == "g" ? &p.g : n == "b" ? &p.b : n == "wqk" ? &p.wqk : n == "bqk" ? &p.bqk : n == "wv" ? &p.wv : n == "bv" ? &p.bv : n == "wo" ? &p.wo : n == "bo" ? &p.bo : nullptr; if (tgt) tgt->assign(fd, fd + f.length); } }
        if (p.wqk.size() != (size_t)p.C * 2 * p.C || p.wv.size() != (size_t)p.C * p.C || p.wo.size() != (size_t)p.C * p.C) { fprintf(stderr, "LQEncSelfAttn: bad fields\n"); return nullptr; }
        return new LQEncSaPlugin(std::move(p));
    }
private: std::vector<PluginField> f_; PluginFieldCollection fc_{};
};
static LQEncSaCreator gEncSa;
extern "C" IPluginCreatorInterface* lq_encsa_creator() { return &gEncSa; }

// ------------------------------------------------------------ LQEncFfn -----------------------------------------------------------------
struct FfnParams { int C = 256, F = 2048; float eps = 1e-5f; std::vector<float> g, b, w1, b1, w2, b2; };
class LQEncFfnPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    FfnParams p; lqg::HW l1, l2; float* dG = nullptr; float* dB = nullptr; lqg::Lt lt;
    explicit LQEncFfnPlugin(FfnParams pp) : p(std::move(pp)) {}
    ~LQEncFfnPlugin() override { l1.free(); l2.free(); if (dG) cudaFree(dG); if (dB) cudaFree(dB); }
    LQG_COMMON(0)
    IPluginV3* clone() noexcept override { return new LQEncFfnPlugin(p); }
    char const* getPluginName() const noexcept override { return "LQEncFfn"; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
        int64_t M = 1; for (int i = 0; i < in[0].desc.dims.nbDims - 1; i++) M *= in[0].desc.dims.d[i] > 0 ? in[0].desc.dims.d[i] : 1;
        return (size_t)M * (p.C + p.F) * 2 + (1 << 20);
    }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        f_.clear(); f_.emplace_back("C", &p.C, PluginFieldType::kINT32, 1); f_.emplace_back("F", &p.F, PluginFieldType::kINT32, 1); f_.emplace_back("eps", &p.eps, PluginFieldType::kFLOAT32, 1);
        const char* nm[6] = {"g", "b", "w1", "b1", "w2", "b2"}; std::vector<float>* vv[6] = {&p.g, &p.b, &p.w1, &p.b1, &p.w2, &p.b2};
        for (int i = 0; i < 6; i++) f_.emplace_back(nm[i], vv[i]->data(), PluginFieldType::kFLOAT32, (int32_t)vv[i]->size());
        fc_.nbFields = (int32_t)f_.size(); fc_.fields = f_.data(); return &fc_;
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const*, void const* const* inputs, void* const* outputs, void* ws, cudaStream_t st) noexcept override {
        int64_t M = 1; for (int i = 0; i < in[0].dims.nbDims - 1; i++) M *= in[0].dims.d[i];
        if (!l1.dW) { l1.w = lqg::to_half(p.w1.data(), (int)p.w1.size()); l1.b = lqg::to_half(p.b1.data(), (int)p.b1.size()); l1.up(); l2.w = lqg::to_half(p.w2.data(), (int)p.w2.size()); l2.b = lqg::to_half(p.b2.data(), (int)p.b2.size()); l2.up();
            cudaMalloc(&dG, p.C * 4); cudaMemcpy(dG, p.g.data(), p.C * 4, cudaMemcpyHostToDevice); cudaMalloc(&dB, p.C * 4); cudaMemcpy(dB, p.b.data(), p.C * 4, cudaMemcpyHostToDevice); }
        const __half* x = (const __half*)inputs[0]; __half* y = (__half*)outputs[0]; __half* n = (__half*)ws; __half* hbuf = n + (size_t)M * p.C;
        if (p.C == 256) lqg::ln_kernel<256><<<(unsigned)((M + 7) / 8), 256, 0, st>>>(x, nullptr, n, nullptr, dG, dB, p.eps, (int)M); else return 1;
        if (!lt.gemm((int)M, p.F, p.C, n, p.C, l1.dW, l1.dB, true, nullptr, 0, hbuf, p.F, st)) return 1;
        return lt.gemm((int)M, p.C, p.F, hbuf, p.F, l2.dW, l2.dB, false, x, p.C, y, p.C, st) ? 0 : 1;
    }
private: std::vector<PluginField> f_; PluginFieldCollection fc_{};
};
class LQEncFfnCreator : public IPluginCreatorV3One {
public:
    LQEncFfnCreator() { const char* nm[9] = {"C", "F", "eps", "g", "b", "w1", "b1", "w2", "b2"}; for (int i = 0; i < 9; i++) f_.emplace_back(nm[i], nullptr, i < 2 ? PluginFieldType::kINT32 : PluginFieldType::kFLOAT32, 0); fc_.nbFields = 9; fc_.fields = f_.data(); }
    char const* getPluginName() const noexcept override { return "LQEncFfn"; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        FfnParams p;
        for (int i = 0; i < fc->nbFields; i++) { const PluginField& f = fc->fields[i]; std::string n = f.name; const float* fd = (const float*)f.data;
            if (n == "C") p.C = *(const int32_t*)f.data; else if (n == "F") p.F = *(const int32_t*)f.data; else if (n == "eps") p.eps = *fd;
            else { std::vector<float>* tgt = n == "g" ? &p.g : n == "b" ? &p.b : n == "w1" ? &p.w1 : n == "b1" ? &p.b1 : n == "w2" ? &p.w2 : n == "b2" ? &p.b2 : nullptr; if (tgt) tgt->assign(fd, fd + f.length); } }
        if (p.w1.size() != (size_t)p.C * p.F || p.w2.size() != (size_t)p.F * p.C) { fprintf(stderr, "LQEncFfn: bad fields\n"); return nullptr; }
        return new LQEncFfnPlugin(std::move(p));
    }
private: std::vector<PluginField> f_; PluginFieldCollection fc_{};
};
static LQEncFfnCreator gEncFfn;
extern "C" IPluginCreatorInterface* lq_encffn_creator() { return &gEncFfn; }
