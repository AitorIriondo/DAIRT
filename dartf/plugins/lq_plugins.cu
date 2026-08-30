// LQ_STAGES: INT8 GEMM pipeline depth. 5 fits the 164 KB shared memory of sm80/sm87/sm90; sm86/sm89 (99 KB per block) need 3 (build.sh sets it).
#ifndef LQ_STAGES
#define LQ_STAGES 5
#endif
// TensorRT 10 IPluginV3 plugin "LQFc1Gelu": Y[M,N] fp16 = GELU( s_a * s_w[n] * (X_int8[M,K] . W_int8[K,N]) + b[n] )
// Input 0: INT8 tensor [..., K] (explicit-quantization: TensorRT passes its scale in PluginTensorDesc::scale).
// Attributes: codes  int8 [N,K] (row-major, K contiguous = column-major [K,N]),  wscale float [N],  bias float [N],
//             gelu   int (0 = erf, 1 = tanh approximation)
// Output 0: FP16 tensor [..., N].  Kernel: CUTLASS int8 tensor-op GEMM 128x256x64, 5 stages, EpilogueWithBroadcast
//   Z = act( (alpha*acc + C) * V ),  alpha = s_a, C = b[n]/s_w[n] (ldc=0 broadcast row, fp16), V = s_w[n] (fp32)
// nvcc -O3 -std=c++17 -arch=sm_87 -Xcompiler -fPIC -shared -I cutlass/include lq_fc1_plugin.cu -o lq_plugins.so -lnvinfer
#include <cuda.h>
typedef CUresult (*PFN_cuTensorMapEncodeTiled)(...);
typedef CUresult (*PFN_cuTensorMapEncodeIm2col)(...);
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/functional.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <vector>
#include <string>
#include <cstdio>
#include <mutex>

using namespace nvinfer1;

template <typename Act>
struct Fc1Kernel {
    using ElementC = cutlass::half_t;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<ElementC, int32_t, float, cutlass::half_t, cutlass::half_t, 8, Act, cutlass::multiplies<float>, false, float>;
    using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<
        int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor, ElementC, cutlass::layout::RowMajor,
        int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, LQ_STAGES>;
    static cutlass::Status run(int M, int N, int K, const int8_t* X, const int8_t* W, const cutlass::half_t* Crow, const float* V, cutlass::half_t* Y, float s_a, void* ws, cudaStream_t st) {
        typename EpilogueOp::Params ep(s_a, 1.0f);
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, ep, X, W, Crow, Y, (void*)V, nullptr,
            (int64_t)M * K, (int64_t)N * K, (int64_t)0, (int64_t)M * N, (int64_t)0, (int64_t)0, K, K, 0, N, 0, 0);
        Gemm gemm; cutlass::Status s = gemm.initialize(args, ws, st); if (s != cutlass::Status::kSuccess) return s; return gemm(st);
    }
    static size_t workspace(int M, int N, int K) {
        typename EpilogueOp::Params ep(1.f, 1.f);
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, ep, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            (int64_t)M * K, (int64_t)N * K, (int64_t)0, (int64_t)M * N, (int64_t)0, (int64_t)0, K, K, 0, N, 0, 0);
        return Gemm::get_workspace_size(args);
    }
};
using ErfK = Fc1Kernel<cutlass::epilogue::thread::GELU<float>>;
using TanhK = Fc1Kernel<cutlass::epilogue::thread::GELU_taylor<float>>;


// ---------------------------------------------------------------- LQGemmRope: Y = RoPE(s_a*s_w[n]*acc + b[n]) ----
#include <cutlass/numeric_conversion.h>
#include <map>
namespace lq {
template <int kElementsPerAccess_>
class RopeDequant {
public:
    using ElementOutput = cutlass::half_t; using ElementC = cutlass::half_t; using ElementAccumulator = int32_t; using ElementCompute = float;
    using ElementZ = cutlass::half_t; using ElementT = cutlass::half_t; using ElementVector = float;
    static int const kElementsPerAccess = kElementsPerAccess_; static int const kCount = kElementsPerAccess;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>; using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
    using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>; using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>; using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;
    using FragmentOutput = FragmentZ;
    static bool const kIsHeavy = false; static bool const kStoreZ = true; static bool const kStoreT = false; static bool const kIsSingleSource = true;
    struct Params { ElementCompute alpha; ElementCompute beta; ElementCompute const* alpha_ptr; ElementCompute const* beta_ptr;
        CUTLASS_HOST_DEVICE Params() : alpha(1), beta(1), alpha_ptr(nullptr), beta_ptr(nullptr) {}
        CUTLASS_HOST_DEVICE Params(ElementCompute a, ElementCompute b) : alpha(a), beta(b), alpha_ptr(nullptr), beta_ptr(nullptr) {} };
private: ElementCompute alpha_, beta_;
public:
    CUTLASS_HOST_DEVICE RopeDequant(Params const& p) { alpha_ = p.alpha_ptr ? *p.alpha_ptr : p.alpha; beta_ = p.beta_ptr ? *p.beta_ptr : p.beta; }
    CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
    CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
    CUTLASS_HOST_DEVICE void operator()(FragmentZ& frag_Z, FragmentT&, FragmentAccumulator const& AB, FragmentC const& frag_C, FragmentCompute const& V) const {
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess> acc2f; cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess> c2f;
        FragmentCompute a = acc2f(AB); FragmentCompute cs = c2f(frag_C); FragmentCompute y, z;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) { unsigned u = __float_as_uint(V[i]); float s = __half2float(__ushort_as_half((unsigned short)(u & 0xffffu))); float b = __half2float(__ushort_as_half((unsigned short)(u >> 16))); y[i] = alpha_ * a[i] * s + b; }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; i += 2) { float c = cs[i], sn = cs[i + 1]; z[i] = y[i] * c - y[i + 1] * sn; z[i + 1] = y[i + 1] * c + y[i] * sn; }
        cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> f2z; frag_Z = f2z(z);
    }
    CUTLASS_HOST_DEVICE void operator()(FragmentZ& frag_Z, FragmentT& frag_T, FragmentAccumulator const& AB, FragmentCompute const& V) const { FragmentC zero; zero.clear(); (*this)(frag_Z, frag_T, AB, zero, V); }
};
}
struct RopeKernel {
    using EpilogueOp = lq::RopeDequant<8>;
    using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor, cutlass::half_t, cutlass::layout::RowMajor,
        int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, LQ_STAGES>;
    static cutlass::Status run(int M, int N, int K, const int8_t* X, const int8_t* W, const cutlass::half_t* Ctab, const float* V, cutlass::half_t* Y, float s_a, void* ws, cudaStream_t st) {
        typename EpilogueOp::Params ep(s_a, 1.0f);
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, ep, X, W, Ctab, Y, (void*)V, nullptr, (int64_t)M * K, (int64_t)N * K, (int64_t)M * N, (int64_t)M * N, (int64_t)0, (int64_t)0, K, K, N, N, 0, 0);
        Gemm gemm; cutlass::Status s = gemm.initialize(args, ws, st); if (s != cutlass::Status::kSuccess) return s; return gemm(st);
    }
    static size_t workspace(int M, int N, int K) {
        typename EpilogueOp::Params ep(1.f, 1.f);
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, ep, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, (int64_t)M * K, (int64_t)N * K, (int64_t)M * N, (int64_t)M * N, (int64_t)0, (int64_t)0, K, K, N, N, 0, 0);
        return Gemm::get_workspace_size(args);
    }
};
// packed [M,N] cos/sin table expansion: C[m, n] = (n even ? cos : sin)[m % Ntok, n % 64]   (cos/sin tables are [Ntok,64] with equal pairs)
__global__ void expand_table(const float* cosT, const float* sinT, int Ntok, int M, int N, __half* out) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; if (idx >= (long)M * N) return; int m = (int)(idx / N), n = (int)(idx % N); int t = m % Ntok, d = n % 64;
    out[idx] = __float2half((n & 1) ? sinT[t * 64 + d] : cosT[t * 64 + d]);
}
static std::map<std::string, __half*> g_tables; static std::mutex g_tables_mu;
static __half* get_packed_table(const std::vector<float>& cosT, const std::vector<float>& sinT, int Ntok, int M, int N) {
    // key: table identity (size + a few samples) + M,N
    char key[256]; snprintf(key, sizeof key, "%d_%d_%d_%.6f_%.6f_%.6f_%.6f", Ntok, M, N, cosT[64 + 3], sinT[64 + 3], cosT[(Ntok - 1) * 64 + 61], sinT[(Ntok / 2) * 64 + 17]);
    std::lock_guard<std::mutex> lk(g_tables_mu); auto it = g_tables.find(key); if (it != g_tables.end()) return it->second;
    float *dc, *ds; cudaMalloc(&dc, cosT.size() * 4); cudaMalloc(&ds, sinT.size() * 4); cudaMemcpy(dc, cosT.data(), cosT.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(ds, sinT.data(), sinT.size() * 4, cudaMemcpyHostToDevice);
    __half* out; cudaMalloc(&out, (size_t)M * N * 2); long total = (long)M * N; expand_table<<<(unsigned)((total + 255) / 256), 256>>>(dc, ds, Ntok, M, N, out); cudaDeviceSynchronize(); cudaFree(dc); cudaFree(ds);
    g_tables[key] = out; return out;
}


namespace lq {
template <typename ElementZ_, typename Act, int kElementsPerAccess_>
class GeluDequantQ {   // Z = q( act(alpha*acc*s + b) ),  (s,b) packed per column in V (fp32 word = two halves); q = identity (fp16 out) or round(x/oscale) clamp (int8 out)
public:
    using ElementOutput = ElementZ_; using ElementC = ElementZ_; using ElementAccumulator = int32_t; using ElementCompute = float;
    using ElementZ = ElementZ_; using ElementT = ElementZ_; using ElementVector = float;
    static int const kElementsPerAccess = kElementsPerAccess_; static int const kCount = kElementsPerAccess;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>; using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
    using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>; using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>; using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;
    using FragmentOutput = FragmentZ;
    static bool const kIsHeavy = true; static bool const kStoreZ = true; static bool const kStoreT = false; static bool const kIsSingleSource = true;
    struct Params { ElementCompute alpha; ElementCompute beta; ElementCompute const* alpha_ptr; ElementCompute const* beta_ptr; ElementCompute inv_oscale;
        CUTLASS_HOST_DEVICE Params() : alpha(1), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr), inv_oscale(0) {}
        CUTLASS_HOST_DEVICE Params(ElementCompute a, ElementCompute inv_os) : alpha(a), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr), inv_oscale(inv_os) {} };
private: ElementCompute alpha_, inv_oscale_;
public:
    CUTLASS_HOST_DEVICE GeluDequantQ(Params const& p) : alpha_(p.alpha), inv_oscale_(p.inv_oscale) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const { return false; }
    CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
    CUTLASS_DEVICE void compute(FragmentCompute& z, FragmentAccumulator const& AB, FragmentCompute const& V) const {
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess> acc2f; FragmentCompute a = acc2f(AB); Act act;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) { unsigned u = __float_as_uint(V[i]); float s = __half2float(__ushort_as_half((unsigned short)(u & 0xffffu))); float b = __half2float(__ushort_as_half((unsigned short)(u >> 16)));
            float y = act(alpha_ * a[i] * s + b); z[i] = (inv_oscale_ > 0.f) ? fminf(fmaxf(rintf(y * inv_oscale_), -127.f), 127.f) : y; }
    }
    CUTLASS_DEVICE void operator()(FragmentZ& frag_Z, FragmentT&, FragmentAccumulator const& AB, FragmentC const&, FragmentCompute const& V) const { FragmentCompute z; compute(z, AB, V); cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> f2z; frag_Z = f2z(z); }
    CUTLASS_DEVICE void operator()(FragmentZ& frag_Z, FragmentT&, FragmentAccumulator const& AB, FragmentCompute const& V) const { FragmentCompute z; compute(z, AB, V); cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> f2z; frag_Z = f2z(z); }
};
}
template <typename ElementZ, typename Act>
struct Fc1KernelQ {
    using EpilogueOp = lq::GeluDequantQ<ElementZ, Act, 8>;
    using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor, ElementZ, cutlass::layout::RowMajor,
        int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, LQ_STAGES>;
    static cutlass::Status run(int M, int N, int K, const int8_t* X, const int8_t* W, const float* V, void* Y, float s_a, float inv_os, void* ws, cudaStream_t st) {
        typename EpilogueOp::Params ep(s_a, inv_os);
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, ep, X, W, nullptr, Y, (void*)V, nullptr, (int64_t)M * K, (int64_t)N * K, (int64_t)0, (int64_t)M * N, (int64_t)0, (int64_t)0, K, K, 0, N, 0, 0);
        Gemm gemm; cutlass::Status s = gemm.initialize(args, ws, st); if (s != cutlass::Status::kSuccess) return s; return gemm(st);
    }
    static size_t workspace(int M, int N, int K) {
        typename EpilogueOp::Params ep(1.f, 0.f);
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, ep, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, (int64_t)M * K, (int64_t)N * K, (int64_t)0, (int64_t)M * N, (int64_t)0, (int64_t)0, K, K, 0, N, 0, 0);
        return Gemm::get_workspace_size(args);
    }
};
using ErfF16 = Fc1KernelQ<cutlass::half_t, cutlass::epilogue::thread::GELU<float>>; using TanhF16 = Fc1KernelQ<cutlass::half_t, cutlass::epilogue::thread::GELU_taylor<float>>;
using ErfI8 = Fc1KernelQ<int8_t, cutlass::epilogue::thread::GELU<float>>; using TanhI8 = Fc1KernelQ<int8_t, cutlass::epilogue::thread::GELU_taylor<float>>;

static const char* kName = "LQFc1Gelu"; static const char* kVersion = "1"; static const char* kNamespace = "";

class LQFc1Plugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    // host copies of the parameters (owned); device copies created lazily
    std::vector<int8_t> codes; std::vector<float> wscale, bias; int N = 0, K = 0, gelu = 0; float in_scale = 0.f, ascale = 0.f, oscale = 0.f;
    int8_t* dW = nullptr; float* dV = nullptr; __half* dC = nullptr; float dC_scale = 0.f;
    LQFc1Plugin(std::vector<int8_t> c, std::vector<float> ws, std::vector<float> b, int n, int k, int g, float as, float os) : codes(std::move(c)), wscale(std::move(ws)), bias(std::move(b)), N(n), K(k), gelu(g), ascale(as), oscale(os) {}
    ~LQFc1Plugin() override { if (dW) cudaFree(dW); if (dV) cudaFree(dV); if (dC) cudaFree(dC); }
    // --- IPluginV3 ---
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
        if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (t == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    }
    IPluginV3* clone() noexcept override { auto* p = new LQFc1Plugin(codes, wscale, bias, N, K, gelu, ascale, oscale); p->in_scale = in_scale; return p; }
    // --- OneCore ---
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    // --- OneBuild ---
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override { in_scale = in[0].desc.scale; return 0; }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override {
        if (io[pos].desc.format != TensorFormat::kLINEAR) return false;
        return pos == 0 ? io[0].desc.type == DataType::kINT8 : io[1].desc.type == (oscale > 0.f ? DataType::kINT8 : DataType::kHALF);
    }
    int32_t getOutputDataTypes(DataType* out, int32_t, DataType const*, int32_t) const noexcept override { out[0] = oscale > 0.f ? DataType::kINT8 : DataType::kHALF; return 0; }
    int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t, IExprBuilder& eb) noexcept override {
        out[0] = in[0]; out[0].d[in[0].nbDims - 1] = eb.constant(N); return 0;
    }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
        int64_t M = 1; for (int i = 0; i < in[0].desc.dims.nbDims - 1; i++) M *= in[0].desc.dims.d[i] > 0 ? in[0].desc.dims.d[i] : 1;
        return ErfF16::workspace((int)M, N, K) + (64 << 10);
    }
    int32_t getValidTactics(int32_t*, int32_t) noexcept override { return 0; }
    int32_t getNbTactics() noexcept override { return 0; }
    char const* getTimingCacheID() noexcept override { return nullptr; }
    int32_t getFormatCombinationLimit() noexcept override { return 1; }
    char const* getMetadataString() noexcept override { return nullptr; }
    // --- OneRuntime ---
    int32_t setTactic(int32_t) noexcept override { return 0; }
    int32_t onShapeChange(PluginTensorDesc const* in, int32_t, PluginTensorDesc const*, int32_t) noexcept override { in_scale = in[0].scale; return 0; }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        fields_.clear();
        fields_.emplace_back("codes", codes.data(), PluginFieldType::kINT8, (int32_t)codes.size());
        fields_.emplace_back("wscale", wscale.data(), PluginFieldType::kFLOAT32, (int32_t)wscale.size());
        fields_.emplace_back("bias", bias.data(), PluginFieldType::kFLOAT32, (int32_t)bias.size());
        fields_.emplace_back("gelu", &gelu, PluginFieldType::kINT32, 1); fields_.emplace_back("ascale", &ascale, PluginFieldType::kFLOAT32, 1); fields_.emplace_back("oscale", &oscale, PluginFieldType::kFLOAT32, 1);
        fc_.nbFields = (int32_t)fields_.size(); fc_.fields = fields_.data(); return &fc_;
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const* out, void const* const* inputs, void* const* outputs, void* ws, cudaStream_t st) noexcept override {
        float s_a = ascale > 0.f ? ascale : (in[0].scale > 0 ? in[0].scale : in_scale);
        if (!dW) upload();
        int64_t M = 1; for (int i = 0; i < in[0].dims.nbDims - 1; i++) M *= in[0].dims.d[i];
        float inv_os = oscale > 0.f ? 1.f / oscale : 0.f; cutlass::Status s;
        if (oscale > 0.f) s = gelu ? TanhI8::run((int)M, N, K, (const int8_t*)inputs[0], dW, dV, outputs[0], s_a, inv_os, ws, st) : ErfI8::run((int)M, N, K, (const int8_t*)inputs[0], dW, dV, outputs[0], s_a, inv_os, ws, st);
        else s = gelu ? TanhF16::run((int)M, N, K, (const int8_t*)inputs[0], dW, dV, outputs[0], s_a, 0.f, ws, st) : ErfF16::run((int)M, N, K, (const int8_t*)inputs[0], dW, dV, outputs[0], s_a, 0.f, ws, st);
        return s == cutlass::Status::kSuccess ? 0 : 1;
    }
private:
    std::vector<PluginField> fields_; PluginFieldCollection fc_{};
    void upload() {
        cudaMalloc(&dW, codes.size()); cudaMemcpy(dW, codes.data(), codes.size(), cudaMemcpyHostToDevice);
        std::vector<unsigned> packed(N); for (int n = 0; n < N; n++) { unsigned short hs = __half_as_ushort(__float2half(wscale[n])), hb = __half_as_ushort(__float2half(bias[n])); packed[n] = (unsigned)hs | ((unsigned)hb << 16); }
        cudaMalloc(&dV, N * 4); cudaMemcpy(dV, packed.data(), N * 4, cudaMemcpyHostToDevice);
    }
};

class LQFc1Creator : public IPluginCreatorV3One {
public:
    LQFc1Creator() {
        fields_ = {PluginField("codes", nullptr, PluginFieldType::kINT8, 0), PluginField("wscale", nullptr, PluginFieldType::kFLOAT32, 0),
                   PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 0), PluginField("gelu", nullptr, PluginFieldType::kINT32, 1),
                   PluginField("ascale", nullptr, PluginFieldType::kFLOAT32, 1), PluginField("oscale", nullptr, PluginFieldType::kFLOAT32, 1)};
        fc_.nbFields = (int32_t)fields_.size(); fc_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        std::vector<int8_t> codes; std::vector<float> ws, b; int gelu = 0; float as = 0.f, os = 0.f;
        for (int i = 0; i < fc->nbFields; i++) {
            const PluginField& f = fc->fields[i]; std::string n = f.name;
            if (n == "codes") codes.assign((const int8_t*)f.data, (const int8_t*)f.data + f.length);
            else if (n == "wscale") ws.assign((const float*)f.data, (const float*)f.data + f.length);
            else if (n == "bias") b.assign((const float*)f.data, (const float*)f.data + f.length);
            else if (n == "gelu") gelu = *(const int32_t*)f.data;
            else if (n == "ascale") as = *(const float*)f.data; else if (n == "oscale") os = *(const float*)f.data;
        }
        if (codes.empty() || ws.empty() || codes.size() % ws.size() != 0) { fprintf(stderr, "LQFc1Gelu: bad fields (codes %zu, wscale %zu)\n", codes.size(), ws.size()); return nullptr; }
        int N = (int)ws.size(), K = (int)(codes.size() / ws.size());
        return new LQFc1Plugin(std::move(codes), std::move(ws), std::move(b), N, K, gelu, as, os);
    }
private:
    std::vector<PluginField> fields_; PluginFieldCollection fc_{};
};

static LQFc1Creator gCreator;
static const char* kRopeName = "LQGemmRope";
class LQRopePlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    std::vector<int8_t> codes; std::vector<float> wscale, bias, cosT, sinT; int N = 0, K = 0, Ntok = 0; float in_scale = 0.f, ascale = 0.f;
    int8_t* dW = nullptr; float* dV = nullptr; __half* dC = nullptr; int dC_M = 0;
    LQRopePlugin(std::vector<int8_t> c, std::vector<float> ws, std::vector<float> b, std::vector<float> ct, std::vector<float> st, int n, int k, int ntok)
        : codes(std::move(c)), wscale(std::move(ws)), bias(std::move(b)), cosT(std::move(ct)), sinT(std::move(st)), N(n), K(k), Ntok(ntok) {}
    ~LQRopePlugin() override { if (dW) cudaFree(dW); if (dV) cudaFree(dV); }
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
        if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (t == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    }
    IPluginV3* clone() noexcept override { auto* p = new LQRopePlugin(codes, wscale, bias, cosT, sinT, N, K, Ntok); p->in_scale = in_scale; p->ascale = ascale; return p; }
    char const* getPluginName() const noexcept override { return kRopeName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override { in_scale = in[0].desc.scale; return 0; }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override {
        if (io[pos].desc.format != TensorFormat::kLINEAR) return false; return pos == 0 ? io[0].desc.type == DataType::kINT8 : io[1].desc.type == DataType::kHALF;
    }
    int32_t getOutputDataTypes(DataType* out, int32_t, DataType const*, int32_t) const noexcept override { out[0] = DataType::kHALF; return 0; }
    int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t, IExprBuilder& eb) noexcept override { out[0] = in[0]; out[0].d[in[0].nbDims - 1] = eb.constant(N); return 0; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
        int64_t M = 1; for (int i = 0; i < in[0].desc.dims.nbDims - 1; i++) M *= in[0].desc.dims.d[i] > 0 ? in[0].desc.dims.d[i] : 1; return RopeKernel::workspace((int)M, N, K) + (64 << 10);
    }
    int32_t getValidTactics(int32_t*, int32_t) noexcept override { return 0; }
    int32_t getNbTactics() noexcept override { return 0; }
    char const* getTimingCacheID() noexcept override { return nullptr; }
    int32_t getFormatCombinationLimit() noexcept override { return 1; }
    char const* getMetadataString() noexcept override { return nullptr; }
    int32_t setTactic(int32_t) noexcept override { return 0; }
    int32_t onShapeChange(PluginTensorDesc const* in, int32_t, PluginTensorDesc const*, int32_t) noexcept override { in_scale = in[0].scale; return 0; }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        fields_.clear();
        fields_.emplace_back("codes", codes.data(), PluginFieldType::kINT8, (int32_t)codes.size()); fields_.emplace_back("wscale", wscale.data(), PluginFieldType::kFLOAT32, (int32_t)wscale.size());
        fields_.emplace_back("bias", bias.data(), PluginFieldType::kFLOAT32, (int32_t)bias.size()); fields_.emplace_back("cos", cosT.data(), PluginFieldType::kFLOAT32, (int32_t)cosT.size());
        fields_.emplace_back("sin", sinT.data(), PluginFieldType::kFLOAT32, (int32_t)sinT.size()); fields_.emplace_back("ascale", &ascale, PluginFieldType::kFLOAT32, 1);
        fc_.nbFields = (int32_t)fields_.size(); fc_.fields = fields_.data(); return &fc_;
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const*, void const* const* inputs, void* const* outputs, void* ws, cudaStream_t st) noexcept override {
        float s_a = ascale > 0.f ? ascale : (in[0].scale > 0 ? in[0].scale : in_scale); int64_t M = 1; for (int i = 0; i < in[0].dims.nbDims - 1; i++) M *= in[0].dims.d[i];
        if (!dW) upload(); if (!dC || dC_M != (int)M) { dC = get_packed_table(cosT, sinT, Ntok, (int)M, N); dC_M = (int)M; }
        cutlass::Status s = RopeKernel::run((int)M, N, K, (const int8_t*)inputs[0], dW, (const cutlass::half_t*)dC, dV, (cutlass::half_t*)outputs[0], s_a, ws, st);
        return s == cutlass::Status::kSuccess ? 0 : 1;
    }
private:
    std::vector<PluginField> fields_; PluginFieldCollection fc_{};
    void upload() {
        cudaMalloc(&dW, codes.size()); cudaMemcpy(dW, codes.data(), codes.size(), cudaMemcpyHostToDevice);
        std::vector<unsigned> packed(N); for (int n = 0; n < N; n++) { unsigned short hs = __half_as_ushort(__float2half(wscale[n])), hb = __half_as_ushort(__float2half(bias[n])); packed[n] = (unsigned)hs | ((unsigned)hb << 16); }
        cudaMalloc(&dV, N * 4); cudaMemcpy(dV, packed.data(), N * 4, cudaMemcpyHostToDevice);
    }
};
class LQRopeCreator : public IPluginCreatorV3One {
public:
    LQRopeCreator() {
        fields_ = {PluginField("codes", nullptr, PluginFieldType::kINT8, 0), PluginField("wscale", nullptr, PluginFieldType::kFLOAT32, 0), PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 0),
                   PluginField("cos", nullptr, PluginFieldType::kFLOAT32, 0), PluginField("sin", nullptr, PluginFieldType::kFLOAT32, 0), PluginField("ascale", nullptr, PluginFieldType::kFLOAT32, 1)};
        fc_.nbFields = (int32_t)fields_.size(); fc_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override { return kRopeName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        std::vector<int8_t> codes; std::vector<float> ws, b, ct, st; float as_ = 0.f;
        for (int i = 0; i < fc->nbFields; i++) { const PluginField& f = fc->fields[i]; std::string n = f.name;
            if (n == "codes") codes.assign((const int8_t*)f.data, (const int8_t*)f.data + f.length); else if (n == "wscale") ws.assign((const float*)f.data, (const float*)f.data + f.length);
            else if (n == "bias") b.assign((const float*)f.data, (const float*)f.data + f.length); else if (n == "cos") ct.assign((const float*)f.data, (const float*)f.data + f.length);
            else if (n == "sin") st.assign((const float*)f.data, (const float*)f.data + f.length); else if (n == "ascale") as_ = *(const float*)f.data; }
        if (codes.empty() || ws.empty() || ct.empty() || ct.size() % 64 != 0 || codes.size() % ws.size() != 0) { fprintf(stderr, "LQGemmRope: bad fields\n"); return nullptr; }
        int N = (int)ws.size(), K = (int)(codes.size() / ws.size()), Ntok = (int)(ct.size() / 64);
        auto* p = new LQRopePlugin(std::move(codes), std::move(ws), std::move(b), std::move(ct), std::move(st), N, K, Ntok); p->ascale = as_; return p;
    }
private: std::vector<PluginField> fields_; PluginFieldCollection fc_{};
};
static LQRopeCreator gRopeCreator;

extern "C" IPluginCreatorInterface* lq_memattn_creator();
extern "C" {
// TensorRT looks for this symbol when a plugin library is passed via --plugins / getPluginRegistry()->loadLibrary()
IPluginCreatorInterface* lq_attn_creator();
IPluginCreatorInterface* lq_mlp_creator();
IPluginCreatorInterface* lq_qkv_creator();
IPluginCreatorInterface* lq_attnproj_creator();
IPluginCreatorInterface* lq_encsa_creator();
IPluginCreatorInterface* lq_encffn_creator();
void setLoggerFinder(ILoggerFinder*) {}   // required for IPluginRegistry::loadLibrary / trtexec --dynamicPlugins
IPluginCreatorInterface* const* getCreators(int32_t& nbCreators) { nbCreators = 8; static IPluginCreatorInterface* list[8] = {&gCreator, &gRopeCreator, lq_attn_creator(), lq_mlp_creator(), lq_qkv_creator(), lq_attnproj_creator(), lq_encsa_creator(), lq_encffn_creator()}; return list; }
// explicit registration for ctypes users (call after loading; never at static-init time)
void lq_register() { auto* reg = getPluginRegistry(); if (reg) { reg->registerCreator(*static_cast<IPluginCreatorV3One*>(lq_memattn_creator()), kNamespace); reg->registerCreator(gCreator, kNamespace); reg->registerCreator(gRopeCreator, kNamespace); reg->registerCreator(*static_cast<IPluginCreatorV3One*>(lq_attn_creator()), kNamespace); reg->registerCreator(*static_cast<IPluginCreatorV3One*>(lq_mlp_creator()), kNamespace); reg->registerCreator(*static_cast<IPluginCreatorV3One*>(lq_qkv_creator()), kNamespace); reg->registerCreator(*static_cast<IPluginCreatorV3One*>(lq_attnproj_creator()), kNamespace); reg->registerCreator(*static_cast<IPluginCreatorV3One*>(lq_encsa_creator()), kNamespace); reg->registerCreator(*static_cast<IPluginCreatorV3One*>(lq_encffn_creator()), kNamespace); } }
}
