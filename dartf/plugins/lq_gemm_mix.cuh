// LQ_STAGES: INT8 GEMM pipeline depth. 5 fits the 164 KB shared memory of sm80/sm87/sm90; sm86/sm89 (99 KB per block) need 3 (build.sh sets it).
#ifndef LQ_STAGES
#define LQ_STAGES 5
#endif
// INT8 GEMM (CUTLASS SM80 multistage mainloop) with a custom shared-memory epilogue that can un-mix a 64-token Hadamard on the output rows,
// then apply per-column scale/bias, optional GELU, optional residual, optional re-mix + INT8 requantization or fp16 store.
//   D[m, n] = epi( sum_k A[m,k] B[n,k] )   A: int8 row-major [M,K] (rows = tokens, possibly Hadamard-mixed in groups of 64),  B: int8 [N,K] (K contiguous)
// Token mixing (Y = T^T (T X) W): the mainloop computes (T X) W; the epilogue applies T^T (64-point Walsh-Hadamard down the rows of each
// 64-row group) before any per-token operation. Requires M % 64 == 0 and tile origins at multiples of 64 (TB M = 128).
#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/epilogue/thread/activation.h"
#include <cuda_fp16.h>
namespace lqmix {
constexpr int TBM = 128, TBN = 256, TBK = 64, STAGES = LQ_STAGES, WARPS_M = 2, WARPS_N = 4, THREADS = 32 * WARPS_M * WARPS_N;   // 128x256 tile, 8 warps of 64x64 (the plugin's CUTLASS config)
using ThreadblockShape = cutlass::gemm::GemmShape<TBM, TBN, TBK>; using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>; using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
using DefaultMma = cutlass::gemm::threadblock::DefaultMma<int8_t, cutlass::layout::RowMajor, 16, int8_t, cutlass::layout::ColumnMajor, 16, int32_t, cutlass::layout::RowMajor,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, STAGES, cutlass::arch::OpMultiplyAddSaturate>;
using Mma = typename DefaultMma::ThreadblockMma;
// ---- epilogue modes ----
enum Mode { kMainloopOnly = 99,
            kFp16Bias = 0,        // D = acc*s[n] + b[n]                       (fp16 out)
            kFp16Residual = 1,    // D = unmix?(acc)*s[n] + b[n] + R[m,n]      (fp16 out; proj / fc2 — inputs may be token-mixed: unmix_in)
            kGeluInt8 = 2 };      // D = sat_i8( mix_out?( gelu(acc*s[n] + b[n]) ) * t[n] )   (int8 out; fc1 -> fc2 input, optionally re-mixed)
struct EpiParams { const float* s; const float* b; const float* t; const __half* R; int ldr; int unmix_in; int mix_out; int gelu_tanh; };
constexpr int LDT = TBN + 8;   // smem row stride (floats) of the staged tile
struct SharedStorage { union { typename Mma::SharedStorage main; float tile[TBM * LDT]; }; };
__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118f)); }
__device__ __forceinline__ float gelu_tanh(float x) { cutlass::epilogue::thread::GELU_taylor<float> g; return g(x); }
// normalized 64-point Walsh-Hadamard in registers: six explicit constant-stride stages (all indices compile-time -> no local memory)
template <int LEN> __device__ __forceinline__ void wht_stage(float* v) {
    #pragma unroll
    for (int j = 0; j < 32; j++) { const int a = (j / LEN) * (2 * LEN) + (j % LEN); float x = v[a], y = v[a + LEN]; v[a] = x + y; v[a + LEN] = x - y; }
}
__device__ __forceinline__ float tmix_sign(int j) { return (((j * 40503 + 17) % 97) & 1) ? 1.f : -1.f; }   // fixed pseudo-random +-1 per token position: T = H D (randomized Hadamard, orthogonal)
__device__ __forceinline__ void wht64(float* v) {
    wht_stage<1>(v); wht_stage<2>(v); wht_stage<4>(v); wht_stage<8>(v); wht_stage<16>(v); wht_stage<32>(v);
    #pragma unroll
    for (int j = 0; j < 64; j++) v[j] *= 0.125f;
}
template <int MODE>
__global__ void __launch_bounds__(THREADS) gemm_mix_kernel(cutlass::gemm::GemmCoord problem, const int8_t* A, int lda, const int8_t* B, int ldb, void* D, int ldd, EpiParams ep) {
    extern __shared__ __align__(16) unsigned char smem_raw[]; SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);
    const int tb_m = blockIdx.y * TBM, tb_n = blockIdx.x * TBN; const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    typename Mma::IteratorA iter_A({lda}, (int8_t*)A, {problem.m(), problem.k()}, tid, {tb_m, 0});
    typename Mma::IteratorB iter_B({ldb}, (int8_t*)B, {problem.k(), problem.n()}, tid, {0, tb_n});
    Mma mma(smem.main, tid, warp, lane); typename Mma::FragmentC acc; acc.clear();
    int k_iters = (problem.k() + TBK - 1) / TBK; mma(k_iters, acc, iter_A, iter_B, acc);
    __syncthreads();
    // ---- stage accumulators as fp32 (CUTLASS warp order: warp_m = warp % WARPS_M; fragment index = m + n*kRow; m16n8 layout inside) ----
    const int warp_m = warp % WARPS_M, warp_n = warp / WARPS_M; const int r_base = warp_m * 64 + (lane >> 2), c_base = warp_n * 64 + 2 * (lane & 3);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) { const int32_t* a4 = &acc[(ni * 4 + mi) * 4]; float* t0 = smem.tile + (r_base + mi * 16) * LDT + c_base + ni * 8;
            *reinterpret_cast<float2*>(t0) = make_float2((float)a4[0], (float)a4[1]); *reinterpret_cast<float2*>(t0 + 8 * LDT) = make_float2((float)a4[2], (float)a4[3]); }
    }
    __syncthreads();
    if (MODE == kMainloopOnly) { if (tid == 0) ((float*)D)[blockIdx.x + blockIdx.y * gridDim.x] = smem.tile[0]; return; }
    // ---- pass 1 (column-owner): unmix -> scale/bias -> (gelu) -> (re-mix), written back to the tile as fp32 ----
    const int n_valid = min(TBN, problem.n() - tb_n), m_valid = min(TBM, problem.m() - tb_m); const int c = tid, gn = tb_n + c;
    if (c < n_valid) {
        const float sc = ep.s[gn], bi = ep.b[gn];
        #pragma unroll 1
        for (int g = 0; g < 2; g++) {
            const int r0 = g * 64; if (r0 >= m_valid) continue; float v[64];
            #pragma unroll
            for (int r = 0; r < 64; r++) v[r] = smem.tile[(r0 + r) * LDT + c];
            if (ep.unmix_in) { wht64(v);
                #pragma unroll
                for (int r = 0; r < 64; r++) v[r] *= tmix_sign(r); }
            #pragma unroll
            for (int r = 0; r < 64; r++) { float x = v[r] * sc + bi; if (MODE == kGeluInt8) x = ep.gelu_tanh ? gelu_tanh(x) : gelu_erf(x); v[r] = x; }
            if (MODE == kGeluInt8 && ep.mix_out) {
                #pragma unroll
                for (int r = 0; r < 64; r++) v[r] *= tmix_sign(r);
                wht64(v); }
            #pragma unroll
            for (int r = 0; r < 64; r++) smem.tile[(r0 + r) * LDT + c] = v[r];
        }
    }
    __syncthreads();
    // ---- pass 2 (row-major, 8 columns per thread): (+residual) -> convert -> 16B/8B coalesced stores ----
    for (int i = tid; i < TBM * (TBN / 8); i += THREADS) {
        const int r = i / (TBN / 8), c8 = 8 * (i % (TBN / 8)); if (r >= m_valid || c8 + 8 > n_valid) continue; const int gm = tb_m + r, gn8 = tb_n + c8;
        const float4 a = *reinterpret_cast<const float4*>(smem.tile + r * LDT + c8), b4 = *reinterpret_cast<const float4*>(smem.tile + r * LDT + c8 + 4);
        float x[8] = {a.x, a.y, a.z, a.w, b4.x, b4.y, b4.z, b4.w};
        if (MODE == kFp16Residual) { uint4 rr = *reinterpret_cast<const uint4*>(ep.R + (size_t)gm * ep.ldr + gn8); const __half2* rh = reinterpret_cast<const __half2*>(&rr);
            #pragma unroll
            for (int q = 0; q < 4; q++) { float2 f = __half22float2(rh[q]); x[2 * q] += f.x; x[2 * q + 1] += f.y; } }
        if (MODE == kGeluInt8) { const float4 t0 = *reinterpret_cast<const float4*>(ep.t + gn8), t1 = *reinterpret_cast<const float4*>(ep.t + gn8 + 4); float tt[8] = {t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w};
            int8_t q8[8];
            #pragma unroll
            for (int q = 0; q < 8; q++) q8[q] = (int8_t)__float2int_rn(fminf(fmaxf(x[q] * tt[q], -127.f), 127.f));
            *reinterpret_cast<uint2*>((int8_t*)D + (size_t)gm * ldd + gn8) = *reinterpret_cast<uint2*>(q8); }
        else { uint4 o; __half2* oh = reinterpret_cast<__half2*>(&o);
            #pragma unroll
            for (int q = 0; q < 4; q++) oh[q] = __floats2half2_rn(x[2 * q], x[2 * q + 1]);
            *reinterpret_cast<uint4*>((__half*)D + (size_t)gm * ldd + gn8) = o; }
    }
}
template <int MODE>
inline cudaError_t launch(int M, int N, int K, const int8_t* A, int lda, const int8_t* B, int ldb, void* D, int ldd, EpiParams ep, cudaStream_t st) {
    static bool attr = false; size_t smem = sizeof(SharedStorage); if (!attr) { cudaFuncSetAttribute(gemm_mix_kernel<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
    dim3 grid((N + TBN - 1) / TBN, (M + TBM - 1) / TBM); gemm_mix_kernel<MODE><<<grid, THREADS, smem, st>>>({M, N, K}, A, lda, B, ldb, D, ldd, ep); return cudaGetLastError();
}
}  // namespace lqmix
