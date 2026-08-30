// LQ memory attention kernel: one head, D = 256, long key sequences, batched objects (SAM 3 tracker memory attention).
//   Q [Bq, Qn, 256] fp16 (Bq = 1 broadcasts one query set to every batch element), K, V [B, Kmax, 256] fp16, nvalid[B] (keys used per element)
//   O [B, Qn, 256] fp16 = softmax(scale * Q K^T) V over the first nvalid[b] keys.
// Flash-attention-2 structure on mma.sync m16n8k16: BR query rows per CTA (BR/16 warps), key tiles of BC rows double-buffered with cp.async
// (zero-filled past nvalid), the score tile of tile t+1 issued before the softmax of tile t (as in lq_attn_kernel.cuh). Q fragments are read
// from shared memory at every k-step instead of being held in registers: with D = 256 the 16 x 256 fp32 output tile already takes 128
// registers per lane. Keys beyond nvalid are masked to -inf in the score tile.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
namespace lqmem {
constexpr int D = 256, LDS = D + 8, KS = D / 16, DT = D / 8, CH = D / 8;
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16_zfill(uint32_t saddr, const void* gmem, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(saddr), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) { __half2 h = __floats2half2_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&h); }
template <int BR, int BC> struct Smem { __half q[BR * LDS]; __half k[2][BC * LDS]; __half v[2][BC * LDS]; };

template <int BR, int BC, bool PIPE>
__global__ void __launch_bounds__(BR * 2) memattn_kernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, __half* __restrict__ O,
                                                        const int* __restrict__ nvalid, int Qn, int Kmax, int q_bstride, float scale_log2e) {
    constexpr int NT = BR * 2, NTILE = BC / 8, KK = BC / 16; constexpr int CHUNKS = BC * CH, PER = (CHUNKS + NT - 1) / NT;
    extern __shared__ __align__(16) unsigned char smem_raw[]; Smem<BR, BC>& sm = *reinterpret_cast<Smem<BR, BC>*>(smem_raw);
    const int b = blockIdx.y, qrow0 = blockIdx.x * BR; const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nk = nvalid[b] < Kmax ? nvalid[b] : Kmax; const int ntiles = (nk + BC - 1) / BC;
    const __half* Qb = Q + (size_t)b * q_bstride; const __half* Kb = K + (size_t)b * Kmax * D; const __half* Vb = V + (size_t)b * Kmax * D; __half* Ob = O + (size_t)b * Qn * D;
    const uint32_t sQ = smem_u32(sm.q), sK0 = smem_u32(sm.k[0]), sK1 = smem_u32(sm.k[1]), sV0 = smem_u32(sm.v[0]), sV1 = smem_u32(sm.v[1]);
    const uint32_t k_lane = (uint32_t)((((lane & 7) + (lane >> 4) * 8) * LDS + ((lane >> 3) & 1) * 8) * 2);
    const uint32_t v_lane = (uint32_t)(((lane & 15) * LDS + (lane >> 4) * 8) * 2);
    const uint32_t q_lane = (uint32_t)(((warp * 16 + (lane & 15)) * LDS + (lane >> 4) * 8) * 2);
    int krow[PER], kch[PER];
    #pragma unroll
    for (int i = 0; i < PER; i++) { int c = tid + i * NT; krow[i] = c < CHUNKS ? c / CH : -1; kch[i] = c % CH; }
    auto load_kv = [&](uint32_t sk, uint32_t sv, int t) {
        #pragma unroll
        for (int i = 0; i < PER; i++) if (krow[i] >= 0) {
            int row = t * BC + krow[i]; bool ok = row < nk; int r = ok ? row : 0; uint32_t so = (uint32_t)((krow[i] * LDS + kch[i] * 8) * 2);
            cp_async16_zfill(sk + so, Kb + (size_t)r * D + kch[i] * 8, ok ? 16 : 0); cp_async16_zfill(sv + so, Vb + (size_t)r * D + kch[i] * 8, ok ? 16 : 0); }
    };
    // prologue: Q rows (rows past Qn are clamped; their outputs are not written), K/V tile 0
    for (int c = tid; c < BR * CH; c += NT) { int row = c / CH, ch = c % CH; int qr = qrow0 + row < Qn ? qrow0 + row : Qn - 1; cp_async16_zfill(sQ + (uint32_t)((row * LDS + ch * 8) * 2), Qb + (size_t)qr * D + ch * 8, 16); }
    load_kv(sK0, sV0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    const int r = lane >> 2;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f; float o[DT][4];
    #pragma unroll
    for (int dt = 0; dt < DT; dt++) { o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.f; }
    auto qk = [&](float (&s)[NTILE][4], uint32_t kbase) {
        #pragma unroll
        for (int nt = 0; nt < NTILE; nt++) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f; }
        const uint32_t kb = kbase + k_lane, qb = sQ + q_lane;
        #pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            uint32_t a[4]; ldsm_x4(a[0], a[1], a[2], a[3], qb + (uint32_t)(ks * 16 * 2));
            #pragma unroll
            for (int nt = 0; nt < NTILE; nt += 2) {
                uint32_t bfr[4]; ldsm_x4(bfr[0], bfr[1], bfr[2], bfr[3], kb + (uint32_t)((nt * 8 * LDS + ks * 16) * 2));
                mma_f16(s[nt], a, bfr); mma_f16(s[nt + 1], a, bfr + 2);
            }
        }
    };
    float s_cur[NTILE][4]; qk(s_cur, sK0);
    for (int t = 0; t < ntiles; t++) {
        const int cur = t & 1; const uint32_t sVcur = cur ? sV1 : sV0, sKnxt = cur ? sK0 : sK1, sVnxt = cur ? sV0 : sV1;
        if (t + 1 < ntiles) { load_kv(sKnxt, sVnxt, t + 1); cp_async_commit(); cp_async_wait_all(); __syncthreads(); }
        float s_nxt[NTILE][4];
        if (PIPE && t + 1 < ntiles) qk(s_nxt, sKnxt);
        // mask keys past nvalid in the current tile
        const int kbase_idx = t * BC;
        if (kbase_idx + BC > nk) {
            #pragma unroll
            for (int nt = 0; nt < NTILE; nt++) { int j0 = kbase_idx + nt * 8 + (lane & 3) * 2; if (j0 >= nk) { s_cur[nt][0] = -1e30f; s_cur[nt][2] = -1e30f; } if (j0 + 1 >= nk) { s_cur[nt][1] = -1e30f; s_cur[nt][3] = -1e30f; } }
        }
        float mx0 = -1e30f, mx1 = -1e30f;
        #pragma unroll
        for (int nt = 0; nt < NTILE; nt++) { mx0 = fmaxf(mx0, fmaxf(s_cur[nt][0], s_cur[nt][1])); mx1 = fmaxf(mx1, fmaxf(s_cur[nt][2], s_cur[nt][3])); }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0 * scale_log2e), mn1 = fmaxf(m1, mx1 * scale_log2e);
        const float a0 = exp2f(m0 - mn0), a1 = exp2f(m1 - mn1); float rs0 = 0.f, rs1 = 0.f; uint32_t pa[KK][4];
        #pragma unroll
        for (int nt = 0; nt < NTILE; nt++) {
            float p0 = exp2f(fmaf(s_cur[nt][0], scale_log2e, -mn0)), p1 = exp2f(fmaf(s_cur[nt][1], scale_log2e, -mn0));
            float p2 = exp2f(fmaf(s_cur[nt][2], scale_log2e, -mn1)), p3 = exp2f(fmaf(s_cur[nt][3], scale_log2e, -mn1));
            rs0 += p0 + p1; rs1 += p2 + p3; int kk = nt >> 1, hi = nt & 1; pa[kk][hi * 2 + 0] = pack_half2(p0, p1); pa[kk][hi * 2 + 1] = pack_half2(p2, p3);
        }
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1); rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2); rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1); rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
        l0 = l0 * a0 + rs0; l1 = l1 * a1 + rs1; m0 = mn0; m1 = mn1;
        #pragma unroll
        for (int dt = 0; dt < DT; dt++) { o[dt][0] *= a0; o[dt][1] *= a0; o[dt][2] *= a1; o[dt][3] *= a1; }
        const uint32_t vbase = sVcur + v_lane;
        #pragma unroll
        for (int kk = 0; kk < KK; kk++) {
            #pragma unroll
            for (int dt = 0; dt < DT; dt += 2) {
                uint32_t bfr[4]; ldsm_x4_trans(bfr[0], bfr[1], bfr[2], bfr[3], vbase + (uint32_t)((kk * 16 * LDS + dt * 8) * 2));
                mma_f16(o[dt], pa[kk], bfr); mma_f16(o[dt + 1], pa[kk], bfr + 2);
            }
        }
        if (PIPE) {
            #pragma unroll
            for (int nt = 0; nt < NTILE; nt++) { s_cur[nt][0] = s_nxt[nt][0]; s_cur[nt][1] = s_nxt[nt][1]; s_cur[nt][2] = s_nxt[nt][2]; s_cur[nt][3] = s_nxt[nt][3]; }
        } else if (t + 1 < ntiles) qk(s_cur, sKnxt);
        __syncthreads();
    }
    const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f; const int row0 = qrow0 + warp * 16 + r;
    #pragma unroll
    for (int dt = 0; dt < DT; dt++) {
        int c = dt * 8 + (lane & 3) * 2;
        if (row0 < Qn) *reinterpret_cast<uint32_t*>(Ob + (size_t)row0 * D + c) = pack_half2(o[dt][0] * inv0, o[dt][1] * inv0);
        if (row0 + 8 < Qn) *reinterpret_cast<uint32_t*>(Ob + (size_t)(row0 + 8) * D + c) = pack_half2(o[dt][2] * inv1, o[dt][3] * inv1);
    }
}
template <int BR = 32, int BC = 32, bool PIPE = true>
inline cudaError_t launch(const __half* Q, const __half* K, const __half* V, __half* O, const int* nvalid, int B, int Bq, int Qn, int Kmax, float scale, cudaStream_t st) {
    size_t smem = sizeof(Smem<BR, BC>); static bool attr_set = false;
    if (!attr_set) { cudaFuncSetAttribute(memattn_kernel<BR, BC, PIPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr_set = true; }
    dim3 grid((Qn + BR - 1) / BR, B); memattn_kernel<BR, BC, PIPE><<<grid, BR * 2, smem, st>>>(Q, K, V, O, nvalid, Qn, Kmax, Bq == 1 ? 0 : Qn * D, scale * 1.4426950408889634f);
    return cudaGetLastError();
}
// pick a configuration by the device's shared memory per block: sm87 (164 KB) takes 4 warps x 32-key tiles, sm86/sm89 (99 KB) 4 warps x 16-key tiles
inline cudaError_t launch_auto(const __half* Q, const __half* K, const __half* V, __half* O, const int* nvalid, int B, int Bq, int Qn, int Kmax, float scale, cudaStream_t st, int cfg = -1) {
    static int smem_max = 0; if (!smem_max) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev); }
    if (cfg < 0) cfg = smem_max >= 150000 ? 0 : 1;
    switch (cfg) {
        case 0: return launch<64, 32, true>(Q, K, V, O, nvalid, B, Bq, Qn, Kmax, scale, st);
        case 1: return launch<64, 16, true>(Q, K, V, O, nvalid, B, Bq, Qn, Kmax, scale, st);
        case 2: return launch<32, 32, true>(Q, K, V, O, nvalid, B, Bq, Qn, Kmax, scale, st);
        case 3: return launch<32, 32, false>(Q, K, V, O, nvalid, B, Bq, Qn, Kmax, scale, st);
        case 4: return launch<64, 32, false>(Q, K, V, O, nvalid, B, Bq, Qn, Kmax, scale, st);
        case 5: return launch<32, 64, true>(Q, K, V, O, nvalid, B, Bq, Qn, Kmax, scale, st);
        case 6: return launch<32, 64, false>(Q, K, V, O, nvalid, B, Bq, Qn, Kmax, scale, st);
        default: return launch<64, 16, false>(Q, K, V, O, nvalid, B, Bq, Qn, Kmax, scale, st);
    }
}
}  // namespace lqmem
