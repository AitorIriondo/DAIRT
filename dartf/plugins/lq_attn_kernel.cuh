// LQ fused windowed attention for SAM3 ViT-H on SM87 (flash-attention-2 structure, hand-written mma.sync / ldmatrix).
//   inputs  Q, K, V : fp16 token-major [M, H*D]  (head h occupies columns h*D .. h*D+D-1), tokens grouped in windows of N
//   output  O       : fp16 token-major [M, H*D]
//   O_w,h = softmax(scale * Q_w,h K_w,h^T) V_w,h   per window w, head h.   D = 64, N multiple of 64 (576 windowed, 5184 global).
// Variant INT8QK: Q and K tiles are quantized to int8 with per-row (per-token) scales inside the kernel; S = Q8 K8^T on the
// int8 tensor cores (m16n8k32), rescaled by sq[i]*sk[j]*scale in fp32 before the softmax.  P·V stays fp16 (m16n8k16).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace lqattn {

constexpr int BC = 64;
template <int D> struct Cfg { static constexpr int LDS = D + 8; static constexpr int LDS8 = D + 16; static constexpr int KS = D / 16; static constexpr int DT = D / 8; };   // fp16/int8 smem row strides, k-steps, d-tiles

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(smem)), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_s8(int32_t* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&h);
}

// load a ROWS x 64 fp16 tile (rows r0.., head columns) from a token-major [M, ld] matrix into smem [ROWS][LDS] via cp.async
template <int ROWS, int NT, int D>
__device__ __forceinline__ void load_tile_async(__half* s, const __half* g, int row0, int ld, int col0) {
    constexpr int CH = D / 8;   // 16B chunks per row
    for (int c = threadIdx.x; c < ROWS * CH; c += NT) { int row = c / CH, ch = c % CH; cp_async16(s + row * Cfg<D>::LDS + ch * 8, g + (size_t)(row0 + row) * ld + col0 + ch * 8); }
}
// load a ROWS x 64 int8 tile from a token-major int8 [M, ld] matrix into smem [ROWS][LDS8]
template <int ROWS, int NT, int D>
__device__ __forceinline__ void load_tile8_async(int8_t* s, const int8_t* g, int row0, int ld, int col0) {
    constexpr int CH = D / 16;
    for (int c = threadIdx.x; c < ROWS * CH; c += NT) { int row = c / CH, ch = c % CH; cp_async16(s + row * Cfg<D>::LDS8 + ch * 16, g + (size_t)(row0 + row) * ld + col0 + ch * 16); }
}
// one-pass row quantizer: X fp16 [M, H*64] -> X8 int8 [M, H*64] + scales [M, H] (per token per head). grid = M*H/8 blocks of 256 (one warp per row-head)
static __global__ void quantize_rows_kernel(const __half* __restrict__ X, int8_t* __restrict__ X8, float* __restrict__ scales, int M, int H) {
    int rh = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31; if (rh >= M * H) return;
    int m = rh / H, h = rh % H; const __half* src = X + (size_t)m * H * 64 + h * 64; __half2 v = *reinterpret_cast<const __half2*>(src + lane * 2); float2 f = __half22float2(v);
    float mx = fmaxf(fabsf(f.x), fabsf(f.y));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sc = mx > 0.f ? mx / 127.f : 1.f, inv = 1.f / sc; if (lane == 0) scales[rh] = sc;
    char2 q; q.x = (int8_t)__float2int_rn(fminf(fmaxf(f.x * inv, -127.f), 127.f)); q.y = (int8_t)__float2int_rn(fminf(fmaxf(f.y * inv, -127.f), 127.f));
    *reinterpret_cast<char2*>(X8 + (size_t)m * H * 64 + h * 64 + lane * 2) = q;
}

template <int BR, bool INT8QK, int D>
struct Smem {
    __half q[INT8QK ? 8 : BR * Cfg<D>::LDS]; int8_t q8[INT8QK ? BR * Cfg<D>::LDS8 : 16];
    __half k[2][INT8QK ? 8 : BC * Cfg<D>::LDS]; int8_t k8[2][INT8QK ? BC * Cfg<D>::LDS8 : 16];
    __half v[2][BC * Cfg<D>::LDS]; float sk[2][BC];
};

// grid: (N/BR, H, nwin); block: BR/16 warps. scale_log2e = softmax scale * log2(e). INT8QK: Q8/K8 int8 + per-(token,head) scales sQ/sK [M,H]
template <int BR, bool INT8QK, bool OUT8 = false, int D = 64>
__global__ void __launch_bounds__(BR * 2) attn_kernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, void* __restrict__ Oout,
                                                       const int8_t* __restrict__ Q8, const int8_t* __restrict__ K8, const float* __restrict__ sQ, const float* __restrict__ sK,
                                                       int N, int H, float scale_log2e, float inv_oscale, int ld_in, int ld_out, int ld_v) {
    __half* O = (__half*)Oout; int8_t* O8 = (int8_t*)Oout;
    constexpr int NT = BR * 2;   // 16 rows per warp
    constexpr int LDS = Cfg<D>::LDS, LDS8 = Cfg<D>::LDS8, KS = Cfg<D>::KS, DT = Cfg<D>::DT;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem<BR, INT8QK, D>& sm = *reinterpret_cast<Smem<BR, INT8QK, D>*>(smem_raw);
    const int ld = ld_in; const int h = blockIdx.y, w = blockIdx.z; const int qrow0 = w * N + blockIdx.x * BR; const int col0 = h * D;   // D = head dim (template); ld_in = row stride of Q/K/V
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // ---- prologue: Q tile, first K/V tiles ----
    if (!INT8QK) { load_tile_async<BR, NT, D>(sm.q, Q, qrow0, ld, col0); load_tile_async<BC, NT, D>(sm.k[0], K, w * N, ld, col0); }
    else { load_tile8_async<BR, NT, D>(sm.q8, Q8, qrow0, ld, col0); load_tile8_async<BC, NT, D>(sm.k8[0], K8, w * N, ld, col0); for (int i = tid; i < BC; i += NT) sm.sk[0][i] = sK[(size_t)(w * N + i) * H + h]; }
    load_tile_async<BC, NT, D>(sm.v[0], V, w * N, ld_v, col0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    uint32_t qa[4][4];
    if (!INT8QK) {
        #pragma unroll
        for (int ks = 0; ks < KS; ks++) ldmatrix_x4(qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], sm.q + (warp * 16 + (lane & 15)) * LDS + ks * 16 + (lane >> 4) * 8);
    } else {
        #pragma unroll
        for (int ks = 0; ks < KS / 2; ks++) ldmatrix_x4(qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], sm.q8 + (warp * 16 + (lane & 15)) * LDS8 + ks * 32 + (lane >> 4) * 16);
    }
    const int r = lane >> 2;                 // rows r and r+8 of this warp's 16-row slab
    float sq0 = 0.f, sq1 = 0.f; if (INT8QK) { sq0 = sQ[(size_t)(qrow0 + warp * 16 + r) * H + h]; sq1 = sQ[(size_t)(qrow0 + warp * 16 + r + 8) * H + h]; }
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f; float o[DT][4];
    #pragma unroll
    for (int dt = 0; dt < DT; dt++) { o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.f; }
    const int ntiles = N / BC;
    for (int t = 0; t < ntiles; t++) {
        const int buf = t & 1;
        if (t + 1 < ntiles) {
            if (!INT8QK) load_tile_async<BC, NT, D>(sm.k[buf ^ 1], K, w * N + (t + 1) * BC, ld, col0);
            else { load_tile8_async<BC, NT, D>(sm.k8[buf ^ 1], K8, w * N + (t + 1) * BC, ld, col0); for (int i = tid; i < BC; i += NT) sm.sk[buf ^ 1][i] = sK[(size_t)(w * N + (t + 1) * BC + i) * H + h]; }
            load_tile_async<BC, NT, D>(sm.v[buf ^ 1], V, w * N + (t + 1) * BC, ld_v, col0); cp_async_commit();
        }
        // ---- S = Q K^T for this warp's 16 rows x 64 cols ----
        float s[8][4];
        if (!INT8QK) {
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f; }
            #pragma unroll
            for (int ks = 0; ks < KS; ks++) {
                #pragma unroll
                for (int nt = 0; nt < 8; nt += 2) {
                    uint32_t b[4]; ldmatrix_x4(b[0], b[1], b[2], b[3], sm.k[buf] + ((nt + (lane >> 4)) * 8 + (lane & 7)) * LDS + ks * 16 + ((lane >> 3) & 1) * 8);
                    mma_f16(s[nt], qa[ks], b); mma_f16(s[nt + 1], qa[ks], b + 2);
                }
            }
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) { s[nt][0] *= scale_log2e; s[nt][1] *= scale_log2e; s[nt][2] *= scale_log2e; s[nt][3] *= scale_log2e; }
        } else {
            int32_t si[8][4];
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) { si[nt][0] = si[nt][1] = si[nt][2] = si[nt][3] = 0; }
            #pragma unroll
            for (int ks = 0; ks < KS / 2; ks++) {
                #pragma unroll
                for (int nt = 0; nt < 8; nt += 2) {
                    uint32_t b[4]; ldmatrix_x4(b[0], b[1], b[2], b[3], sm.k8[buf] + ((nt + (lane >> 4)) * 8 + (lane & 7)) * LDS8 + ks * 32 + ((lane >> 3) & 1) * 16);
                    mma_s8(si[nt], qa[ks], b); mma_s8(si[nt + 1], qa[ks], b + 2);
                }
            }
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                int c = nt * 8 + (lane & 3) * 2; float k0 = sm.sk[buf][c] * scale_log2e, k1 = sm.sk[buf][c + 1] * scale_log2e;
                s[nt][0] = (float)si[nt][0] * sq0 * k0; s[nt][1] = (float)si[nt][1] * sq0 * k1; s[nt][2] = (float)si[nt][2] * sq1 * k0; s[nt][3] = (float)si[nt][3] * sq1 * k1;
            }
        }
        // ---- online softmax (rows r, r+8) ----
        float mx0 = -1e30f, mx1 = -1e30f;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) { mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1])); mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3])); }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1); float a0 = exp2f(m0 - mn0), a1 = exp2f(m1 - mn1); float rs0 = 0.f, rs1 = 0.f;
        uint32_t pa[4][4];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            float p0 = exp2f(s[nt][0] - mn0), p1 = exp2f(s[nt][1] - mn0), p2 = exp2f(s[nt][2] - mn1), p3 = exp2f(s[nt][3] - mn1);
            rs0 += p0 + p1; rs1 += p2 + p3;
            int kk = nt >> 1, hi = nt & 1; pa[kk][hi * 2 + 0] = pack_half2(p0, p1); pa[kk][hi * 2 + 1] = pack_half2(p2, p3);
        }
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1); rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2); rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1); rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
        l0 = l0 * a0 + rs0; l1 = l1 * a1 + rs1; m0 = mn0; m1 = mn1;
        #pragma unroll
        for (int dt = 0; dt < DT; dt++) { o[dt][0] *= a0; o[dt][1] *= a0; o[dt][2] *= a1; o[dt][3] *= a1; }
        // ---- O += P V ----
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            #pragma unroll
            for (int dt = 0; dt < DT; dt += 2) {
                uint32_t b[4]; ldmatrix_x4_trans(b[0], b[1], b[2], b[3], sm.v[buf] + (kk * 16 + (lane & 15)) * LDS + (dt + (lane >> 4)) * 8);
                mma_f16(o[dt], pa[kk], b); mma_f16(o[dt + 1], pa[kk], b + 2);
            }
        }
        cp_async_wait_all(); __syncthreads();
    }
    // ---- epilogue: normalize and store ----
    float inv0 = 1.f / l0, inv1 = 1.f / l1; const int row0 = qrow0 + warp * 16 + r;
    #pragma unroll
    for (int dt = 0; dt < DT; dt++) {
        int c = col0 + dt * 8 + (lane & 3) * 2;
        if (!OUT8) {
            *reinterpret_cast<uint32_t*>(O + (size_t)row0 * ld_out + c) = pack_half2(o[dt][0] * inv0, o[dt][1] * inv0);
            *reinterpret_cast<uint32_t*>(O + (size_t)(row0 + 8) * ld_out + c) = pack_half2(o[dt][2] * inv1, o[dt][3] * inv1);
        } else {
            char2 a, b; a.x = (int8_t)__float2int_rn(fminf(fmaxf(o[dt][0] * inv0 * inv_oscale, -127.f), 127.f)); a.y = (int8_t)__float2int_rn(fminf(fmaxf(o[dt][1] * inv0 * inv_oscale, -127.f), 127.f));
            b.x = (int8_t)__float2int_rn(fminf(fmaxf(o[dt][2] * inv1 * inv_oscale, -127.f), 127.f)); b.y = (int8_t)__float2int_rn(fminf(fmaxf(o[dt][3] * inv1 * inv_oscale, -127.f), 127.f));
            *reinterpret_cast<char2*>(O8 + (size_t)row0 * ld_out + c) = a; *reinterpret_cast<char2*>(O8 + (size_t)(row0 + 8) * ld_out + c) = b;
        }
    }
}

// INT8QK launch expects workspace for Q8, K8 (M*H*64 bytes each) and sQ, sK (M*H floats each): ws >= M*H*(128 + 8) bytes
template <int BR, bool INT8QK, bool OUT8 = false, int D = 64>
inline cudaError_t launch(const __half* Q, const __half* K, const __half* V, void* O, int M, int N, int H, float scale, void* ws, cudaStream_t st, float oscale = 0.f, int ld_in = 0, int ld_out = 0, int ld_v = 0) {
    if (ld_in == 0) ld_in = H * D; if (ld_out == 0) ld_out = H * D; if (ld_v == 0) ld_v = ld_in;
    size_t smem = sizeof(Smem<BR, INT8QK, D>); static bool attr_set = false;
    if (!attr_set) { cudaFuncSetAttribute(attn_kernel<BR, INT8QK, OUT8, D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr_set = true; }
    const int8_t* Q8 = nullptr; const int8_t* K8 = nullptr; const float* sQ = nullptr; const float* sK = nullptr;
    if (INT8QK) {
        int8_t* q8 = (int8_t*)ws; int8_t* k8 = q8 + (size_t)M * H * D; float* sq = (float*)(k8 + (size_t)M * H * D); float* sk = sq + (size_t)M * H;
        quantize_rows_kernel<<<(M * H + 7) / 8, 256, 0, st>>>(Q, q8, sq, M, H); quantize_rows_kernel<<<(M * H + 7) / 8, 256, 0, st>>>(K, k8, sk, M, H);
        Q8 = q8; K8 = k8; sQ = sq; sK = sk;
    }
    dim3 grid(N / BR, H, M / N); attn_kernel<BR, INT8QK, OUT8, D><<<grid, BR * 2, smem, st>>>(Q, K, V, O, Q8, K8, sQ, sK, N, H, scale * 1.4426950408889634f, oscale > 0.f ? 1.f / oscale : 0.f, ld_in, ld_out, ld_v);
    return cudaGetLastError();
}
// ============================================================================================================================
// MT variant: each warp owns MT*16 query rows (MT m-tiles), so every K/V fragment fetched by ldmatrix feeds MT mma pairs.
// Rationale (SM87 roofline): with 16 rows/warp the kernel is shared-memory-bandwidth bound (each warp streams all of K and V);
// MT=2 halves that traffic at the price of ~2x accumulator registers. fp16 Q/K only. Block = BR/(16*MT) warps.
template <int BR, int MT, bool OUT8, int D>
__global__ void __launch_bounds__(BR * 2 / MT) attn_kernel_mt(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, void* __restrict__ Oout,
                                                              int N, int H, float scale_log2e, float inv_oscale, int ld_in, int ld_out, int ld_v) {
    static_assert(BR % (16 * MT) == 0, "BR must be a multiple of 16*MT");
    __half* O = (__half*)Oout; int8_t* O8 = (int8_t*)Oout;
    constexpr int NT = BR * 2 / MT, WR = 16 * MT;
    constexpr int LDS = Cfg<D>::LDS, KS = Cfg<D>::KS, DT = Cfg<D>::DT;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem<BR, false, D>& sm = *reinterpret_cast<Smem<BR, false, D>*>(smem_raw);
    const int ld = ld_in; const int h = blockIdx.y, w = blockIdx.z; const int qrow0 = w * N + blockIdx.x * BR; const int col0 = h * D;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    load_tile_async<BR, NT, D>(sm.q, Q, qrow0, ld, col0); load_tile_async<BC, NT, D>(sm.k[0], K, w * N, ld, col0);
    load_tile_async<BC, NT, D>(sm.v[0], V, w * N, ld_v, col0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    uint32_t qa[MT][KS][4];
    #pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        #pragma unroll
        for (int ks = 0; ks < KS; ks++) ldmatrix_x4(qa[mt][ks][0], qa[mt][ks][1], qa[mt][ks][2], qa[mt][ks][3], sm.q + (warp * WR + mt * 16 + (lane & 15)) * LDS + ks * 16 + (lane >> 4) * 8);
    }
    const int r = lane >> 2;
    float m[MT][2], l[MT][2], o[MT][DT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; mt++) { m[mt][0] = m[mt][1] = -1e30f; l[mt][0] = l[mt][1] = 0.f;
        #pragma unroll
        for (int dt = 0; dt < DT; dt++) { o[mt][dt][0] = o[mt][dt][1] = o[mt][dt][2] = o[mt][dt][3] = 0.f; } }
    const int ntiles = N / BC;
    for (int t = 0; t < ntiles; t++) {
        const int buf = t & 1;
        if (t + 1 < ntiles) { load_tile_async<BC, NT, D>(sm.k[buf ^ 1], K, w * N + (t + 1) * BC, ld, col0); load_tile_async<BC, NT, D>(sm.v[buf ^ 1], V, w * N + (t + 1) * BC, ld_v, col0); cp_async_commit(); }
        float s[MT][8][4];
        #pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) { s[mt][nt][0] = s[mt][nt][1] = s[mt][nt][2] = s[mt][nt][3] = 0.f; } }
        #pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            #pragma unroll
            for (int nt = 0; nt < 8; nt += 2) {
                uint32_t b[4];
#ifdef LQ_ABL_NOLDK
                b[0] = qa[0][0][0] ^ nt; b[1] = qa[0][0][1]; b[2] = qa[0][0][2]; b[3] = qa[0][0][3] ^ ks;
#else
                ldmatrix_x4(b[0], b[1], b[2], b[3], sm.k[buf] + ((nt + (lane >> 4)) * 8 + (lane & 7)) * LDS + ks * 16 + ((lane >> 3) & 1) * 8);
#endif
#ifndef LQ_ABL_NOQK
                #pragma unroll
                for (int mt = 0; mt < MT; mt++) { mma_f16(s[mt][nt], qa[mt][ks], b); mma_f16(s[mt][nt + 1], qa[mt][ks], b + 2); }
#else
                #pragma unroll
                for (int mt = 0; mt < MT; mt++) { s[mt][nt][0] += __uint_as_float(b[0]) * 1e-30f; s[mt][nt + 1][1] += __uint_as_float(b[2]) * 1e-30f; }
#endif
            }
        }
        uint32_t pa[MT][4][4];
        #pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            float mx0 = -1e30f, mx1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) { s[mt][nt][0] *= scale_log2e; s[mt][nt][1] *= scale_log2e; s[mt][nt][2] *= scale_log2e; s[mt][nt][3] *= scale_log2e;
                mx0 = fmaxf(mx0, fmaxf(s[mt][nt][0], s[mt][nt][1])); mx1 = fmaxf(mx1, fmaxf(s[mt][nt][2], s[mt][nt][3])); }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            float mn0 = fmaxf(m[mt][0], mx0), mn1 = fmaxf(m[mt][1], mx1); float a0 = exp2f(m[mt][0] - mn0), a1 = exp2f(m[mt][1] - mn1); float rs0 = 0.f, rs1 = 0.f;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
#ifdef LQ_ABL_NOEXP
                float p0 = (s[mt][nt][0] - mn0), p1 = (s[mt][nt][1] - mn0), p2 = (s[mt][nt][2] - mn1), p3 = (s[mt][nt][3] - mn1);
#else
                float p0 = exp2f(s[mt][nt][0] - mn0), p1 = exp2f(s[mt][nt][1] - mn0), p2 = exp2f(s[mt][nt][2] - mn1), p3 = exp2f(s[mt][nt][3] - mn1);
#endif
                rs0 += p0 + p1; rs1 += p2 + p3;
                int kk = nt >> 1, hi = nt & 1; pa[mt][kk][hi * 2 + 0] = pack_half2(p0, p1); pa[mt][kk][hi * 2 + 1] = pack_half2(p2, p3);
            }
            rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1); rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2); rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1); rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
            l[mt][0] = l[mt][0] * a0 + rs0; l[mt][1] = l[mt][1] * a1 + rs1; m[mt][0] = mn0; m[mt][1] = mn1;
            #pragma unroll
            for (int dt = 0; dt < DT; dt++) { o[mt][dt][0] *= a0; o[mt][dt][1] *= a0; o[mt][dt][2] *= a1; o[mt][dt][3] *= a1; }
        }
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            #pragma unroll
            for (int dt = 0; dt < DT; dt += 2) {
                uint32_t b[4];
#ifdef LQ_ABL_NOLDV
                b[0] = pa[0][kk][0] ^ dt; b[1] = pa[0][kk][1]; b[2] = pa[0][kk][2]; b[3] = pa[0][kk][3] ^ kk;
#else
                ldmatrix_x4_trans(b[0], b[1], b[2], b[3], sm.v[buf] + (kk * 16 + (lane & 15)) * LDS + (dt + (lane >> 4)) * 8);
#endif
#ifndef LQ_ABL_NOPV
                #pragma unroll
                for (int mt = 0; mt < MT; mt++) { mma_f16(o[mt][dt], pa[mt][kk], b); mma_f16(o[mt][dt + 1], pa[mt][kk], b + 2); }
#else
                #pragma unroll
                for (int mt = 0; mt < MT; mt++) { o[mt][dt][0] += __uint_as_float(b[0] ^ pa[mt][kk][0]) * 1e-30f; o[mt][dt + 1][1] += __uint_as_float(b[2] ^ pa[mt][kk][1]) * 1e-30f; }
#endif
            }
        }
        cp_async_wait_all(); __syncthreads();
    }
    #pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        float inv0 = 1.f / l[mt][0], inv1 = 1.f / l[mt][1]; const int row0 = qrow0 + warp * WR + mt * 16 + r;
        #pragma unroll
        for (int dt = 0; dt < DT; dt++) {
            int c = col0 + dt * 8 + (lane & 3) * 2;
            if (!OUT8) {
                *reinterpret_cast<uint32_t*>(O + (size_t)row0 * ld_out + c) = pack_half2(o[mt][dt][0] * inv0, o[mt][dt][1] * inv0);
                *reinterpret_cast<uint32_t*>(O + (size_t)(row0 + 8) * ld_out + c) = pack_half2(o[mt][dt][2] * inv1, o[mt][dt][3] * inv1);
            } else {
                char2 a, b; a.x = (int8_t)__float2int_rn(fminf(fmaxf(o[mt][dt][0] * inv0 * inv_oscale, -127.f), 127.f)); a.y = (int8_t)__float2int_rn(fminf(fmaxf(o[mt][dt][1] * inv0 * inv_oscale, -127.f), 127.f));
                b.x = (int8_t)__float2int_rn(fminf(fmaxf(o[mt][dt][2] * inv1 * inv_oscale, -127.f), 127.f)); b.y = (int8_t)__float2int_rn(fminf(fmaxf(o[mt][dt][3] * inv1 * inv_oscale, -127.f), 127.f));
                *reinterpret_cast<char2*>(O8 + (size_t)row0 * ld_out + c) = a; *reinterpret_cast<char2*>(O8 + (size_t)(row0 + 8) * ld_out + c) = b;
            }
        }
    }
}
template <int BR, int MT, bool OUT8 = false, int D = 64>
inline cudaError_t launch_mt(const __half* Q, const __half* K, const __half* V, void* O, int M, int N, int H, float scale, cudaStream_t st, float oscale = 0.f, int ld_in = 0, int ld_out = 0, int ld_v = 0) {
    if (ld_in == 0) ld_in = H * D; if (ld_out == 0) ld_out = H * D; if (ld_v == 0) ld_v = ld_in;
    size_t smem = sizeof(Smem<BR, false, D>); static bool attr_set = false;
    if (!attr_set) { cudaFuncSetAttribute(attn_kernel_mt<BR, MT, OUT8, D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr_set = true; }
    dim3 grid(N / BR, H, M / N); attn_kernel_mt<BR, MT, OUT8, D><<<grid, BR * 2 / MT, smem, st>>>(Q, K, V, O, N, H, scale * 1.4426950408889634f, oscale > 0.f ? 1.f / oscale : 0.f, ld_in, ld_out, ld_v);
    return cudaGetLastError();
}
// ============================================================================================================================
// PP variant: software-pipelined flash attention. Iteration t issues S_{t+1} = Q K_{t+1}^T (tensor pipe) BEFORE the softmax of
// S_t (FP32 ALU) so that mma.sync work overlaps the softmax inside one warp; K runs one tile ahead of V.  Scale folded into
// the exp argument (one FFMA per score). 16 rows per warp (MT=1). fp16 only.

__device__ __forceinline__ uint32_t ex2_h2(float lo, float hi) {   // 2^lo, 2^hi computed on packed half2 (MUFU.EX2 f16x2), returns packed half2
    uint32_t x = pack_half2(lo, hi), y; asm("ex2.approx.f16x2 %0, %1;\n" : "=r"(y) : "r"(x)); return y;
}
__device__ __forceinline__ uint32_t hadd2_u(uint32_t a, uint32_t b) { uint32_t c; asm("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b)); return c; }
__device__ __forceinline__ float h2_sum(uint32_t a) { __half2 h = *reinterpret_cast<__half2*>(&a); float2 f = __half22float2(h); return f.x + f.y; }
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void cp_async16_a(uint32_t saddr, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gmem));
}
__device__ __forceinline__ float tmix_sign(int j) { return (((j * 40503 + 17) % 97) & 1) ? 1.f : -1.f; }   // fixed pseudo-random +-1 per token position: T = H D (randomized Hadamard, orthogonal)
template <int LEN> __device__ __forceinline__ void wht_stage64(float* v) {   // one butterfly stage of a 64-point Walsh-Hadamard, all indices compile-time
    #pragma unroll
    for (int j = 0; j < 32; j++) { const int a = (j / LEN) * (2 * LEN) + (j % LEN); float x = v[a], y = v[a + LEN]; v[a] = x + y; v[a + LEN] = x - y; }
}
template <int BR, bool OUT8, int D, bool H2EXP = false, bool MIX = false>
__global__ void __launch_bounds__(BR * 2) attn_kernel_pp(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, void* __restrict__ Oout,
                                                          int N, int H, float scale_log2e, float inv_oscale, int ld_in, int ld_out, int ld_v) {
    __half* O = (__half*)Oout; int8_t* O8 = (int8_t*)Oout;
    constexpr int NT = BR * 2; constexpr int LDS = Cfg<D>::LDS, KS = Cfg<D>::KS, DT = Cfg<D>::DT, CH = D / 8;
    constexpr int CHUNKS = BC * CH, PER = (CHUNKS + NT - 1) / NT;          // cp.async 16B chunks per K/V tile, per thread
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem<BR, false, D>& sm = *reinterpret_cast<Smem<BR, false, D>*>(smem_raw);
    const int ld = ld_in; const int h = blockIdx.y, w = blockIdx.z; const int qrow0 = w * N + blockIdx.x * BR; const int col0 = h * D;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31; const int ntiles = N / BC;
    // ---- hoisted addressing: 32-bit shared bases, per-lane ldmatrix offsets, per-thread cp.async chunk pointers ----
    const uint32_t sQ = smem_u32(sm.q), sK0 = smem_u32(sm.k[0]), sK1 = smem_u32(sm.k[1]), sV0 = smem_u32(sm.v[0]), sV1 = smem_u32(sm.v[1]);
    const uint32_t k_lane = (uint32_t)((((lane & 7) + (lane >> 4) * 8) * LDS + ((lane >> 3) & 1) * 8) * 2);      // K fragment: rows nt*8.., cols ks*16..
    const uint32_t v_lane = (uint32_t)(((lane & 15) * LDS + (lane >> 4) * 8) * 2);                                 // V fragment (trans): rows kk*16.., cols dt*8..
    const __half* kg[PER]; const __half* vg[PER]; uint32_t so[PER]; bool cv[PER];
    #pragma unroll
    for (int i = 0; i < PER; i++) { int c = tid + i * NT; cv[i] = c < CHUNKS; int row = c / CH, ch = c % CH; if (!cv[i]) { row = 0; ch = 0; }
        kg[i] = K + (size_t)(w * N + row) * ld + col0 + ch * 8; vg[i] = V + (size_t)(w * N + row) * ld_v + col0 + ch * 8; so[i] = (uint32_t)((row * LDS + ch * 8) * 2); }
    const size_t kstep = (size_t)BC * ld, vstep = (size_t)BC * ld_v;
    auto load_k = [&](uint32_t sbase, int t) {
        #pragma unroll
        for (int i = 0; i < PER; i++) if (cv[i]) cp_async16_a(sbase + so[i], kg[i] + (size_t)t * kstep); };
    auto load_v = [&](uint32_t sbase, int t) {
        #pragma unroll
        for (int i = 0; i < PER; i++) if (cv[i]) cp_async16_a(sbase + so[i], vg[i] + (size_t)t * vstep); };
    // prologue: Q, K0, V0, K1
    load_tile_async<BR, NT, D>(sm.q, Q, qrow0, ld, col0); load_k(sK0, 0); load_v(sV0, 0); if (ntiles > 1) load_k(sK1, 1);
    cp_async_commit(); cp_async_wait_all(); __syncthreads();
    uint32_t qa[KS][4];
    #pragma unroll
    for (int ks = 0; ks < KS; ks++) ldsm_x4(qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], sQ + (uint32_t)(((warp * 16 + (lane & 15)) * LDS + ks * 16 + (lane >> 4) * 8) * 2));
    const int r = lane >> 2;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f; float o[DT][4];
    #pragma unroll
    for (int dt = 0; dt < DT; dt++) { o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.f; }
    auto qk = [&](float (&s)[8][4], uint32_t kbase) {
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f; }
        const uint32_t kb = kbase + k_lane;
        #pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            #pragma unroll
            for (int nt = 0; nt < 8; nt += 2) {
                uint32_t b[4]; ldsm_x4(b[0], b[1], b[2], b[3], kb + (uint32_t)((nt * 8 * LDS + ks * 16) * 2));
                mma_f16(s[nt], qa[ks], b); mma_f16(s[nt + 1], qa[ks], b + 2);
            }
        }
    };
    float s_cur[8][4]; qk(s_cur, sK0);
    for (int t = 0; t < ntiles; t++) {
        const int vb = t & 1, kb = (t + 1) & 1;
        const uint32_t sVcur = vb ? sV1 : sV0, sVnxt = vb ? sV0 : sV1, sKnxt = kb ? sK1 : sK0, sKnn = kb ? sK0 : sK1;
        if (t + 1 < ntiles) load_v(sVnxt, t + 1);
        if (t + 2 < ntiles) load_k(sKnn, t + 2);
        cp_async_commit();
        float s_nxt[8][4];
        if (t + 1 < ntiles) qk(s_nxt, sKnxt);
        float mx0 = -1e30f, mx1 = -1e30f;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) { mx0 = fmaxf(mx0, fmaxf(s_cur[nt][0], s_cur[nt][1])); mx1 = fmaxf(mx1, fmaxf(s_cur[nt][2], s_cur[nt][3])); }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0 * scale_log2e), mn1 = fmaxf(m1, mx1 * scale_log2e);
        const float a0 = exp2f(m0 - mn0), a1 = exp2f(m1 - mn1); float rs0 = 0.f, rs1 = 0.f;
        uint32_t pa[4][4];
        if (H2EXP) {
            uint32_t acc0 = 0u, acc1 = 0u;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                int kk = nt >> 1, hi = nt & 1;
                uint32_t q0 = ex2_h2(fmaf(s_cur[nt][0], scale_log2e, -mn0), fmaf(s_cur[nt][1], scale_log2e, -mn0));
                uint32_t q1 = ex2_h2(fmaf(s_cur[nt][2], scale_log2e, -mn1), fmaf(s_cur[nt][3], scale_log2e, -mn1));
                pa[kk][hi * 2 + 0] = q0; pa[kk][hi * 2 + 1] = q1; acc0 = hadd2_u(acc0, q0); acc1 = hadd2_u(acc1, q1);
            }
            rs0 = h2_sum(acc0); rs1 = h2_sum(acc1);
        } else {
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                float p0 = exp2f(fmaf(s_cur[nt][0], scale_log2e, -mn0)), p1 = exp2f(fmaf(s_cur[nt][1], scale_log2e, -mn0));
                float p2 = exp2f(fmaf(s_cur[nt][2], scale_log2e, -mn1)), p3 = exp2f(fmaf(s_cur[nt][3], scale_log2e, -mn1));
                rs0 += p0 + p1; rs1 += p2 + p3;
                int kk = nt >> 1, hi = nt & 1; pa[kk][hi * 2 + 0] = pack_half2(p0, p1); pa[kk][hi * 2 + 1] = pack_half2(p2, p3);
            }
        }
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1); rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2); rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1); rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
        l0 = l0 * a0 + rs0; l1 = l1 * a1 + rs1; m0 = mn0; m1 = mn1;
        #pragma unroll
        for (int dt = 0; dt < DT; dt++) { o[dt][0] *= a0; o[dt][1] *= a0; o[dt][2] *= a1; o[dt][3] *= a1; }
        const uint32_t vbase = sVcur + v_lane;
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            #pragma unroll
            for (int dt = 0; dt < DT; dt += 2) {
                uint32_t b[4]; ldsm_x4_trans(b[0], b[1], b[2], b[3], vbase + (uint32_t)((kk * 16 * LDS + dt * 8) * 2));
                mma_f16(o[dt], pa[kk], b); mma_f16(o[dt + 1], pa[kk], b + 2);
            }
        }
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) { s_cur[nt][0] = s_nxt[nt][0]; s_cur[nt][1] = s_nxt[nt][1]; s_cur[nt][2] = s_nxt[nt][2]; s_cur[nt][3] = s_nxt[nt][3]; }
        cp_async_wait_all(); __syncthreads();
    }
    float inv0 = 1.f / l0, inv1 = 1.f / l1; const int row0 = qrow0 + warp * 16 + r;
    if (MIX && OUT8) {   // token-mixed INT8 output: stage the BR x D fp32 tile, 64-point Hadamard down each column (rows = tokens), quantize
        static_assert(!MIX || BR == 64, "token mixing needs 64-row tiles");
        constexpr int LDM = D + 4; float* tile = reinterpret_cast<float*>(smem_raw);   // reuses the (now idle) Q/K/V buffers: 64 x (D+4) floats
        __syncthreads();
        #pragma unroll
        for (int dt = 0; dt < DT; dt++) { int c = dt * 8 + (lane & 3) * 2; int rr = warp * 16 + r;
            tile[rr * LDM + c] = o[dt][0] * inv0; tile[rr * LDM + c + 1] = o[dt][1] * inv0; tile[(rr + 8) * LDM + c] = o[dt][2] * inv1; tile[(rr + 8) * LDM + c + 1] = o[dt][3] * inv1; }
        __syncthreads();
        if (tid < D) {
            float v[64];
            #pragma unroll
            for (int j = 0; j < 64; j++) v[j] = tile[j * LDM + tid];
            #pragma unroll
            for (int j = 0; j < 64; j++) v[j] *= tmix_sign(j);
            wht_stage64<1>(v); wht_stage64<2>(v); wht_stage64<4>(v); wht_stage64<8>(v); wht_stage64<16>(v); wht_stage64<32>(v);
            #pragma unroll
            for (int j = 0; j < 64; j++) O8[(size_t)(qrow0 + j) * ld_out + col0 + tid] = (int8_t)__float2int_rn(fminf(fmaxf(v[j] * 0.125f * inv_oscale, -127.f), 127.f));
        }
        return;
    }
    #pragma unroll
    for (int dt = 0; dt < DT; dt++) {
        int c = col0 + dt * 8 + (lane & 3) * 2;
        if (!OUT8) {
            *reinterpret_cast<uint32_t*>(O + (size_t)row0 * ld_out + c) = pack_half2(o[dt][0] * inv0, o[dt][1] * inv0);
            *reinterpret_cast<uint32_t*>(O + (size_t)(row0 + 8) * ld_out + c) = pack_half2(o[dt][2] * inv1, o[dt][3] * inv1);
        } else {
            char2 a, b; a.x = (int8_t)__float2int_rn(fminf(fmaxf(o[dt][0] * inv0 * inv_oscale, -127.f), 127.f)); a.y = (int8_t)__float2int_rn(fminf(fmaxf(o[dt][1] * inv0 * inv_oscale, -127.f), 127.f));
            b.x = (int8_t)__float2int_rn(fminf(fmaxf(o[dt][2] * inv1 * inv_oscale, -127.f), 127.f)); b.y = (int8_t)__float2int_rn(fminf(fmaxf(o[dt][3] * inv1 * inv_oscale, -127.f), 127.f));
            *reinterpret_cast<char2*>(O8 + (size_t)row0 * ld_out + c) = a; *reinterpret_cast<char2*>(O8 + (size_t)(row0 + 8) * ld_out + c) = b;
        }
    }
}
template <int BR, bool OUT8 = false, int D = 64, bool H2EXP = false, bool MIX = false>
inline cudaError_t launch_pp(const __half* Q, const __half* K, const __half* V, void* O, int M, int N, int H, float scale, cudaStream_t st, float oscale = 0.f, int ld_in = 0, int ld_out = 0, int ld_v = 0) {
    if (ld_in == 0) ld_in = H * D; if (ld_out == 0) ld_out = H * D; if (ld_v == 0) ld_v = ld_in;
    size_t smem = sizeof(Smem<BR, false, D>); static bool attr_set = false;
    if (!attr_set) { cudaFuncSetAttribute(attn_kernel_pp<BR, OUT8, D, H2EXP, MIX>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr_set = true; }
    dim3 grid(N / BR, H, M / N); attn_kernel_pp<BR, OUT8, D, H2EXP, MIX><<<grid, BR * 2, smem, st>>>(Q, K, V, O, N, H, scale * 1.4426950408889634f, oscale > 0.f ? 1.f / oscale : 0.f, ld_in, ld_out, ld_v);
    return cudaGetLastError();
}
// ============================================================================================================================
// PP8: the pipelined kernel with INT8 Q.K^T (per-(token,head) dynamic scales from quantize_rows_kernel; mma.sync s8 m16n8k32).
// P.V stays fp16. The score dequant (sq_row * sk_col * scale_log2e) is folded into the exp FFMA's multiplier per (row, col).
template <int BR, bool OUT8, int D>
__global__ void __launch_bounds__(BR * 2) attn_kernel_pp8(const int8_t* __restrict__ Q8, const int8_t* __restrict__ K8, const float* __restrict__ sQ, const float* __restrict__ sK,
                                                           const __half* __restrict__ V, void* __restrict__ Oout, int N, int H, float scale_log2e, float inv_oscale, int ld_v, int ld_out) {
    static_assert(D == 64, "pp8: D=64 only");
    __half* O = (__half*)Oout; int8_t* O8 = (int8_t*)Oout;
    constexpr int NT = BR * 2; constexpr int LDS = Cfg<D>::LDS, LDS8 = Cfg<D>::LDS8, DT = Cfg<D>::DT, KS8 = D / 32;
    constexpr int CH8 = D / 16, CHUNKS8 = BC * CH8, PER8 = (CHUNKS8 + NT - 1) / NT, CH = D / 8, CHUNKS = BC * CH, PER = (CHUNKS + NT - 1) / NT;
    struct SmemPP8 { int8_t q8[BR * LDS8]; int8_t k8[2][BC * LDS8]; float sk[2][BC]; __half v[2][BC * LDS]; };
    extern __shared__ __align__(16) unsigned char smem_raw[]; SmemPP8& sm = *reinterpret_cast<SmemPP8*>(smem_raw);
    const int h = blockIdx.y, w = blockIdx.z; const int qrow0 = w * N + blockIdx.x * BR; const int col0 = h * D; const int ldq = H * D;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31; const int ntiles = N / BC;
    const uint32_t sQb = smem_u32(sm.q8), sK0 = smem_u32(sm.k8[0]), sK1 = smem_u32(sm.k8[1]), sV0 = smem_u32(sm.v[0]), sV1 = smem_u32(sm.v[1]);
    const uint32_t k_lane = (uint32_t)((((lane & 7) + (lane >> 4) * 8) * LDS8 + ((lane >> 3) & 1) * 16));      // int8: 16 B per ks-half
    const uint32_t v_lane = (uint32_t)(((lane & 15) * LDS + (lane >> 4) * 8) * 2);
    const int8_t* kg[PER8]; uint32_t ko[PER8]; bool kc[PER8]; const __half* vg[PER]; uint32_t vo[PER]; bool vc[PER];
    #pragma unroll
    for (int i = 0; i < PER8; i++) { int c = tid + i * NT; kc[i] = c < CHUNKS8; int row = kc[i] ? c / CH8 : 0, ch = kc[i] ? c % CH8 : 0; kg[i] = K8 + (size_t)(w * N + row) * ldq + col0 + ch * 16; ko[i] = (uint32_t)(row * LDS8 + ch * 16); }
    #pragma unroll
    for (int i = 0; i < PER; i++) { int c = tid + i * NT; vc[i] = c < CHUNKS; int row = vc[i] ? c / CH : 0, ch = vc[i] ? c % CH : 0; vg[i] = V + (size_t)(w * N + row) * ld_v + col0 + ch * 8; vo[i] = (uint32_t)((row * LDS + ch * 8) * 2); }
    const size_t kstep = (size_t)BC * ldq, vstep = (size_t)BC * ld_v;
    auto load_k = [&](uint32_t sb, float* skb, int t) {
        #pragma unroll
        for (int i = 0; i < PER8; i++) if (kc[i]) cp_async16_a(sb + ko[i], kg[i] + (size_t)t * kstep);
        for (int i = tid; i < BC; i += NT) skb[i] = sK[(size_t)(w * N + t * BC + i) * H + h]; };
    auto load_v = [&](uint32_t sb, int t) {
        #pragma unroll
        for (int i = 0; i < PER; i++) if (vc[i]) cp_async16_a(sb + vo[i], vg[i] + (size_t)t * vstep); };
    // Q tile (int8) via cp.async
    for (int c = tid; c < BR * CH8; c += NT) { int row = c / CH8, ch = c % CH8; cp_async16_a(sQb + (uint32_t)(row * LDS8 + ch * 16), Q8 + (size_t)(qrow0 + row) * ldq + col0 + ch * 16); }
    load_k(sK0, sm.sk[0], 0); load_v(sV0, 0); if (ntiles > 1) load_k(sK1, sm.sk[1], 1);
    cp_async_commit(); cp_async_wait_all(); __syncthreads();
    uint32_t qa[KS8][4];
    #pragma unroll
    for (int ks = 0; ks < KS8; ks++) ldsm_x4(qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], sQb + (uint32_t)((warp * 16 + (lane & 15)) * LDS8 + ks * 32 + (lane >> 4) * 16));
    const int r = lane >> 2; const float sq0 = sQ[(size_t)(qrow0 + warp * 16 + r) * H + h] * scale_log2e, sq1 = sQ[(size_t)(qrow0 + warp * 16 + r + 8) * H + h] * scale_log2e;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f; float o[DT][4];
    #pragma unroll
    for (int dt = 0; dt < DT; dt++) { o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.f; }
    auto qk = [&](float (&s)[8][4], uint32_t kbase, const float* skb) {   // s = (q8.k8) * sq_row * sk_col * scale (log2 units)
        int32_t si[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) { si[nt][0] = si[nt][1] = si[nt][2] = si[nt][3] = 0; }
        const uint32_t kb = kbase + k_lane;
        #pragma unroll
        for (int ks = 0; ks < KS8; ks++) {
            #pragma unroll
            for (int nt = 0; nt < 8; nt += 2) {
                uint32_t b[4]; ldsm_x4(b[0], b[1], b[2], b[3], kb + (uint32_t)(nt * 8 * LDS8 + ks * 32));
                mma_s8(si[nt], qa[ks], b); mma_s8(si[nt + 1], qa[ks], b + 2);
            }
        }
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) { int c = nt * 8 + (lane & 3) * 2; float k0 = skb[c], k1 = skb[c + 1];
            s[nt][0] = (float)si[nt][0] * sq0 * k0; s[nt][1] = (float)si[nt][1] * sq0 * k1; s[nt][2] = (float)si[nt][2] * sq1 * k0; s[nt][3] = (float)si[nt][3] * sq1 * k1; }
    };
    float s_cur[8][4]; qk(s_cur, sK0, sm.sk[0]);
    for (int t = 0; t < ntiles; t++) {
        const int vb = t & 1, kb = (t + 1) & 1;
        const uint32_t sVcur = vb ? sV1 : sV0, sVnxt = vb ? sV0 : sV1, sKnxt = kb ? sK1 : sK0, sKnn = kb ? sK0 : sK1; float* sknxt = sm.sk[kb]; float* sknn = sm.sk[kb ^ 1];
        if (t + 1 < ntiles) load_v(sVnxt, t + 1);
        float s_nxt[8][4];
        if (t + 1 < ntiles) qk(s_nxt, sKnxt, sknxt);          // consumes K_{t+1} and its scales before they are overwritten below
        __syncthreads();                                      // all warps done reading K_{t+1}'s scale row before K_{t+2} lands in that buffer? (scales buffer kb^1 holds K_t's, already consumed)
        if (t + 2 < ntiles) load_k(sKnn, sknn, t + 2);
        cp_async_commit();
        float mx0 = -1e30f, mx1 = -1e30f;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) { mx0 = fmaxf(mx0, fmaxf(s_cur[nt][0], s_cur[nt][1])); mx1 = fmaxf(mx1, fmaxf(s_cur[nt][2], s_cur[nt][3])); }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1); const float a0 = exp2f(m0 - mn0), a1 = exp2f(m1 - mn1); float rs0 = 0.f, rs1 = 0.f;
        uint32_t pa[4][4];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            float p0 = exp2f(s_cur[nt][0] - mn0), p1 = exp2f(s_cur[nt][1] - mn0), p2 = exp2f(s_cur[nt][2] - mn1), p3 = exp2f(s_cur[nt][3] - mn1);
            rs0 += p0 + p1; rs1 += p2 + p3; int kk = nt >> 1, hi = nt & 1; pa[kk][hi * 2 + 0] = pack_half2(p0, p1); pa[kk][hi * 2 + 1] = pack_half2(p2, p3);
        }
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1); rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2); rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1); rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
        l0 = l0 * a0 + rs0; l1 = l1 * a1 + rs1; m0 = mn0; m1 = mn1;
        #pragma unroll
        for (int dt = 0; dt < DT; dt++) { o[dt][0] *= a0; o[dt][1] *= a0; o[dt][2] *= a1; o[dt][3] *= a1; }
        const uint32_t vbase = sVcur + v_lane;
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            #pragma unroll
            for (int dt = 0; dt < DT; dt += 2) { uint32_t b[4]; ldsm_x4_trans(b[0], b[1], b[2], b[3], vbase + (uint32_t)((kk * 16 * LDS + dt * 8) * 2)); mma_f16(o[dt], pa[kk], b); mma_f16(o[dt + 1], pa[kk], b + 2); }
        }
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) { s_cur[nt][0] = s_nxt[nt][0]; s_cur[nt][1] = s_nxt[nt][1]; s_cur[nt][2] = s_nxt[nt][2]; s_cur[nt][3] = s_nxt[nt][3]; }
        cp_async_wait_all(); __syncthreads();
    }
    float inv0 = 1.f / l0, inv1 = 1.f / l1; const int row0 = qrow0 + warp * 16 + r;
    #pragma unroll
    for (int dt = 0; dt < DT; dt++) {
        int c = col0 + dt * 8 + (lane & 3) * 2;
        if (!OUT8) { *reinterpret_cast<uint32_t*>(O + (size_t)row0 * ld_out + c) = pack_half2(o[dt][0] * inv0, o[dt][1] * inv0); *reinterpret_cast<uint32_t*>(O + (size_t)(row0 + 8) * ld_out + c) = pack_half2(o[dt][2] * inv1, o[dt][3] * inv1); }
        else { char2 a, b; a.x = (int8_t)__float2int_rn(fminf(fmaxf(o[dt][0] * inv0 * inv_oscale, -127.f), 127.f)); a.y = (int8_t)__float2int_rn(fminf(fmaxf(o[dt][1] * inv0 * inv_oscale, -127.f), 127.f));
            b.x = (int8_t)__float2int_rn(fminf(fmaxf(o[dt][2] * inv1 * inv_oscale, -127.f), 127.f)); b.y = (int8_t)__float2int_rn(fminf(fmaxf(o[dt][3] * inv1 * inv_oscale, -127.f), 127.f));
            *reinterpret_cast<char2*>(O8 + (size_t)row0 * ld_out + c) = a; *reinterpret_cast<char2*>(O8 + (size_t)(row0 + 8) * ld_out + c) = b; }
    }
}
// workspace: M*H*64*2 (q8,k8) + M*H*4*2 (scales) bytes
template <int BR, bool OUT8 = false>
inline cudaError_t launch_pp8(const __half* Q, const __half* K, const __half* V, void* O, int M, int N, int H, float scale, void* ws, cudaStream_t st, float oscale = 0.f, int ld_v = 0, int ld_out = 0) {
    constexpr int D = 64; if (ld_v == 0) ld_v = H * D; if (ld_out == 0) ld_out = H * D;
    int8_t* q8 = (int8_t*)ws; int8_t* k8 = q8 + (size_t)M * H * D; float* sq = (float*)(k8 + (size_t)M * H * D); float* sk = sq + (size_t)M * H;
    quantize_rows_kernel<<<(M * H + 7) / 8, 256, 0, st>>>(Q, q8, sq, M, H); quantize_rows_kernel<<<(M * H + 7) / 8, 256, 0, st>>>(K, k8, sk, M, H);
    struct SmemPP8 { int8_t q8[BR * Cfg<D>::LDS8]; int8_t k8[2][BC * Cfg<D>::LDS8]; float sk[2][BC]; __half v[2][BC * Cfg<D>::LDS]; };
    size_t smem = sizeof(SmemPP8); static bool attr_set = false;
    if (!attr_set) { cudaFuncSetAttribute(attn_kernel_pp8<BR, OUT8, D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr_set = true; }
    dim3 grid(N / BR, H, M / N); attn_kernel_pp8<BR, OUT8, D><<<grid, BR * 2, smem, st>>>(q8, k8, sq, sk, V, O, N, H, scale * 1.4426950408889634f, oscale > 0.f ? 1.f / oscale : 0.f, ld_v, ld_out);
    return cudaGetLastError();
}
}  // namespace lqattn
