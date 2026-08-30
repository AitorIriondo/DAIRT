// small torch-free helpers for the SAM 3 tracker driver: row gather (fp16 rows of 128 B) from a table of device pointers
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
// dst[b, j, :] = srcs[src_row_ptr[b*Kn + j]] (row = 64 fp16 = 128 B); rows past n_valid[b] are zeroed. src_row_ptr holds device addresses.
__global__ void gather_rows64(__half* __restrict__ dst, const unsigned long long* __restrict__ src_row_ptr, const int* __restrict__ n_valid, int Kn) {
    int j = blockIdx.x * blockDim.x / 8 + threadIdx.x / 8; int lane = threadIdx.x & 7; int b = blockIdx.y; if (j >= Kn) return;
    uint4* d = reinterpret_cast<uint4*>(dst + ((size_t)b * Kn + j) * 64) + lane;                 // 8 x 16 B per row
    if (j < n_valid[b]) { const uint4* s = reinterpret_cast<const uint4*>(src_row_ptr[(size_t)b * Kn + j]) + lane; *d = *s; }
    else *d = make_uint4(0, 0, 0, 0);
}
extern "C" int lq_gather_rows64(void* dst, const void* src_row_ptr, const void* n_valid, int B, int Kn, cudaStream_t st) {
    dim3 grid((Kn * 8 + 255) / 256, B); gather_rows64<<<grid, 256, 0, st>>>((__half*)dst, (const unsigned long long*)src_row_ptr, (const int*)n_valid, Kn); return (int)cudaGetLastError();
}
// dst[b] = src_ptr[b] rows of `bytes` (mask rows etc.), generic byte copy by 16 B chunks
__global__ void gather_blocks(uint8_t* __restrict__ dst, const unsigned long long* __restrict__ src_ptr, size_t bytes) {
    int b = blockIdx.y; size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 16; if (i >= bytes) return;
    *reinterpret_cast<uint4*>(dst + (size_t)b * bytes + i) = *reinterpret_cast<const uint4*>((const uint8_t*)src_ptr[b] + i);
}
extern "C" int lq_gather_blocks(void* dst, const void* src_ptr, int B, size_t bytes, cudaStream_t st) {
    dim3 grid((unsigned)((bytes / 16 + 255) / 256), B); gather_blocks<<<grid, 256, 0, st>>>((uint8_t*)dst, (const unsigned long long*)src_ptr, bytes); return (int)cudaGetLastError();
}
// [64, 5184] channel-major memory map (fp32 or fp16 as produced by the engine) -> [5184, 64] token-major fp16, B maps
template <typename T> __global__ void transpose64(__half* __restrict__ dst, const T* __restrict__ src) {
    int b = blockIdx.y; int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= 5184) return;
    const T* s = src + (size_t)b * 64 * 5184 + t; __half* d = dst + ((size_t)b * 5184 + t) * 64;
    #pragma unroll
    for (int c = 0; c < 64; c++) d[c] = __float2half((float)s[(size_t)c * 5184]);
}
extern "C" int lq_transpose64(void* dst, const void* src, int B, int src_fp32, cudaStream_t st) {
    dim3 grid((5184 + 255) / 256, B);
    if (src_fp32) transpose64<float><<<grid, 256, 0, st>>>((__half*)dst, (const float*)src); else transpose64<__half><<<grid, 256, 0, st>>>((__half*)dst, (const __half*)src);
    return (int)cudaGetLastError();
}
