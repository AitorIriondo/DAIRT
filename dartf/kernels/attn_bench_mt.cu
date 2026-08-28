// MT (rows-per-warp) sweep for lqattn::attn_kernel_mt vs the baseline kernel. nvcc -O3 -std=c++17 -arch=sm_87 -Xptxas -v attn_bench_mt.cu -o attn_bench_mt
#include "lq_attn_kernel.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
template <int BR, int MT, int D>
static void run_mt(const char* name, int M, int N, int H, __half* dQ, __half* dK, __half* dV, __half* dO, const std::vector<__half>& Oref, int check_rows) {
    const float scale = 1.f / sqrtf((float)D); const int ld = H * D;
    cudaError_t e = lqattn::launch_mt<BR, MT, false, D>(dQ, dK, dV, dO, M, N, H, scale, 0); cudaDeviceSynchronize();
    if (e != cudaSuccess || cudaGetLastError() != cudaSuccess) { printf("%s: launch error %s\n", name, cudaGetErrorString(e)); return; }
    std::vector<__half> Oh((size_t)M * ld); cudaMemcpy(Oh.data(), dO, Oh.size() * 2, cudaMemcpyDeviceToHost);
    double num = 0, den = 0; for (int i = 0; i < check_rows; i++) for (int c = 0; c < ld; c++) { double g = __half2float(Oh[(size_t)i * ld + c]), r = __half2float(Oref[(size_t)i * ld + c]); num += (g - r) * (g - r); den += r * r; }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) lqattn::launch_mt<BR, MT, false, D>(dQ, dK, dV, dO, M, N, H, scale, 0);
    cudaEventRecord(e0); for (int i = 0; i < 20; i++) lqattn::launch_mt<BR, MT, false, D>(dQ, dK, dV, dO, M, N, H, scale, 0); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 20; double flops = 4.0 * (M / N) * H * (double)N * N * D;
    printf("%s: %.4f ms  (%.1f TFLOPS-equiv)  rel_vs_base=%.2e\n", name, ms, flops / (ms * 1e-3) / 1e12, sqrt(num / den));
}
template <int BR, int D, bool H2 = false>
static void run_pp(const char* name, int M, int N, int H, __half* dQ, __half* dK, __half* dV, __half* dO, const std::vector<__half>& Oref, int check_rows) {
    const float scale = 1.f / sqrtf((float)D); const int ld = H * D;
    cudaError_t e = lqattn::launch_pp<BR, false, D, H2>(dQ, dK, dV, dO, M, N, H, scale, 0); cudaDeviceSynchronize();
    if (e != cudaSuccess || cudaGetLastError() != cudaSuccess) { printf("%s: launch error %s\n", name, cudaGetErrorString(e)); return; }
    std::vector<__half> Oh((size_t)M * ld); cudaMemcpy(Oh.data(), dO, Oh.size() * 2, cudaMemcpyDeviceToHost);
    double num = 0, den = 0; for (int i = 0; i < check_rows; i++) for (int c = 0; c < ld; c++) { double g = __half2float(Oh[(size_t)i * ld + c]), r = __half2float(Oref[(size_t)i * ld + c]); num += (g - r) * (g - r); den += r * r; }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) lqattn::launch_pp<BR, false, D, H2>(dQ, dK, dV, dO, M, N, H, scale, 0);
    cudaEventRecord(e0); for (int i = 0; i < 20; i++) lqattn::launch_pp<BR, false, D, H2>(dQ, dK, dV, dO, M, N, H, scale, 0); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 20; double flops = 4.0 * (M / N) * H * (double)N * N * D;
    printf("%s: %.4f ms  (%.1f TFLOPS-equiv)  rel_vs_base=%.2e\n", name, ms, flops / (ms * 1e-3) / 1e12, sqrt(num / den));
}
template <int BR, int D>
static std::vector<__half> base(const char* name, int M, int N, int H, __half* dQ, __half* dK, __half* dV, __half* dO) {
    const float scale = 1.f / sqrtf((float)D); const int ld = H * D; static void* ws = nullptr; if (!ws) cudaMalloc(&ws, (size_t)M * 16 * 136 + 1024);
    lqattn::launch<BR, false, false, D>(dQ, dK, dV, dO, M, N, H, scale, ws, 0); cudaDeviceSynchronize();
    std::vector<__half> Oh((size_t)M * ld); cudaMemcpy(Oh.data(), dO, Oh.size() * 2, cudaMemcpyDeviceToHost);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) lqattn::launch<BR, false, false, D>(dQ, dK, dV, dO, M, N, H, scale, ws, 0);
    cudaEventRecord(e0); for (int i = 0; i < 20; i++) lqattn::launch<BR, false, false, D>(dQ, dK, dV, dO, M, N, H, scale, ws, 0); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 20; printf("%s: %.4f ms (baseline MT=1)\n", name, ms); return Oh;
}
int main() {
    const int H = 16, ld = H * 64, M = 5184;
    std::vector<__half> Qh((size_t)M * ld), Kh(Qh.size()), Vh(Qh.size()); srand(7); auto rnd = [] { return (rand() / (float)RAND_MAX - 0.5f) * 4.f; };
    for (size_t i = 0; i < Qh.size(); i++) { Qh[i] = __float2half(rnd()); Kh[i] = __float2half(rnd()); Vh[i] = __float2half(rnd()); }
    __half *dQ, *dK, *dV, *dO; cudaMalloc(&dQ, Qh.size() * 2); cudaMalloc(&dK, Qh.size() * 2); cudaMalloc(&dV, Qh.size() * 2); cudaMalloc(&dO, Qh.size() * 2);
    cudaMemcpy(dQ, Qh.data(), Qh.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dK, Kh.data(), Qh.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dV, Vh.data(), Qh.size() * 2, cudaMemcpyHostToDevice);
    printf("windowed N=576 d64 H16 (TRT 0.79 ms)\n");
    { auto ref = base<96, 64>("  base BR96", M, 576, H, dQ, dK, dV, dO);
      run_mt<64, 2, 64>("  MT2 BR64 ", M, 576, H, dQ, dK, dV, dO, ref, 1152); run_mt<96, 2, 64>("  MT2 BR96 ", M, 576, H, dQ, dK, dV, dO, ref, 1152);
      run_mt<192, 2, 64>("  MT2 BR192", M, 576, H, dQ, dK, dV, dO, ref, 1152); run_mt<96, 1, 64>("  MT1 BR96 ", M, 576, H, dQ, dK, dV, dO, ref, 1152);
      run_pp<64, 64>("  PP  BR64 ", M, 576, H, dQ, dK, dV, dO, ref, 1152); run_pp<96, 64>("  PP  BR96 ", M, 576, H, dQ, dK, dV, dO, ref, 1152); run_pp<192, 64>("  PP  BR192", M, 576, H, dQ, dK, dV, dO, ref, 1152);
      run_pp<64, 64, true>("  PPH BR64 ", M, 576, H, dQ, dK, dV, dO, ref, 1152); run_pp<96, 64, true>("  PPH BR96 ", M, 576, H, dQ, dK, dV, dO, ref, 1152); run_pp<192, 64, true>("  PPH BR192", M, 576, H, dQ, dK, dV, dO, ref, 1152); }
    printf("global N=5184 d64 H16 (TRT 5.31 ms)\n");
    { auto ref = base<64, 64>("  base BR64", M, 5184, H, dQ, dK, dV, dO);
      run_mt<64, 2, 64>("  MT2 BR64 ", M, 5184, H, dQ, dK, dV, dO, ref, 256); run_mt<96, 2, 64>("  MT2 BR96 ", M, 5184, H, dQ, dK, dV, dO, ref, 256);
      run_mt<192, 2, 64>("  MT2 BR192", M, 5184, H, dQ, dK, dV, dO, ref, 256);
      run_pp<64, 64>("  PP  BR64 ", M, 5184, H, dQ, dK, dV, dO, ref, 256); run_pp<96, 64>("  PP  BR96 ", M, 5184, H, dQ, dK, dV, dO, ref, 256); run_pp<192, 64>("  PP  BR192", M, 5184, H, dQ, dK, dV, dO, ref, 256);
      run_pp<64, 64, true>("  PPH BR64 ", M, 5184, H, dQ, dK, dV, dO, ref, 256); run_pp<96, 64, true>("  PPH BR96 ", M, 5184, H, dQ, dK, dV, dO, ref, 256); run_pp<192, 64, true>("  PPH BR192", M, 5184, H, dQ, dK, dV, dO, ref, 256); }
    printf("grounding N=5184 d32 H8 (TRT ~1.8 ms/layer)\n");
    { auto ref = base<64, 32>("  base BR64", M, 5184, 8, dQ, dK, dV, dO);
      run_mt<64, 2, 32>("  MT2 BR64 ", M, 5184, 8, dQ, dK, dV, dO, ref, 256); run_mt<96, 2, 32>("  MT2 BR96 ", M, 5184, 8, dQ, dK, dV, dO, ref, 256);
      run_mt<192, 2, 32>("  MT2 BR192", M, 5184, 8, dQ, dK, dV, dO, ref, 256); run_pp<64, 32>("  PP  BR64 ", M, 5184, 8, dQ, dK, dV, dO, ref, 256); run_pp<96, 32>("  PP  BR96 ", M, 5184, 8, dQ, dK, dV, dO, ref, 256); run_pp<192, 32>("  PP  BR192", M, 5184, 8, dQ, dK, dV, dO, ref, 256);
      run_pp<64, 32, true>("  PPH BR64 ", M, 5184, 8, dQ, dK, dV, dO, ref, 256); run_pp<96, 32, true>("  PPH BR96 ", M, 5184, 8, dQ, dK, dV, dO, ref, 256); run_pp<192, 32, true>("  PPH BR192", M, 5184, 8, dQ, dK, dV, dO, ref, 256); }
    return 0;
}
