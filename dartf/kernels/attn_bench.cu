// correctness (vs CPU fp32 reference) + timing for lqattn::attn_kernel on SAM3 shapes.
// nvcc -O3 -std=c++17 -arch=sm_87 attn_bench.cu -o attn_bench
#include "lq_attn_kernel.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

static void reference(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V, std::vector<float>& O, int M, int N, int H, float scale, int max_rows, int D = 64) {
    const int ld = H * D; std::vector<float> s(N);
    for (int w = 0; w < M / N; w++) for (int h = 0; h < H; h++) for (int i = 0; i < N; i++) {
        int gi = w * N + i; if (gi >= max_rows) continue;
        float mx = -1e30f;
        for (int j = 0; j < N; j++) { float acc = 0; for (int d = 0; d < D; d++) acc += Q[(size_t)gi * ld + h * D + d] * K[(size_t)(w * N + j) * ld + h * D + d]; s[j] = acc * scale; mx = std::max(mx, s[j]); }
        float sum = 0; for (int j = 0; j < N; j++) { s[j] = expf(s[j] - mx); sum += s[j]; }
        for (int d = 0; d < D; d++) { float acc = 0; for (int j = 0; j < N; j++) acc += s[j] * V[(size_t)(w * N + j) * ld + h * D + d]; O[(size_t)gi * ld + h * D + d] = acc / sum; }
    }
}
template <int BR, bool I8, int D = 64>
static void run_case(const char* name, int M, int N, int H, __half* dQ, __half* dK, __half* dV, __half* dO, const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V, int check_rows) {
    const float scale = 1.f / sqrtf((float)D); const int ld = H * D;
    static void* ws = nullptr; if (!ws) cudaMalloc(&ws, (size_t)M * H * 136 + 1024);
    cudaError_t e = lqattn::launch<BR, I8, false, D>(dQ, dK, dV, dO, M, N, H, scale, ws, 0); cudaDeviceSynchronize();
    if (e != cudaSuccess || cudaGetLastError() != cudaSuccess) { printf("%s: launch error %s\n", name, cudaGetErrorString(e)); return; }
    if (check_rows > 0) {
        std::vector<__half> Oh((size_t)M * ld); cudaMemcpy(Oh.data(), dO, Oh.size() * 2, cudaMemcpyDeviceToHost);
        std::vector<float> Oref((size_t)M * ld, 0.f); reference(Q, K, V, Oref, M, N, H, scale, check_rows, D);
        double num = 0, den = 0, maxabs = 0;
        for (int i = 0; i < check_rows; i++) for (int c = 0; c < ld; c++) { double g = __half2float(Oh[(size_t)i * ld + c]), r = Oref[(size_t)i * ld + c]; num += (g - r) * (g - r); den += r * r; maxabs = std::max(maxabs, fabs(g - r)); }
        printf("%s: rel_l2=%.3e max_abs=%.3e (rows checked %d)\n", name, sqrt(num / den), maxabs, check_rows);
    }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) lqattn::launch<BR, I8, false, D>(dQ, dK, dV, dO, M, N, H, scale, ws, 0);
    cudaEventRecord(e0); for (int i = 0; i < 20; i++) lqattn::launch<BR, I8, false, D>(dQ, dK, dV, dO, M, N, H, scale, ws, 0); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 20; double flops = 4.0 * (M / N) * H * (double)N * N * D;
    printf("%s: %.4f ms  (%.1f TFLOPS-equiv)\n", name, ms, flops / (ms * 1e-3) / 1e12);
}
int main() {
    const int H = 16, ld = H * 64, M = 5184;
    std::vector<float> Q((size_t)M * ld), K((size_t)M * ld), V((size_t)M * ld); std::vector<__half> Qh(Q.size()), Kh(Q.size()), Vh(Q.size());
    srand(7); auto rnd = [] { return (rand() / (float)RAND_MAX - 0.5f) * 4.f; };
    for (size_t i = 0; i < Q.size(); i++) { Qh[i] = __float2half(rnd()); Kh[i] = __float2half(rnd()); Vh[i] = __float2half(rnd()); Q[i] = __half2float(Qh[i]); K[i] = __half2float(Kh[i]); V[i] = __half2float(Vh[i]); }
    // make a few channels "outlier" so int8 per-row quantization is exercised
    for (int i = 0; i < M; i++) { Qh[(size_t)i * ld + 5] = __float2half(rnd() * 8.f); Q[(size_t)i * ld + 5] = __half2float(Qh[(size_t)i * ld + 5]); Kh[(size_t)i * ld + 77] = __float2half(rnd() * 8.f); K[(size_t)i * ld + 77] = __half2float(Kh[(size_t)i * ld + 77]); }
    __half *dQ, *dK, *dV, *dO; cudaMalloc(&dQ, Q.size() * 2); cudaMalloc(&dK, Q.size() * 2); cudaMalloc(&dV, Q.size() * 2); cudaMalloc(&dO, Q.size() * 2);
    cudaMemcpy(dQ, Qh.data(), Q.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dK, Kh.data(), Q.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dV, Vh.data(), Q.size() * 2, cudaMemcpyHostToDevice);
    printf("windowed N=576 (9 windows x 16 heads), TRT fused MHA reference 0.79 ms\n");
    run_case<64, false>("  fp16 BR64       ", M, 576, H, dQ, dK, dV, dO, Q, K, V, 1152);
    run_case<96, false>("  fp16 BR96       ", M, 576, H, dQ, dK, dV, dO, Q, K, V, 1152);
    run_case<192, false>("  fp16 BR192      ", M, 576, H, dQ, dK, dV, dO, Q, K, V, 1152);
    run_case<96, true >("  int8QK BR96     ", M, 576, H, dQ, dK, dV, dO, Q, K, V, 1152);
    printf("global N=5184 (1 x 16 heads), TRT fused MHA reference 5.31 ms\n");
    run_case<64, false>("  fp16 BR64       ", M, 5184, H, dQ, dK, dV, dO, Q, K, V, 128);
    run_case<96, false>("  fp16 BR96       ", M, 5184, H, dQ, dK, dV, dO, Q, K, V, 128);
    printf("grounding encoder: global N=5184, H=8, D=32 (TRT MHA ~11 ms/class for 6 layers -> ~1.8 ms/layer)\n");
    run_case<64, false, 32>("  fp16 BR64 d32   ", M, 5184, 8, dQ, dK, dV, dO, Q, K, V, 128);
    run_case<96, false, 32>("  fp16 BR96 d32   ", M, 5184, 8, dQ, dK, dV, dO, Q, K, V, 128);
    return 0;
}
