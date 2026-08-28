// CUTLASS INT8 tensor-op GEMM microbench for SAM3 trunk shapes on SM87: D[M,N] (fp16) = alpha * A_int8[M,K] . B_int8[K,N] + C
// A row-major (K contiguous), B stored as [N,K] row-major == column-major [K,N] (K contiguous) -> the int8 TN layout.
// Sweeps threadblock tile shapes; reports best ms and TOPS per shape.
// nvcc -O3 -std=c++17 -arch=sm_87 -I cutlass/include cutlass_int8_bench.cu -o cutlass_int8_bench
#include <cuda.h>
// CUDA 13.2 no longer exposes the TMA PFN typedefs that CUTLASS 3.5's host adapter names; they are never called on SM87.
typedef CUresult (*PFN_cuTensorMapEncodeTiled)(...);
typedef CUresult (*PFN_cuTensorMapEncodeIm2col)(...);
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/numeric_types.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>

template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int STAGES>
double run_cfg(int M, int N, int K, void* dA, void* dB, void* dC, void* dD, const char* label) {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        int8_t, cutlass::layout::RowMajor,            // A [M,K]
        int8_t, cutlass::layout::ColumnMajor,         // B [K,N] col-major == [N,K] row-major
        cutlass::half_t, cutlass::layout::RowMajor,   // C/D [M,N]
        int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<TBM, TBN, TBK>, cutlass::gemm::GemmShape<WM, WN, WK>, cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, int32_t, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, STAGES>;
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1,
        {1e-4f, 1.0f}, dA, dB, dC, dD, (int64_t)M * K, (int64_t)N * K, (int64_t)0, (int64_t)M * N,   // batch strides
        K, K, 0 /*ldc=0: bias row broadcast*/, N);
    Gemm gemm; size_t ws = Gemm::get_workspace_size(args); void* dws = nullptr; if (ws) cudaMalloc(&dws, ws);
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) { printf("    %-28s cannot implement\n", label); return -1; }
    if (gemm.initialize(args, dws) != cutlass::Status::kSuccess) { printf("    %-28s init failed\n", label); return -1; }
    for (int i = 0; i < 3; i++) gemm();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
    for (int i = 0; i < 20; i++) gemm();
    cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 20;
    cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { printf("    %-28s cuda error %s\n", label, cudaGetErrorString(err)); return -1; }
    double tops = 2.0 * M * N * K / (ms * 1e-3) / 1e12; printf("    %-28s %.4f ms  %.1f TOPS\n", label, ms, tops);
    if (dws) cudaFree(dws); return ms;
}

int main() {
    int shapes[4][3] = {{5184, 1024, 1024}, {5184, 5120, 1024}, {5184, 4736, 1024}, {5184, 1024, 4736}};   // M,N,K
    const char* nm[4] = {"proj (TRT 0.22 ms)", "qkv+rope 5120 (TRT 1.02/1.44)", "fc1 4736 (TRT 1.44/1.02)", "fc2 K=4736 (TRT 0.77)"};
    void *dA, *dB, *dC, *dD; cudaMalloc(&dA, 5184 * 4736); cudaMalloc(&dB, 5120 * 4736); cudaMalloc(&dC, 5120 * 2); cudaMalloc(&dD, (size_t)5184 * 5120 * 2);
    cudaMemset(dA, 1, 5184 * 4736); cudaMemset(dB, 1, 5120 * 4736); cudaMemset(dC, 0, 5120 * 2);
    for (int s = 0; s < 4; s++) {
        int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2]; printf("%s  M=%d N=%d K=%d\n", nm[s], M, N, K);
        run_cfg<128, 256, 64, 64, 64, 64, 3>(M, N, K, dA, dB, dC, dD, "128x256x64 w64x64 s3");
        run_cfg<128, 256, 64, 64, 64, 64, 4>(M, N, K, dA, dB, dC, dD, "128x256x64 w64x64 s4");
        run_cfg<128, 256, 64, 64, 64, 64, 5>(M, N, K, dA, dB, dC, dD, "128x256x64 w64x64 s5");
        run_cfg<128, 256, 128, 64, 64, 128, 3>(M, N, K, dA, dB, dC, dD, "128x256x128 w64x64 s3");
        run_cfg<256, 256, 64, 64, 64, 64, 3>(M, N, K, dA, dB, dC, dD, "256x256x64 w64x64 s3");
        run_cfg<128, 256, 64, 32, 128, 64, 3>(M, N, K, dA, dB, dC, dD, "128x256x64 w32x128 s3");
        run_cfg<256, 128, 128, 64, 64, 128, 3>(M, N, K, dA, dB, dC, dD, "256x128x128 w64x64 s3");
    }
    // correctness spot check on the last run: A=1,B=1 -> acc=K -> D = 1e-4*K + C(0)
    __half h; cudaMemcpy(&h, dD, 2, cudaMemcpyDeviceToHost); printf("spot: D[0]=%.4f (expect %.4f)\n", __half2float(h), 1e-4f * 4736);
    return 0;
}
