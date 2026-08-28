# Kernels and plugins

## Block-level TensorRT plugins
TensorRT's explicit-quantization optimizer does not pass a QuantizeLinear scale to a plugin and does not accept INT8 plugin outputs, so a plugin that replaces a single fused kernel becomes an island. DARTF therefore internalizes whole sub-blocks behind fp16 residual-stream edges:
- `LQQkvRope`: masked RMSNorm → INT8 quantization → q/k GEMMs with RoPE in the epilogue → v GEMM.
- `LQAttnProj`: fused attention with an INT8 epilogue → proj GEMM with bias and residual add.
- `LQMlp`: norm → quantization → fc1 with GELU and requantization in the epilogue (optionally with per-channel fc2 scales) → fc2 with bias and residual add.
GEMMs are CUTLASS 3.5 INT8 tensor-op kernels (128×256×64 tiles, 5 pipeline stages). The library exports `getCreators` and `setLoggerFinder`, so `trtexec --dynamicPlugins` and the TensorRT Python registry both load it; `plugins/build.sh` builds it for any sm80+ target.

## Fused attention kernel (`kernels/lq_attn_kernel.cuh`)
Flash-attention-2 structure for SM87 with token-major `[M, H·d]` inputs and outputs (no head transposes), fp32 online softmax with exp2, INT8 output option for the following proj GEMM, head dimensions 64 and 32. The kernel is software pipelined: Q·K_{t+1}ᵀ is issued on the tensor pipe before the softmax of S_t runs on the FP32 pipe, with the scores double-buffered in registers; shared-memory addresses are hoisted to 32-bit bases with compile-time immediates. On the Orin it runs the windowed block (N=576) in 0.625 ms and the global block (N=5184) in 4.82 ms, against 0.79 and 5.31 ms for TensorRT's fused MHA, exact to fp16 rounding.

## Grounding head
See `HEAD.md`.

## Shared memory on sm86 and sm89

The INT8 GEMMs use 128 x 256 x 64 tiles with a 5-stage `cp.async` pipeline, which needs 120 KB of shared memory per block. That fits sm80, sm87 and sm90 (164 KB or more) but not sm86/sm89 (99 KB per block), where the kernels would fail at enqueue. The stage count is the `LQ_STAGES` macro; `plugins/build.sh` sets it to 3 for `ARCH=86` and `ARCH=89` and to 5 otherwise.
