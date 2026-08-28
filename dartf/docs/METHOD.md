# Method

DARTF quantizes the SAM3 ViT-H trunk to INT8 weights and INT8 activations (W8A8) and runs it with TensorRT plugins. Every step below is either exact (bit-level rewrite of the network) or measured against the FP32 engine at detection level.

## Quantization grammar
Symmetric INT8, zero point 0. Weights: one scale per output channel, rounded by activation-aware GPTQ. Activations: one static scale per tensor (p99.999 of 16 calibration images). Attention (Q·Kᵀ, softmax, P·V), normalizations, GELU, RoPE and the FPN neck stay FP16. Quantized sites per block: the shared q/k/v input, the attention-output (proj) input, the fc1 input, the fc2 input and the four weight matrices. Block 0 stays FP16 in the default recipe (INT8 in the speed variant).

## Exact rewrites (verified against PyTorch to rel 5e-6)
1. **Rotated residual stream.** For any orthogonal Q whose first row is the ones direction, LayerNorm(x)·Q equals an RMSNorm over the rotated coordinates 1…C−1 with coordinate 0 zeroed. The rotation therefore folds into the q/k/v/fc1 weights (right), the proj/fc2 weights (left) and one rotation of the patch embedding; the network runs a masked RMSNorm instead of LayerNorm. With a Walsh–Hadamard Q the crest factor of the LayerNorm-fed activations drops from 28.4 to 4.07, which is what makes static per-tensor INT8 work at these sites.
2. **Per-head V rotation.** The attention output a = P·V is linear in V, so a per-head orthogonal R folds into v_proj (rows) and o_proj (columns) and lowers the crest factor of the proj input.
3. **RoPE fold.** The pairwise rotation is a signed column permutation of the q/k weights, exact even on INT8 codes; the rotated branch becomes a second projection, which the plugins fold into the GEMM epilogue.
4. **Window-major token layout.** Tokens are partitioned into windows once after the patch embedding and un-partitioned once before the neck; global blocks use permuted RoPE tables. All 32 blocks become structurally identical.
5. **Per-channel fc2 activation scales.** x_k/c_k with the k-th row of W2 scaled by c_k (SmoothQuant on the one site where rotation cannot reach); the plugin applies 1/c_k in the fc1 requantization epilogue.

## Activation-aware GPTQ with bias correction
Blocks are quantized in order. Each block is fed the fake-quantized output of the already quantized prefix; the site's activation quantizer is applied to the inputs before forming the Hessian, and each output channel's bias is corrected by the mean quantization error. The emitted scales, codes and biases are consumed verbatim by the ONNX quantizer, so the deployed graph is exactly the one the Hessians saw.

## Pipeline
HF weights → DART ONNX export in the rotated, window-major basis (`export/`) → FP32 TensorRT calibration engine (`quant/collect_calib_hf.py`) → activation-aware GPTQ on CPU (`quant/gptq_actaware.py --hadamard --head-rot --fc2-chan 0.5`) → ONNX quantization, GELU fusion, RoPE fold, plugin splices (`quant/`, `plugins/`) → TensorRT build with `lq_plugins.so` → verification against onnxruntime FP32 and detection-level evaluation (`runtime/`). `scripts/pipeline.sh` runs all of it.
