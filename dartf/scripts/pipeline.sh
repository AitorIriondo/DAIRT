#!/usr/bin/env bash
# End-to-end: HF weights -> Hadamard + window-major ONNX -> calibration -> activation-aware GPTQ -> quantize/fold/splice -> TensorRT engine -> verify
# Default recipe: INT8 weights and activations in all 32 blocks (block 0 included). To keep block 0's activations in FP16 (164 ms instead of 158 ms on the Orin,
# same COCO AP) use --act-blocks 1-31 in both quantization steps and drop --no-int8 at build.
# usage: DART_ROOT=... CALIB_IMAGES=dir CALIB_IDS=ids.json CHECK_IMG=img.jpg ./pipeline.sh out_dir
set -euo pipefail
OUT=${1:-out}; mkdir -p "$OUT"; R="$(cd "$(dirname "$0")/.." && pwd)"; PY=${PY:-python3}
: "${DART_ROOT:?set DART_ROOT}" "${CALIB_IMAGES:?}" "${CALIB_IDS:?}" "${CHECK_IMG:?}"
PL="$R/plugins/lq_plugins.so"; [ -f "$PL" ] || (cd "$R/plugins" && ./build.sh)
LQ_HEAD_ROT=1 $PY "$R/export/export_hf_had_nowin.py" "$DART_ROOT" "$OUT/hadnw" 1008 "$CHECK_IMG"
$PY "$R/export/rename_hf_io.py" "$OUT/hadnw/hf_backbone.onnx" "$OUT/vision_1008_hf_hadnw.onnx" vision_1008_hf_hadnw.onnx.data && mv "$OUT/hadnw/hf_backbone.onnx.data" "$OUT/vision_1008_hf_hadnw.onnx.data"
(cd "$OUT" && $PY "$R/quant/collect_calib_hf.py" vision_1008_hf_hadnw.onnx had 1008)      # writes calib_had.json/.npz in $OUT
LQ_HEAD_ROT=1 $PY "$R/quant/gptq_actaware.py" --images "$CALIB_IMAGES" --ids "$CALIB_IDS" --out "$OUT/gptq_aah" --blocks 0-31 --act-blocks 0-31 --smooth 0 --hadamard --head-rot
$PY "$R/quant/quantize_hf.py" "$OUT/vision_1008_hf_hadnw.onnx" "$OUT/q0.onnx" "$OUT/calib_had.json" --calib-npz "$OUT/calib_had.npz" --act p99999 --blocks 0-31 --act-blocks 0-31 --always-act-fams proj \
   --gptq-cache "$OUT/gptq_aah_cache.npz" --act-override "$OUT/gptq_aah_act.json" --bias-override "$OUT/gptq_aah_bias.json"
$PY "$R/quant/fuse_gelu.py" "$OUT/q0.onnx" "$OUT/q0g.onnx" && mv -f "$OUT/q0g.onnx" "$OUT/q0.onnx" && mv -f "$OUT/q0g.onnx.data" "$OUT/q0.onnx.data" 2>/dev/null; $PY "$R/quant/fix_ext_location.py" "$OUT/q0.onnx" q0.onnx.data
$PY "$R/quant/fold_rope.py" "$OUT/q0.onnx" "$OUT/q1.onnx"
$PY "$R/plugins/splice_rope_plugin.py" "$OUT/q1.onnx" "$OUT/q2.onnx"
$PY "$R/plugins/splice_attn_plugin.py" "$OUT/q2.onnx" "$OUT/q3.onnx"
$PY "$R/plugins/splice_mlp_plugin.py" "$OUT/q3.onnx" "$OUT/q4.onnx" --gelu tanh
$PY "$R/plugins/splice_block_plugins.py" "$OUT/q4.onnx" "$OUT/vision_int8.onnx" --qkv --attnproj
$PY "$R/runtime/build_engine.py" "$OUT/vision_int8.onnx" "$OUT/vision_int8.plan" --no-int8 --plugins "$PL"    # all 32 blocks are plugins: no Q/DQ remains for TensorRT
LQ_PLUGINS="$PL" $PY "$R/runtime/verify_plan.py" "$OUT/vision_int8.plan"
echo "done: $OUT/vision_int8.plan"
