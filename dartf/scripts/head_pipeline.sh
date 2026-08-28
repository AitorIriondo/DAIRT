#!/usr/bin/env bash
# Grounding head: exact prefix sharing (+ optional INT8 encoder FFN) from the exported bucket graphs to validated engines.
# usage: head_pipeline.sh <ground_c1.onnx> <ground_cK.onnx> <out_dir> [--int8-ffn calib_ground.json]
set -euo pipefail; C1=$1; CK=$2; OUT=$3; shift 3; INT8=""; [ "${1:-}" = "--int8-ffn" ] && INT8=$2
R="$(cd "$(dirname "$0")/.." && pwd)"; PY=${PY:-python3}; H=${HANDOFF:-$HOME/sam3_orin_agx64_jp72_handoff}; mkdir -p "$OUT"
$PY "$R/export/split_ground_prefix.py" "$C1" "$C1" "$OUT/ground_pre.onnx" "$OUT/_unused_post.onnx"; rm -f "$OUT/_unused_post.onnx"*      # prefix from the K=1 export (B=1)
$PY "$R/export/split_ground_c16.py" "$CK" "$H/goldens/inputs" "$OUT/ground_post.onnx"                                                   # post from the K-bucket export
if [ -n "$INT8" ]; then $PY "$R/quant/quantize_ground_ffn_c16.py" "$OUT/ground_post.onnx" "$INT8" "$OUT/ground_post_int8ffn.onnx"; POST="$OUT/ground_post_int8ffn.onnx"; else POST="$OUT/ground_post.onnx"; fi
$PY "$R/runtime/build_engine.py" "$OUT/ground_pre.onnx" "$OUT/ground_pre.plan" --no-int8
if [ -n "$INT8" ]; then $PY "$R/runtime/build_engine.py" "$POST" "$OUT/ground_post.plan"; else $PY "$R/runtime/build_engine.py" "$POST" "$OUT/ground_post.plan" --no-int8; fi
$PY "$R/runtime/ground_split_cmp.py" "$H/run_orin/engine/ground_c16.plan" "$OUT/ground_pre.plan" "$OUT/ground_post.plan"                  # timing + rel-L2 vs the monolithic bucket engine
echo "engines: $OUT/ground_pre.plan (once per image) + $OUT/ground_post.plan (per bucket); evaluate with runtime/eval_pipeline.py --ground <post> --ground-pre <pre>"
