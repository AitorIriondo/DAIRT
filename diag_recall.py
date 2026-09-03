#!/usr/bin/env python3
"""Per-class detection recall over a video.

Answers "which class prompt actually fires, and how often" — the question that
sits underneath both tracking churn and contact recall. Detection-only, so it
runs at full TRT speed; masks are not needed to count detections.

Example:
    PYTHONIOENCODING=utf-8 python diag_recall.py \\
        --video person_tool_video.mp4 \\
        --classes person hand glove "power tool" \\
        --checkpoint sam3.pt --trt hf_backbone_fp16.engine --stride 3
"""

import argparse
import time

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--trt", default=None)
    ap.add_argument("--imgsz", type=int, default=1008)
    ap.add_argument("--confidence", type=float, default=0.2)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint, device="cuda",
        eval_mode=True, resolution=args.imgsz,
    )
    predictor = Sam3MultiClassPredictorFast(
        model, device="cuda", resolution=args.imgsz, use_fp16=True,
        detection_only=True, trt_engine_path=args.trt,
    )
    predictor.set_classes(args.classes)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    present = {c: 0 for c in args.classes}    # frames with >=1 detection
    counts = {c: [] for c in args.classes}    # detections per analysed frame
    scores = {c: [] for c in args.classes}    # best score per analysed frame
    analysed = 0
    idx = 0
    t0 = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok or (args.max_frames and idx >= args.max_frames):
            break
        if idx % args.stride != 0:
            idx += 1
            continue

        # PIL, not numpy: set_image() reads numpy shapes as CHW.
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = predictor.predict_image(pil, confidence_threshold=args.confidence)

        per = {c: [] for c in args.classes}
        for i, nm in enumerate(res["class_names"]):
            if nm in per:
                per[nm].append(float(res["scores"][i]))
        for c in args.classes:
            counts[c].append(len(per[c]))
            if per[c]:
                present[c] += 1
                scores[c].append(max(per[c]))

        analysed += 1
        idx += 1
        if analysed % 50 == 0:
            print(f"  {analysed} frames ({analysed/(time.time()-t0):.1f} fps)")

    cap.release()
    el = time.time() - t0
    print(f"\nAnalysed {analysed} frames (stride {args.stride}) in {el:.1f}s\n")

    w = max(len(c) for c in args.classes) + 1
    print(f"{'class':<{w}} {'present':>9} {'recall':>8} {'mean/fr':>8} "
          f"{'med score':>10} {'p90':>7}")
    print("-" * (w + 46))
    for c in args.classes:
        rec = 100.0 * present[c] / analysed if analysed else 0.0
        mean_n = float(np.mean(counts[c])) if counts[c] else 0.0
        s = sorted(scores[c])
        med = s[len(s) // 2] if s else 0.0
        p90 = s[int(0.9 * (len(s) - 1))] if s else 0.0
        print(f"{c:<{w}} {present[c]:>4}/{analysed:<4} {rec:>7.1f}% "
              f"{mean_n:>8.2f} {med:>10.3f} {p90:>7.3f}")

    print("\nrecall = % of analysed frames with at least one detection")


if __name__ == "__main__":
    main()
