#!/usr/bin/env python3
"""One-shot hand-object contact analysis.

Launched by "RUN ANALYSIS.bat". Asks for a video and the objects to look for,
then runs the whole pipeline and opens the report:

    video + prompts
        -> masks video      (demo_video.py --masks --track)
        -> per-frame overlap + centroids  (analyze_contact.py -> .json)
        -> compressed clip  (ffmpeg)
        -> standalone HTML  (video embedded, no server needed)

Everything lands in  outputs\\<video name>\\.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

# Engines are GPU- and TensorRT-version specific; they live beside this script.
BACKBONE_ENGINE = HERE / "hf_backbone_fp16.engine"

BANNER = "=" * 66


def find_checkpoint():
    """Locate the cached SAM3 checkpoint (gated download, already fetched)."""
    cache = Path.home() / ".cache/huggingface/hub/models--facebook--sam3/snapshots"
    if cache.is_dir():
        for snap in cache.iterdir():
            ckpt = snap / "sam3.pt"
            if ckpt.is_file():
                return ckpt
    local = HERE / "sam3.pt"
    return local if local.is_file() else None


def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    info = {
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "n": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def ask_inputs():
    """Small dialog for the video and the object list. Returns (path, classes)."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    state = {"video": None, "classes": None}

    root = tk.Tk()
    root.title("DART - hand/object contact analysis")
    root.resizable(False, False)
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    pad = {"padx": 12, "pady": 6}
    tk.Label(root, text="1. Choose a video", font=("Segoe UI", 10, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", **pad)

    vpath = tk.StringVar(value="(no video chosen)")
    tk.Label(root, textvariable=vpath, width=58, anchor="w",
             relief="sunken").grid(row=1, column=0, columnspan=2, sticky="w", **pad)

    def browse():
        f = filedialog.askopenfilename(
            title="Choose a video",
            initialdir=str(HERE),
            filetypes=[("Video", "*.mp4 *.mov *.MOV *.avi *.mkv *.m4v"),
                       ("All files", "*.*")])
        if f:
            state["video"] = f
            vpath.set(f)

    tk.Button(root, text="Browse...", command=browse, width=12).grid(
        row=1, column=2, **pad)

    tk.Label(root, text="2. Objects to look for", font=("Segoe UI", 10, "bold")).grid(
        row=2, column=0, columnspan=3, sticky="w", **pad)
    tk.Label(root, justify="left", fg="#555", text=(
        'Space separated. Put quotes round anything multi-word:\n'
        '    apple "soda can" "drinking glass"\n'
        '"hand" is added automatically - everything else is tested against it.\n'
        'Specific words beat generic ones: "soda can" not "can".')).grid(
        row=3, column=0, columnspan=3, sticky="w", padx=12)

    entry = tk.Entry(root, width=70)
    entry.grid(row=4, column=0, columnspan=3, sticky="w", **pad)
    entry.insert(0, '"power tool"')
    entry.focus_set()

    def run():
        if not state["video"]:
            messagebox.showwarning("No video", "Choose a video first.")
            return
        try:
            names = shlex.split(entry.get())
        except ValueError:
            messagebox.showerror("Bad input", "Unbalanced quotes in the object list.")
            return
        names = [n for n in names if n.strip()]
        if not names:
            messagebox.showwarning("No objects", "Name at least one object.")
            return
        state["classes"] = names
        root.destroy()

    tk.Button(root, text="Run analysis", command=run, width=18,
              font=("Segoe UI", 10, "bold")).grid(row=5, column=0, columnspan=3, pady=14)

    root.bind("<Return>", lambda e: run())
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 3
    root.geometry(f"+{x}+{y}")
    root.mainloop()

    return state["video"], state["classes"]


def probe_detections(video, classes, ckpt, conf, frames=12):
    """Estimate detections per frame before committing to a full render.

    Mask generation allocates per-detection logits on the GPU, so a broad prompt
    can OOM an 8 GB card: one clip hit 89 detections in frame 0 and died. A short
    probe costs ~40 s and turns that into an adjusted setting instead of a crash.

    Returns total detections/frame across all classes, or None if the probe fails.
    """
    cmd = [str(PY), str(HERE / "diag_recall.py"), "--video", str(video),
           "--classes", *classes, "--checkpoint", str(ckpt),
           "--trt", str(BACKBONE_ENGINE), "--stride", "5",
           "--max-frames", str(frames * 5), "--confidence", str(conf)]
    try:
        r = subprocess.run(cmd, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                           capture_output=True, text=True, timeout=600)
    except Exception:
        return None
    if r.returncode != 0:
        return None

    total = 0.0
    seen = False
    for line in r.stdout.splitlines():
        parts = line.split()
        # rows look like:  <class...>  123/233  94.0%  1.82  0.781  0.872
        if len(parts) >= 5 and parts[-4].endswith("%") and "/" in parts[-5]:
            try:
                total += float(parts[-3])
                seen = True
            except ValueError:
                pass
    return total if seen else None


def pause():
    """Wait for the user, but do not crash when there is no console attached."""
    try:
        input("\nPress Enter to close...")
    except (EOFError, KeyboardInterrupt):
        pass


def step(n, total, label):
    print(f"\n{BANNER}\n[{n}/{total}]  {label}\n{BANNER}", flush=True)


def run_cmd(args, label):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([str(a) for a in args], env=env)
    if r.returncode != 0:
        print(f"\n!! {label} failed (exit {r.returncode})")
        return False
    return True


def main():
    print(BANNER)
    print("  DART - hand/object contact analysis")
    print(BANNER)

    ckpt = find_checkpoint()
    if ckpt is None:
        print("\nERROR: SAM3 checkpoint not found.")
        print("  Accept the licence at https://huggingface.co/facebook/sam3")
        print("  then run:  hf auth login")
        pause()
        return 1
    if not BACKBONE_ENGINE.is_file():
        print(f"\nERROR: TensorRT backbone engine missing: {BACKBONE_ENGINE.name}")
        print("  Build it with:  python scripts/export_hf_backbone.py --image x.jpg --imgsz 1008")
        pause()
        return 1

    # Headless mode: passing --video/--classes skips the dialog, so the pipeline
    # is scriptable and testable. Without them, the GUI asks.
    ap = argparse.ArgumentParser(description="Hand/object contact analysis.")
    ap.add_argument("--video", help="Skip the dialog and use this video.")
    ap.add_argument("--classes", nargs="+", help="Objects to look for.")
    ap.add_argument("--stride", type=int, default=3,
                    help="Analyse every Nth frame (default: 3).")
    ap.add_argument("--no-render", action="store_true",
                    help="Skip the masks video; the report shows the raw clip.")
    args = ap.parse_args()

    if args.video and args.classes:
        video, classes = args.video, list(args.classes)
    else:
        video, classes = ask_inputs()
    if not video:
        print("\nCancelled.")
        return 0

    video = Path(video)
    if "hand" not in classes:
        classes = ["hand"] + classes

    info = video_info(video)
    if info is None:
        print(f"\nERROR: cannot open {video}")
        pause()
        return 1

    # Contact geometry runs at a fixed 640 px working width inside
    # analyze_contact.py, so dilation is resolution-independent and the default
    # is used as-is. Working full-size instead churned ~100 MB per frame and
    # could exhaust system RAM on a long clip.
    dilate = 13

    out = HERE / "outputs" / video.stem
    out.mkdir(parents=True, exist_ok=True)
    masks_mp4 = out / f"{video.stem}_masks.mp4"
    small_mp4 = out / f"{video.stem}_small.mp4"
    js = out / f"{video.stem}.json"
    html = out / f"{video.stem}_report.html"

    print(f"\n  video    {video.name}")
    print(f"  size     {info['w']}x{info['h']}  {info['fps']:.1f} fps  "
          f"{info['n']} frames  ({info['n']/info['fps']:.1f}s)")
    print(f"  objects  {', '.join(repr(c) for c in classes)}")
    print(f"  dilation {dilate} px (at the 640 px working width)")
    print(f"  output   {out}")

    # The masks render is the expensive half and scales with frames x classes x
    # output pixels. Say so up front rather than letting it surprise you an hour in.
    if not args.no_render:
        cost = info["n"] * len(classes) * (info["w"] * info["h"]) / 1e9
        if cost > 4:
            mins = cost * 2.5
            print(f"\n  NOTE: big render job ({info['n']} frames x {len(classes)} "
                  f"classes at {info['w']}x{info['h']}).")
            print(f"        Expect roughly {mins:.0f}-{mins*2:.0f} minutes.")
            print("        Use --no-render for a first pass: you still get the")
            print("        timeline, JSON and report, just without the mask video.")

    t0 = time.time()
    total = 4

    conf = 0.25
    if args.no_render:
        total = 3
        print("\n(skipping the masks render; report will show the raw clip)")
        masks_mp4 = video
    else:
        step(1, total, "Checking how many detections these prompts produce")
        dets = probe_detections(video, classes, ckpt, conf)
        if dets is None:
            print("  probe inconclusive - going ahead with the defaults")
        else:
            print(f"  ~{dets:.1f} detections/frame at confidence {conf}")
            # Each detection needs its own mask logits on the GPU. Past roughly
            # 30 per frame an 8 GB card runs out during mask generation.
            while dets > 30 and conf < 0.6:
                conf = round(conf + 0.15, 2)
                print(f"  too many for mask generation - retrying at {conf}")
                d2 = probe_detections(video, classes, ckpt, conf)
                if d2 is None:
                    break
                dets = d2
                print(f"  ~{dets:.1f} detections/frame at confidence {conf}")
            if dets > 30:
                print("  still high. The render may fail; the analysis will")
                print("  still run and you will get the report either way.")
                print("  A more specific prompt is the real fix - see docs/03.")

        step(1, total, "Rendering masks video (the slow part)")
        ok = run_cmd([PY, HERE / "demo_video.py",
                      "--video", video, "--classes", *classes,
                      "--checkpoint", ckpt, "--trt", BACKBONE_ENGINE,
                      "--imgsz", 1008, "--masks", "--track",
                      "--class-agnostic-nms", 0.7, "--confidence", conf,
                      "-o", masks_mp4], "render")
        # The render is the optional half. Losing it must not cost the user the
        # timeline, the JSON and the report, which is what they actually asked for.
        if not ok:
            print("\n  Render failed - continuing without it. The report will")
            print("  show the original clip instead of the mask overlay.")
            masks_mp4 = video

    step(2, total, "Measuring hand/object contact")
    ok = run_cmd([PY, HERE / "analyze_contact.py",
                  "--video", video, "--classes", *classes, "--hand-class", "hand",
                  "--checkpoint", ckpt, "--trt", BACKBONE_ENGINE,
                  "--confidence", conf, "--dilate", dilate, "--stride", args.stride,
                  "--json", js, "--report-video", masks_mp4, "-o", html], "analysis")
    if not ok:
        pause()
        return 1

    step(3, total, "Compressing the clip for embedding")
    ff = find_ffmpeg()
    embedded = False
    if ff is None:
        print("  imageio-ffmpeg not installed - the report will reference the")
        print("  mp4 instead of embedding it (needs a web server to view).")
    else:
        r = subprocess.run([str(ff), "-y", "-i", str(masks_mp4),
                            "-vf", "scale=854:-2", "-c:v", "libx264",
                            "-crf", "30", "-preset", "medium", "-an",
                            "-movflags", "+faststart", str(small_mp4)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        embedded = r.returncode == 0 and small_mp4.is_file()
        if embedded:
            mb = small_mp4.stat().st_size / 1048576
            print(f"  {masks_mp4.stat().st_size/1048576:.1f} MB -> {mb:.1f} MB")
        else:
            print("  compression failed - falling back to a linked video.")

    step(4, total, "Building the report")
    args = [PY, HERE / "analyze_contact.py", "--from-json", js, "-o", html]
    if embedded:
        args += ["--video", small_mp4, "--embed-video"]
    else:
        args += ["--video", masks_mp4]
    if not run_cmd(args, "report"):
        pause()
        return 1

    print(f"\n{BANNER}")
    print(f"  Done in {time.time()-t0:.0f}s")
    print(f"{BANNER}")
    print(f"  report  {html}")
    print(f"  data    {js}")
    print(f"  video   {masks_mp4}")
    print("\n  Tune the thresholds with the sliders in the report - they")
    print("  recompute in the browser, so no re-run is needed.")

    try:
        os.startfile(str(html))
    except Exception:
        print(f"\n  Open it yourself: {html}")

    pause()
    return 0


if __name__ == "__main__":
    sys.exit(main())
