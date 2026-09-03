# Running it

## The easy way: double-click

**`RUN ANALYSIS.bat`** in the repository root.

It opens a small window asking for two things:

1. **A video** — file picker.
2. **The objects to look for** — space separated, quotes round anything
   multi-word:
   ```
   apple "soda can" "drinking glass"
   ```
   `hand` is added automatically; everything else is tested against it.

Then it runs the whole pipeline and opens the report when done.

Results land in `outputs\<video name>\`:

| file | what |
|---|---|
| `<name>_masks.mp4` | full-quality video with masks, boxes and track IDs |
| `<name>_small.mp4` | compressed copy (854 px) used inside the report |
| `<name>.json` | raw per-frame overlap and centroid tracks |
| `<name>_report.html` | **standalone report** — double-click, no server |

### What it does automatically

- Finds the gated SAM3 checkpoint in the HuggingFace cache
- Adds `hand` if you forgot it
- **Probes your prompts first** and raises `--confidence` if they produce too many
  detections for mask generation to fit in VRAM
- Analyses every 3rd frame (10 Hz effective — well above what a grasp signal needs)
- Computes contact geometry at a fixed 640 px working width, so memory stays
  bounded and `--dilate` is resolution-independent
- Compresses and embeds the video so the HTML works from `file://`
- **Carries on if the render fails**, so you still get the timeline, JSON and report

### Timing

Roughly **2× the video duration** for a 1080p clip, dominated by the mask render.
A 30-second clip takes 4–6 minutes.

## Before a real run: check your prompts

```bash
python diag_recall.py --video myclip.mp4 \
    --classes hand "power tool" screwdriver "impact wrench" \
    --checkpoint <ckpt> --trt hf_backbone_fp16.engine --stride 3
```

60–90 seconds, and it tells you whether the prompts detect anything. **Do this
first.** Contact numbers computed on a class that only fires in 40% of frames are
meaningless, and nothing downstream will warn you.

Read both columns — recall *and* detections/frame. See
[03-prompt-findings.md](03-prompt-findings.md).

## Retuning without re-running

All filter parameters live in the report and recompute in the browser. Drag the
sliders and the lanes, stats and event log update instantly.

To rebuild the HTML itself (new defaults, different video, template changes) — no
GPU, instant:

```bash
python analyze_contact.py --from-json outputs/clip/clip.json \
    --video outputs/clip/clip_small.mp4 --embed-video -o report.html
```

## Report controls

| control | default | what it does |
|---|---|---|
| Enter contact | 0.165 | overlap needed to start a grasp |
| Leave contact | 0.165 | overlap below which it ends |
| Bridge gaps | 15 fr | fill dropouts shorter than this |
| Drop blips | 0 fr | delete grasps shorter than this |
| Smoothing | 0.30 | EMA on the raw signal |
| Motion window | 20 fr | window for the motion test |
| Motion strictness | 0.30 | how much of the hand's motion the object must follow |
| **Must move with the hand** | on | per-object. Untick for tables, fixed machines, conveyors. |

Clicking a timeline lane or an event-log line seeks the video. "Copy as text"
exports the log.

## Running the pieces manually

```bash
CKPT="$HOME/.cache/huggingface/hub/models--facebook--sam3/snapshots/<hash>/sam3.pt"

# 1. masks video (slow)
python demo_video.py --video clip.mp4 --classes hand apple "soda can" \
    --checkpoint "$CKPT" --trt hf_backbone_fp16.engine --imgsz 1008 \
    --masks --track --class-agnostic-nms 0.7 --confidence 0.25 -o clip_masks.mp4

# 2. contact analysis -> json + html
python analyze_contact.py --video clip.mp4 --classes hand apple "soda can" \
    --hand-class hand --checkpoint "$CKPT" --trt hf_backbone_fp16.engine \
    --confidence 0.25 --dilate 13 --stride 3 \
    --json clip.json --report-video clip_masks.mp4 -o clip.html
```

Always prefix with `PYTHONIOENCODING=utf-8` on Windows.

### Useful flags

`analyze_contact.py`:

| flag | default | note |
|---|---|---|
| `--hand-class` | `hand` | what everything else is tested against |
| `--contact-classes` | all but hand and `person` | override the auto choice |
| `--dilate` | 13 | at the working width, so independent of source resolution |
| `--work-width` | 640 | resolution for contact geometry; bounds memory |
| `--max-instances` | 8 | cap per class per frame, highest score first |
| `--stride` | 1 | 3 is plenty and 3× faster |
| `--report-video` | = `--video` | point at the mask render |
| `--embed-video` | off | self-contained HTML |
| `--from-json` | — | rebuild the report, no GPU |
| `--max-jump` | 0.08 | centroid association limit for motion gating |

## Speed notes

- `--stride 3` costs almost nothing in quality (the filters smooth over 15+
  frames anyway) and cuts runtime by ~3×.
- The mask render is the slow half. Skip it if you only need numbers — pass
  `--report-video` the raw clip.
- `analyze_contact.py` does not pipeline the way `demo_video.py` does, but since
  contact geometry moved to a 640 px working width it reaches ~14 FPS at
  `--stride 3`. Batching the mask GPU→CPU transfer would gain a little more.
- Use `--no-render` for first passes. The mask video is the expensive half and
  you usually want to check the prompts work before paying for it.
