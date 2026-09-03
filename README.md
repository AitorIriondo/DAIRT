# DAIRT — Detect Any Interaction in Real Time

**DAIRT** answers a question that object detection alone cannot: *is the person
actually holding this thing, and when?*

It takes a video and a list of objects described in plain English, and returns a
timeline of every grasp and release, with timestamps:

```
00:09.9  Person grabs the hammer
00:15.5  Person releases the hammer      (held 5.6s)
00:16.2  Person grabs the hammer
00:18.5  Person releases the hammer      (held 2.3s)
```

You get an interactive HTML report with the video, a timeline lane per object, a
text event log, and live sliders that retune the detection thresholds in the
browser without re-running anything.

---

## Built on DART

DAIRT is a layer on top of **[DART: Detect Anything in Real Time](https://github.com/mkturkcan/DART)**
by Mehmet Kerem Turkcan, which converts Meta's SAM 3 into a real-time
open-vocabulary multi-class detector. All detection and segmentation is DART's
work; the upstream README is preserved here as
[`README-DART-upstream.md`](README-DART-upstream.md).

**DART detects objects. DAIRT detects *interactions* between them.**

What DAIRT adds:

| | |
|---|---|
| Contact measurement | Dilated mask-overlap metric between hands and objects |
| Temporal filtering | EMA, hysteresis and morphological open/close over the raw signal |
| Motion gating | Optional per-object test that a held object travels with the hand |
| Reporting | Self-contained HTML with synced video, timeline lanes and an event log |
| Prompt diagnostics | A tool that measures whether your class prompts actually work |
| One-click running | A launcher that asks for a video and objects and does the rest |
| Mask support in video | A patch to DART's `demo_video.py`, which was detection-only |

```bibtex
@misc{turkcan2026detectrealtimesingleprompt,
      title={Detect Anything in Real Time: From Single-Prompt Segmentation to Multi-Class Detection},
      author={Mehmet Kerem Turkcan},
      year={2026},
      eprint={2603.11441},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.11441},
}
```

---

## Contents

- [What it produces](#what-it-produces)
- [Hardware requirements](#hardware-requirements)
- [Setup](#setup) — **read the two traps**
- [Running it](#running-it)
- [How long things take](#how-long-things-take)
- [How it works](#how-it-works)
- [The filters, in detail](#the-filters-in-detail)
- [Choosing prompts](#choosing-prompts)
- [Known limits](#known-limits)
- [Roadmap](#roadmap) — depth discrimination with MoGe-2
- [Repository layout](#repository-layout)

Longer write-ups live in [`docs/`](docs/).

---

## What it produces

For every run you get four files in `outputs/<video name>/`:

| File | Contents |
|---|---|
| `<name>_report.html` | **The deliverable.** Self-contained: video, timeline, event log, live sliders. Opens by double-click, no web server. |
| `<name>.json` | Raw per-frame overlap and centroid tracks. Analysis is fully reproducible from this without a GPU. |
| `<name>_masks.mp4` | The video with segmentation masks, boxes and track IDs drawn on. |
| `<name>_small.mp4` | Compressed copy embedded in the report. |

The report contains:

- **Video player** with the mask overlay
- **A timeline lane per object**, green where contact is detected, click to seek
- **An event log** with timestamps and hold durations, click a line to jump, copy as text
- **A statistics table** — contact time, % of clip, number of grasps, mean and longest grasp
- **Seven live controls** that recompute everything in the browser (see [the filters](#the-filters-in-detail))

---

## Hardware requirements

Developed and measured on:

| | |
|---|---|
| GPU | NVIDIA RTX 5070 Laptop, **8 GB**, Blackwell (sm_120), 95 W |
| CPU / RAM | AMD Ryzen AI 9 365 / 31 GB |
| OS | Windows 11 |

**Minimum: an NVIDIA GPU with 8 GB VRAM.** The pipeline was built and tuned
against that ceiling, so more VRAM only helps. About 16 GB of system RAM is
comfortable; the analysis stage holds ~4 GB.

Disk: ~10 GB for the environment, ~3.5 GB for the SAM 3 checkpoint, ~1 GB for
the TensorRT engines.

**This is not real time on a laptop GPU.** The name comes from DART. Expect
roughly 2× the video duration end to end — fine for offline analysis, not for
live feedback. See [timings](#how-long-things-take).

---

## Setup

### 1. Create the environment

```bash
conda create -n dairt python=3.11 -y
conda activate dairt
```

Python 3.11 or 3.12. Nothing newer — PyTorch wheels lag.

### 2. Install PyTorch — READ THIS, THE UPSTREAM COMMAND IS WRONG FOR MODERN GPUs

DART's README says `--index-url .../cu126`. On any Blackwell card (RTX 50-series,
compute capability 12.0) **that installs cleanly and then fails at the first CUDA
call**, because cu126 wheels are built only up to sm_90.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify before going further:

```bash
python -c "import torch; print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list())"
```

Your capability must appear in the arch list. On this machine:

```
(12, 0)
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

Stay on **CUDA 12.x**, not 13.x, so TensorRT and PyTorch share a runtime.

### 3. Install DART and its dependencies

```bash
pip install -e .
```

### 4. Install TensorRT — THE SECOND TRAP

`pip install -e ".[tensorrt]"` reads `tensorrt>=10.9.0` and resolves to
**TensorRT 11.x built against CUDA 13**. That breaks twice over: TRT 11 removed
APIs DART calls (`platform_has_fast_fp16`, `EXPLICIT_BATCH`), and CUDA 13
conflicts with your CUDA 12.8 PyTorch.

```bash
pip install "tensorrt-cu12==10.13.3.9" onnx_graphsurgeon polygraphy
```

If you already hit it:

```bash
pip uninstall -y tensorrt tensorrt_cu13 tensorrt_cu13_bindings tensorrt_cu13_libs
pip install "tensorrt-cu12==10.13.3.9"
```

### 5. Install the extras DAIRT uses

```bash
pip install imageio-ffmpeg     # bundles ffmpeg for compressing the report video
```

Node.js is optional — only needed for `check_report.js`, which syntax-checks
generated reports.

### 6. Get the SAM 3 checkpoint (gated)

The model requires accepting Meta's licence:

1. Accept at **https://huggingface.co/facebook/sam3**
2. Create a read token at https://huggingface.co/settings/tokens
3. `hf auth login` and paste it

First run downloads ~3.5 GB to
`~/.cache/huggingface/hub/models--facebook--sam3/`. DAIRT finds it automatically.

### 7. Build the TensorRT engines (one-time)

```bash
CKPT=~/.cache/huggingface/hub/models--facebook--sam3/snapshots/<hash>/sam3.pt

# Encoder-decoder — about 2 minutes, peaks at ~3.3 GB VRAM
python -m sam3.trt.export_enc_dec --checkpoint "$CKPT" \
    --output enc_dec.onnx --max-classes 4 --imgsz 1008
python -m sam3.trt.build_engine --onnx enc_dec.onnx \
    --output enc_dec_fp16.engine --fp16 --mixed-precision none

# ViT-H backbone — about 3 minutes, produces a ~900 MB engine
python scripts/export_hf_backbone.py --image x.jpg --imgsz 1008
```

`--mixed-precision none` matters for the encoder-decoder: the auto-detect
heuristic applies backbone rules that are wrong there.

**Engines are tied to your exact GPU and TensorRT version.** They cannot be
copied between machines. Rebuild after any driver or TensorRT change.

#### Verify FP16 quality

The backbone export prints a cosine similarity against the PyTorch reference.
**It must be > 0.999:**

```
Cosine: 0.999875   0.999632   0.999517
```

If you see ~0.058, TensorRT dispatched to an FP16 kernel that accumulates
badly. The engine still runs at full speed and produces numerically worthless
features — the worst kind of failure. Use the `export_hf_backbone.py` path,
which restructures attention so TensorRT pattern-matches correctly.

### 8. Point the launcher at your environment

Edit `RUN ANALYSIS.bat` and set `PYEXE` to your `python.exe`.

---

## Running it

### The easy way

Double-click **`RUN ANALYSIS.bat`**. A window asks for two things:

1. **A video** — file picker
2. **The objects** — space separated, quotes round anything multi-word:
   ```
   apple "soda can" "drinking glass"
   ```

`hand` is added automatically; every other class is tested against it.

Then it runs everything and opens the report.

### Check your prompts first (strongly recommended)

```bash
python diag_recall.py --video myclip.mp4 \
    --classes hand hammer mallet "rubber mallet" \
    --checkpoint "$CKPT" --trt hf_backbone_fp16.engine --stride 3
```

90 seconds, and it prints what actually detects:

```
class            present   recall  mean/fr  med score
hand            132/192     68.8%     1.07      0.895
hammer          154/192     80.2%     0.86      0.941
mallet          149/192     77.6%     0.84      0.918
rubber mallet   124/192     64.6%     0.66      0.925
```

**Do this before any real analysis.** Contact percentages computed on a class
that only fires half the time are meaningless, and nothing downstream will warn
you. See [choosing prompts](#choosing-prompts).

### Headless / scripted

```bash
python run_analysis.py --video clip.mp4 --classes hand hammer
python run_analysis.py --video clip.mp4 --classes hand hammer --no-render --stride 5
```

`--no-render` skips the mask video — the expensive half. You still get the
timeline, JSON and report. **Use it for first passes.**

### Retuning without re-running

Every filter parameter lives in the report and recomputes in the browser. To
rebuild the page itself (new defaults, different video) — no GPU, instant:

```bash
python analyze_contact.py --from-json outputs/clip/clip.json \
    --video outputs/clip/clip_small.mp4 --embed-video -o report.html
```

---

## How long things take

Measured on the reference machine (RTX 5070 Laptop 8 GB).

### One-time setup

| Step | Time |
|---|---|
| conda env + PyTorch cu128 | 5–10 min (≈3 GB download) |
| `pip install -e .` + TensorRT | 3–5 min |
| SAM 3 checkpoint download | 2–5 min (3.5 GB) |
| Encoder-decoder engine | **105 s** |
| ViT-H backbone engine | **194 s** |
| **Total** | **~20–30 min** |

### Per-frame throughput

| Path | ms/frame | FPS | Masks |
|---|---|---|---|
| PyTorch, boxes only | 762 | 1.3 | no |
| Full TensorRT, boxes only | 150 | 6.7 | no |
| TRT backbone + masks | 296 | 3.4 | **yes** |
| Contact analysis (stride 3) | ~70 | 14.6 | yes |

Backbone alone: **748 ms → 110 ms**, a 6.8× speedup from TensorRT.

### Whole-pipeline examples

| Video | Frames | Classes | With render | Analysis only |
|---|---|---|---|---|
| 19 s, 1080p | 576 | 2 | **272 s** | ~45 s |
| 7 s, 1080p | 222 | 2 | **228 s** | ~30 s |
| 77 s, 1080p | 2322 | 4 | *>47 min* | **159 s** |

**Rule of thumb: with the mask render, budget roughly 2× the video duration.
Without it, roughly 0.2×.**

The render cost scales with `frames × classes × output pixels` and is by far the
expensive half. The launcher warns and estimates before starting a large one.

### Why masks halve the speed

DART's TensorRT encoder-decoder emits only scores and boxes — the mask decoder
needs hidden states that are not in the exported graph
(`sam3/model/sam3_multiclass_fast.py:229`). With masks you keep the TensorRT
**backbone**, which is the larger win, and fall back to PyTorch for the
encoder-decoder.

It is worth it. Boxes are actively misleading for contact: an air hose's
bounding box covers half the frame while its mask is a thin curve, so
box-overlap logic reports constant false contact.

---

## How it works

```
video ──► DART (SAM 3 + TensorRT)  ──► per-frame masks per class
                                          │
                                          ▼
                             hand ∩ object overlap  (dilated, normalised)
                                          │
                                          ▼
                    EMA ─► hysteresis ─► motion gate ─► close ─► open
                                          │
                                          ▼
                          contact timeline + event log + report
```

### The key design decision: contact needs no object IDs

It looks like this needs reliable tracking, so you can say "hand #2 holds tool
#5". It does not. **Ask only: does any hand mask touch any instance of this
class, this frame?**

That sidesteps the tracking problem entirely — ByteTrack produces 200–370 unique
tracks for a scene with ~5 real objects on handheld footage, and no amount of
tuning fixes it, because it associates on box IoU with no appearance model.

Left/right hand attribution *would* need identity. That is a later refinement,
not a prerequisite.

### Inference and analysis are separated

Inference takes minutes; thresholding takes milliseconds. The analysis writes
**raw per-frame values** to JSON, and every filter parameter is applied
client-side in the report. Retuning never re-runs the model.

---

## The filters, in detail

Seven controls, all live in the report.

### 1. The contact metric (fixed at analysis time)

**Dilation is mandatory, not a refinement.** Measured on a frame where the
operator is unambiguously gripping a wrench:

```
hand (area 13409) ∩ tool (area 8059) = 0 pixels
```

Zero. SAM assigns each contact pixel to one object *or* the other, so a gripping
hand and tool produce **adjacent** masks, never overlapping ones. Raw
intersection reads zero at exactly the moment of contact.

Both masks are dilated, then intersected, then normalised **by the dilated hand
area** so the result is bounded to [0, 1]:

```
contact = |dilate(hand) ∩ dilate(object)| / |dilate(hand)|
```

Normalising by the *raw* hand area instead lets the value exceed 1 and inflates
exactly the small, distant hands that are least reliable. Not IoU — a hand and a
tool differ too much in area for IoU to mean anything.

Geometry runs at a fixed **640 px working width**, which bounds memory and makes
`--dilate` resolution-independent. Default **13 px**.

| flag | default | effect |
|---|---|---|
| `--dilate` | 13 | dilation in px at the working width |
| `--work-width` | 640 | resolution for contact geometry |
| `--max-instances` | 8 | most instances per class per frame, highest score first |
| `--confidence` | 0.25 | detection threshold |
| `--stride` | 3 | analyse every Nth frame |

### 2. Smoothing — EMA (default 0.30)

Exponential moving average over the raw overlap. Removes frame-to-frame mask
jitter. 0 disables it.

### 3–4. Enter / leave contact — hysteresis (defaults 0.165 / 0.165)

Two thresholds with a dead band between them. Contact starts when the smoothed
signal rises above **enter**, and ends only when it falls below **leave**.

This is the stage people skip. With a single threshold, a signal hovering near
it flips state every few frames. Setting `leave` below `enter` stops the
chatter.

Setting them **equal** (the shipped default) disables the dead band — good on
clean footage where the signal moves decisively, more prone to chatter on noisy
footage.

### 5. Bridge gaps — morphological close (default 15 frames, 0.5 s)

Fills `false` gaps shorter than the window. A real grasp survives brief
detection dropouts from motion blur or occlusion.

### 6. Drop blips — morphological open (default 0 frames)

Deletes `true` runs shorter than the window. Removes phantom grasps.

Together, close and open express "ignore state changes shorter than N frames".
Raise `open` to ~8 frames if a clip shows sub-second flicker.

### 7–8. Motion gating — per object, on by default

A held object moves *with* the hand; a static one does not. Over a window, the
object's displacement is projected onto the hand's:

```
agreement = (Δobject · Δhand) / |Δhand|²
```

Held → ~1. Static → ~0. Contact is vetoed where agreement falls below **motion
strictness** (default 0.30) over a **motion window** (default 20 frames).

**When the hand barely moves, the test abstains** rather than vetoing —
otherwise holding something still would read as "not holding", a worse bug than
the one being fixed. Motion only ever removes contacts where the hand
demonstrably moved and the object demonstrably did not.

Each object has a checkbox, **ticked by default**. Untick anything that stays
put when touched — a table, a fixed machine, a conveyor — because motion gating
would wrongly veto every real contact with those.

**Honest limitation:** motion gating needs object identity over time, even
though the contact boolean does not. Where a prompt fires on lookalikes
elsewhere in the scene, the tracked centroid drifts onto one of them and the
test becomes unreliable. It works well for cleanly-detected objects and is worth
unticking otherwise. Full analysis in
[`docs/04-contact-detection.md`](docs/04-contact-detection.md).

### Defaults summary

```
Smoothing (EMA)        0.30
Enter contact          0.165
Leave contact          0.165
Bridge gaps            15 frames (0.50 s)
Drop blips              0 frames
Motion window          20 frames (0.67 s)
Motion strictness      0.30
Must move with hand    on, per object
```

---

## Choosing prompts

**Prompt wording changes results more than every other knob combined.** Measured
findings, with the full tables in
[`docs/03-prompt-findings.md`](docs/03-prompt-findings.md).

### Read two columns, not one

`diag_recall.py` reports **recall** (% of frames with a detection) and
**detections/frame**. A prompt fails by being too narrow *or* too broad, and
only both columns together tell you which.

### Specific nouns beat generic ones

One glass in a kitchen:

| prompt | recall | dets/frame |
|---|---|---|
| `glass` | 100% | **23.30** |
| `drinking glass` | 100% | **2.66** |

`glass` names a material as well as a vessel, so it matched the oven door, the
window, the tiles and the blender jar. This corrupted a downstream number:
contact read 46.7% with `glass`, 26.8% with `drinking glass`.

### Low recall can be correct

| prompt | recall | dets/frame |
|---|---|---|
| `power tool` | 97.2% | 6.17 |
| `screwdriver` | 47.3% | 1.20 |

In a workshop with several tools, `power tool` fires on all of them;
`screwdriver` isolates one. **Low recall of a specific prompt is
discrimination, not failure** — decide whether you want coverage or specificity
before reading the numbers.

### Qualifiers can make it worse

`screwdriver` 47.3% → `pneumatic screwdriver` **25.2%**. More precise language
is not automatically a better prompt.

### Prompts do not transfer between scenes

`power tool`: **97.2%** on a workshop close-up, **0.0%** on a factory wide shot.
Always re-run `diag_recall.py` on new footage.

### Rules of thumb

- Aim for **~1 detection/frame** per single object
- Above ~3/frame will corrupt contact numbers — the metric takes the max over
  instances, so a spurious detection near a hand reads as contact
- Above ~30/frame will exhaust an 8 GB GPU during mask generation. The launcher
  probes for this and raises confidence automatically, but a better prompt is
  the real fix

---

## Known limits

### 2D masks cannot distinguish contact from depth alignment

**The most important limitation, and no parameter fixes it.**

In one test clip the operator stands behind a table with a can in front of him.
His hands hang at roughly the same *image height* as the can, so the masks
overlap substantially in 2D while being half a metre apart in 3D.

Result: a **13.7-second false "grab"** before he touches it. It survives every
threshold, because the measured overlap genuinely is large. The information is
simply not in the 2D masks.

This is the motivation for the [roadmap](#roadmap) below.

Camera placement matters more than any setting: a viewpoint that does not align
hands with background objects avoids the problem entirely.

### Open-vocabulary detection has a vocabulary boundary

Custom industrial fixtures are not reachable by text prompt. On factory footage,
**21 prompts** were tested against a stainless lifting beam and a pneumatic
nutrunner:

| prompt | recall |
|---|---|
| `chain` | 100% (but it's the hoist chain, not the device) |
| `handle` | 61.8% |
| `hoist` | 9.4% |
| `lifting device` | 7.3% |
| `pneumatic screwdriver`, `nutrunner`, `air tool`, `power tool` | **0.0%** |

Higher input resolution does not help — 1008, 1400 and 1736 px were tested and
the model is tuned for ~1008.

The boundary is what the internet has many pictures of. Apples and hammers work;
a bespoke stainless fixture does not, however precisely you name it. See the
[roadmap](#roadmap) for image exemplars.

### Tracking IDs churn on handheld footage

| condition | unique tracks (scene has ~5 objects) |
|---|---|
| Handheld, default | 371 |
| Handheld, tuned | 296 |
| Handheld, tuned + masks | 215 |
| **Static camera** | **52** |

ByteTrack associates on box IoU with no appearance model, so two visually
identical gloved hands cannot be told apart. Contact detection does not depend
on this, but motion gating does.

---

## Roadmap

### 1. Depth as a discriminator — MoGe-2 (highest priority)

The single most valuable addition. It directly fixes the depth-alignment false
positives described above.

**Approach.** Estimate per-frame depth, then require the hand and object to be
close *in depth* as well as overlapping in pixels.

**Why it is cheaper than it sounds — 2D proposes, depth disposes.** Mask overlap
is already a good *candidate* detector; it just cannot reject depth-aligned
false positives. So run depth **only on frames where 2D overlap is already
non-zero** — typically 20–40% of frames. Depth becomes a verification stage on a
minority of frames rather than doubling the cost of every frame.

**Design notes, learned the hard way:**

- **Avoid absolute distance thresholds.** Monocular depth is scale-ambiguous, so
  "within 10 cm" is not a stable number across clips. Instead compare the hand's
  median depth against the **object's own depth spread** — scale-free and needing
  no calibration.
- **Erode masks before sampling depth.** Depth is unreliable at object
  boundaries, where it bleeds between foreground and background.
- **Use the median, not the mean.** A few bad boundary pixels otherwise drag the
  estimate badly.
- **Transparent objects stay hard.** Monocular depth on a drinking glass is
  poor, and stereo/ToF sensors struggle for the same physical reason.

**Two sources, and both are worth supporting:**

| | |
|---|---|
| **MoGe-2** (monocular, estimated) | Works on footage already shot, including archives. Start here. |
| **RealSense** (measured) | Categorically better, but only for new recordings. |

**Suggested implementation order:**

1. Add a `--depth moge2` flag to `analyze_contact.py`, loading MoGe-2 lazily
2. Compute depth only for candidate-contact frames
3. Store the depth-agreement value per frame in the JSON, **alongside** the
   existing overlap — not replacing it
4. Add a **Depth strictness** slider and a per-object checkbox in the report,
   mirroring the motion gate, so it is tunable without re-running
5. Validate on the known false-positive clip, where the correct answer is known

Point 3 matters: keeping depth as a separate stored signal means it tunes in the
browser like everything else, and a bad depth estimate never destroys the
underlying overlap data.

### 2. Image exemplars instead of text prompts

For objects outside the text vocabulary — custom fixtures, specific tools.

The plumbing already exists upstream. `sam3/model/sam3_image.py:204`:

```python
prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
```

Text, geometry and visual prompts are interchangeable entries in one prompt
sequence. DART currently passes **zeros** for `visual_prompt_embed`, but the
path is wired through, and there is an unused `--class-method prototype`.

The TensorRT encoder-decoder takes `text_feats` as an input tensor and does not
care where the numbers came from, so **a visual embedding occupying that slot
needs no engine rebuild**.

Estimated effort: half a day to confirm the checkpoint contains SAM 3's exemplar
encoder (the consumer exists; the producer is unverified), then 1–2 days to wire
it up and add a "mark the object once" step.

Caveat: exemplars fix the vocabulary problem, not the scale problem.

### 3. Left / right hand attribution

Wrist keypoints from a pose model give left and right wrist as distinct
anatomical joints that survive blur and motion — the identity signal ByteTrack
lacks. Combined with DAIRT's masks this yields *which hand* holds *which object*.

### 4. Smaller items

- Batch the mask GPU→CPU transfer (analysis runs ~3 FPS vs `demo_video.py`'s 3.4
  because it does not pipeline)
- Store overlap at several dilation values so **dilation becomes a slider** —
  it is currently the least-validated parameter
- Per-frame hand-detection flag in the JSON, so the report can distinguish "no
  hand visible" from "hand visible, not touching"
- Optional CSV export of the event log

---

## Repository layout

| Path | Purpose |
|---|---|
| `RUN ANALYSIS.bat` | Double-click launcher |
| `run_analysis.py` | Orchestrator: probes prompts, renders, analyses, reports |
| `analyze_contact.py` | Contact measurement and HTML report generation |
| `diag_recall.py` | **Per-class detection recall. Run before trusting any analysis.** |
| `check_report.js` | Node syntax check for generated reports |
| `demo_video.py` | Upstream DART, **patched** to support masks |
| `demo_multiclass.py` | Upstream DART, single-image detection |
| `sam3/` | Upstream DART model code |
| `scripts/` | Upstream DART export, evaluation and benchmark scripts |
| `docs/` | Detailed write-ups |
| `outputs/` | Per-video results (gitignored) |

### Local patches to upstream

`demo_video.py` hardcoded `detection_only=True` and had no mask path. Added:

- `--masks` and `--mask-alpha` flags
- mask blending in `draw_detections_cv2`, drawn *under* boxes so labels stay legible
- a guard rejecting `--masks` with `--trt-enc-dec`, which cannot produce masks

**Reapply after any upstream merge.**

### Documentation

| Document | Contents |
|---|---|
| [`docs/README.md`](docs/README.md) | Index |
| [`docs/01-setup-and-hardware.md`](docs/01-setup-and-hardware.md) | Environment, install traps, engines, measured speed |
| [`docs/02-running-it.md`](docs/02-running-it.md) | Launcher, scripts, flags, report controls |
| [`docs/03-prompt-findings.md`](docs/03-prompt-findings.md) | **Seven measured findings on prompt wording** |
| [`docs/04-contact-detection.md`](docs/04-contact-detection.md) | The metric, filtering and motion gating in depth |
| [`docs/05-limits-and-gotchas.md`](docs/05-limits-and-gotchas.md) | Hard limits and bugs that cost real time |

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| CUDA error on first call, install looked fine | cu126 wheels on a Blackwell GPU. Install cu128. |
| `platform_has_fast_fp16` AttributeError | TensorRT 11 installed. Downgrade to `tensorrt-cu12==10.13.3.9`. |
| Backbone cosine ≈ 0.058 | FP16 accumulation. Use `scripts/export_hf_backbone.py`, not a naive export. |
| `CUDA out of memory` during mask generation | Too many detections. Use a more specific prompt, or raise `--confidence`. The launcher probes and adjusts automatically. |
| Machine runs out of RAM | Ensure `--work-width` is set (default 640). Working at full resolution churns ~100 MB per frame. |
| Report loads but the video does not | `file://` pages cannot load sibling videos. Use `--embed-video` (the launcher does) or serve over HTTP. |
| Report is blank below the video | JavaScript syntax error. Run `node check_report.js <report>`. |
| A class shows ~100% contact at p50 = 1.0 | It segments the same object as the hand (e.g. `glove` vs `hand`). Remove it. |
| Engine fails to load after a driver update | Engines are TensorRT- and GPU-specific. Rebuild them. |

---

## Licence

DAIRT follows the licence of the upstream DART repository — see [`LICENSE`](LICENSE).
SAM 3 model weights are subject to Meta's own licence, accepted at download time.
