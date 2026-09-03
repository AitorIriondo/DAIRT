# Limits and gotchas

## Hard limits

### 1. 2D masks cannot tell contact from depth alignment

**The most important limitation, and no parameter fixes it.**

In `soda_apple.mp4` the operator stands behind a table with a can in front of him.
His hands hang at roughly the same **image height** as the can. In 2D projection
the masks overlap substantially; in 3D they are half a metre apart.

Result: a **13.7-second false "grab"** starting at 00:00.5, before he touches it.
It survives every threshold tested including 0.40/0.25, because the measured
overlap genuinely is large.

The information is not in the 2D masks. Threshold, dilation and filtering are all
powerless here.

**Why the workshop footage suffered less:** the camera was close and roughly level
with the work, so image proximity implied real proximity. Camera placement matters
more than any setting.

**Ways forward, best first:**

1. **Depth.** `FastSAM3DToOpenSim` already has MoGe (monocular depth) and a
   RealSense (`record_realsense.py`) for measured depth. Gate contact on hand and
   object being close *in depth*, not just overlapping in pixels.
   - Run depth only on frames where 2D overlap is already non-zero — 2D proposes,
     depth disposes. That makes it a verification stage on a minority of frames
     rather than doubling the cost of every frame.
   - Avoid absolute distance thresholds; monocular depth is scale-ambiguous.
     Compare the hand's median depth against the **object's own depth spread** —
     scale-free, no calibration.
   - Erode masks before sampling depth (it is unreliable at edges) and use
     **median, not mean**.
   - Weak on transparent objects — a drinking glass defeats both MoGe and
     RealSense for the same physical reason.
2. **Motion correlation.** Implemented, half works — see
   [04-contact-detection.md](04-contact-detection.md). Inherits the object
   identity problem.
3. **Camera placement.** For footage you control, a viewpoint that does not align
   hands with background objects avoids the problem entirely.

### 2. Open-vocabulary detection has a vocabulary boundary

Custom industrial fixtures are not detectable by text prompt at any wording — 21
prompts tested on a stainless lifting beam and a pneumatic nutrunner, with the
best result being `chain` (the thing it hangs from) rather than the device.

See [03-prompt-findings.md](03-prompt-findings.md) for the full table.

**The fix is image exemplars rather than text.** The model already supports this —
`sam3/model/sam3_image.py:204`:

```python
prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
```

Text, geometry and visual prompts are interchangeable entries in one prompt
sequence. DART currently passes **zeros** for `visual_prompt_embed` (line 195),
but the path is wired through. There is also an unused
`--class-method prototype` with a calibrated-prototypes file.

**Why this is cheaper than it looks:** the TRT enc-dec takes `text_feats` as an
input tensor and does not care where the numbers came from. If a visual embedding
occupies that slot, **the existing engines work unchanged** — no re-export. Only
*appending* visual prompts alongside text would change the sequence length and
force a rebuild.

Estimated effort: half a day to confirm the checkpoint contains SAM 3's exemplar
encoder (the consumer exists; the producer is unverified), then 1–2 days to wire
it up and add a "mark the object once" step. Treat the first as a genuine go/no-go.

This suits fixed recurring objects — the same factory fixture in every recording —
far better than text, which has to name a category that barely exists in web
imagery.

**Caveat: exemplars fix the vocabulary problem, not the scale problem.** If an
object is failing partly because it is small and distant, an exemplar will not
rescue that half.

### 3. ByteTrack has no appearance model

Unique tracks for a scene containing ~5 real objects:

| condition | tracks |
|---|---|
| Handheld, default params | 371 |
| Handheld, tuned | 296 |
| Handheld, tuned + masks | 215 |
| **Static camera (kitchen)** | **52** |

`person` reliably holds ID #1; small fast objects churn. Two visually identical
gloved hands cannot be told apart by box IoU alone, so tuning gives diminishing
returns. **This is a method ceiling, not a config problem.**

Masks *improve* tracking (215 vs 296 on identical parameters) because tighter
detections give cleaner NMS.

For per-hand identity, wrist keypoints from YOLO11m-pose (already in
`FastSAM3DToOpenSim`) are the durable answer — pose gives left and right wrist as
distinct anatomical joints that survive blur and motion.

## Gotchas that cost real time

### Doing contact geometry at full resolution can exhaust system RAM

The worst bug in this work: it **froze the whole machine**.

`analyze_contact.py` originally extracted masks at source resolution and dilated
them there. On a 1920x1080 clip with ~10 detections per frame:

- each mask is 2 MB
- ~21 hand-object pairs per frame, each doing 2 `cv2.dilate` calls with a 38x38
  kernel on 2-megapixel arrays
- **~100 MB of allocation churn per frame**, over 2322 frames

**Fix: compute contact geometry at a fixed 640 px working width** (`--work-width`).
Contact is a coarse geometric signal; full resolution buys nothing.

| | before | after |
|---|---|---|
| speed (2322 frames) | did not finish | **159 s (14.6 FPS)** |
| RAM | grew until the machine froze | **flat at 3.8-4.3 GB** |

A useful side effect: `--dilate` is now **resolution-independent** (13 px at the
640 px working width) instead of needing to be scaled per video.

Also added `--max-instances` (default 8, highest score first) so a runaway prompt
like `metal bar` at 122 detections/frame cannot explode the pair loop.

**The general lesson: bound per-frame work by construction, not by hoping.** Cost
scaled with frames x detections x pixels, and all three were larger on the factory
clip than on anything it had been tested against.

### A broad prompt can OOM the GPU during mask generation

A warehouse clip full of plastic bins, prompted with `box`, produced **93.8
detections per frame**. Mask generation allocates per-detection logits on the
GPU, so this died with `CUDA error: out of memory` on frame 0.

`run_analysis.py` now probes the prompts on a few frames first and raises
confidence until the count is manageable:

```
~93.8 detections/frame at confidence 0.25
too many for mask generation - retrying at 0.4
~55.1 detections/frame at confidence 0.4
too many for mask generation - retrying at 0.55
~35.9 detections/frame at confidence 0.55
too many for mask generation - retrying at 0.7
~18.6 detections/frame at confidence 0.7      <- proceeds
```

The probe costs ~40 s and replaces a crash several minutes in. Raising confidence
is a blunt fix, though: **a more specific prompt is the real answer** — see
[03-prompt-findings.md](03-prompt-findings.md).

**Render failure is now non-fatal.** It previously aborted the whole run, so a
failed render cost the user the timeline, JSON and report as well. Those are the
actual deliverable; the mask video is a convenience. The pipeline now warns and
carries on with the original clip in the report.

### The masks render is the expensive half

A 2322-frame 1080p clip with 4 classes at confidence 0.25 took **over 47 minutes**
to render and was still going. The analysis of the same clip takes **under 3
minutes**.

Use `--no-render` for a first pass. You still get the timeline, JSON and report -
only the mask video is skipped. `run_analysis.py` now prints a warning and a time
estimate before starting a large render.


### `set_image()` reads numpy arrays as CHW

```python
elif isinstance(image, (torch.Tensor, np.ndarray)):
    height, width = image.shape[-2:]
```

An OpenCV **HWC** frame of shape `(720, 1280, 3)` is read as
**height=1280, width=3**. Masks come back rescaled to a 3-pixel-wide image — areas
of 84, 31, 9 pixels instead of ~13,000.

**Always pass PIL images**, which carry `.size` correctly. This silently produced
wrong contact numbers for an entire round of analysis, and looked like a method
problem rather than a bug.

### Browsers block `file://` pages from loading sibling video

```
Unsafe attempt to load URL file:///...mp4 from frame with URL file:///...html
```

Every `file://` document is a unique opaque origin, so a page cannot load an mp4
in its own folder. Two fixes:

- **Embed the video as a base64 data URI** (`--embed-video`). Part of the
  document, so no cross-origin check. Base64 adds ~33%, so compress first —
  854 px at CRF 30 turned 53 MB into 3.0 MB, giving a 4 MB self-contained page.
- Serve over HTTP (`python -m http.server --bind 127.0.0.1`).

The launcher embeds by default.

### Always syntax-check generated JavaScript

A patch introduced a literal newline inside a JS string:

```js
LAST_LINES.join("
");
```

That is a hard syntax error, so the **entire** `<script>` block fails to parse and
every dynamic element vanishes — timeline, log, sliders, stats. The static video
element still renders, which makes it look like a styling problem rather than a
fatal error.

Checking that expected markers were present in the HTML did **not** catch it.
Use `check_report.js`, which parses the script block with Node:

```bash
node check_report.js outputs/clip/clip_report.html
```

Prefer forms that cannot be mangled by escaping — `String.fromCharCode(10)` over
an escaped newline.

### A class that *is* the hand looks like a perfect result

`glove` measured against `hand` gives 86.4% contact at p50 = 1.000. They segment
the same physical object; that is self-overlap, not contact.
`analyze_contact.py` now warns when a contact class overlaps the hand at ≥0.95 in
more than half of frames.

### `person` is auto-excluded from contact classes

Hands are always inside the person mask, so the lane would saturate. Override with
`--contact-classes` if you really want it.

### The report shows whichever video you point it at

`--video` means "video to analyse". Use **`--report-video`** to make the HTML
display the mask-annotated render instead of the raw input. Without it the report
shows the plain video and the masks look missing.

### Naming collision: SAM 3 vs SAM 3D Body

`C:\Users\aitor\Desktop\FastSAM3DToOpenSim\checkpoints\sam-3d-body-dinov3\` is
**SAM 3D Body** — 3D human mesh recovery. DART needs **SAM 3** — open-vocabulary
2D detection. Different architectures, different weights, nothing shared.
