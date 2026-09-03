# Roadmap

Ordered by value. Item 1 addresses the single biggest weakness in the method.

---

## 1. Depth as a discriminator (MoGe-2)

### The problem it solves

2D mask overlap cannot tell contact from depth alignment. In one test clip the
operator stands behind a table with a can in front of him; his hands hang at
roughly the same **image height** as the can, so the masks overlap substantially
while the objects are half a metre apart in 3D.

The result is a **13.7-second false "grab"** before he touches it, which survives
every threshold tested including 0.40/0.25 — the measured overlap genuinely is
large. No parameter fixes this, because the information is not in the 2D masks.

Motion gating was tried first as the cheaper option. It helps (30 events → 22 on
that clip) but does not fix this case, because it needs object identity over time
and the tracked centroid drifts onto lookalike detections. **Depth is per-frame
and does not care which instance is which**, which is exactly why it is the
better investment.

### Approach

Require the hand and object to be close **in depth** as well as overlapping in
pixels.

**2D proposes, depth disposes.** Mask overlap is already a good *candidate*
detector; it just cannot reject depth-aligned false positives. Run depth **only
on frames where 2D overlap is already non-zero** — typically 20–40% of frames.
That turns depth from "double the cost of every frame" into a verification stage
on a minority of them.

### Design notes

These are the non-obvious parts, and getting them wrong will waste days.

**Do not use an absolute distance threshold.** Monocular depth is
scale-ambiguous, so "within 10 cm" is not stable across clips or even across
shots. Instead compare the hand's median depth against the **object's own depth
spread**: sample depth inside the object mask, take its range, and ask whether
the hand's median depth falls inside that range plus a margin. Scale-free, and
needs no calibration.

**Erode both masks before sampling depth.** Depth estimates are unreliable at
object boundaries, where they bleed between foreground and background. The
interior is trustworthy; the edge is not.

**Use the median, not the mean.** A handful of bad boundary pixels otherwise
drags the estimate badly, and the mask edges are exactly where depth is worst.

**Transparent objects will stay hard.** Monocular depth on a drinking glass is
poor because the network largely sees through it. Stereo and time-of-flight
sensors fail for the same physical reason, so this is not fixed by better
hardware.

### Two sources, both worth supporting

| | Notes |
|---|---|
| **MoGe-2** (monocular, estimated) | Works on footage already shot, including archives. Start here. |
| **RealSense** (measured) | Categorically better where you control the recording. |

### Suggested implementation order

1. Add `--depth moge2` to `analyze_contact.py`, loading the model lazily so
   nothing changes for runs that do not ask for it.
2. Compute depth only for frames where overlap is already non-zero.
3. Store a per-frame depth-agreement value in the JSON, **alongside** the
   existing overlap rather than replacing it.
4. Add a **Depth strictness** slider and a per-object checkbox to the report,
   mirroring the motion gate.
5. Validate on the known false-positive clip, where the right answer is known in
   advance.

**Step 3 is the important one.** Keeping depth as a separate stored signal means
it tunes in the browser like every other parameter, and a bad depth estimate can
never destroy the underlying overlap data. It also makes the two gates
independently switchable, so their contributions can be compared honestly.

### Expected cost

MoGe-2 adds a second network pass. Running it on ~30% of frames should cost
roughly 30–50% more analysis time — acceptable given analysis is currently the
fast half (~14 FPS at `--stride 3`).

---

## 2. Image exemplars instead of text prompts

### The problem it solves

Open-vocabulary detection has a vocabulary boundary. On factory footage, **21
prompts** failed to find a stainless lifting beam or a pneumatic nutrunner; the
best available handle was `chain` (the hoist chain the device hangs from, not the
device). `pneumatic screwdriver`, `nutrunner`, `air tool` and `power tool` all
scored **0.0%** recall.

Higher input resolution does not help — 1008, 1400 and 1736 px were tested and
the model is tuned for ~1008.

### Why it is tractable

The plumbing already exists upstream. `sam3/model/sam3_image.py:204`:

```python
prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
```

Text, geometry and visual prompts are interchangeable entries in one prompt
sequence. DART passes **zeros** for `visual_prompt_embed` (line 195), but the
path is wired all the way through. There is also an unused
`--class-method prototype` with a calibrated-prototypes file.

**The TensorRT engines need no rebuild.** The encoder-decoder takes `text_feats`
as an input tensor and has no idea where the numbers came from. A visual
embedding occupying that slot works with the engines you already have. Only
*appending* visual prompts alongside text would change the sequence length and
force a re-export.

### Effort

- **Half a day**: confirm the checkpoint contains SAM 3's exemplar encoder. The
  consumer exists; the producer is unverified. Treat this as a genuine go/no-go.
- **1–2 days** if it does: wire it up, add a "mark the object once" step (a box
  on a single frame is enough to start), cache the embedding to a `.pt` exactly
  like the existing text cache.
- If it does not: fall back to prototypes — crop the object from several frames,
  embed the crops, average. SAM 3's text encoder is CLIP-derived (the
  `bpe_simple_vocab_16e6.txt.gz` asset is the CLIP BPE vocab), so a paired image
  encoder in the same space very likely exists.

### Why it suits this use case

Exemplars work best when the same physical object recurs. A fixture in a
particular factory is the same unit in every recording — mark it once, reuse the
embedding forever. Far better matched than text, which has to name a category
that barely exists in web imagery.

**Caveat: exemplars fix the vocabulary problem, not the scale problem.** If an
object is failing partly because it is small and distant, an exemplar will not
rescue that half.

---

## 3. Left / right hand attribution

Contact detection deliberately needs no object identity. Saying *which* hand does
need it.

ByteTrack cannot supply it — it associates on box IoU with no appearance model,
so two visually identical gloved hands are indistinguishable. On handheld
footage it produced 215–371 unique tracks for a scene with ~5 real objects.

Wrist keypoints from a pose model are the durable answer: they give left and
right wrist as distinct **anatomical** joints, inferred from body kinematics, so
they survive motion blur and partial occlusion where an appearance detector
fails. Combined with DAIRT's masks this yields which hand holds which object.

---

## 4. Smaller items

**Make dilation a slider.** It is currently the least-validated parameter in the
pipeline — the default was chosen from a handful of probe frames. Storing overlap
at several dilation values during analysis would make it tunable in the browser
like everything else, at negligible extra cost.

**Record hand detection per frame.** The JSON stores overlap but not whether a
hand was detected at all, so the report cannot distinguish "no hand visible" from
"hand visible, not touching". Cheap to add and makes failure modes legible rather
than guessable.

**Batch the mask GPU→CPU transfer.** `analyze_contact.py` copies masks one at a
time, forcing a sync per detection.

**CSV export of the event log.** The report copies as text; a CSV would drop
straight into analysis tools.

**Warn on suspicious contact classes.** Already done for classes that overlap the
hand at ≥0.95 in most frames (the `glove` vs `hand` self-overlap trap). The same
idea could flag classes whose detections/frame is high enough to corrupt the
max-over-instances metric.
