# Contact detection: how it works and why

The goal: from a video, produce a timeline of when a person is holding each
object, plus a readable event log.

## The core insight: contact does not need object IDs

Early on this looked like it needed reliable tracking, so that "hand #2 holds
tool #5" could be stated. It does not.

**Per-frame, ask only: does any hand mask touch any instance of this class?**
That is a boolean per frame per class. Track identity is irrelevant, which
sidesteps the tracking problem entirely (ByteTrack produced 215–371 unique tracks
for a scene with ~5 real objects on handheld footage).

Left/right hand attribution *would* need identity, but that is a later refinement,
not a prerequisite.

## Step 1: the overlap metric

### Dilation is mandatory, not a refinement

Measured on a frame where the operator is unambiguously gripping the wrench:

```
hand (area 13409) x tool (area 8059):  raw intersection = 0
```

**Zero.** SAM assigns each contact pixel to one object or the other, so a gripping
hand and tool produce *adjacent* masks, not overlapping ones. Raw intersection
reads zero at exactly the moment of contact.

Both masks are dilated before intersecting. At 1280×720:

| dilation | measured overlap on that frame |
|---|---|
| 0 px | 0.000 |
| 9 px | 0.068 |
| 25 px | 0.225 |

### Dilation must scale with resolution

25 px was tuned at 1280 wide. `run_analysis.py` scales it automatically:

```python
dilate = max(9, round(25 * frame_width / 1280))    # 38 px at 1920
```

### Normalise by the *dilated* hand area

The metric is "what fraction of the hand is touching". An early version divided
by the **raw** hand area while intersecting **dilated** masks, so values reached
**6.0** on a quantity defined to be ≤ 1.

That was not merely cosmetic: dilation grows a small mask proportionally more, so
the bug inflated exactly the small, distant hands that are least reliable. Divide
by the dilated area and it is bounded to [0, 1].

### Why not IoU

A hand and a tool differ too much in area for IoU to mean anything — it stays low
during a firm grip because the union is dominated by the larger object.

## Step 2: temporal filtering

Raw per-frame overlap is noisy. Three stages, each fixing a different failure,
all applied **in the browser** so they retune without re-running the model:

| stage | fixes |
|---|---|
| **EMA** (default 0.30) | mask jitter frame to frame |
| **Hysteresis** (enter/leave thresholds) | chattering when the value sits near the threshold |
| **Morphological close** (15 fr) | dropouts in the middle of a real grasp |
| **Morphological open** (0 fr default) | short phantom grasps |

Close and open are the natural expression of "ignore state changes shorter than N
frames" — close bridges gaps inside a `true` run, open deletes short `true` runs.

Hysteresis is the stage people skip. With one threshold, a signal hovering near it
flips every few frames. Two thresholds with a dead band stop that.

### Defaults in use

```
Enter contact  0.165     Bridge gaps  15 fr (0.50 s)
Leave contact  0.165     Drop blips    0 fr
Smoothing      0.30
```

Chosen by inspection against the kitchen footage. Note `enter == leave` disables
the dead band — fine on clean footage where the signal moves decisively, more
prone to chatter on noisy footage. With **Drop blips = 0** nothing suppresses
short events; raising it to ~8 frames removes sub-second flicker if a clip needs it.

## Step 3: motion gating (optional, per object)

A held object moves *with* the hand. A static one does not. Over a window, project
the object's displacement onto the hand's:

```
agreement = (dObj · dHand) / |dHand|²
```

Held → ~1. Static → ~0.

**Key design decision: when the hand barely moves, the test abstains.** It returns
inconclusive rather than vetoing, because otherwise holding something still would
read as "not holding" — a worse bug than the one being fixed. Motion only ever
removes contacts where the hand demonstrably moved and the object demonstrably
did not.

### Per-object checkboxes

Each object gets a checkbox, **on by default**. Untick anything that stays put
when touched — a table, a fixed machine, a conveyor. Motion gating would wrongly
veto every real contact with those.

### Honest assessment: it half works

On `soda_apple` it cut events from 30 to 22, removing real noise. It did **not**
fix the false positive that motivated it (a 13.7 s "grab" of a can sitting on a
table).

**Why: motion gating needs object identity, even though the contact boolean does
not.** The tracked centroid must follow one physical object over time. Two
failures were found and one remains:

1. *(fixed)* Recording the centroid of whichever instance best overlaps the hand
   makes the track follow the hand **by construction** — the test becomes
   circular. Replaced with nearest-neighbour association to the previous frame.
2. *(fixed)* Sudden jumps to lookalikes elsewhere — capped by `--max-jump`.
3. *(unsolved)* **Gradual drift.** With `soda can` also firing on counter bottles,
   the track walks from the real can onto a bottle near the hands, ~0.03/frame.
   Too slow for a jump limit to catch. Agreement then reads **+1.08** — "moves
   perfectly with the hand" — for an object that never moved.

So motion gating is reliable **only for cleanly-detected objects** (one instance
per frame, no lookalikes). That makes the per-object checkbox the right control:
leave it on for the apple, off for the can.

This also reverses an earlier judgement: motion was proposed as the cheap
alternative to depth, but it inherits the identity problem. **Depth is per-frame
and does not care which instance is which**, so it is now the better investment.

## Step 4: the report

`analyze_contact.py` writes a `.json` of raw per-frame values plus a standalone
HTML.

**The architecture that matters: inference and analysis are fully separated.**
Inference takes minutes; thresholding takes milliseconds. All filter parameters
live in the browser and recompute client-side, so tuning never re-runs the model.

```bash
# rebuild the report from existing data — no GPU, instant
python analyze_contact.py --from-json outputs/clip/clip.json -o report.html
```

The report contains: the mask video, one timeline lane per object, live sliders, a
stats table (contact time, % of clip, grasp count, mean and longest grasp) and a
text event log with timestamps.

### Example output

```
00:00.5  Person grabs the soda can
00:05.7  Person releases the soda can      (held 5.2s)
00:06.6  Person grabs the apple
00:10.9  Person releases the apple         (held 4.3s)
00:12.1  Person grabs the drinking glass
00:17.0  Person releases the drinking glass (held 4.9s)
```

Clicking a line seeks the video. "Copy as text" puts the log on the clipboard.

## Masks are required, and cost the TRT enc-dec

`sam3/model/sam3_multiclass_fast.py:229`:

> `trt_enc_dec_engine_path requires detection_only=True` — *TRT enc-dec does not
> produce hidden states for mask generation*

The exported enc-dec graph emits only `scores` and `boxes`. With masks you keep
the TRT **backbone** (the larger win) and fall back to PyTorch for the enc-dec:
**150 ms → 296 ms per frame**.

Worth it. Boxes are actively misleading here — the air hose box covers half the
frame while its mask is a thin curve, so box-based contact logic reports constant
false contact with hands, legs and tools.

`demo_video.py` hardcoded `detection_only=True` and had no mask path at all. It is
**locally patched** to add `--masks` / `--mask-alpha`, mask blending under the
boxes, and a guard rejecting `--masks` with `--trt-enc-dec`. **Reapply after any
`git pull`.**
