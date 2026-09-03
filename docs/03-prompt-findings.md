# Prompt findings

**The single most important document here.** Across this work, changing the
wording of a class prompt moved results far more than any threshold, filter
parameter or model setting. Everything below is measured with `diag_recall.py`.

## How to read the numbers

`diag_recall.py` reports two things that matter, and they fail in opposite
directions:

- **recall** — % of frames with at least one detection. Low recall = the prompt
  misses the object.
- **detections/frame** — how many instances fire. If one object is present and
  this reads 8, the prompt is matching things you did not ask for.

**A prompt can fail by being too narrow (low recall) or too broad (high
detections/frame). You need both columns to tell which.** Score alone will not
show you either problem.

## Finding 1: specific nouns beat generic ones — for precision

Kitchen scene, one can and one glass present, 1920×1080:

| prompt | recall | dets/frame | median score |
|---|---|---|---|
| `can` | 100% | **8.09** | 0.959 |
| **`soda can`** | 100% | **2.55** | 0.952 |
| `glass` | 100% | **23.30** | 0.950 |
| **`drinking glass`** | 100% | **2.66** | 0.941 |
| `apple` | 100% | 1.22 | 0.945 |

`glass` fired **23 times per frame** for one glass. In English it names a
material as well as a vessel, so it matched the oven door, the window, the wall
tiles, the blender jar and the bottles. `drinking glass` can only be the vessel:
**8.8× fewer spurious detections, no loss of recall or confidence.**

This directly corrupted a downstream number. Contact for `glass` read 46.7%;
with `drinking glass` it read 26.8%. The first figure was mostly hands passing
near oven doors.

`apple` at 1.22/frame is what a clean prompt looks like.

## Finding 2: low recall can be *correct* — it means selectivity

Workshop video, operator using a pneumatic impact wrench near a pegboard of other
tools:

| prompt | recall | dets/frame | median score |
|---|---|---|---|
| `power tool` | 97.2% | **6.17** | 0.812 |
| `impact wrench` | 95.6% | **7.06** | 0.800 |
| `screwdriver` | **47.3%** | **1.20** | 0.365 |

Initially read as a failure: `screwdriver` misses half the frames. **That reading
was wrong.** The operator uses several tools; the goal was to isolate one.
`power tool` at 6.17/frame fires on every tool in the room. `screwdriver` at
1.20/frame fires on the one being asked about.

**Low recall of a specific prompt is discrimination, not failure.** Which column
you want depends on whether you need coverage or specificity — decide that before
reading the numbers, or you will misinterpret them.

## Finding 3: adding qualifiers can make things worse

| prompt | recall |
|---|---|
| `screwdriver` | 47.3% |
| `pneumatic screwdriver` | **25.2%** |

More precise language is not automatically a better prompt. The model matches
against visual concepts as they appear in training data, not against your
technical vocabulary. Test, don't reason.

## Finding 4: the same prompt is not portable between scenes

`power tool`:

| video | recall |
|---|---|
| Workshop close-up (1280×720) | **97.2%** |
| Factory wide shot (1920×1080) | **0.0%** |

Zero. Same prompt, same model, same settings. **Always re-run `diag_recall.py`
when you change footage.** A prompt validated on one video tells you nothing
about another.

## Finding 5: custom industrial fixtures are outside the vocabulary

Volvo Skövde factory: an operator uses a chain-hung stainless spreader beam to
move cylinder heads, and a pistol-grip pneumatic nutrunner on a balancer hose.
**21 prompts tested.**

The lifting device:

| prompt | recall | dets/frame | score |
|---|---|---|---|
| **`chain`** | **100%** | 3.81 | **0.835** |
| `handle` | 61.8% | 3.63 | 0.354 |
| `clamp` | 36.5% | 1.01 | 0.200 |
| `hoist` | 9.4% | 0.13 | 0.386 |
| `lifting device` | 7.3% | 0.08 | 0.366 |
| `lifting beam` | 5.6% | 0.07 | 0.346 |
| `crane` | 0.4% | 0.00 | 0.320 |
| `gripper`, `crane hook` | 0.0% | — | — |
| `metal bar` | 100% | **122.76** | 0.761 |

The pneumatic tool:

| prompt | recall |
|---|---|
| `pneumatic screwdriver`, `nutrunner`, `air tool`, `power tool` | **0.0%** |
| `drill` | 1.3% |

**Nothing describes the nutrunner.** The best handle on the lifting device is
`chain` — not the device itself, but the hoist chain it hangs from, which *is* a
common visual category. Note the operator grips the beam handles, not the chain,
so `chain` localises the device without being what she touches.

`metal bar` at **122 detections per frame** is the instructive failure: in a
factory, everything is a metal bar.

**The boundary is what the internet has many pictures of.** Apples and soda cans
work. A custom stainless lifting fixture does not, however precisely you name it.
For objects like these, text prompting is the wrong tool — see
[05-limits-and-gotchas.md](05-limits-and-gotchas.md) on image exemplars.

## Finding 6: higher input resolution does not rescue small objects

Tested whether the factory failures were about scale, on one frame:

| `--imgsz` | best score, any lifting prompt |
|---|---|
| 1008 | 0.110 |
| 1400 | 0.000 |
| 1736 | 0.100 |

**Higher resolution did not help and slightly hurt.** The model is tuned for
~1008 px. Do not build a larger engine hoping to find small objects.

## Finding 7: `hand` is reliable; `glove` is not better

| prompt | recall | median score |
|---|---|---|
| `hand` | 90.5% | 0.856 |
| `glove` | 87.7% | 0.873 |
| `work glove` | 86.4% | 0.868 |

Tested because the operator wears grey work gloves. They are equivalent —
`hand` marginally ahead on recall, `glove` marginally on score. `hand` reaches
94–97% on the static-camera videos.

**Do not use `glove` as a contact class alongside `hand`.** They segment the same
physical object, so overlap saturates at 1.0 and looks like a perfect result.
`analyze_contact.py` now warns when a class overlaps the hand at ≥0.95 in more
than half of frames.

## Practical recipe

1. Run `diag_recall.py` with 4–8 candidate prompts. 60–90 seconds.
2. Want **coverage**? Take high recall, accept extra detections/frame.
   Want **one specific object**? Take the prompt nearest 1.0 detections/frame.
3. Anything above ~3 detections/frame for a single object will corrupt contact
   numbers — the metric takes the max over instances, so a spurious detection
   near a hand reads as contact.
4. Re-test whenever the footage changes.
