# DAIRT — engineering notes

Working notes behind **DAIRT (Detect Any Interaction in Real Time)**, a
hand/object contact pipeline built on top of
[DART](https://github.com/mkturkcan/DART).

Start with the [main README](../README.md) for setup and usage. These documents
record *why* things are the way they are, and what was measured to get there.

Everything here is **measured on this machine**, not quoted from the upstream
README. Where our numbers disagree with upstream, ours are the ones that were
observed here.

## The short version

| | |
|---|---|
| Does it run on an 8 GB laptop GPU? | Yes. 6.7 FPS boxes, 3.4 FPS with masks, 4 classes at 1008 px. |
| Biggest setup trap | The README's install command is wrong for this GPU. See [01](01-setup-and-hardware.md). |
| Biggest quality lever | The wording of the class prompt. Not thresholds, not filtering. See [03](03-prompt-findings.md). |
| Biggest unsolved limit | 2D masks cannot tell contact from depth alignment. See [05](05-limits-and-gotchas.md). |

## Read in this order

1. **[Setup and hardware](01-setup-and-hardware.md)** — the environment, the two
   install traps, the TensorRT engines, measured speed.
2. **[Running it](02-running-it.md)** — the double-click launcher, the scripts,
   and what each one produces.
3. **[Prompt findings](03-prompt-findings.md)** — the single highest-value
   document here. Prompt wording changed results more than every other knob
   combined.
4. **[Contact detection](04-contact-detection.md)** — how the overlap metric,
   temporal filtering and motion gating actually work, and why each choice was made.
5. **[Limits and gotchas](05-limits-and-gotchas.md)** — what does not work, and
   the bugs that cost real time.
6. **[Roadmap](06-roadmap.md)** — depth discrimination with MoGe-2, image
   exemplars, and what else is worth building.

## Files in `C:\Dart`

| File | What it is |
|---|---|
| `RUN ANALYSIS.bat` | Double-click. Asks for a video and objects, runs everything. |
| `run_analysis.py` | The orchestrator behind the launcher. |
| `analyze_contact.py` | Contact measurement + HTML report generator. |
| `diag_recall.py` | Per-class detection recall. **Run this before trusting any analysis.** |
| `check_report.js` | Node syntax check for generated reports. |
| `demo_video.py` | Upstream, **locally patched** to support masks. |
| `*.engine` | TensorRT engines. GPU- and TRT-version specific, not portable. |
| `outputs\<video>\` | Per-video results: masks mp4, json, html report. |

## The one habit worth keeping

**Run `diag_recall.py` before any full analysis.** It takes 60–90 seconds and
tells you whether your prompts actually detect the objects. Three separate times
in this work, a downstream number looked like a method problem and was really a
prompt problem. Contact percentages computed on a class that only fires in 40% of
frames are meaningless, and nothing later in the pipeline will tell you that.
