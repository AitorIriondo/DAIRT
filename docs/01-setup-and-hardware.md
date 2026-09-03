# Setup and hardware

## The reference machine

| | |
|---|---|
| GPU | NVIDIA RTX 5070 Laptop, **8 GB**, **Blackwell (sm_120)**, 95 W TGP |
| CPU / RAM | AMD Ryzen AI 9 365 / 31 GB |
| OS | Windows 11 |
| Driver | 592.82 (CUDA 13.1 capable) |

Free VRAM is about **6.7–7.3 GB** — Windows keeps ~1.2 GB for the desktop.

## Environment

conda env `dartsam3` at `C:\Users\aitor\anaconda3\envs\dartsam3`

```
Python 3.11          torch 2.11.0+cu128      torchvision 0.26.0+cu128
tensorrt-cu12 10.13.3.9                      transformers 5.16.1
numpy 1.26.4         opencv 4.11.0           imageio-ffmpeg (bundled ffmpeg)
```

Always run with `PYTHONIOENCODING=utf-8` — DART prints Unicode that cp1252 cannot
encode, and it crashes without it.

## Trap 1: the README's PyTorch command is wrong for this GPU

Upstream says:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**This installs cleanly and then fails at the first CUDA call.** cu126 wheels are
built up to sm_90 (Hopper). This GPU is sm_120. Use cu128:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verified — cu128 includes the needed architecture:

```
capability: (12, 0)
arch list:  ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

Stay on CUDA **12.x**, not 13.x, so the TensorRT wheels (built against CUDA 12)
share a runtime with torch.

## Trap 2: pip resolves TensorRT to a broken major version

`pip install -e ".[tensorrt]"` reads `tensorrt>=10.9.0` and installs **TensorRT
11.2 built against CUDA 13**. Two problems at once:

- TRT 11 removed APIs DART calls — `platform_has_fast_fp16`, `EXPLICIT_BATCH`
- CUDA 13 conflicts with torch cu128

Install the version DART targets instead:

```bash
pip uninstall -y tensorrt tensorrt_cu13 tensorrt_cu13_bindings tensorrt_cu13_libs
pip install "tensorrt-cu12==10.13.3.9"
```


## The checkpoint is gated

`facebook/sam3` requires accepting Meta's licence:

1. Accept at https://huggingface.co/facebook/sam3
2. `hf auth login` with a read token

It lands at
`C:\Users\aitor\.cache\huggingface\hub\models--facebook--sam3\snapshots\<hash>\sam3.pt`
(3.45 GB). `run_analysis.py` finds it automatically.

## TensorRT engines

```bash
# encoder-decoder — 105 s build, peak 3321 MiB VRAM
python -m sam3.trt.export_enc_dec --checkpoint <ckpt> --output enc_dec.onnx --max-classes 4 --imgsz 1008
python -m sam3.trt.build_engine --onnx enc_dec.onnx --output enc_dec_fp16.engine --fp16 --mixed-precision none

# ViT-H backbone — 194 s build, 917 MB engine
python scripts/export_hf_backbone.py --image x.jpg --imgsz 1008
```

`--mixed-precision none` matters for the enc-dec: the auto-detect heuristic
applies backbone rules that are wrong there.

**Engines are specific to this GPU *and* this TensorRT version.** They cannot be
copied to another machine. ONNX files are portable; engines are not.

### FP16 accuracy on Blackwell — verified good

Upstream documents that naive FP16 export of the ViT-H backbone produces
numerically worthless features (cosine 0.058) that still run at full speed. Their
fix is the restructured attention in `export_hf_backbone.py`, validated on Ada.

It transfers to sm_120 under TRT 10.13:

```
Cosine: 0.999875   0.999632   0.999517     (the three FPN outputs)
```

No FP32 fallback needed. Worth re-checking if you ever change TRT version — a
silently-wrong engine that still runs fast is the worst failure mode here.

## VRAM ceiling: 4 classes per pass

Engine build costs roughly 1 GB per class slot.

| `--max-classes` | VRAM | On this machine |
|---|---|---|
| 4 | ~3.3 GB measured | fits comfortably |
| 8 | ~8 GB | borderline |
| 16 | more | will not build |

`--max-classes` does **not** cap what you can detect — the predictor chunks larger
class sets into multiple passes. It caps how many run per pass. The 80-class COCO
setup (`scripts/build_coco_engine.py`) is out of reach; upstream notes it OOMs
even on 16 GB.

If a build OOMs, lower `--workspace` (default 4 GB, `sam3/trt/build_engine.py:513`).

## Measured performance

949-frame 1280×720 video, 4 classes, 1008 px:

| Path | ms/frame | FPS | Masks |
|---|---|---|---|
| PyTorch, boxes | 762 | 1.3 | no |
| **Full TRT (backbone + enc-dec), boxes** | **150** | **6.7** | no |
| **TRT backbone + PyTorch enc-dec** | **296** | **3.4** | **yes** |

Backbone alone: **748 ms → 110 ms (6.8×)**.

For reference the paper reports 15.8 FPS on a desktop RTX 4080. This 95 W laptop
lands roughly 2.4× behind. **It is not real time**, and won't be on this hardware.
For offline analysis that is fine — a 30-second clip takes 2–5 minutes.

## Do not run alongside the EasyErgo service

`C:\FastSam3dBodyServer\EasyErgo_FastSam3DService` targets the same 8 GB GPU, and
its SAM 3D Body checkpoint alone is 2.1 GB. They fit individually, not together.

Note the naming collision: that project uses **SAM 3D Body** (3D human mesh
recovery). DART uses **SAM 3** (open-vocabulary 2D detection). Different models,
different weights, nothing shared.
