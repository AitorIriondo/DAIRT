# DARTF: Detect Anything in Real Time Faster

W8A8 INT8 deployment of the SAM3 ViT-H detector on TensorRT: 158 ms per 1008 px frame on a Jetson AGX Orin (DART FP16: 275 ms) at FP32-level detection quality (COCO val2017: 56.0 AP vs 56.1 FP32).

## Quickstart

```bash
# 0) plugins (ARCH=87 Orin; 80/86/89/90 for other GPUs) — needs TensorRT 10, CUDA 12.4+/13, CUTLASS 3.5.1 headers
cd plugins && ARCH=87 ./build.sh && cd ..

# 1) full pipeline: HF weights -> exact rewrites -> calibration -> activation-aware GPTQ -> INT8 ONNX -> TensorRT engine -> verification
DART_ROOT=/path/to/DART CALIB_IMAGES=data/calib CALIB_IDS=calib/calib_ids.json CHECK_IMG=data/check.jpg scripts/pipeline.sh out

# 2) evaluate against the FP32 engine on COCO images (detection counts, recall, IoU)
python runtime/eval_pipeline.py --cand out/vision_int8.plan --ref out/vision_fp32.plan --text <text.plan> --ground <ground.plan> --dev-images 96

# 3) video demo: detector head (K=4 prompt bucket) and, optionally, the SAM3 segmentation head as TensorRT engines
python demo/export_ground_mask.py --dart /path/to/DART --ckpt sam3.pt --out assets --phase-conv   # ground_c4m.onnx, maskhead_q32_phase.onnx (exact, 24% faster head), img_pos_c4.npy
python runtime/build_engine.py assets/ground_c4m.onnx assets/ground_c4m_fp16.plan --no-int8
python demo/build_mask_engine.py assets/maskhead_q32_phase.onnx assets/maskhead_q32_phase_fp16.plan 1 4 8
LQ_PLUGINS=plugins/lq_plugins.so python demo/run_video.py input.mp4 out.mp4 "lemon,strawberry" --assets assets --heads masks --tracker light --name "Your Name" --affiliation "Your lab"
#   --heads boxes         detector only, no segmentation head (skips its engine and its ms per frame)
#   --tracker none        raw per-frame detections;  --tracker light  lightweight tracker (default);  --tracker sam3  SAM3 native propagation for lost tracks (needs step 4)

# 4) optional: SAM3 native tracker engines (for --tracker sam3) and an INT8 mask head
python demo/export_tracker.py --dart /path/to/DART --ckpt sam3.pt --out assets --export --v2 && python demo/build_trk.py assets/trk_step_v2.onnx assets/trk_step_v2_fp16.plan "B=1:2:8,Kn=1024:15552:36288,P=1:6:19"
python demo/build_trk.py assets/trk_neck.onnx assets/trk_neck_fp16.plan "B=1:1:1" && python demo/build_trk.py assets/trk_init.onnx assets/trk_init_fp16.plan "B=1:4:32"
python demo/dump_maskhead_calib.py --assets assets calib.npz input.mp4:lemon,strawberry:20,140,260 && python demo/quant_maskhead.py assets/maskhead_q32.onnx calib.npz assets/maskhead_q32_int8.onnx
```

## Installation

- TensorRT 10.x (tested 10.16.2 on JetPack 7.2 and 10.16.1 on x86), CUDA 12.4+ or 13.x, an sm80+ NVIDIA GPU (tested: Jetson AGX Orin SM87, RTX 4090 SM89; `build.sh` picks a 3-stage GEMM pipeline on sm86/sm89, whose 99 KB of shared memory per block cannot hold the 5-stage tiles).
- CUTLASS 3.5.1 headers: `git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git third_party/cutlass`
- Python 3.10+: `pip install onnx onnxruntime numpy tensorrt torch transformers pillow` (CPU torch is enough for the export and quantization pipeline; the video demo needs CUDA torch, plus `opencv-python scipy ffmpeg`).
- DART (this repository) for the HF backbone ONNX export; `facebook/sam3` weights from the Hugging Face hub.

## Heads and trackers

Two switches of `demo/run_video.py` select what runs per frame; everything else (backbone, text encoder once per prompt set, detector head) is the same. `--pipeline` overlaps the backbone of the next frame with the head, tracker and output of the current one (a decode thread plus a second CUDA stream, as in DART's runner): 2.0× / 1.7× on the 4090 and 1.24× / 1.17× on the Orin for boxes / masks in headless runs.

| `--heads` | what runs | cost per frame (4 prompts) | use |
|---|---|---|---|
| `boxes` | detector head only | RTX 4090: 16 ms head; Orin: 94 ms head (K=4 bucket) | detection, counting, box tracking |
| `masks` (default) | detector head + SAM3 segmentation head on the top 32 queries of each active prompt | adds 12 ms on the 4090, 31 ms on the Orin (phase head; 29 ms INT8) | masks, mask-based association |

| `--tracker` | what it does | MOT17 half split, zero shot (HOTA / MOTA / IDF1) | cost |
|---|---|---|---|
| `none` | per-frame detections, no identities | detection only: DetA 34.7 | none |
| `light` (default) | association by mask IoU (or box IoU with `--heads boxes`), constant-velocity prediction and cosine similarity of the DETR query embeddings; coasting, class vote | boxes 40.0 / 26.9 / 46.6, masks 41.8 / 24.9 / 49.0 (score 0.5); boxes at score 0.7: 41.7 / 33.7 / 49.6 | CPU only, negligible |
| `sam3` | `light` plus the SAM3 native tracker (`demo/sam3_track_np.py`: no torch needed; raw CUDA buffers, two small gather kernels, CUDA-graph replay of the engine calls): memories refreshed from detector masks every `--refresh` frames, propagation with memory attention for confirmed tracks the detector missed (`--sam3-budget` per frame), memory keys pruned to the object's neighbourhood plus a background grid (`--sam3-prune`, 0 = full memory), `--sam3-adaptive` for a shallower memory on stable objects | masks 42.5 / 16.8 / 49.2: best association (AssA 50.6) but it keeps occluded objects alive, which MOT17 counts as false positives | 4090 ~6 ms per propagated object, Orin ~33 ms; +25 ms/frame on MOT17 (4090); the memory bank stays on the device |

## MOT17 evaluation (train half, TrackEval)

`demo/eval_mot.py` runs any configuration headless on the standard validation half of the seven MOT17 training sequences (second halves, 2622 frames; the Hugging Face mirror `Morrison1025/MOT17` ships it as `ablation/`) and scores it with [TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA, CLEAR, Identity). Prompt "person", no training on MOT17.

```bash
python demo/eval_mot.py --mot /data/MOT17 --split ablation --trackeval /path/to/TrackEval --out out/mot --runner-args "--assets assets --ground assets/ground_c1m_fp16.plan --img-pos assets/img_pos_c1.npy" \
  --configs "boxes_light:--heads boxes --tracker light" "masks_light:--tracker light" "masks_sam3:--tracker sam3 --sam3-budget 8"
python demo/mot_table.py out/mot
```

| config | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | RTX 4090 ms/frame | AGX Orin ms/frame |
|---|---|---|---|---|---|---|---|---|
| boxes, no tracker | 12.6 | 34.7 | 4.8 | -14.0 | 9.8 | 18644 | 25 | 186 |
| boxes, light | 40.0 | 37.5 | 44.0 | 26.9 | 46.6 | 352 | 25 | 186 |
| boxes, light, score 0.7 | 41.7 | 38.5 | 46.4 | 33.7 | 49.6 | 249 | 25 | 186* |
| masks, light | 41.8 | 37.7 | 47.6 | 24.9 | 49.0 | 288 | 33 | 239 |
| masks, light, score 0.6 | 42.3 | 38.3 | 47.9 | 28.1 | 50.0 | 248 | 33 | 239* |
| masks, sam3 | 42.5 | 36.6 | 50.6 | 16.8 | 49.2 | 263 | 62 | 416 |

ms are wall time per frame with `--pipeline`, headless, single prompt (K=1 head bucket), measured on MOT17-02 (299 frames); * = the detection score does not change the per-frame cost, so the base configuration's time is reported. Without the pipeline the 4090 takes 52 / 56 / 85 ms and the Orin 231 / 279 / 459 ms (boxes / masks / sam3). The sam3 rows use the torch-free driver (CUDA-graph replay of every tracker engine call, memory bank and key gathers on the device); on the Orin the tracker propagates 4.6 objects per frame on this crowded benchmark and the step engine, not the driver, sets its cost (about 33 ms per object). The Orin runs use the same engines and agree with the 4090 within noise (masks + light 41.9 vs 41.8 HOTA; masks + sam3 42.0 vs 42.5, both with the torch-free driver; the torch driver gives 42.3 on the 4090). Mask association is worth about +2 HOTA over box association; SAM3 propagation buys association at the price of false positives on this benchmark.

## Examples

```bash
# build the grounding head with exact prefix sharing (one prefix engine per image, one post engine per class bucket)
scripts/head_pipeline.sh ground_c1.onnx ground_c16.onnx out_head

# verify an engine against the onnxruntime FP32 golden (rel-L2 of the finest FPN level, latency)
python runtime/verify_plan.py out/vision_int8.plan

# energy per frame on the Orin
python runtime/energy_bench.py out/vision_int8.plan 20

# CUDA-graph replay in the DART TensorRT runner (exact): apply patches/runner_cuda_graph.patch and run with LQ_CUDA_GRAPH=1
```

Ready-made calibration artifacts for the shipped recipe (activation scales, corrected biases) are in `calib/`; the GPTQ weight cache is regenerated by step 3 of the pipeline (about 20 minutes on a CPU).

## Results (Jetson AGX Orin, 1008 px, batch 1, detection metrics vs the FP32 engine on 100 COCO images)

| engine | ms | J/frame | dets / R@0.5 / R@0.75 / IoU (FP32: 440 / 268 / 227 / 0.794) |
|---|---|---|---|
| DART FP16 | 275 | 13.2 | — |
| **DARTF W8A8 (all 32 blocks INT8, default)** | **158** | **7.7** | 440 / 270 / 232 / 0.807 |
| DARTF W8A8, block 0 activations in FP16 | 164 | 8.0 | 441 / 269 / 228 / 0.808 |

COCO val2017 box AP (5000 images, 80 category prompts, official SAM3 protocol: no NMS, no threshold, top 100 per image; same INT8 graphs on an RTX 4090):

| engine | AP | AP50 | AP75 | APs / APm / APl | AR100 |
|---|---|---|---|---|---|
| FP32 reference | 56.10 | 73.88 | 61.79 | 40.63 / 60.28 / 71.08 | 72.85 |
| **DARTF W8A8 (default)** | **56.01** | 73.88 | 61.75 | 40.65 / 60.20 / 70.90 | 72.81 |
| DARTF W8A8, block 0 activations in FP16 | 55.97 | 73.75 | 61.73 | 40.57 / 60.20 / 70.90 | 72.72 |

Grounding head: 21.9 ms per class after exact prefix sharing (24.2 before); CUDA-graph replay takes the trunk to 162.6 ms.

On an RTX 4090 the same INT8 engine runs the backbone in about 23 ms per frame; with the K=4 detector bucket (16 ms) and the segmentation head (12 ms at four prompts) the video demo runs at 20 to 25 frames per second end to end, including decoding and rendering; `--heads boxes` removes the segmentation head.

## Documentation

- [docs/METHOD.md](docs/METHOD.md): what is done and why it is exact.
- [docs/KERNELS.md](docs/KERNELS.md): the attention kernel and the block-level plugins.
- [docs/HEAD.md](docs/HEAD.md): grounding encoder/decoder deployment.
- [docs/references.bib](docs/references.bib): bibliography (fetched and verified by `docs/check_references.py`).

## Layout

`export/` ONNX export and exact rewrites · `quant/` calibration, GPTQ, quantizer · `plugins/` TensorRT plugins and ONNX splices · `kernels/` attention kernel and micro-benchmarks · `runtime/` engine build, verification, evaluation · `demo/` segmentation head export and the video demo · `scripts/` pipelines · `patches/` DART runner patches · `calib/` artifacts of the shipped recipe.

## Citation

See the DART citation in the repository root; a DARTF paper is in preparation.
