"""DARTF demo on a video: W8A8 vision backbone (TensorRT + LQ plugins) -> detector head (FP16, K=4 prompt bucket, emits queries and
encoder states) -> segmentation head (FP16, dynamic prompt axis, top-Q queries) -> lightweight tracker -> boxes + masks at 1920x1080.
The text encoder runs once per prompt set (cached); the detector head is image conditioned and runs per frame.
usage: LQ_PLUGINS=plugins/lq_plugins.so python run_video.py video.mp4 out.mp4 "lemon,strawberry" --assets <dir> [--duration 15] [--snap I]
<dir> holds vision_int8.plan, ground_c4m_fp16.plan, maskhead_q32_fp16.plan, img_pos_c4.npy, text_c16.onnx(+.data), bpe_simple_vocab_16e6.txt.gz
(file names can be overridden per flag). Needs tensorrt, torch (CUDA), onnxruntime, opencv, scipy, pillow, ffmpeg and a libcudart.so on the library path."""
import sys, os, json, time, subprocess, argparse, numpy as np
HR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HR); sys.path.insert(0, os.path.join(HR, "..", "runtime"))
import tensorrt as trt, cv2
from scipy.optimize import linear_sum_assignment
from trt_util import load_engine, Runner, TRT_LOGGER, NP_DTYPE
from cuda_alloc import DevBuf
from tokenize_classes import Sam3Tokenizer, CONTEXT_LENGTH, BUCKET
from PIL import Image, ImageDraw, ImageFont
import torch, torch.nn.functional as Fn
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("out"); ap.add_argument("prompts"); ap.add_argument("--assets", default="assets")
ap.add_argument("--vision", default="vision_int8.plan"); ap.add_argument("--ground", default="ground_c4m_fp16.plan"); ap.add_argument("--mask", default="maskhead_q32_fp16.plan")
ap.add_argument("--text-onnx", default="text_c16.onnx"); ap.add_argument("--img-pos", default="img_pos_c4.npy"); ap.add_argument("--bpe", default="bpe_simple_vocab_16e6.txt.gz")
ap.add_argument("--font-dir", default="/usr/share/fonts/truetype/lato"); ap.add_argument("--title", default="DARTF: Detect Anything in Real Time Faster"); ap.add_argument("--name", default=""); ap.add_argument("--affiliation", default="")
ap.add_argument("--max-frames", type=int, default=0); ap.add_argument("--thr", type=float, default=0.5, help="score to start a track"); ap.add_argument("--low", type=float, default=0.25, help="score to continue a track")
ap.add_argument("--nms", type=float, default=0.6); ap.add_argument("--fps-out", type=int, default=30); ap.add_argument("--gpu-name", default="RTX 4090")
ap.add_argument("--stats", default=None); ap.add_argument("--snap", default=None); ap.add_argument("--no-lower-third", action="store_true"); ap.add_argument("--lt-slide", action="store_true", help="slide the lower third in (first clip); otherwise it fades in")
ap.add_argument("--duration", type=float, default=15.0, help="seconds of source video to process"); ap.add_argument("--no-swipe", action="store_true", help="disable the with/without wipe")
ap.add_argument("--lt-in", type=float, default=0.5); ap.add_argument("--lt-out", type=float, default=8.0)
a = ap.parse_args()
for k in ("vision", "ground", "mask", "text_onnx", "img_pos", "bpe"):
    if not os.path.isabs(getattr(a, k)): setattr(a, k, os.path.join(a.assets, getattr(a, k)))
OW, OH = 1920, 1080; QSEL = 32
PAL = [(0, 200, 255), (255, 196, 0), (0, 230, 118), (255, 82, 82), (171, 71, 188), (255, 145, 0), (29, 233, 182), (236, 64, 122)]
def F(name, size):
    try: return ImageFont.truetype(os.path.join(a.font_dir, f"Lato-{name}.ttf"), size)
    except OSError: return ImageFont.load_default(size)
FONT_LAB = F("Semibold", 26); FONT_LAB_S = F("Semibold", 21); FONT_T1 = F("Bold", 40); FONT_T2 = F("Semibold", 30); FONT_T3 = F("Regular", 25); FONT_INFO = F("Regular", 24); FONT_INFO_B = F("Semibold", 24)
prompts = [p.strip() for p in a.prompts.split(",") if p.strip()]; K = len(prompts); assert 1 <= K <= 4, "the K=4 bucket holds at most 4 prompts"

# ---- text encoder: once per prompt set (onnxruntime CPU; rows are independent, so slicing to K is exact) ----
cache = os.path.join(a.assets, "text_" + "_".join(p.replace(" ", "-") for p in prompts) + ".npz")
if os.path.exists(cache): z = np.load(cache); tf, tm = z["tf"], z["tm"]
else:
    import onnxruntime as ort
    tok = Sam3Tokenizer(a.bpe); ids = np.zeros((BUCKET, CONTEXT_LENGTH), np.int32); ids[:, 0] = 49406; ids[:, 1] = 49407
    for i, p in enumerate(prompts): e = tok.encode(p); ids[i, :] = 0; ids[i, :len(e)] = e
    sess = ort.InferenceSession(a.text_onnx, providers=["CPUExecutionProvider"]); tf, tm = sess.run(None, {"token_ids": ids}); np.savez(cache, tf=tf, tm=tm)
tf4 = np.ascontiguousarray(tf[:, :4, :]).astype(np.float16); tm4 = np.ascontiguousarray(tm[:4]).astype(np.float32)
for i in range(K, 4): tm4[i, :] = tm[BUCKET - 1]; tf4[:, i, :] = tf[:, BUCKET - 1, :]
tfK = np.ascontiguousarray(tf4[:, :K, :]); tmK = np.ascontiguousarray(tm4[:K])

# ---- engines ----
vision = Runner(load_engine(a.vision)); ground = Runner(load_engine(a.ground)); img_pos = np.load(a.img_pos)
fpn0_name, fpn1_name = vision.alias.get("fpn_0", "fpn_0"), vision.alias.get("fpn_1", "fpn_1")
class DynRunner:
    def __init__(self, engine):
        self.engine = engine; self.ctx = engine.create_execution_context(); self.names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.inputs = [n for n in self.names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]; self.outputs = [n for n in self.names if n not in self.inputs]; self.bufs = {}
        for n in self.inputs:
            shape = engine.get_tensor_profile_shape(n, 0)[2] if -1 in tuple(engine.get_tensor_shape(n)) else tuple(engine.get_tensor_shape(n))
            dt = NP_DTYPE[engine.get_tensor_dtype(n)]; self.bufs[n] = (DevBuf(int(np.prod(shape)) * np.dtype(dt).itemsize), dt); self.ctx.set_input_shape(n, shape)
        for n in self.outputs: shape = tuple(self.ctx.get_tensor_shape(n)); dt = NP_DTYPE[engine.get_tensor_dtype(n)]; self.bufs[n] = (DevBuf(int(np.prod(shape)) * np.dtype(dt).itemsize), dt)
        for n in self.names: self.ctx.set_tensor_address(n, self.bufs[n][0].ptr)
        self.external = {}
    def bind(self, name, ptr, shape): self.external[name] = ptr; self.ctx.set_tensor_address(name, ptr); self.ctx.set_input_shape(name, shape)
    def __call__(self, feeds):
        for n in self.inputs:
            if n in self.external: continue
            x = np.ascontiguousarray(feeds[n], dtype=self.bufs[n][1]); self.ctx.set_input_shape(n, tuple(x.shape)); self.bufs[n][0].upload(x)
        assert self.ctx.execute_async_v3(0); DevBuf.sync()
        return {n: self.bufs[n][0].download(tuple(self.ctx.get_tensor_shape(n)), self.bufs[n][1]) for n in self.outputs}
mask = DynRunner(load_engine(a.mask)); mask.bind("fpn_0", vision.bufs[fpn0_name][0].ptr, tuple(vision.bufs[fpn0_name][1])); mask.bind("fpn_1", vision.bufs[fpn1_name][0].ptr, tuple(vision.bufs[fpn1_name][1]))

# ---- tracker: mask IoU Hungarian matching, low score continuation, coasting, EMA smoothing, class voting ----
dev = torch.device("cuda")
class Track:
    __slots__ = ("id", "box", "mask", "votes", "hits", "misses", "p")
    def __init__(self, tid, det): self.id = tid; self.box = det["box"].copy(); self.mask = det["mask"].clone(); self.votes = np.zeros(K); self.votes[det["k"]] = det["p"]; self.hits = 1; self.misses = 0; self.p = det["p"]
    def update(self, det): self.box = 0.5 * self.box + 0.5 * det["box"]; self.mask = 0.3 * self.mask + 0.7 * det["mask"]; self.votes[det["k"]] += det["p"]; self.hits += 1; self.misses = 0; self.p = 0.7 * self.p + 0.3 * det["p"]
    @property
    def cls(self): return int(np.argmax(self.votes))
def iou_matrix(A, B):
    if len(A) == 0 or len(B) == 0: return np.zeros((len(A), len(B)))
    A = (torch.stack(A) > 0).float().flatten(1); B = (torch.stack(B) > 0).float().flatten(1); inter = A @ B.t(); return (inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-6)).cpu().numpy()
def match(tracks, dets, min_iou):
    if not tracks or not dets: return [], list(range(len(tracks))), list(range(len(dets)))
    iou = iou_matrix([t.mask for t in tracks], [d["mask"] for d in dets]); r, c = linear_sum_assignment(1 - iou); pairs = [(i, j) for i, j in zip(r, c) if iou[i, j] >= min_iou]
    ti = {i for i, _ in pairs}; di = {j for _, j in pairs}; return pairs, [i for i in range(len(tracks)) if i not in ti], [j for j in range(len(dets)) if j not in di]
class Tracker:
    def __init__(self): self.tracks = []; self.next_id = 1
    def step(self, dets):
        for t in self.tracks: t.votes *= 0.92
        high = [d for d in dets if d["p"] >= a.thr]; low = [d for d in dets if d["p"] < a.thr]
        pairs, un_t, un_d = match(self.tracks, high, 0.25)
        for i, j in pairs: self.tracks[i].update(high[j])
        rest = [self.tracks[i] for i in un_t]; pairs2, un_t2, _ = match(rest, low, 0.3)
        for i, j in pairs2: rest[i].update(low[j])
        for i in un_t2: rest[i].misses += 1
        for j in un_d: self.tracks.append(Track(self.next_id, high[j])); self.next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= 15]
        return [t for t in self.tracks if t.hits >= 5 and t.misses <= 8 and t.p >= 0.4]
tracker = Tracker()

# ---- overlays ----
def rounded_box(img, x0, y0, x1, y1, color, t=3, r=14):
    x0, y0, x1, y1 = [int(round(v)) for v in (x0, y0, x1, y1)]; r = max(2, min(r, (x1 - x0) // 3, (y1 - y0) // 3)); c = color; AA = cv2.LINE_AA
    cv2.line(img, (x0 + r, y0), (x1 - r, y0), c, t, AA); cv2.line(img, (x0 + r, y1), (x1 - r, y1), c, t, AA); cv2.line(img, (x0, y0 + r), (x0, y1 - r), c, t, AA); cv2.line(img, (x1, y0 + r), (x1, y1 - r), c, t, AA)
    cv2.ellipse(img, (x0 + r, y0 + r), (r, r), 180, 0, 90, c, t, AA); cv2.ellipse(img, (x1 - r, y0 + r), (r, r), 270, 0, 90, c, t, AA); cv2.ellipse(img, (x1 - r, y1 - r), (r, r), 0, 0, 90, c, t, AA); cv2.ellipse(img, (x0 + r, y1 - r), (r, r), 90, 0, 90, c, t, AA)
def lower_third():
    l1, l2, l3 = a.title, a.name or " ", a.affiliation or " "
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10))); w = int(max(tmp.textlength(l1, FONT_T1), tmp.textlength(l2, FONT_T2), tmp.textlength(l3, FONT_T3))) + 76; h = 150
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=14, fill=(8, 10, 14, 225)); d.rounded_rectangle([0, 0, 8, h - 1], radius=4, fill=(255, 196, 0, 255))
    d.text((30, 14), l1, font=FONT_T1, fill=(255, 255, 255, 255)); d.text((30, 66), l2, font=FONT_T2, fill=(235, 235, 235, 255)); d.text((30, 106), l3, font=FONT_T3, fill=(190, 190, 190, 255))
    return im
LT = None if a.no_lower_third else lower_third()
def sigmoid(x): return 1 / (1 + np.exp(-x.astype(np.float64)))
def ease(x): x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)

# ---- video I/O ----
probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,nb_frames", "-of", "csv=p=0", a.video], capture_output=True, text=True).stdout.strip().split(",")
num, den = probe[0].split("/"); src_fps = float(num) / float(den); n_frames = int(probe[1]) if probe[1].isdigit() else 0
dec_m = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.video, "-vf", "scale=1008:1008:flags=bilinear", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8)
dec_d = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.video, "-vf", f"scale={OW}:{OH}:flags=bicubic", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8)
enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-framerate", f"{src_fps:.6f}", "-i", "-", "-r", str(a.fps_out), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "medium", a.out], stdin=subprocess.PIPE)
stats = {"frames": 0, "vision_ms": [], "head_ms": [], "mask_ms": [], "tracks": [], "new_ids": [], "appear": []}; prev_shown = set()
XG = np.arange(OW)[None, :, None]
src_dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.video], capture_output=True, text=True).stdout.strip() or 0)
clip_len = min(a.duration, src_dur) if a.duration else src_dur; cycle = clip_len >= 13.0      # the off/on cycle needs 4 s and must finish 2 s before the cut
T_OFF, T_ON = clip_len - 6.0, clip_len - 4.0
def wipe_boundary(t):
    """x boundary of the with/without wipe: left of it the overlay is shown. None = full overlay, 0 = raw.
    0 to 0.5 s raw; 0.5 to 4.5 s the overlay sweeps in from the left; for clips of 13 s or more it sweeps out over [L-6, L-4] and back in over [L-4, L-2]."""
    if a.no_swipe: return None
    if t < 0.5: return 0
    if t < 4.5: return int(OW * ease((t - 0.5) / 4.0)) + 1
    if not cycle or t < T_OFF or t >= T_ON + 2.0: return None
    if t < T_ON: return int(OW * (1 - ease((t - T_OFF) / 2.0)))       # overlay retreats to the left
    return int(OW * ease((t - T_ON) / 2.0)) + 1
t_start = time.time(); fi = 0
while True:
    bm = dec_m.stdout.read(1008 * 1008 * 3); bd = dec_d.stdout.read(OW * OH * 3)
    if len(bm) < 1008 * 1008 * 3 or len(bd) < OW * OH * 3 or (a.max_frames and fi >= a.max_frames) or (a.duration and fi >= int(a.duration * src_fps)): break
    x = np.frombuffer(bm, np.uint8).reshape(1008, 1008, 3).transpose(2, 0, 1).astype(np.float32) * (1 / 127.5) - 1.0
    t0 = time.perf_counter(); v = vision({"images": x[None]}, want=["fpn_2"]); DevBuf.sync(); t1 = time.perf_counter()
    g = ground({"img_feat": np.repeat(v["fpn_2"].astype(np.float16), 4, axis=0), "img_pos": img_pos, "text_feats": tf4, "text_mask": tm4}); DevBuf.sync(); t2 = time.perf_counter()
    probs = sigmoid(g["scores"][:K, :, 0]) * sigmoid(g["presence"][:K, 0])[:, None]
    active = [k for k in range(K) if (probs[k] > a.low).any()]; sel = {k: np.argsort(-probs[k])[:QSEL] for k in active}; dets = []
    if active:
        hs_sel = np.stack([g["hs"][k, sel[k]] for k in active]).astype(np.float16)
        m = mask({"enc": np.ascontiguousarray(g["enc"][:, active, :]), "text_feats": np.ascontiguousarray(tfK[:, active, :]), "text_mask": np.ascontiguousarray(tmK[active]), "hs_sel": hs_sel})["masks"]
        mt = torch.from_numpy(m.astype(np.float32)).to(dev)
        cand = sorted([(float(probs[k, q]), k, q, ai, qi) for ai, k in enumerate(active) for qi, q in enumerate(sel[k]) if probs[k, q] > a.low], key=lambda c: -c[0])
        if cand:
            M = (torch.stack([mt[c[3], c[4]] for c in cand]) > 0).float().flatten(1); inter = (M @ M.t()); area = M.sum(1)
            iou = (inter / (area[:, None] + area[None, :] - inter + 1e-6)).cpu().numpy(); cont = (inter / (torch.minimum(area[:, None], area[None, :]) + 1e-6)).cpu().numpy(); keep = []
            ins = (inter / (area[:, None] + 1e-6)).cpu().numpy(); ar = area.cpu().numpy(); n = len(cand)
            groups = {j for j in range(n) if sum(1 for i in range(n) if i != j and ins[i, j] > 0.7 and ar[i] < 0.5 * ar[j] and cand[i][0] >= a.thr) >= 2}   # contains two confident objects: a group, prefer the parts
            for i in range(n):              # highest score first: drop groups, duplicates (mask IoU) and nested parts of an already kept object (containment)
                if i in groups: continue
                if a.nms < 1 and any(iou[i, j] > a.nms or cont[i, j] > 0.85 for j in keep): continue
                keep.append(i)
            for i in keep:
                p, k, q, ai, qi = cand[i]; cx, cy, w, h = g["boxes"][k, q].astype(np.float64)
                dets.append({"p": p, "k": k, "mask": mt[ai, qi], "box": np.array([(cx - w / 2) * OW, (cy - h / 2) * OH, (cx + w / 2) * OW, (cy + h / 2) * OH])})
    t3 = time.perf_counter(); n0 = tracker.next_id; shown = tracker.step(dets); stats["new_ids"].append(tracker.next_id - n0); ids = {t.id for t in shown}; stats["appear"].append(len(ids - prev_shown)); prev_shown = ids
    # ---- render: soft masks, anti-aliased contours and rounded boxes, name labels, lower third ----
    frame = torch.from_numpy(np.frombuffer(bd, np.uint8).reshape(OH, OW, 3).copy()).to(dev).float(); contours = []
    dense = len(shown) > 15; boxes = {}
    for t in shown:
        col = torch.tensor(PAL[t.cls % len(PAL)], device=dev).float(); soft = torch.sigmoid(Fn.interpolate(t.mask[None, None], size=(OH, OW), mode="bilinear", align_corners=False)[0, 0] * 1.5)
        al = (soft * 0.30)[..., None]; frame = frame * (1 - al) + col * al; b = (soft > 0.5).byte().cpu().numpy(); contours.append((b, PAL[t.cls % len(PAL)]))
        ys, xs = np.where(b)
        boxes[t.id] = (xs.min() - 3, ys.min() - 3, xs.max() + 3, ys.max() + 3) if len(xs) > 30 else tuple(t.box)     # box from the smoothed mask: consistent with what is drawn
    img = np.ascontiguousarray(frame.clamp(0, 255).byte().cpu().numpy())
    for bm_, c in contours:
        cs, _ = cv2.findContours(bm_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cs = [cc for cc in cs if cv2.contourArea(cc) > 150]
        if cs: cv2.polylines(img, cs, True, c, 2, cv2.LINE_AA)
    for t in shown: rounded_box(img, *boxes[t.id], PAL[t.cls % len(PAL)], t=2 if dense else 3, r=10 if dense else 14)
    pil = Image.fromarray(img); d = ImageDraw.Draw(pil, "RGBA"); fl = FONT_LAB_S if dense else FONT_LAB; lh = 26 if dense else 32
    for t in shown:
        c = PAL[t.cls % len(PAL)]; lab = prompts[t.cls]; tw = d.textlength(lab, font=fl); x0, y0 = boxes[t.id][0], boxes[t.id][1]; ly = y0 - lh - 4 if y0 - lh - 4 > 4 else y0 + 3; lx = max(4, min(x0, OW - tw - 20))
        d.rounded_rectangle([lx, ly, lx + tw + 14, ly + lh], radius=7, fill=c + (255,)); d.text((lx + 7, ly + 1), lab, font=fl, fill=(10, 10, 10, 255))
    tsec = fi / src_fps; xb = wipe_boundary(tsec); raw = np.frombuffer(bd, np.uint8).reshape(OH, OW, 3)
    if xb is not None:                                   # with/without wipe: overlay left of the boundary, raw video right of it
        ov = np.asarray(pil.convert("RGB")); comp = np.where(XG < xb, ov, raw); pil = Image.fromarray(np.ascontiguousarray(comp)); d = ImageDraw.Draw(pil, "RGBA")
        if 0 < xb < OW: d.rectangle([xb - 5, 0, xb + 5, OH], fill=(255, 196, 0, 70)); d.rectangle([xb - 2, 0, xb + 1, OH], fill=(255, 196, 0, 255))
    vm = np.mean(stats["vision_ms"][-30:]) if stats["vision_ms"] else (t1 - t0) * 1000; pa = ease((tsec - 0.5) / 1.0)
    if pa > 0:
        info1 = "prompts: " + ", ".join(prompts); info2 = f"SAM 3 ViT H, W8A8 in TensorRT, 1008 px; backbone {vm:.0f} ms per frame on an {a.gpu_name}"
        w1, w2 = d.textlength(info1, font=FONT_INFO_B), d.textlength(info2, font=FONT_INFO); pw = int(max(w1, w2)) + 40; A = lambda v: int(v * pa)
        d.rounded_rectangle([OW - pw - 40, 40, OW - 40, 118], radius=12, fill=(8, 10, 14, A(215))); d.text((OW - pw - 20, 50), info1, font=FONT_INFO_B, fill=(255, 196, 0, A(255))); d.text((OW - pw - 20, 84), info2, font=FONT_INFO, fill=(220, 220, 220, A(255)))
    if LT is not None:
        e = ease((tsec - a.lt_in) / 0.8) * (1 - ease((tsec - (a.lt_out - 1.0)) / 1.0))     # in after lt_in, gone by lt_out
        if e > 0:
            lt = LT if e >= 1 else LT.copy()
            if e < 1: lt.putalpha(lt.getchannel("A").point(lambda v: int(v * e)))
            slide = int((1 - ease((tsec - a.lt_in) / 0.8)) * 60) if a.lt_slide else 0
            pil.paste(lt, (40 - slide, OH - LT.height - 40), lt)
    out = np.asarray(pil.convert("RGB")); enc.stdin.write(out.tobytes())
    if a.snap and fi == int(a.snap): pil.convert("RGB").save(a.out + f".frame{fi}.png")
    stats["vision_ms"].append((t1 - t0) * 1000); stats["head_ms"].append((t2 - t1) * 1000); stats["mask_ms"].append((t3 - t2) * 1000); stats["tracks"].append(len(shown)); stats["frames"] += 1; fi += 1
    if fi % 50 == 0: print(f"{fi}/{n_frames}  backbone {np.mean(stats['vision_ms'][-50:]):.1f}  head {np.mean(stats['head_ms'][-50:]):.1f}  masks {np.mean(stats['mask_ms'][-50:]):.1f} ms  tracks {len(shown)}  {time.time() - t_start:.0f}s", flush=True)
enc.stdin.close(); enc.wait(); dec_m.kill(); dec_d.kill()
summary = {"video": a.video, "prompts": prompts, "frames": stats["frames"], "backbone_ms": float(np.mean(stats["vision_ms"])), "head_ms": float(np.mean(stats["head_ms"])), "mask_ms": float(np.mean(stats["mask_ms"])), "tracks_mean": float(np.mean(stats["tracks"])), "new_ids_per_frame_after_warmup": float(np.mean(stats["new_ids"][10:])) if len(stats["new_ids"]) > 10 else None, "tracks_appearing_per_frame": float(np.mean(stats["appear"][10:])) if len(stats["appear"]) > 10 else None, "src_fps": src_fps}
print(json.dumps(summary)); json.dump(summary, open(a.stats or a.out + ".json", "w"), indent=1)
