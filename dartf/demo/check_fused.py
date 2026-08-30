"""fused ground+mask engine vs the two-engine path on real frames: query sets, mask agreement, per-frame head+mask time (host copies included).
usage: python check_fused.py <vision.plan> <ground_cK.plan> <maskhead.plan> <groundmask_cK.plan> <img_pos_cK.npy> <text_cache.npz> <video | %06d.jpg pattern> [n_frames]"""
import sys, os, time, subprocess, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam3_track_np import DynRunner, load_engine
from cuda_alloc import DevBuf
vis_p, gr_p, mk_p, fu_p, pos_p, txt_p, video = sys.argv[1:8]; n_frames = int(sys.argv[8]) if len(sys.argv) > 8 else 20; QSEL = 32
vision = DynRunner(load_engine(vis_p)); ground = DynRunner(load_engine(gr_p)); mask = DynRunner(load_engine(mk_p)); fused = DynRunner(load_engine(fu_p)); img_pos = np.load(pos_p)
K = int(fused.engine.get_tensor_shape("text_feats")[1]); z = np.load(txt_p); tf = np.ascontiguousarray(z["tf"][:, :K, :]).astype(np.float16); tm = np.ascontiguousarray(z["tm"][:K]).astype(np.float32)
names = {n: n for n in vision.outputs}; f0 = [n for n in vision.outputs if "fpn_0" in n or n.endswith("fpn_0")][0]; f1 = [n for n in vision.outputs if "fpn_1" in n][0]; f2 = [n for n in vision.outputs if "fpn_2" in n][0]
for r in (mask, fused):
    r.bind("fpn_0", vision.bufs[f0][0].ptr, vision.bufs[f0][1]); r.bind("fpn_1", vision.bufs[f1][0].ptr, vision.bufs[f1][1])
fused.bind("fpn_2", vision.bufs[f2][0].ptr, vision.bufs[f2][1])
sig = lambda x: 1 / (1 + np.exp(-x.astype(np.float64)))
if "%" in video:
    from PIL import Image
    def frames():
        i = 1
        while True:
            p = video % i
            if not os.path.exists(p): return
            im = Image.open(p).convert("RGB").resize((1008, 1008), Image.BILINEAR); yield np.asarray(im); i += 1
else:
    def frames():
        dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-vf", "scale=1008:1008:flags=bilinear", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8)
        while True:
            b = dec.stdout.read(1008 * 1008 * 3)
            if len(b) < 1008 * 1008 * 3: return
            yield np.frombuffer(b, np.uint8).reshape(1008, 1008, 3)
t2e, tfu, same_sel, ious, rels = [], [], [], [], []
for fi, im in enumerate(frames()):
    if fi >= n_frames: break
    x = (im.transpose(2, 0, 1).astype(np.float32) * (1 / 127.5) - 1.0)[None]; vision({"images": x}, want=[]); DevBuf.sync()
    # two engines: fpn_2 download, repeat, upload; enc download, upload of the active subset (all K here)
    t0 = time.perf_counter(); v = vision.bufs[f2][0].download(tuple(vision.bufs[f2][1]), vision.bufs[f2][2])
    g = ground({"img_feat": np.repeat(v.astype(np.float16), K, axis=0), "img_pos": img_pos, "text_feats": tf, "text_mask": tm}); probs = sig(g["scores"][:, :, 0]) * sig(g["presence"][:, 0])[:, None]
    sel = np.stack([np.argsort(-probs[k])[:QSEL] for k in range(K)]); hs_sel = np.stack([g["hs"][k, sel[k]] for k in range(K)]).astype(np.float16)
    m2 = mask({"enc": np.ascontiguousarray(g["enc"]), "text_feats": tf, "text_mask": tm, "hs_sel": hs_sel}, want=["masks"])["masks"]; DevBuf.sync(); t1 = time.perf_counter()
    o = fused({"text_feats": tf, "text_mask": tm}); DevBuf.sync(); t2 = time.perf_counter()
    if fi >= 2: t2e.append((t1 - t0) * 1000); tfu.append((t2 - t1) * 1000)
    for k in range(K):
        if not (probs[k] > 0.25).any(): continue
        s2, sf = list(sel[k]), list(o["sel"][k]); common = [q for q in s2 if q in sf]; same_sel.append(len(common) / QSEL)
        for q in common[:8]:
            a_, b_ = m2[k, s2.index(q)].astype(np.float32), o["masks"][k, sf.index(q)].astype(np.float32); A, B = a_ > 0, b_ > 0
            ious.append((A & B).sum() / max(1, (A | B).sum())); rels.append(np.linalg.norm(a_ - b_) / (np.linalg.norm(a_) + 1e-6))
    if fi == 0: print("scores rel %.2e boxes rel %.2e hs rel %.2e" % (np.linalg.norm(o["scores"].astype(np.float32) - g["scores"].astype(np.float32)) / np.linalg.norm(g["scores"].astype(np.float32)), np.linalg.norm(o["boxes"].astype(np.float32) - g["boxes"].astype(np.float32)) / np.linalg.norm(g["boxes"].astype(np.float32)), np.linalg.norm(o["hs"].astype(np.float32) - g["hs"].astype(np.float32)) / np.linalg.norm(g["hs"].astype(np.float32))))
print(f"K={K} frames {len(t2e) + 2}: two engines {np.mean(t2e):.2f} ms/frame (incl. fpn_2 and enc round trips), fused {np.mean(tfu):.2f} ms/frame | query sets identical {np.mean(same_sel) * 100:.1f} % | masks: mean IoU {np.mean(ious):.4f} min {np.min(ious):.4f}, rel {np.mean(rels):.2e}")
