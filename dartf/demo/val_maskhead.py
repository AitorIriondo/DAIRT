"""compare two mask-head plans on validation samples: mask IoU (logits > 0) of every query and logit error. usage: python val_maskhead.py ref.plan cand.plan val.npz"""
import sys, os, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam3_track import DynRunner; from trt_util import load_engine
ref = DynRunner(load_engine(sys.argv[1])); cand = DynRunner(load_engine(sys.argv[2])); z = np.load(sys.argv[3]); n = int(z["n"]); ious = []; rels = []; flips = []
for i in range(n):
    feeds = {k: z[f"{k}_{i}"] for k in ("fpn_0", "fpn_1", "enc", "text_feats", "text_mask", "hs_sel")}
    a = ref(feeds)["masks"].astype(np.float32); b = cand(feeds)["masks"].astype(np.float32); A = a > 0; B = b > 0
    inter = (A & B).sum((2, 3)); uni = (A | B).sum((2, 3)); iou = np.where(uni > 0, inter / np.maximum(uni, 1), 1.0); ious.append(iou[uni > 200]); rels.append(np.linalg.norm(a - b) / np.linalg.norm(a)); flips.append((A != B).mean())
ious = np.concatenate(ious); print("masks compared %d | IoU mean %.4f min %.4f p5 %.4f | logit rel %.4f | pixel flip rate %.5f" % (len(ious), ious.mean(), ious.min(), np.percentile(ious, 5), np.mean(rels), np.mean(flips)))
