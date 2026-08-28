"""Compare baseline ground_cK engine vs (pre @B=1 + post @K) on the goldens; time both.  usage: ground_split_cmp.py base_cK.plan pre.plan post.plan"""
import sys, os, time, numpy as np
from trt_util import load_engine, Runner
H = os.path.expanduser("~/sam3_orin_agx64_jp72_handoff")
base, pre, post = [Runner(load_engine(p)) for p in sys.argv[1:4]]
K = base.bufs[base.inputs[0]][1][0]
gi = lambda n, dt, sh: np.fromfile(f"{H}/goldens/inputs/ground_c16__{n}.bin", dtype=dt).reshape(sh)
feat, pos, tf, tm = gi("img_feat", np.float16, (16, 256, 72, 72)), gi("img_pos", np.float16, (16, 256, 72, 72)), gi("text_feats", np.float16, (32, 16, 256)), gi("text_mask", np.float32, (16, 32))
fb = {"img_feat": feat[:K], "img_pos": pos[:K], "text_feats": np.ascontiguousarray(tf[:, :K]), "text_mask": tm[:K]}
fpre = {"img_feat": feat[:1], "img_pos": pos[:1]}
def timeit(f, n=20):
    for _ in range(3): f()
    t = time.time()
    for _ in range(n): f()
    return (time.time() - t) / n * 1000
ob = base(fb); tb = timeit(lambda: base(fb))
def split():
    o = pre(fpre); return post({"x1_in": o["/encoder/layers.0/Add_1_output_0"], "pos_in": o["/encoder/Concat_3_output_0"], "text_feats": fb["text_feats"], "text_mask": fb["text_mask"]})
os_ = split(); tp = timeit(lambda: pre(fpre)); ts = timeit(split)
print(f"K={K}: baseline {tb:.2f} ms | pre {tp:.2f} ms + post = {ts:.2f} ms total  (per class: {tb/K:.2f} -> {ts/K:.2f})")
for k in ob:
    x, y = ob[k].astype(np.float32).ravel(), os_[k].astype(np.float32).ravel()
    print(f"  {k:10s} rel_l2={np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-9):.3e} max_abs={np.abs(x - y).max():.3e}")
