"""compare a v2 step plan against the CPU reference of the pruned run saved by export_tracker.py --v2 --check; also time full-key steps"""
import sys, os, time, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam3_track import DynRunner; from trt_util import load_engine
plan = sys.argv[1]; r = np.load("exp/trk/ref_check_v2.npz"); step = DynRunner(load_engine(plan))
def rel(x, y): return float(np.linalg.norm(x.astype(np.float64) - y) / (np.linalg.norm(y) + 1e-9))
feeds = {"feat72": r["feat72"], "hr0": r["hr0"], "hr1": r["hr1"], "memtok": r["memtok"], "mempos": r["mempos"], "keyidx": r["keyidx"], "nvalid": r["nvalid"], "ptrs": r["ptrs"], "ptr_pos": r["ptr_pos"]}
o = step(feeds); A = o["masks"] > 0; Bm = r["low"] > 0
print("pruned check: masks rel %.2e (binary IoU %.4f) obj_ptr %.2e obj_score %.2e iou %.2e maskmem %.2e" % (rel(o["masks"], r["low"]), float((A & Bm).sum() / max(1, (A | Bm).sum())), rel(o["obj_ptr"], r["optr"]), rel(o["obj_score"], r["oscore"]), rel(o["iou"], r["ious"]), rel(o["maskmem"], r["memn"])))
N = 5184
for B, M, P in [(1, 7, 6), (4, 7, 6), (1, 3, 3), (1, 1, 1)]:
    Kn = M * N; f = {"feat72": r["feat72"], "hr0": r["hr0"], "hr1": r["hr1"], "memtok": np.repeat(np.tile(r["memtok"][:1, :N], (1, M, 1)), B, 0), "mempos": np.repeat(np.tile(r["mempos"][:1, :N], (1, M, 1)), B, 0),
         "keyidx": np.repeat(np.tile(np.arange(N)[None], (1, M)), B, 0).astype(np.int64), "nvalid": np.full((B,), Kn, np.int32), "ptrs": np.repeat(r["ptrs"][:1, :P], B, 0), "ptr_pos": r["ptr_pos"][:P]}
    step(f); t0 = time.perf_counter(); n = 10
    for _ in range(n): step(f)
    print(f"{os.path.basename(plan)} B={B} M={M} P={P}: {(time.perf_counter() - t0) / n * 1000:.1f} ms per call incl. host copies")
