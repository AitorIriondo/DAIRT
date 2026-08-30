"""compare the three tracker TensorRT plans against the CPU reference saved by export_tracker.py --check
usage: python check_trk_trt.py <dir with trk_*_fp16.plan and ref_check.npz>"""
import sys, os, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtime"))
from sam3_track import DynRunner; from trt_util import load_engine
D = sys.argv[1] if len(sys.argv) > 1 else "exp/trk"; r = np.load(f"{D}/ref_check.npz")
def rel(x, y): return float(np.linalg.norm(x.astype(np.float64) - y) / (np.linalg.norm(y) + 1e-9))
neck = DynRunner(load_engine(f"{D}/trk_neck_fp16.plan")); o = neck({"trunk": r["trunk"]}); print("neck: feat72 %.2e hr0 %.2e hr1 %.2e" % (rel(o["feat72"], r["feat72"]), rel(o["hr0"], r["hr0"]), rel(o["hr1"], r["hr1"])))
init = DynRunner(load_engine(f"{D}/trk_init_fp16.plan")); oi = init({"feat72": r["feat72"], "hr0": r["hr0"], "hr1": r["hr1"], "mask_in": r["masks0"]}); print("init: obj_ptr %.2e maskmem %.2e" % (rel(oi["obj_ptr"], r["optr0"]), rel(oi["maskmem"], r["mem0"])))
step = DynRunner(load_engine(f"{D}/trk_step_fp16.plan")); os_ = step({"feat72": r["feat72"], "hr0": r["hr0"], "hr1": r["hr1"], "mem": r["mem"], "tpos_idx": r["tpos_idx"], "ptrs": r["ptrs"], "ptr_pos": r["ptr_pos"]})
print("step: masks %.2e (binary IoU %.4f) obj_ptr %.2e obj_score %.2e (%s vs %s) iou %.2e maskmem %.2e" % (rel(os_["masks"], r["low"]), float(((os_["masks"] > 0) & (r["low"] > 0)).sum() / max(1, ((os_["masks"] > 0) | (r["low"] > 0)).sum())), rel(os_["obj_ptr"], r["optr"]), rel(os_["obj_score"], r["oscore"]), os_["obj_score"].ravel().round(3).tolist(), r["oscore"].ravel().round(3).tolist(), rel(os_["iou"], r["ious"]), rel(os_["maskmem"], r["memn"])))
import time
for B, M, P in [(1, 7, 6), (4, 7, 6), (16, 7, 6), (1, 1, 1)]:
    feeds = {"feat72": r["feat72"], "hr0": r["hr0"], "hr1": r["hr1"], "mem": np.repeat(r["mem"][:1, :M], B, 0), "tpos_idx": r["tpos_idx"][:M], "ptrs": np.repeat(r["ptrs"][:1, :P], B, 0), "ptr_pos": r["ptr_pos"][:P]}
    step(feeds); t0 = time.perf_counter(); n = 10
    for _ in range(n): step(feeds)
    print(f"step B={B} M={M} P={P}: {(time.perf_counter() - t0) / n * 1000:.1f} ms per call (incl. host copies)")
t0 = time.perf_counter()
for _ in range(10): neck({"trunk": r["trunk"]})
print("neck: %.1f ms" % ((time.perf_counter() - t0) / 10 * 1000)); t0 = time.perf_counter()
for _ in range(10): init({"feat72": r["feat72"], "hr0": r["hr0"], "hr1": r["hr1"], "mask_in": r["masks0"]})
print("init B=2: %.1f ms" % ((time.perf_counter() - t0) / 10 * 1000))
