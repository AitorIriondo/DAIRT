"""query-side pruning (step v3) vs the full step (v2) on a real consecutive pair: init 12 objects from real masks at frame t, propagate at t+1 with both engines
(same bank), report per-object mask IoU / score deltas and the query fraction. usage: python check_qprune.py [radii...]"""
import sys, time, numpy as np; sys.path.insert(0, ".")
from cuda_alloc import DevBuf; import sam3_track_np as N
radii = [int(x) for x in sys.argv[1:]] or [2, 4, 6, 8, 12]
f0, f1, mk = np.load("exp/trk/vp_200.npz"), np.load("exp/trk/vp_201.npz"), np.load("exp/trk/vp_200_masks.npz")
tb = DevBuf(f0["trunk"].nbytes); pe = np.load("exp/trk/pe_mem.npy"); tp = np.load("exp/trk/tpos_enc.npy"); D = "exp/trk"
masks = mk["masks"].astype(np.float16); mb = DevBuf(masks.nbytes); mb.upload(masks); rows = [mb.ptr + i * 288 * 288 * 2 for i in range(len(masks))]; ids = list(range(1, len(masks) + 1))
def run(step_plan, qr, qg=0, prune_r=4, reps=5):
    reps = max(1, reps)
    tn = N.Sam3TrackerNP(f"{D}/trk_neck_fp16.plan", f"{D}/trk_init288_fp16.plan", step_plan, tb.ptr, f0["trunk"].shape, pe, tp, "plugins/lq_util_sm89.so", prune_r=prune_r, qprune_r=qr, qgrid_stride=qg)
    tb.upload(f0["trunk"].astype(np.float32)); tn.run_neck(); tn.refresh(0, ids, rows)
    for i, oid in enumerate(ids): tn.objs[oid].mask = masks[i].astype(np.float32)        # last mask (drives key and query selection), as in tracking
    tb.upload(f1["trunk"].astype(np.float32)); tn.run_neck(); out = tn.propagate(1, ids)
    t0 = time.perf_counter()
    for _ in range(reps): tn.propagate(1, ids)
    ms = (time.perf_counter() - t0) * 1000 / reps / len(ids)
    return {k: (v[0] > 0, v[1], v[2]) for k, v in out.items()}, ms, tn.q_used / max(1, tn.q_full), tn.keys_used / max(1, tn.keys_full)
ref, ms2, _, kf = run(f"{D}/trk_step_v2_fp16.plan", 0); print(f"v2 (all queries, keys {kf:.2f}): {ms2:.2f} ms/object")
def iou(a, b): return float((a & b).sum() / max(1, (a | b).sum()))
full, ms3, qf, _ = run(f"{D}/trk_step_v3_fp16.plan", 0); print(f"v3 all queries: {ms3:.2f} ms/object, IoU vs v2 min {min(iou(full[k][0], ref[k][0]) for k in ref):.4f} (exactness check)")
for r in radii:
    for qg in (0, 4):
        o, ms, qf, _ = run(f"{D}/trk_step_v3_fp16.plan", r, qg); ious = [iou(o[k][0], ref[k][0]) for k in ref]; ds = [abs(o[k][1] - ref[k][1]) for k in ref]
        print(f"v3 r={r:2d} grid={qg}: queries {qf:.2f}  {ms:.2f} ms/object  IoU vs v2 mean {np.mean(ious):.4f} min {min(ious):.4f}  |dscore| max {max(ds):.2f}")
