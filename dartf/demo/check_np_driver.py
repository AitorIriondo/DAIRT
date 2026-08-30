"""compare the torch-free SAM 3 driver with the torch driver on identical inputs (real trunk features, rectangle masks)"""
import sys, os, numpy as np, torch; sys.path.insert(0, ".")
from cuda_alloc import DevBuf; import sam3_track as T; import sam3_track_np as N
trunk = np.load("exp/trk/frame100.npz")["trunk"].astype(np.float32); tb = DevBuf(trunk.nbytes); tb.upload(trunk)
class VR: bufs = {"permute_4": (tb, trunk.shape, np.float32)}
pe = np.load("exp/trk/pe_mem.npy"); tp = np.load("exp/trk/tpos_enc.npy"); D = "exp/trk"
tt = T.Sam3Tracker(f"{D}/trk_neck_fp16.plan", f"{D}/trk_init_fp16.plan", f"{D}/trk_step_v2_fp16.plan", VR, "permute_4", pe_mem=pe, tpos_enc=tp, prune_r=0)
tn = N.Sam3TrackerNP(f"{D}/trk_neck_fp16.plan", f"{D}/trk_init288_fp16.plan", f"{D}/trk_step_v2_fp16.plan", tb.ptr, trunk.shape, pe, tp, "plugins/lq_util_sm89.so", prune_r=0)
tt.run_neck(); tn.run_neck()
def rel(x, y): x = np.asarray(x, np.float64); y = np.asarray(y, np.float64); return float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-9))
print("neck feat72 rel", rel(tn.neck.bufs["feat72"][0].download((1, 256, 72, 72), np.float32), tt.neck.bufs["feat72"].cpu().numpy()))
# init from the same mask: logits at 288 (rectangle), torch path upsamples to 1008 > 0; np path gathers the fp16 288 logits
m288 = np.full((2, 288, 288), -10.0, np.float32); m288[0, 90:170, 100:190] = 10; m288[1, 150:230, 250:370] = 10
mask_dev = DevBuf(m288.astype(np.float16).nbytes); mask_dev.upload(m288.astype(np.float16)); rows = [mask_dev.ptr, mask_dev.ptr + 288 * 288 * 2]
m1008 = torch.nn.functional.interpolate(torch.from_numpy(m288).cuda()[:, None], size=(1008, 1008), mode="bilinear", align_corners=False)[:, 0] > 0
tt.refresh(0, [1, 2], m1008); tn.refresh(0, [1, 2], rows)
for oid in (1, 2):
    mt, pt = tt.objs[oid].conds[0]; mn, pn = tn.objs[oid].conds[0]; mn_np = mn.download((5184, 64), np.float16).astype(np.float32).T.reshape(64, 72, 72)
    print(f"obj {oid}: init maskmem rel {rel(mn_np, mt.float().cpu().numpy()):.2e}  obj_ptr rel {rel(pn, pt.cpu().numpy()):.2e}")
# one propagation step at frame 1 for both objects (cond only), masks as numpy from both
tt.objs[1].mask = torch.from_numpy(m288[0]).cuda(); tt.objs[2].mask = torch.from_numpy(m288[1]).cuda(); tn.objs[1].mask = m288[0]; tn.objs[2].mask = m288[1]
ot = tt.propagate(1, [1, 2]); on = tn.propagate(1, [1, 2])
for oid in (1, 2):
    a = ot[oid][0].cpu().numpy(); b = on[oid][0]; print(f"obj {oid}: step masks rel {rel(b, a):.2e} (IoU {((a > 0) & (b > 0)).sum() / max(1, ((a > 0) | (b > 0)).sum()):.4f}) score {ot[oid][1]:.3f} vs {on[oid][1]:.3f} iou {ot[oid][2]:.3f} vs {on[oid][2]:.3f}")
    mt = tt.objs[oid].hist[1][0].float().cpu().numpy(); mn = tn.objs[oid].hist[1][0].download((5184, 64), np.float16).astype(np.float32).T.reshape(64, 72, 72); print(f"        step maskmem rel {rel(mn, mt):.2e}")
# ---- localize: raw engine output before the transpose, and the gathered mask input ----
raw_n = tn.init.bufs["maskmem"][0].download((2, 64, 72, 72), tn.init.bufs["maskmem"][2]).astype(np.float32); raw_t = tt.init.bufs["maskmem"][:2].float().cpu().numpy()
print("raw init maskmem (engine output) rel", rel(raw_n, raw_t)); mg = tn.mask_gather.download((2, 288, 288), np.float16).astype(np.float32); print("gathered mask input rel", rel(mg, m288))
tr = tn.mem_t.download((2, 5184, 64), np.float16).astype(np.float32).transpose(0, 2, 1).reshape(2, 64, 72, 72); print("transposed buffer rel", rel(tr, raw_n))
c1 = tn.objs[1].conds[0][0].download((5184, 64), np.float16).astype(np.float32).T.reshape(64, 72, 72); print("stored cond copy rel vs raw", rel(c1, raw_n[0]), "vs transposed", rel(c1, tr[0]))
