"""Host-side driver of the SAM 3 native tracker on TensorRT (trk_neck / trk_init / trk_step engines), device-resident memory bank.
Memories, pointers and masks live in CUDA tensors (torch); the per-step key gather (with mask-guided pruning) runs on the GPU and the engines are
bound directly to those tensors, so nothing but the tiny valid-count array and the per-object scalars crosses the PCIe bus.
Memory rules mirror Sam3TrackerBase with memory selection on: conditioning frames (from detector masks) plus recent valid non-conditioning frames
(slots NM-1 for cond, NM-t_pos-1 for the rest); pointers = conditioning frames by distance + the most recent valid frames by recency.
Objects are batched by identical (M, P). Association with detections and object birth/death are done in run_video."""
import numpy as np, torch, torch.nn.functional as Fn, time
import tensorrt as trt
from trt_util import load_engine, NP_DTYPE
NP_DTYPE = dict(NP_DTYPE); NP_DTYPE[trt.int64] = np.int64
TORCH_DTYPE = {np.float32: torch.float32, np.float16: torch.float16, np.int8: torch.int8, np.int32: torch.int32, np.int64: torch.int64, np.bool_: torch.bool}
dev = torch.device("cuda")
class DynRunner:
    """execution context with one optimization profile. Inputs/outputs are CUDA torch tensors sized at the profile max; feeds may be numpy
    (uploaded) or CUDA tensors (bound in place, zero copy); outputs are returned as views of the output tensors (no download unless numpy=True)."""
    def __init__(self, engine):
        self.engine = engine; self.ctx = engine.create_execution_context(); self.names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.inputs = [n for n in self.names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]; self.outputs = [n for n in self.names if n not in self.inputs]; self.bufs = {}
        for n in self.inputs:
            shape = tuple(engine.get_tensor_profile_shape(n, 0)[2]) if -1 in tuple(engine.get_tensor_shape(n)) else tuple(engine.get_tensor_shape(n))
            dt = NP_DTYPE[engine.get_tensor_dtype(n)]; self.bufs[n] = torch.empty(shape, dtype=TORCH_DTYPE[dt], device=dev); self.ctx.set_input_shape(n, shape)
        for n in self.outputs: shape = tuple(self.ctx.get_tensor_shape(n)); dt = NP_DTYPE[engine.get_tensor_dtype(n)]; self.bufs[n] = torch.empty(shape, dtype=TORCH_DTYPE[dt], device=dev)
        for n in self.names: self.ctx.set_tensor_address(n, self.bufs[n].data_ptr())
        self.external = set(); self._keep = []
    def bind(self, name, ptr, shape): self.external.add(name); self.ctx.set_tensor_address(name, ptr); self.ctx.set_input_shape(name, tuple(shape))
    def __call__(self, feeds, want=None, numpy=None):
        self._keep = []; any_torch = any(isinstance(v, torch.Tensor) for v in feeds.values())
        if numpy is None: numpy = not any_torch                                   # numpy in -> numpy out (old scripts); tensors in -> tensors out
        for n in self.inputs:
            if n in self.external: continue
            x = feeds[n]
            if isinstance(x, torch.Tensor):
                x = x.contiguous().to(self.bufs[n].dtype); self.ctx.set_input_shape(n, tuple(x.shape)); self.ctx.set_tensor_address(n, x.data_ptr()); self._keep.append(x)   # bound in place
            else:
                x = torch.from_numpy(np.ascontiguousarray(x)).to(self.bufs[n].dtype); self.ctx.set_input_shape(n, tuple(x.shape)); self.bufs[n][tuple(slice(0, d) for d in x.shape)].copy_(x); self.ctx.set_tensor_address(n, self.bufs[n].data_ptr())
        torch.cuda.synchronize(); assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
        out = {}
        for n in (want or self.outputs):
            shp = tuple(self.ctx.get_tensor_shape(n)); v = self.bufs[n][tuple(slice(0, d) for d in shp)]; out[n] = v.cpu().numpy() if numpy else v
        return out
class Obj:
    """SAM 3 memory state of one tracked object: conditioning memories (from detector masks, several frames) and non-conditioning memories (from steps); all CUDA tensors"""
    __slots__ = ("id", "conds", "hist", "mask", "score", "iou", "last_init", "last_step")
    def __init__(self, oid): self.id = oid; self.conds = {}; self.hist = {}; self.mask = None; self.score = 10.0; self.iou = 1.0; self.last_init = -999; self.last_step = -999
class Sam3Tracker:
    NM, MAXPTR, MF_THR, MAX_COND = 7, 16, 0.01, 4
    def __init__(self, neck_plan, init_plan, step_plan, vision_runner, trunk_name, max_hist=24, pe_mem=None, tpos_enc=None, prune_r=4, grid_stride=2):
        self.neck = DynRunner(load_engine(neck_plan)); self.init = DynRunner(load_engine(init_plan)); self.step = DynRunner(load_engine(step_plan)); self.max_hist = max_hist
        self.neck.bind("trunk", vision_runner.bufs[trunk_name][0].ptr, vision_runner.bufs[trunk_name][1])          # vision -> neck on device
        for eng in (self.init, self.step):                                                                          # neck -> init/step on device (inputs the exporter pruned are skipped)
            for n in ("feat72", "hr0", "hr1"):
                if n in eng.inputs: eng.bind(n, self.neck.bufs[n].data_ptr(), tuple(self.neck.engine.get_tensor_shape(n)))
        self.objs = {}; self.dropped_ids = set(); self.t_ms = {"neck": 0.0, "init": 0.0, "step": 0.0}; self.n_init = 0; self.n_step = 0; self.n_step_calls = 0; self.keys_used = 0; self.keys_full = 0
        self.v2 = "memtok" in self.step.inputs; self.p_max = int(self.step.engine.get_tensor_profile_shape("ptrs", 0)[2][1]); self.adaptive = False; self.prune_r = prune_r
        self.m_max = 7 if self.v2 else int(self.step.engine.get_tensor_profile_shape("mem", 0)[2][1])
        if self.v2:
            self.kn_max = int(self.step.engine.get_tensor_profile_shape("memtok", 0)[2][1])
            self.pe_flat = torch.from_numpy(np.asarray(pe_mem).reshape(64, 5184).T.copy()).to(dev).float(); self.tpos_flat = torch.from_numpy(np.asarray(tpos_enc).reshape(-1, 64).copy()).to(dev).float()
            yy, xx = torch.meshgrid(torch.arange(72), torch.arange(72), indexing="ij"); self.bg_keep = ((yy % grid_stride == 0) & (xx % grid_stride == 0)).reshape(-1).to(dev)
    def run_neck(self):
        t0 = time.perf_counter(); assert self.neck.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize(); self.t_ms["neck"] += (time.perf_counter() - t0) * 1000
    @staticmethod
    def mem_score(obj_score, iou): return (max(0.0, 1 / (1 + np.exp(-obj_score)) * 2 - 1)) * iou
    def obj(self, oid): return self.objs.setdefault(oid, Obj(oid))
    def refresh(self, t, ids, masks1008):
        """conditioning memories from detector masks (trk_init, batched): masks1008 CUDA tensor [B,1008,1008] (bool/float)"""
        if not len(ids): return
        bmax = int(self.init.engine.get_tensor_profile_shape("mask_in", 0)[2][0])
        for c in range(0, len(ids), bmax):
            t0 = time.perf_counter(); o = self.init({"mask_in": masks1008[c:c + bmax, None].float()}); self.t_ms["init"] += (time.perf_counter() - t0) * 1000; self.n_init += len(ids[c:c + bmax])
            mm = o["maskmem"].clone(); op = o["obj_ptr"].float().clone()                                          # own copies: the engine buffers are reused next call
            for i, oid in enumerate(ids[c:c + bmax]):
                ob = self.obj(oid); ob.conds[t] = (mm[i], op[i]); ob.last_init = t
                for f in sorted(ob.conds)[:-self.MAX_COND]: del ob.conds[f]
    def _bank(self, ob, t):
        conds = sorted(ob.conds, key=lambda f: abs(t - f))[:self.MAX_COND]
        valid = [f for f in sorted(ob.hist) if f < t and ob.hist[f][2] > self.MF_THR][-(self.MAXPTR - 1):]
        m_max = self.m_max if not (self.adaptive and ob.iou >= 0.85 and ob.score > 4.0) else min(self.m_max, 3)
        mems = [ob.conds[f][0] for f in conds]; tpos = [self.NM - 1] * len(conds); slots = valid[-min(self.NM - 1, m_max - len(conds)):] if m_max > len(conds) else []
        for i, f in enumerate(slots): t_pos = self.NM - len(slots) + i; mems.append(ob.hist[f][0]); tpos.append(self.NM - t_pos - 1)
        ptrs = [ob.conds[f][1] for f in conds]; ppos = [float(t - f) for f in conds]
        for d in range(1, min(self.MAXPTR, len(valid), self.p_max - len(conds) + 1)): ptrs.append(ob.hist[valid[-d]][1]); ppos.append(float(d))
        return mems, tpos, torch.stack(ptrs), np.array(ppos, np.float32) / (self.MAXPTR - 1)
    def _keep(self, ob):
        keep = torch.ones(5184, dtype=torch.bool, device=dev)
        if self.prune_r > 0 and ob.mask is not None:
            m72 = (ob.mask > 0).reshape(1, 1, 288, 288)[:, :, ::4, ::4].float(); reg = Fn.max_pool2d(m72, 2 * self.prune_r + 1, 1, self.prune_r)[0, 0] > 0; keep = reg.reshape(-1) | self.bg_keep
        return keep
    def propagate(self, t, ids):
        """trk_step for the objects `ids`, batched by identical (M, P); returns {id: (mask288 CUDA tensor, obj_score, iou)}"""
        groups = {}; out = {}
        for oid in ids:
            ob = self.objs.get(oid)
            if ob is None or not ob.conds: continue
            mems, tpos, ptrs, ppos = self._bank(ob, t); groups.setdefault((len(tpos), len(ppos)), []).append((ob, mems, tpos, ptrs, ppos))
        bmax = int(self.step.engine.get_tensor_profile_shape("memtok" if self.v2 else "mem", 0)[2][0])
        for (M, P), items in groups.items():
            for c in range(0, len(items), bmax):
                chunk = items[c:c + bmax]; t0 = time.perf_counter(); Bc = len(chunk)
                if self.v2:
                    sels = [torch.nonzero(self._keep(ob))[:, 0] for (ob, *_r) in chunk]; nv = np.array([M * len(s) for s in sels], np.int32); Kn = int(nv.max())
                    memtok = torch.zeros(Bc, Kn, 64, dtype=torch.float16, device=dev); mempos = torch.zeros(Bc, Kn, 64, dtype=torch.float16, device=dev); keyidx = torch.zeros(Bc, Kn, dtype=torch.int64, device=dev)
                    for i, (ob, mems, tpos, _p, _pp) in enumerate(chunk):
                        sel = sels[i]; n = len(sel); tk = torch.stack([m.reshape(64, 5184).T[sel] for m in mems]).reshape(M * n, 64)          # [M*n, 64] gathered on the GPU
                        ps = torch.stack([self.pe_flat[sel] + self.tpos_flat[tp] for tp in tpos]).reshape(M * n, 64)
                        memtok[i, :M * n] = tk.half(); mempos[i, :M * n] = ps.half(); keyidx[i, :M * n] = sel.repeat(M)
                    self.keys_used += int(nv.sum()); self.keys_full += int(M * 5184 * Bc)
                    o = self.step({"memtok": memtok, "mempos": mempos, "keyidx": keyidx, "nvalid": nv, "ptrs": torch.stack([it[3] for it in chunk]), "ptr_pos": chunk[0][4]})
                else:
                    o = self.step({"mem": torch.stack([torch.stack(it[1]) for it in chunk]), "tpos_idx": np.array(chunk[0][2], np.int64), "ptrs": torch.stack([it[3] for it in chunk]), "ptr_pos": chunk[0][4]})
                masks = o["masks"].float().clone(); mm = o["maskmem"].clone(); op = o["obj_ptr"].float().clone(); sc = o["obj_score"].float().cpu().numpy(); io = o["iou"].float().cpu().numpy()
                self.t_ms["step"] += (time.perf_counter() - t0) * 1000; self.n_step_calls += 1; self.n_step += Bc
                for i, (ob, *_r) in enumerate(chunk):
                    ob.mask = masks[i, 0]; ob.score = float(sc[i, 0]); ob.iou = float(io[i, 0]); ob.last_step = t
                    ob.hist[t] = (mm[i], op[i], self.mem_score(ob.score, ob.iou))
                    for f in sorted(ob.hist)[:-self.max_hist]: del ob.hist[f]
                    out[ob.id] = (ob.mask, ob.score, ob.iou)
        return out
    def prune(self, live_ids):
        for oid in [o for o in self.objs if o not in live_ids]: del self.objs[oid]
        self.dropped_ids &= set(live_ids)
    def drop(self, oid): self.dropped_ids.add(oid)
    def dropped(self, oid): return oid in self.dropped_ids
