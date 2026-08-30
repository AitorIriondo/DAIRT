"""Torch-free driver of the SAM 3 native tracker on TensorRT (trk_neck / trk_init288 / trk_step_v2), device-resident memory bank.
Memories live in raw CUDA buffers; the per-step key gather runs in a small CUDA kernel (lq_util.so) driven by a row-pointer table built on the
CPU; the object masks that prompt the initialization are gathered straight from the segmentation head's output buffer. Only bookkeeping arrays
(a few KB per step) and the 288x288 mask logits of propagated objects cross the bus. Same memory rules and batching as sam3_track.py."""
import os, numpy as np, time, ctypes, os
import tensorrt as trt
from trt_util import load_engine, NP_DTYPE
from cuda_alloc import DevBuf, Stream, CudaGraph, memcpy_d2d, _lib as _cudart
NP_DTYPE = dict(NP_DTYPE); NP_DTYPE[trt.int64] = np.int64
class DynRunner:
    """TensorRT execution context with buffers at the profile max; feeds are numpy (uploaded) or ("ptr", address, shape) tuples (bound in place)"""
    def __init__(self, engine):
        self.engine = engine; self.ctx = engine.create_execution_context(); self.names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.inputs = [n for n in self.names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]; self.outputs = [n for n in self.names if n not in self.inputs]; self.bufs = {}
        for n in self.inputs:
            shape = tuple(engine.get_tensor_profile_shape(n, 0)[2]) if -1 in tuple(engine.get_tensor_shape(n)) else tuple(engine.get_tensor_shape(n))
            dt = NP_DTYPE[engine.get_tensor_dtype(n)]; self.bufs[n] = (DevBuf(int(np.prod(shape)) * np.dtype(dt).itemsize), shape, dt); self.ctx.set_input_shape(n, shape)
        for n in self.outputs: shape = tuple(self.ctx.get_tensor_shape(n)); dt = NP_DTYPE[engine.get_tensor_dtype(n)]; self.bufs[n] = (DevBuf(int(np.prod(shape)) * np.dtype(dt).itemsize), shape, dt)
        for n in self.names: self.ctx.set_tensor_address(n, self.bufs[n][0].ptr)
        self.external = set(); self.graphs = {} if os.environ.get("LQ_TRK_GRAPH", "1") == "1" else None; self.stream = Stream(blocking=True)   # one CUDA graph per (shapes, addresses) key, captured on first use
    def _run(self):
        if self.graphs is None: assert self.ctx.execute_v2([self.ctx.get_tensor_address(n) for n in self.names]); DevBuf.sync(); return
        key = tuple((tuple(self.ctx.get_tensor_shape(n)), self.ctx.get_tensor_address(n)) for n in self.names); g = self.graphs.get(key)
        if g is None:
            assert self.ctx.execute_async_v3(self.stream.ptr); self.stream.sync()      # TensorRT needs one run at new shapes before a capture
            try: g = CudaGraph(self.stream, lambda: self.ctx.execute_async_v3(self.stream.ptr))
            except Exception as e: print("[sam3_np] graph capture failed, executing directly:", e); g = False
            self.graphs[key] = g
        if g: g.launch(self.stream); self.stream.sync()
        else: assert self.ctx.execute_v2([self.ctx.get_tensor_address(n) for n in self.names]); DevBuf.sync()
    def bind(self, name, ptr, shape): self.external.add(name); self.ctx.set_tensor_address(name, ptr); self.ctx.set_input_shape(name, tuple(shape))
    def __call__(self, feeds, want=None, download=True):
        for n in self.inputs:
            if n in self.external: continue
            x = feeds[n]
            if isinstance(x, tuple) and x[0] == "ptr": self.ctx.set_input_shape(n, tuple(x[2])); self.ctx.set_tensor_address(n, x[1])
            else:
                x = np.ascontiguousarray(x, dtype=self.bufs[n][2]); self.ctx.set_input_shape(n, tuple(x.shape)); self.bufs[n][0].upload(x); self.ctx.set_tensor_address(n, self.bufs[n][0].ptr)
        self._run()
        out = {}
        for n in (want or self.outputs):
            shp = tuple(self.ctx.get_tensor_shape(n)); out[n] = self.bufs[n][0].download(shp, self.bufs[n][2]) if download else (self.bufs[n][0].ptr, shp, self.bufs[n][2])
        return out
class Obj:
    __slots__ = ("id", "conds", "hist", "mask", "score", "iou", "last_init", "last_step")
    def __init__(self, oid): self.id = oid; self.conds = {}; self.hist = {}; self.mask = None; self.score = 10.0; self.iou = 1.0; self.last_init = -999; self.last_step = -999
class Sam3TrackerNP:
    NM, MAXPTR, MF_THR, MAX_COND = 7, 16, 0.01, 4
    def __init__(self, neck_plan, init288_plan, step_plan, vision_bufptr, vision_shape, pe_mem, tpos_enc, util_so, max_hist=24, prune_r=4, grid_stride=2):
        self.util = ctypes.CDLL(util_so); self.util.lq_gather_rows64.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]; self.util.lq_gather_blocks.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
        self.util.lq_transpose64.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.neck = DynRunner(load_engine(neck_plan)); self.init = DynRunner(load_engine(init288_plan)); self.step = DynRunner(load_engine(step_plan)); self.max_hist = max_hist; self.prune_r = prune_r; self.adaptive = False
        self.neck.bind("trunk", vision_bufptr, vision_shape)
        for eng in (self.init, self.step):
            for n in ("feat72", "hr0", "hr1"):
                if n in eng.inputs: eng.bind(n, self.neck.bufs[n][0].ptr, tuple(self.neck.engine.get_tensor_shape(n)))
        self.objs = {}; self.dropped_ids = set(); self.t_ms = {"neck": 0.0, "init": 0.0, "step": 0.0}; self.prof = {k: 0.0 for k in ("prep", "gather", "engine", "post")}; self.PROF = bool(os.environ.get("LQ_PROF")); self.n_init = 0; self.n_step = 0; self.n_step_calls = 0; self.keys_used = 0; self.keys_full = 0
        self.p_max = int(self.step.engine.get_tensor_profile_shape("ptrs", 0)[2][1]); self.kn_max = int(self.step.engine.get_tensor_profile_shape("memtok", 0)[2][1]); self.m_max = 7; self.v2 = True
        self.pe_flat = np.asarray(pe_mem).reshape(64, 5184).T.astype(np.float32); self.tpos_flat = np.asarray(tpos_enc).reshape(-1, 64).astype(np.float32)
        yy, xx = np.meshgrid(np.arange(72), np.arange(72), indexing="ij"); self.bg_keep = ((yy % grid_stride == 0) & (xx % grid_stride == 0)).ravel()
        self.b_max = int(self.step.engine.get_tensor_profile_shape("memtok", 0)[2][0]); self.b_init = int(self.init.engine.get_tensor_profile_shape("mask288", 0)[2][0])
        self.ptr_tab = DevBuf(8 * self.b_max * self.kn_max); self.nv_dev = DevBuf(4 * self.b_max); self.mask_gather = DevBuf(self.b_init * 288 * 288 * 2); self.mask_ptrs = DevBuf(8 * self.b_init)
        self.mem_t = DevBuf(max(self.b_max, self.b_init) * 64 * 5184 * 2)      # token-major transposed memories of one call
        pet = np.ascontiguousarray(np.stack([self.pe_flat + self.tpos_flat[tp] for tp in range(self.NM)]).astype(np.float16)); self.pet = DevBuf(pet.nbytes); self.pet.upload(pet)   # [NM, 5184, 64] fp16: key position rows, gathered on the device
        self.pos_tab = DevBuf(8 * self.b_max * self.kn_max); self.pool = []; self.kn_bucket = int(os.environ.get("LQ_KN_BUCKET", 2048))       # free list of memory-frame buffers (no cudaMalloc per step)
        assert self.step.bufs["mempos"][2] == np.float16
    def _alloc(self): return self.pool.pop() if self.pool else DevBuf(64 * 5184 * 2)
    def _forget(self, d, keys):
        for f in keys: self.pool.append(d[f][0]); del d[f]
    def run_neck(self):
        t0 = time.perf_counter(); self.neck._run(); self.t_ms["neck"] += (time.perf_counter() - t0) * 1000
    @staticmethod
    def mem_score(obj_score, iou): return (max(0.0, 1 / (1 + np.exp(-obj_score)) * 2 - 1)) * iou
    def obj(self, oid): return self.objs.setdefault(oid, Obj(oid))
    def refresh(self, t, ids, mask_rows):
        """conditioning memories from the segmentation head's masks: mask_rows = device addresses of 288x288 fp16 logit rows, one per object id"""
        if not len(ids): return
        for c in range(0, len(ids), self.b_init):
            sel = ids[c:c + self.b_init]; B = len(sel); t0 = time.perf_counter()
            self.mask_ptrs.upload(np.asarray(mask_rows[c:c + self.b_init], np.uint64)); assert self.util.lq_gather_blocks(self.mask_gather.ptr, self.mask_ptrs.ptr, B, 288 * 288 * 2, None) == 0
            o = self.init({"mask288": ("ptr", self.mask_gather.ptr, (B, 288, 288))}, download=False); mm_ptr, mm_shape, _ = o["maskmem"]; op = self.init.bufs["obj_ptr"][0].download((B, 256), np.float32)
            assert self.util.lq_transpose64(self.mem_t.ptr, mm_ptr, B, int(self.init.bufs["maskmem"][2] == np.float32), None) == 0; DevBuf.sync(); self.t_ms["init"] += (time.perf_counter() - t0) * 1000; self.n_init += B
            for i, oid in enumerate(sel):
                ob = self.obj(oid); buf = self._alloc(); memcpy_d2d(buf.ptr, self.mem_t.ptr + i * 64 * 5184 * 2, 64 * 5184 * 2); ob.conds[t] = (buf, op[i].astype(np.float32)); ob.last_init = t
                self._forget(ob.conds, sorted(ob.conds)[:-self.MAX_COND])
    def _keep(self, ob):
        keep = np.ones(5184, bool)
        if self.prune_r > 0 and ob.mask is not None:
            m72 = (np.asarray(ob.mask) > 0).reshape(288, 288)[::4, ::4]; r = self.prune_r; pad = np.pad(m72, r); reg = np.zeros_like(m72)
            for dy in range(2 * r + 1):
                for dx in range(2 * r + 1): reg |= pad[dy:dy + 72, dx:dx + 72]
            keep = reg.ravel() | self.bg_keep
        return keep
    def _bank(self, ob, t):
        conds = sorted(ob.conds, key=lambda f: abs(t - f))[:self.MAX_COND]
        valid = [f for f in sorted(ob.hist) if f < t and ob.hist[f][2] > self.MF_THR][-(self.MAXPTR - 1):]
        m_max = self.m_max if not (self.adaptive and ob.iou >= 0.85 and ob.score > 4.0) else min(self.m_max, 3)
        mems = [ob.conds[f][0] for f in conds]; tpos = [self.NM - 1] * len(conds); slots = valid[-min(self.NM - 1, m_max - len(conds)):] if m_max > len(conds) else []
        for i, f in enumerate(slots): t_pos = self.NM - len(slots) + i; mems.append(ob.hist[f][0]); tpos.append(self.NM - t_pos - 1)
        ptrs = [ob.conds[f][1] for f in conds]; ppos = [float(t - f) for f in conds]
        for d in range(1, min(self.MAXPTR, len(valid), self.p_max - len(conds) + 1)): ptrs.append(ob.hist[valid[-d]][1]); ppos.append(float(d))
        return mems, tpos, np.stack(ptrs), np.array(ppos, np.float32) / (self.MAXPTR - 1)
    def propagate(self, t, ids):
        groups = {}; out = {}
        for oid in ids:
            ob = self.objs.get(oid)
            if ob is None or not ob.conds: continue
            mems, tpos, ptrs, ppos = self._bank(ob, t); groups.setdefault((len(tpos), len(ppos)), []).append((ob, mems, tpos, ptrs, ppos))
        for (M, P), items in groups.items():
            for c in range(0, len(items), self.b_max):
                chunk = items[c:c + self.b_max]; Bc = len(chunk); t0 = time.perf_counter()
                sels = [np.nonzero(self._keep(ob))[0] for (ob, *_r) in chunk]; nv = np.array([M * len(s) for s in sels], np.int32); Kn = int(min(self.kn_max, -(-int(nv.max()) // self.kn_bucket) * self.kn_bucket))   # rounded up to a bucket so CUDA graphs are reused across calls
                table = np.zeros((Bc, Kn), np.uint64); ptab = np.zeros((Bc, Kn), np.uint64); keyidx = np.zeros((Bc, Kn), np.int64)
                for i, (ob, mems, tpos, _p, _pp) in enumerate(chunk):
                    sel = sels[i]; n = len(sel); rows = np.concatenate([m.ptr + sel.astype(np.uint64) * 128 for m in mems]); table[i, :M * n] = rows           # memory rows are token-major [5184, 64] fp16 after the transpose below
                    ptab[i, :M * n] = np.concatenate([self.pet.ptr + (tp * 5184 + sel).astype(np.uint64) * 128 for tp in tpos]); keyidx[i, :M * n] = np.tile(sel, M)
                if self.PROF: DevBuf.sync(); t1 = time.perf_counter(); self.prof["prep"] += t1 - t0
                self.ptr_tab.upload(table); self.nv_dev.upload(nv); memtok_buf = self.step.bufs["memtok"][0]
                assert self.util.lq_gather_rows64(memtok_buf.ptr, self.ptr_tab.ptr, self.nv_dev.ptr, Bc, Kn, None) == 0
                self.pos_tab.upload(ptab); mempos_buf = self.step.bufs["mempos"][0]; assert self.util.lq_gather_rows64(mempos_buf.ptr, self.pos_tab.ptr, self.nv_dev.ptr, Bc, Kn, None) == 0
                self.keys_used += int(nv.sum()); self.keys_full += int(M * 5184 * Bc)
                if self.PROF: DevBuf.sync(); t2 = time.perf_counter(); self.prof["gather"] += t2 - t1
                o = self.step({"memtok": ("ptr", memtok_buf.ptr, (Bc, Kn, 64)), "mempos": ("ptr", mempos_buf.ptr, (Bc, Kn, 64)), "keyidx": keyidx, "nvalid": nv, "ptrs": np.stack([it[3] for it in chunk]).astype(np.float32), "ptr_pos": chunk[0][4]}, want=["masks", "obj_ptr", "obj_score", "iou"])
                if self.PROF: DevBuf.sync(); t3 = time.perf_counter(); self.prof["engine"] += t3 - t2
                mm_ptr = self.step.bufs["maskmem"][0].ptr; assert self.util.lq_transpose64(self.mem_t.ptr, mm_ptr, Bc, int(self.step.bufs["maskmem"][2] == np.float32), None) == 0; DevBuf.sync(); self.t_ms["step"] += (time.perf_counter() - t0) * 1000; self.n_step_calls += 1; self.n_step += Bc
                for i, (ob, *_r) in enumerate(chunk):
                    ob.mask = o["masks"][i, 0].astype(np.float32); ob.score = float(o["obj_score"][i, 0]); ob.iou = float(o["iou"][i, 0]); ob.last_step = t
                    buf = self._alloc(); memcpy_d2d(buf.ptr, self.mem_t.ptr + i * 64 * 5184 * 2, 64 * 5184 * 2); ob.hist[t] = (buf, o["obj_ptr"][i].astype(np.float32), self.mem_score(ob.score, ob.iou))
                    self._forget(ob.hist, sorted(ob.hist)[:-self.max_hist])
                    out[ob.id] = (ob.mask, ob.score, ob.iou)
                if self.PROF: self.prof["post"] += time.perf_counter() - t3
        return out
    def prune(self, live_ids):
        for oid in [o for o in self.objs if o not in live_ids]: ob = self.objs.pop(oid); self._forget(ob.conds, list(ob.conds)); self._forget(ob.hist, list(ob.hist))
        self.dropped_ids &= set(live_ids)
    def drop(self, oid): self.dropped_ids.add(oid)
    def dropped(self, oid): return oid in self.dropped_ids
