"""Structural INT8 (explicit Q/DQ) quantizer for the DART/HF SAM3 backbone export (hf_backbone.onnx).

Sites are found structurally (nodes are unnamed): per trunk block, three [1024,1024] MatMuls sharing one input
(q, k, v), one [1024,1024] proj, fc1 [1024,4736] (input = LayerNormalization), fc2 [4736,1024] (input = Gelu).
Reuses the Meta-graph calibration statistics (same activation tensors per block/family) and, optionally, the Meta
GPTQ code cache (qkv codes are column slices of the fused [1024,3072] matrix; GPTQ is column-independent).
"""
import onnx, numpy as np, argparse, json, re, collections, sys
from onnx import numpy_helper, helper

ap = argparse.ArgumentParser()
ap.add_argument("onnx_in"); ap.add_argument("onnx_out"); ap.add_argument("calib_json")
ap.add_argument("--calib-npz", default=None)
ap.add_argument("--act", default="p99999", choices=["max", "p999", "p9999", "p99999", "none"])
ap.add_argument("--smooth", type=float, default=0.0)
ap.add_argument("--smooth-keep-max", action="store_true")
ap.add_argument("--blocks", default="0-31"); ap.add_argument("--act-blocks", default=None)
ap.add_argument("--always-act-fams", default=""); ap.add_argument("--skip", default="")
ap.add_argument("--gptq-cache", default=None, help="Meta-graph GPTQ cache (gptq_a05.npz); qkv codes are sliced")
ap.add_argument("--check-weights", action="store_true", help="assert HF q/k/v weights equal Meta qkv column slices (needs --meta-onnx)")
ap.add_argument("--meta-onnx", default=None)
ap.add_argument("--act-override", default=None, help="json {block{b}.{fam}: scale} (e.g. from recon_blocks.py) overriding calibrated activation scales")
ap.add_argument("--neck-int8", default=None, help="calib json with calib.neck.<tensor> entries: quantize the neck Conv/ConvTranspose weights (per-output-channel) and inputs (per-tensor)")
ap.add_argument("--smooth-fams", default="qkv,fc1,proj", help="families the SmoothQuant pass folds (subset of qkv,fc1,proj)"); ap.add_argument("--smooth-override", default=None, help="json {block{b}.{qkv|fc1|proj}: [C] smoothing vector} used instead of the calib-derived SmoothQuant vectors")
ap.add_argument("--bias-override", default=None, help="json {block{b}.{fam}[:role]: [N]} replacing the bias Add constants (e.g. gptq_actaware bias correction)")
ap.add_argument("--act-skip-sites", default="", help="comma list of block{b}.{fam} sites whose activation stays FP (e.g. block0.fc1)")
ap.add_argument("--smooth-blocks", default=None, help="restrict SmoothQuant folds to these blocks (default: all --blocks)")
ap.add_argument("--out-q", default=None, help="fc1out_stats.json: add Q/DQ after the fc1 bias Add (lets TRT fuse GEMM+bias+requant)")
ap.add_argument("--out-q-stat", default="p99999")
ap.add_argument("--gemm", action="store_true", help="re-express each quantized trunk MatMul(+bias Add) as Reshape->Gemm(A,B,C=bias)->Reshape so TensorRT can fuse the INT8 epilogue")
a = ap.parse_args()

calib = json.load(open(a.calib_json)); npz = np.load(a.calib_npz) if a.calib_npz else None
cache = dict(np.load(a.gptq_cache)) if a.gptq_cache else None
b0, b1 = [int(v) for v in a.blocks.split("-")]; blocks = set(range(b0, b1 + 1))
ACT_SKIP = set(v for v in a.act_skip_sites.split(",") if v)
BIAS_OVER = json.load(open(a.bias_override)) if a.bias_override else {}
SMOOTH_OVER = json.load(open(a.smooth_override)) if a.smooth_override else {}
ab = a.act_blocks or a.blocks; ab0, ab1 = [int(v) for v in ab.split("-")]; act_blocks = set(range(ab0, ab1 + 1))
always = set(f for f in a.always_act_fams.split(",") if f); skip = set(f for f in a.skip.split(",") if f)
ACT_OVER = json.load(open(a.act_override)).get("act_scales", {}) if a.act_override else {}
if a.smooth_blocks: sb0, sb1 = [int(v) for v in a.smooth_blocks.split("-")]; smooth_blocks = set(range(sb0, sb1 + 1))
else: smooth_blocks = None

m = onnx.load(a.onnx_in); g = m.graph
# static shapes of every tensor, taken from the ORIGINAL graph (before any int8 edits)
SHAPES = {}
try:
    import onnx.shape_inference as _si
    _mi = _si.infer_shapes(m, strict_mode=False)
    for vi in list(_mi.graph.value_info) + list(_mi.graph.output) + list(_mi.graph.input):
        d = [x.dim_value for x in vi.type.tensor_type.shape.dim]
        if d and all(v > 0 for v in d): SHAPES[vi.name] = d
except Exception as e: print("shape inference:", e)
for vi in list(g.value_info) + list(g.output) + list(g.input):
    d = [x.dim_value for x in vi.type.tensor_type.shape.dim]
    if d and all(v > 0 for v in d): SHAPES.setdefault(vi.name, d)
print("static shapes known:", len(SHAPES))
inits = {t.name: t for t in g.initializer}; prod = {o: n for n in g.node for o in n.output}
cons = collections.defaultdict(list)
for n in g.node:
    for i in n.input: cons[i].append(n)
def W_of(name): return numpy_helper.to_array(inits[name]).astype(np.float32)
def W_set(name, arr): inits[name].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), name))

# ---- structural site discovery -------------------------------------------------------------------------------
wm = [n for n in g.node if n.op_type == "MatMul" and n.input[1] in inits and "_had" not in n.input[1]]   # skip Hadamard rotation matmuls
attn_mm = [n for n in g.node if n.op_type == "MatMul" and n.input[1] not in inits and n.input[0] not in inits]
def reaches(node, target, max_depth=12):
    """does node's output reach `target` MatMul through non-MatMul nodes?"""
    frontier = [node]; seen = set()
    for _ in range(max_depth):
        nxt = []
        for f in frontier:
            for c in cons.get(f.output[0], []):
                if c is target: return True
                if c.op_type != "MatMul" and id(c) not in seen: seen.add(id(c)); nxt.append(c)
        frontier = nxt
        if not frontier: break
    return False
sites = []  # dicts: block, fam, node, x (input tensor name), w (weight init name)
blk = -1; trio = []
for n in wm:
    K, N = list(inits[n.input[1]].dims)
    if (K, N) == (1024, 4736): blk += 1; sites.append(dict(block=blk, fam="fc1", node=n, x=n.input[0], w=n.input[1]))
    elif (K, N) == (4736, 1024): sites.append(dict(block=blk, fam="fc2", node=n, x=n.input[0], w=n.input[1]))
    elif (K, N) == (1024, 1024):
        if len(cons[n.input[0]]) >= 3 and sum(1 for c in cons[n.input[0]] if c.op_type == "MatMul") == 3:
            trio.append(n)
            if len(trio) == 3:
                # this trio belongs to block blk+1 (fc1 of that block comes later in topological order)
                b = blk + 1
                qk = [t for t in attn_mm if any(reaches(x, t) for x in trio) and t.input[0] not in [x.output[0] for x in trio] or True]
                roles = {}
                for x in trio:
                    tgt_qk = next((t for t in attn_mm if prod.get(t.input[0]) is not None and reaches(x, t)), None)
                    roles[x.name] = None
                # classify: v reaches the PV matmul (whose input[0] comes from Softmax); q/k reach the QK matmul
                pv = [t for t in attn_mm if prod.get(t.input[0]) is not None and prod[t.input[0]].op_type == "Softmax"]
                qkm = [t for t in attn_mm if t not in pv]
                v = next(x for x in trio if any(reaches(x, t) for t in pv))
                qk_nodes = [x for x in trio if x is not v]
                qkt = next(t for t in qkm if any(reaches(x, t) for x in qk_nodes))
                q = next(x for x in qk_nodes if reaches(x, qkt) and any(reaches(x, t) for t in [qkt]) and _first_input_reaches(x, qkt)) if False else None
                # q is the one reaching input[0] of the QK matmul: walk back from qkt.input[0]
                def back_reaches(tensor, node, depth=12):
                    fr = [tensor]
                    for _ in range(depth):
                        nx = []
                        for t in fr:
                            p = prod.get(t)
                            if p is None: continue
                            if p is node: return True
                            if p.op_type != "MatMul": nx.extend(p.input)
                        fr = nx
                        if not fr: break
                    return False
                q = next(x for x in qk_nodes if back_reaches(qkt.input[0], x)); k = next(x for x in qk_nodes if x is not q)
                for role, node in (("q", q), ("k", k), ("v", v)):
                    sites.append(dict(block=b, fam="qkv", role=role, node=node, x=node.input[0], w=node.input[1]))
                trio = []
        else:
            sites.append(dict(block=blk + 1 if not any(s["block"] == blk + 1 and s["fam"] == "fc1" for s in sites) else blk, fam="proj", node=n, x=n.input[0], w=n.input[1]))
# fix proj block assignment: proj appears after the qkv trio and before fc1 of the same block
for s in sites:
    if s["fam"] == "proj":
        s["block"] = max(x["block"] for x in sites if x["fam"] == "qkv" and x["node"].name < s["node"].name or True) if False else s["block"]
nblk = max(s["block"] for s in sites) + 1
fams = collections.Counter((s["block"], s["fam"]) for s in sites)
assert all(fams[(b, "qkv")] == 3 and fams[(b, "proj")] == 1 and fams[(b, "fc1")] == 1 and fams[(b, "fc2")] == 1 for b in range(nblk)), fams
print(f"structural sites: {nblk} blocks x (q,k,v,proj,fc1,fc2) =", len(sites))
NBLK = nblk

if a.check_weights and a.meta_onnx:
    mm = onnx.load(a.meta_onnx, load_external_data=True); mi = {t.name: t for t in mm.graph.initializer}
    mn = {n.name: n for n in mm.graph.node}
    for b in (0, 5, 31):
        fused = numpy_helper.to_array(mi[mn[f"/trunk/blocks.{b}/attn/qkv/MatMul"].input[1]])
        for role, off in (("q", 0), ("k", 1024), ("v", 2048)):
            s = next(x for x in sites if x["block"] == b and x["fam"] == "qkv" and x["role"] == role)
            d = np.abs(W_of(s["w"]) - fused[:, off:off + 1024]).max()
            print(f"  block{b} {role}: max|HF-Meta| = {d:.2e}")
        for fam, nm in (("proj", "attn/proj/MatMul"), ("fc1", "mlp/MatMul"), ("fc2", "mlp/fc2/MatMul")):
            s = next(x for x in sites if x["block"] == b and x["fam"] == fam)
            d = np.abs(W_of(s["w"]) - numpy_helper.to_array(mi[mn[f"/trunk/blocks.{b}/{nm}"].input[1]])).max(); print(f"  block{b} {fam}: max|HF-Meta| = {d:.2e}")

# ---- helpers ---------------------------------------------------------------------------------------------------
def find_ln(tensor, depth=8):
    t = tensor
    for _ in range(depth):
        p = prod.get(t)
        if p is None: return None
        if p.op_type == "LayerNormalization": return p
        if p.op_type in ("Reshape", "Transpose", "Pad", "Slice"): t = p.input[0]; continue
        return None
    return None
def act_scale(block, fam):
    c = calib[f"calib.block{block}.{fam}"]
    return float(c["absmax" if a.act == "max" else a.act]) / 127.0
def q_rtn(W):
    amax = np.abs(W).max(axis=0); amax[amax == 0] = 1.0; s = (amax / 127.0).astype(np.float32)
    return np.clip(np.round(W / s), -127, 127).astype(np.int8), s

new_nodes = []; report = []; smooth_s = {}
active = [s for s in sites if (s["block"] in blocks and s["fam"] not in skip) or s["fam"] in always]
for s in active: s["wq"] = s["block"] in blocks and s["fam"] not in skip   # weights INT8 only inside --blocks
# ---- pass 1: SmoothQuant folds (weights stay FP32 here) ---------------------------------------------------------
if a.smooth > 0 and npz is not None:
    for b in range(NBLK):
        if b not in blocks or (smooth_blocks is not None and b not in smooth_blocks): continue
        # qkv: shared input; one s vector folded into the producing LN, applied to all three weights
        SF = set(a.smooth_fams.split(","))
        trio_s = [s for s in active if s["block"] == b and s["fam"] == "qkv"] if "qkv" in SF else []
        if trio_s:
            chmax = npz[f"calib.block{b}.qkv::chmax"].astype(np.float32)
            wmax = np.max([np.abs(W_of(s["w"])).max(axis=1) for s in trio_s], axis=0)
            sv = np.clip(np.power(np.maximum(chmax, 1e-5), a.smooth) / np.power(np.maximum(wmax, 1e-5), 1 - a.smooth), 1e-2, 1e2).astype(np.float32)
            if f"block{b}.qkv" in SMOOTH_OVER: sv = np.asarray(SMOOTH_OVER[f"block{b}.qkv"], np.float32)
            ln = find_ln(trio_s[0]["x"])
            if ln is not None:
                W_set(ln.input[1], W_of(ln.input[1]) / sv); W_set(ln.input[2], W_of(ln.input[2]) / sv)
                for s in trio_s: W_set(s["w"], W_of(s["w"]) * sv[:, None])
                smooth_s[(b, "qkv")] = sv
        for fam in ("fc1",):
            s1 = next((s for s in active if s["block"] == b and s["fam"] == fam), None) if fam in SF else None
            if s1 is None: continue
            chmax = npz[f"calib.block{b}.{fam}::chmax"].astype(np.float32); W = W_of(s1["w"]); wmax = np.abs(W).max(axis=1)
            sv = np.clip(np.power(np.maximum(chmax, 1e-5), a.smooth) / np.power(np.maximum(wmax, 1e-5), 1 - a.smooth), 1e-2, 1e2).astype(np.float32)
            if f"block{b}.{fam}" in SMOOTH_OVER: sv = np.asarray(SMOOTH_OVER[f"block{b}.{fam}"], np.float32)
            ln = find_ln(s1["x"])
            if ln is not None:
                W_set(ln.input[1], W_of(ln.input[1]) / sv); W_set(ln.input[2], W_of(ln.input[2]) / sv); W_set(s1["w"], W * sv[:, None]); smooth_s[(b, fam)] = sv
        sp = next((s for s in active if s["block"] == b and s["fam"] == "proj"), None) if "proj" in SF else None
        if sp is not None:
            chmax = npz[f"calib.block{b}.proj::chmax"].astype(np.float32); W = W_of(sp["w"]); wmax = np.abs(W).max(axis=1)
            sv = np.clip(np.power(np.maximum(chmax, 1e-5), a.smooth) / np.power(np.maximum(wmax, 1e-5), 1 - a.smooth), 1e-2, 1e2).astype(np.float32)
            if f"block{b}.proj" in SMOOTH_OVER: sv = np.asarray(SMOOTH_OVER[f"block{b}.proj"], np.float32)
            vsite = next(s for s in sites if s["block"] == b and s["fam"] == "qkv" and s["role"] == "v")
            W_set(vsite["w"], W_of(vsite["w"]) / sv[None, :])
            addn = next((c for c in cons[vsite["node"].output[0]] if c.op_type == "Add"), None)
            if addn is not None:
                bname = addn.input[0] if addn.input[0] in inits else addn.input[1]
                if bname in inits: W_set(bname, W_of(bname) / sv)
            W_set(sp["w"], W * sv[:, None]); smooth_s[(b, "proj")] = sv

def C(name, arr): g.initializer.append(numpy_helper.from_array(np.asarray(arr), name)); return name
# ---- pass 1b: per-input-channel scales on fc2 (block{b}.fc2 in --smooth-override): x/c on the activation side (Mul before the Q
#      node; folded into fc1's requant epilogue by the MLP plugin), W2 rows * c on the weight side (the GPTQ cache is already folded)
n_fc2c = 0
for s in active:
    if s["fam"] != "fc2" or f"block{s['block']}.fc2" not in SMOOTH_OVER: continue
    cv = np.asarray(SMOOTH_OVER[f"block{s['block']}.fc2"], np.float32); key = f"block{s['block']}.fc2"
    cn = C(key + "_chan_inv", (1.0 / cv).astype(np.float32)); mo = key + "_xchan"
    new_nodes.append(helper.make_node("Mul", [s["x"], cn], [mo], name=key + "_chanMul"))
    for i_, inp in enumerate(s["node"].input):
        if inp == s["x"]: s["node"].input[i_] = mo
    W_set(s["w"], W_of(s["w"]) * cv[:, None]); s["x"] = mo; n_fc2c += 1
if n_fc2c: print("fc2 per-channel activation scales applied:", n_fc2c)
# ---- pass 2: weights INT8 (cache slice or RTN) + activation Q/DQ ----------------------------------------------
qdq_done = {}; node_w_map = {}
n_w = n_a = 0
for s in active:
    b, fam = s["block"], s["fam"]; key = f"block{b}.{fam}"; W = W_of(s["w"])
    if not s["wq"]:
        want_act = a.act != "none"; x = s["x"]
        if want_act and x not in qdq_done:
            sa = act_scale(b, fam); san = C(f"{key}_act_scale", np.array(sa, dtype=np.float32)); zn = C(f"{key}_act_zp", np.array(0, dtype=np.int8))
            new_nodes.append(helper.make_node("QuantizeLinear", [x, san, zn], [f"{key}_xq"], name=f"{key}_xQ")); new_nodes.append(helper.make_node("DequantizeLinear", [f"{key}_xq", san, zn], [f"{key}_xdq"], name=f"{key}_xDQ")); qdq_done[x] = f"{key}_xdq"
        if want_act: s["node"].input[0] = qdq_done[x]; n_a += 1; report.append((key, "a8only"))
        continue
    if cache is not None and f"{key}::codes" in cache:
        codes, sc = cache[f"{key}::codes"], cache[f"{key}::scale"]
        if fam == "qkv":
            off = {"q": 0, "k": 1024, "v": 2048}[s["role"]]; codes, sc = codes[:, off:off + 1024], sc[off:off + 1024]
        q, ws = np.ascontiguousarray(codes), np.ascontiguousarray(sc)
    else:
        q, ws = q_rtn(W)
    inits[s["w"]].CopyFrom(numpy_helper.from_array(q, s["w"])); wsn = C(s["w"] + "_scale", ws)
    wdq = s["w"] + "_dq"; new_nodes.append(helper.make_node("DequantizeLinear", [s["w"], wsn], [wdq], name=s["w"] + "_wDQ", axis=1)); s["node"].input[1] = wdq; n_w += 1
    node_w_map[f"block{b}.{fam}" + (":" + s["role"] if fam == "qkv" else "")] = wdq
    want_act = a.act != "none" and (b in act_blocks or fam in always) and f"block{b}.{fam}" not in ACT_SKIP
    if not want_act: report.append((key + (":" + s.get("role", "") if fam == "qkv" else ""), "w8")); continue
    x = s["x"]
    if x not in qdq_done:
        sa = ACT_OVER.get(f"block{b}.{fam}", act_scale(b, fam))
        if (b, fam) in smooth_s and f"block{b}.{fam}" not in ACT_OVER:
            chmax = npz[f"calib.block{b}.{fam}::chmax"].astype(np.float32); new_absmax = float((chmax / smooth_s[(b, fam)]).max())
            ratio = 1.0 if (a.act == "max" or a.smooth_keep_max) else min(1.0, sa * 127.0 / max(calib[f"calib.block{b}.{fam}"]["absmax"], 1e-9))
            sa = new_absmax * ratio / 127.0
        san = C(f"{key}_act_scale", np.array(sa, dtype=np.float32)); zn = C(f"{key}_act_zp", np.array(0, dtype=np.int8))
        new_nodes.append(helper.make_node("QuantizeLinear", [x, san, zn], [f"{key}_xq"], name=f"{key}_xQ"))
        new_nodes.append(helper.make_node("DequantizeLinear", [f"{key}_xq", san, zn], [f"{key}_xdq"], name=f"{key}_xDQ")); qdq_done[x] = f"{key}_xdq"
    s["node"].input[0] = qdq_done[x]; n_a += 1
    report.append((key + (":" + s.get("role", "") if fam == "qkv" else ""), "int8"))

if a.out_q:
    OQ = json.load(open(a.out_q)); n_oq = 0
    for s_ in active:
        if s_["fam"] != "fc1" or not s_["wq"]: continue
        b = s_["block"]; addn = next((c for c in cons[s_["node"].output[0]] if c.op_type == "Add"), None)
        if addn is None: continue
        t = addn.output[0]; sc = float(OQ[f"block{b}.fc1out"][a.out_q_stat]) / 127.0
        san = C(f"block{b}.fc1out_scale", np.array(sc, dtype=np.float32)); zn = C(f"block{b}.fc1out_zp", np.array(0, dtype=np.int8))
        new_nodes.append(helper.make_node("QuantizeLinear", [t, san, zn], [t + "_oq"], name=f"block{b}_fc1out_Q"))
        new_nodes.append(helper.make_node("DequantizeLinear", [t + "_oq", san, zn], [t + "_odq"], name=f"block{b}_fc1out_DQ"))
        for c in list(cons[t]):
            if c is not addn:
                for i, inp in enumerate(c.input):
                    if inp == t: c.input[i] = t + "_odq"
        n_oq += 1
    print("fc1 output Q/DQ inserted:", n_oq)
if a.gemm:
    shapes = SHAPES; n_gemm = 0; n_noshape = 0
    for s_ in active:
        node = s_["node"]; out = node.output[0]; addn = next((c for c in cons[out] if c.op_type == "Add"), None)
        if addn is None or not s_.get("wq", False): continue
        if out not in shapes: n_noshape += 1; continue
        bias_name = addn.input[0] if addn.input[0] in inits else addn.input[1]
        if bias_name not in inits: continue
        oshape = shapes[out]; K, N = list(inits[s_["w"]].dims); key = f"block{s_['block']}.{s_['fam']}" + (":" + s_["role"] if s_["fam"] == "qkv" else "")
        x_in = node.input[0]; r2 = C(f"{key}_rs2d", np.array([-1, K], dtype=np.int64)); rback = C(f"{key}_rsback", np.array(oshape, dtype=np.int64))
        new_nodes.append(helper.make_node("Reshape", [x_in, r2], [f"{key}_x2d"], name=f"{key}_Reshape2D"))
        node.op_type = "Gemm"; del node.input[:]; node.input.extend([f"{key}_x2d", node_w_map[key], bias_name]); del node.attribute[:]
        node.output[0] = f"{key}_y2d"
        new_nodes.append(helper.make_node("Reshape", [f"{key}_y2d", rback], [addn.output[0]], name=f"{key}_ReshapeBack"))
        g.node.remove(addn); n_gemm += 1
    print("Gemm reformulated sites:", n_gemm, "| sites without static shape:", n_noshape)
if a.neck_int8:
    NECK = json.load(open(a.neck_int8)); n_neck = 0; neck_q = {}
    for n in list(g.node):
        if n.op_type not in ("Conv", "ConvTranspose") or n.input[0] in ("pixel_values", "images") or n.input[1] not in inits: continue
        key = f"calib.neck.{n.input[0]}"
        if key not in NECK: continue
        Wc = W_of(n.input[1]); axis = 0 if n.op_type == "Conv" else 1
        red = tuple(i for i in range(Wc.ndim) if i != axis); ws = np.maximum(np.abs(Wc).max(axis=red), 1e-12).astype(np.float32) / 127.0
        shp = [1] * Wc.ndim; shp[axis] = -1; q = np.clip(np.round(Wc / ws.reshape(shp)), -127, 127).astype(np.int8)
        inits[n.input[1]].CopyFrom(numpy_helper.from_array(q, n.input[1])); wsn = C(n.input[1] + "_scale", ws)
        new_nodes.append(helper.make_node("DequantizeLinear", [n.input[1], wsn], [n.input[1] + "_dq"], name=n.input[1] + "_wDQ", axis=axis)); n.input[1] = n.input[1] + "_dq"
        x = n.input[0]
        if x not in neck_q:
            sa = float(NECK[key][a.act if a.act != "none" else "p99999"]) / 127.0
            san = C(f"neck_{x}_act_scale", np.array(sa, dtype=np.float32)); zn = C(f"neck_{x}_act_zp", np.array(0, dtype=np.int8))
            new_nodes.append(helper.make_node("QuantizeLinear", [x, san, zn], [f"neck_{x}_xq"], name=f"neck_{x}_xQ")); new_nodes.append(helper.make_node("DequantizeLinear", [f"neck_{x}_xq", san, zn], [f"neck_{x}_xdq"], name=f"neck_{x}_xDQ")); neck_q[x] = f"neck_{x}_xdq"
        n.input[0] = neck_q[x]; n_neck += 1
    print(f"neck INT8: {n_neck} convs, {len(neck_q)} input Q/DQ")
n_bias = 0
for s_ in sites:
    k_ = f"block{s_['block']}.{s_['fam']}" + (":" + s_["role"] if s_["fam"] == "qkv" else "")
    if k_ not in BIAS_OVER: continue
    addn = next((c for c in cons.get(s_["node"].output[0], []) if c.op_type == "Add"), None)
    if addn is None: continue
    bname = next((i_ for i_ in addn.input if i_ in inits), None)
    if bname is None: continue
    inits[bname].CopyFrom(numpy_helper.from_array(np.asarray(BIAS_OVER[k_], dtype=np.float32), bname)); n_bias += 1
if BIAS_OVER: print(f"bias overrides applied: {n_bias}/{len(BIAS_OVER)}")
# topological insertion (new nodes before first consumer)
out2new = {n.output[0]: n for n in new_nodes}; final = []
def emit(n):
    for i in n.input:
        if i in out2new and out2new[i] is not None:
            nn = out2new[i]; out2new[i] = None; emit(nn)
    final.append(n)
for n in list(g.node): emit(n)
del g.node[:]; g.node.extend(final)
onnx.save(m, a.onnx_out, save_as_external_data=True, all_tensors_to_one_file=True, location=a.onnx_out.split("/")[-1] + ".data")
json.dump(report, open(a.onnx_out + ".report.json", "w"), indent=1)
print(f"HF quantized: {n_w} weight sites INT8, {n_a} activation Q/DQ, smoothed {len(smooth_s)} -> {a.onnx_out}")
