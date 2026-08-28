"""K5b/K5c splices (run AFTER splice_rope_plugin.py and splice_attn_plugin.py):
  --qkv :  r -> norm -> Q(s) -> { LQGemmRope(q) , LQGemmRope(k) , DQ -> MatMul(v) -> Add(bv) }   ==>  LQQkvRope(r) -> (q_rot, k_rot, v)
  --attnproj :  LQAttention(q,k,v) -> Reshape -> Q(sp) -> DQ -> MatMul(proj) -> Add(bp) -> Add(r)   ==>  LQAttnProj(q,k,v,r) -> out
usage: splice_block_plugins.py in.onnx out.onnx [--qkv] [--attnproj]
"""
import sys, os, argparse, onnx, numpy as np
from onnx import numpy_helper, helper
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--qkv", action="store_true"); ap.add_argument("--tmix", type=int, default=0, help="experimental option; default 0"); ap.add_argument("--attnproj", action="store_true"); ap.add_argument("--fused", action="store_true", help="merged-QKV plugin (one [.,3072] output) + LQAttnProj(qkv, r)"); a = ap.parse_args()
if a.fused: a.qkv = True; a.attnproj = True
m = onnx.load(a.src); g = m.graph; inits = {i.name: i for i in g.initializer}
def rebuild():
    global prod, cons; prod = {o: n for n in g.node for o in n.output}; cons = {}
    for n in g.node:
        for i in n.input: cons.setdefault(i, []).append(n)
rebuild(); arr = lambda name: numpy_helper.to_array(inits[name])
def const_in(n): return next((i for i in n.input if i in inits), None)
def attr(n, name):
    for at in n.attribute:
        if at.name == name: return numpy_helper.to_array(at.t) if at.type == onnx.AttributeProto.TENSOR else (at.f if at.type == onnx.AttributeProto.FLOAT else at.i)
    return None
def norm_chain(t, r_expect=None):
    """returns (mode, gamma, beta, eps, chain nodes, r) for a norm output tensor t"""
    nn = prod.get(t)
    if nn is not None and nn.op_type == "LayerNormalization":
        return 0, arr(nn.input[1]).astype(np.float32), arr(nn.input[2]).astype(np.float32), float(next((at.f for at in nn.attribute if at.name == "epsilon"), 1e-6)), [nn], nn.input[0]
    if nn is None or nn.op_type != "Mul": return None
    ins = [prod.get(i) for i in nn.input]; rcp = next((x for x in ins if x is not None and x.op_type == "Reciprocal"), None); xm = next((x for x in ins if x is not None and x.op_type == "Mul"), None)
    if rcp is None or xm is None: return None
    sq = prod.get(rcp.input[0]); adde = prod.get(sq.input[0]) if sq is not None and sq.op_type == "Sqrt" else None
    if adde is None or adde.op_type != "Add": return None
    eps = float(arr(const_in(adde)).reshape(-1)[0]); div = prod.get([i for i in adde.input if i not in inits][0]); rs = prod.get(div.input[0]); sqm = prod.get(rs.input[0])
    r = xm.input[0] if xm.input[1] in inits else xm.input[1]
    return 1, np.zeros(0, np.float32), np.zeros(0, np.float32), eps, [nn, rcp, sq, adde, div, rs, sqm, xm], r
kill = set(); new_nodes = []; n_qkv = n_ap = 0
if a.qkv:
    for q1 in [n for n in g.node if n.op_type == "QuantizeLinear"]:
        cs = cons.get(q1.output[0], []); ropes = [c for c in cs if c.op_type == "LQGemmRope"]; dqs = [c for c in cs if c.op_type == "DequantizeLinear"]
        if len(ropes) != 2 or len(dqs) != 1: continue
        mmv = next((c for c in cons.get(dqs[0].output[0], []) if c.op_type == "MatMul"), None)
        if mmv is None: continue
        wdqv = prod.get(mmv.input[1]); addv = next((c for c in cons.get(mmv.output[0], []) if c.op_type == "Add"), None)
        if wdqv is None or wdqv.op_type != "DequantizeLinear" or addv is None: continue
        codes_v = arr(wdqv.input[0])
        if codes_v.shape != (1024, 1024): continue
        nc = norm_chain(q1.input[0])
        if nc is None: continue
        mode, gamma, beta, eps, chain, r = nc
        # q/k rope plugins: which is q? the attention plugin's first input is q
        rq, rk = ropes
        attn = next((c for c in cons.get(rq.output[0], []) if c.op_type == "LQAttention"), None)
        if attn is not None and attn.input[0] != rq.output[0]: rq, rk = rk, rq
        s = float(arr(q1.input[1])); tag = f"lqqkv_{n_qkv}"
        kw = dict(mode=mode, eps=eps, s=s, gamma=numpy_helper.from_array(gamma, tag + "_g"), beta=numpy_helper.from_array(beta, tag + "_b"),
                  cos=numpy_helper.from_array(attr(rq, "cos").astype(np.float32), tag + "_cos"), sin=numpy_helper.from_array(attr(rq, "sin").astype(np.float32), tag + "_sin"))
        for pre, node in (("q_", rq), ("k_", rk)):
            kw[pre + "codes"] = numpy_helper.from_array(np.ascontiguousarray(attr(node, "codes")), tag + pre + "c"); kw[pre + "ws"] = numpy_helper.from_array(attr(node, "wscale").astype(np.float32), tag + pre + "ws"); kw[pre + "b"] = numpy_helper.from_array(attr(node, "bias").astype(np.float32), tag + pre + "bb")
        kw["v_codes"] = numpy_helper.from_array(np.ascontiguousarray(codes_v.T), tag + "_vc"); kw["v_ws"] = numpy_helper.from_array(arr(wdqv.input[1]).astype(np.float32), tag + "_vws"); kw["v_b"] = numpy_helper.from_array(arr(const_in(addv)).astype(np.float32), tag + "_vb")
        if a.fused:
            node = helper.make_node("LQQkvRope", [r], [tag + "_qkv"], name=tag, domain="lq", **kw)
            # rewire the attention consumer to read the merged tensor: LQAttention(q,k,v) -> LQAttention(qkv, qkv, qkv) marker (consumed by --attnproj below)
            if attn is not None: attn.input[0] = tag + "_qkv"; attn.input[1] = tag + "_qkv"; attn.input[2] = tag + "_qkv"
        else:
            node = helper.make_node("LQQkvRope", [r], [rq.output[0], rk.output[0], addv.output[0]], name=tag, domain="lq", **kw)
        new_nodes.append(node); kill |= {x.name for x in chain} | {q1.name, rq.name, rk.name, dqs[0].name, wdqv.name, mmv.name, addv.name}; n_qkv += 1
    keep = [n for n in g.node if n.name not in kill]; del g.node[:]; g.node.extend(keep + new_nodes); kill = set(); new_nodes = []; rebuild()
if a.attnproj:
    for at in [n for n in g.node if n.op_type == "LQAttention"]:
        resh = next((c for c in cons.get(at.output[0], []) if c.op_type == "Reshape"), None); qn = resh and next((c for c in cons.get(resh.output[0], []) if c.op_type == "QuantizeLinear"), None)
        dq = qn and next((c for c in cons.get(qn.output[0], []) if c.op_type == "DequantizeLinear"), None); mm = dq and next((c for c in cons.get(dq.output[0], []) if c.op_type == "MatMul"), None)
        if not mm: continue
        wdq = prod.get(mm.input[1]); addb = next((c for c in cons.get(mm.output[0], []) if c.op_type == "Add"), None); addr = addb and next((c for c in cons.get(addb.output[0], []) if c.op_type == "Add"), None)
        if wdq is None or wdq.op_type != "DequantizeLinear" or not addr: continue
        codes = arr(wdq.input[0])
        if codes.shape != (1024, 1024): continue
        r = [i for i in addr.input if i != addb.output[0]][0]; s = float(arr(qn.input[1])); tag = f"lqap_{n_ap}"
        ins = [at.input[0], r] if (a.fused and at.input[0] == at.input[1] == at.input[2]) else [at.input[0], at.input[1], at.input[2], r]
        node = helper.make_node("LQAttnProj", ins, [addr.output[0]], name=tag, domain="lq", tmix=a.tmix, window=attr(at, "window"), heads=attr(at, "heads"), scale=float(attr(at, "scale")), s=s,
                                proj_codes=numpy_helper.from_array(np.ascontiguousarray(codes.T), tag + "_c"), proj_ws=numpy_helper.from_array(arr(wdq.input[1]).astype(np.float32), tag + "_ws"), proj_b=numpy_helper.from_array(arr(const_in(addb)).astype(np.float32), tag + "_b"))
        new_nodes.append(node); kill |= {at.name, resh.name, qn.name, dq.name, wdq.name, mm.name, addb.name, addr.name}; n_ap += 1
    keep = [n for n in g.node if n.name not in kill]; del g.node[:]; g.node.extend(keep + new_nodes); rebuild()
# topological order
nodes = list(g.node); prodf = {o: n for n in nodes for o in n.output}; ordered = []; done = set()
def visit(n):
    if id(n) in done: return
    done.add(id(n))
    for i in n.input:
        if i in prodf: visit(prodf[i])
    ordered.append(n)
for n in nodes: visit(n)
del g.node[:]; g.node.extend(ordered)
print(f"spliced qkv={n_qkv} attnproj={n_ap}")
if os.path.exists(a.src + ".data"): onnx.save(m, a.dst, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(a.dst) + ".data", size_threshold=1024)
else: onnx.save(m, a.dst)
