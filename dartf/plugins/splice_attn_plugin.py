"""K4 splice: replace each block's eager attention subgraph
   q_tm -> Reshape -> Transpose ─┐
   k_tm -> Reshape -> Transpose -> Transpose(kᵀ) ─┤ MatMul -> Mul(scale) -> Softmax -> MatMul(P,v) -> Transpose -> Reshape -> out
   v_tm -> Reshape -> Transpose ───────────────────┘
by  LQAttention(q_tm, k_tm, v_tm) -> Reshape(out shape)   (fp16 token-major in/out; window N and heads from the q Reshape shape).
usage: splice_attn_plugin.py in.onnx out.onnx
"""
import sys, os, onnx, numpy as np
from onnx import numpy_helper, helper
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph; inits = {i.name: i for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}; cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
def back(t, ops):
    out = []
    for op in ops:
        n = prod.get(t)
        if n is None or n.op_type != op: return None
        out.append(n); t = n.input[0]
    return out
n_spliced = 0; kill = set(); new_nodes = []
for sm in [n for n in g.node if n.op_type == "Softmax"]:
    mul = prod.get(sm.input[0]); scale = 1.0; mm_qk = mul
    if mul is not None and mul.op_type == "Mul":
        cname = mul.input[1] if mul.input[1] in inits else (mul.input[0] if mul.input[0] in inits else None)
        if cname is None: continue
        scale = float(numpy_helper.to_array(inits[cname]).reshape(-1)[0]); mm_qk = prod.get([i for i in mul.input if i != cname][0])
    if mm_qk is None or mm_qk.op_type != "MatMul": continue
    qc = back(mm_qk.input[0], ["Transpose", "Reshape"]); kc = back(mm_qk.input[1], ["Transpose", "Transpose", "Reshape"])
    if qc is None or kc is None: continue
    mm_pv = next((c for c in cons.get(sm.output[0], []) if c.op_type == "MatMul"), None)
    if mm_pv is None: continue
    vc = back(mm_pv.input[1], ["Transpose", "Reshape"])
    if vc is None: continue
    oc = cons.get(mm_pv.output[0], [])
    if len(oc) != 1 or oc[0].op_type != "Transpose": continue
    otr = oc[0]; orc = cons.get(otr.output[0], [])
    if len(orc) != 1 or orc[0].op_type != "Reshape": continue
    orsh = orc[0]
    q_tm, k_tm, v_tm = qc[-1].input[0], kc[-1].input[0], vc[-1].input[0]
    shp = numpy_helper.to_array(inits[qc[-1].input[1]]).reshape(-1).tolist()      # q reshape target: [B*w, N, H, 64]
    N, H = int(shp[1]), int(shp[2])
    if N <= 0 or H <= 0 or int(shp[3]) != 64: print("unexpected q reshape", shp); continue
    tag = f"lqattn_{n_spliced}"
    plug = helper.make_node("LQAttention", [q_tm, k_tm, v_tm], [tag + "_out"], name=tag, domain="lq", window=N, heads=H, scale=scale)
    resh = helper.make_node("Reshape", [tag + "_out", orsh.input[1]], [orsh.output[0]], name=tag + "_reshape")
    new_nodes += [plug, resh]
    kill |= {n.name for n in (sm, mm_qk, mm_pv, otr, orsh)} | {n.name for n in qc + kc + vc}
    if mul is not None and mul.op_type == "Mul": kill.add(mul.name)
    n_spliced += 1
keep = [n for n in g.node if n.name not in kill]
prodf = {o: n for n in keep + new_nodes for o in n.output}; ordered = []; done = set()
def visit(n):
    if id(n) in done: return
    done.add(id(n))
    for i in n.input:
        if i in prodf: visit(prodf[i])
    ordered.append(n)
for n in keep + new_nodes: visit(n)
del g.node[:]; g.node.extend(ordered)
if not any(o.domain == "lq" for o in m.opset_import): m.opset_import.append(helper.make_opsetid("lq", 1))
print(f"spliced {n_spliced} attention blocks -> LQAttention")
if os.path.exists(src + ".data"): onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(dst) + ".data", size_threshold=1024)
else: onnx.save(m, dst)
