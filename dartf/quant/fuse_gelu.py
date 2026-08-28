"""Exact fusion of the erf-GELU decomposition emitted by newer exporters into ONNX Gelu (opset 20):
   Div(x, sqrt2) -> Erf -> Add(1) -> Mul(0.5) -> Mul(x, .)   ==>   Gelu(x)      usage: fuse_gelu.py in.onnx out.onnx"""
import sys, onnx, numpy as np
from onnx import helper, numpy_helper
src, dst = sys.argv[1], sys.argv[2]; m = onnx.load(src); g = m.graph
inits = {i.name: i for i in g.initializer}; consts = {n.output[0]: n for n in g.node if n.op_type == "Constant"}; prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
def cval(t):
    if t in consts: a = consts[t].attribute[0]; return float(numpy_helper.to_array(a.t).ravel()[0]) if a.name == "value" else None
    if t in inits: return float(numpy_helper.to_array(inits[t]).ravel()[0]) if numpy_helper.to_array(inits[t]).size == 1 else None
    return None
def single(t, op): c = cons.get(t, []); return c[0] if len(c) == 1 and c[0].op_type == op else None
kill, new, nf = set(), [], 0
for erf in [n for n in g.node if n.op_type == "Erf"]:
    div = prod.get(erf.input[0])
    if div is None or div.op_type != "Div": continue
    x = div.input[0]; c = cval(div.input[1])
    if c is None or abs(c - np.sqrt(2)) > 1e-3: continue
    add = single(erf.output[0], "Add")
    if add is None or not any(abs((cval(i) or 0) - 1.0) < 1e-6 for i in add.input): continue
    mulh = single(add.output[0], "Mul")
    if mulh is None or not any(abs((cval(i) or 0) - 0.5) < 1e-6 for i in mulh.input): continue
    mulx = single(mulh.output[0], "Mul")
    if mulx is None or x not in mulx.input: continue
    new.append(helper.make_node("Gelu", [x], [mulx.output[0]], name=mulx.name + "_gelu", approximate="none"))
    kill |= {div.name, erf.name, add.name, mulh.name, mulx.name}; nf += 1
keep = [n for n in g.node if n.name not in kill]
used = {i for n in keep + new for i in n.input} | {o.name for o in g.output}
keep = [n for n in keep if not (n.op_type == "Constant" and n.output[0] not in used)]
nodes = keep + new; prodf = {o: n for n in nodes for o in n.output}; ordered, seen = [], set()
def visit(n):
    if id(n) in seen: return
    seen.add(id(n))
    for i in n.input:
        if i in prodf: visit(prodf[i])
    ordered.append(n)
for n in nodes: visit(n)
del g.node[:]; g.node.extend(ordered)
op = next((o for o in m.opset_import if o.domain in ("", "ai.onnx")), None)
if op is not None and op.version < 20: op.version = 20
print(f"fused {nf} erf-GELU chains -> Gelu (opset {op.version if op else '?'})"); onnx.save(m, dst)
