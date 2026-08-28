"""K2 splice: in a quantized + RoPE-folded HF graph, replace each q (and k) projection pair
   DQ(Wq)·MatMul + Add(b) -> Reshape -> Transpose -> Mul(cos)      and      DQ(WqR)·MatMul + Add(bR) -> Reshape -> Transpose -> Mul(sin)   -> Add
by  LQGemmRope(Q(x)) -> Reshape -> Transpose   (the plugin emits the rotated projection in token-major [.., 1024]).
Attributes: codes [N,K] int8 (of the un-rotated projection), wscale, bias, and the layer's cos/sin tables [Ntok,64].
usage: splice_rope_plugin.py in.onnx out.onnx
"""
import sys, os, onnx, numpy as np
from onnx import numpy_helper, helper
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph; inits = {i.name: i for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}; cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
sin_names = {k for k in inits if k.endswith("rope_embeddings_sin")}; cos_names = {k for k in inits if k.endswith("rope_embeddings_cos")}
def chain_back(t, ops):
    out = []
    for op in ops:
        n = prod.get(t)
        if n is None or n.op_type != op: return None
        out.append(n); t = n.input[0]
    return out
n_spliced = 0; kill = set(); new_nodes = []
for mul_sin in [n for n in g.node if n.op_type == "Mul" and any(i in sin_names for i in n.input)]:
    sin_name = [i for i in mul_sin.input if i in sin_names][0]; cos_name = sin_name[:-3] + "cos"
    add_out = cons[mul_sin.output[0]][0]                                          # Add(mul_cos, mul_sin) -> rotated q/k
    if add_out.op_type != "Add": continue
    mul_cos = prod[[i for i in add_out.input if i != mul_sin.output[0]][0]]
    if mul_cos.op_type != "Mul" or not any(i in cos_names for i in mul_cos.input): continue
    # rotated branch: Mul(sin) <- Transpose <- Reshape <- Add(bR) <- MatMul(x, DQ(WR))
    rb = chain_back([i for i in mul_sin.input if i != sin_name][0], ["Transpose", "Reshape", "Add", "MatMul"])
    ub = chain_back([i for i in mul_cos.input if i not in cos_names][0], ["Transpose", "Reshape", "Add", "MatMul"])
    if rb is None or ub is None: continue
    T, R, addb, mm = ub; TR, RR, addbR, mmR = rb
    xdq = prod.get(mm.input[0]); wdq = prod.get(mm.input[1])
    if xdq is None or xdq.op_type != "DequantizeLinear" or wdq is None or wdq.op_type != "DequantizeLinear": continue
    codes = numpy_helper.to_array(inits[wdq.input[0]]); wscale = numpy_helper.to_array(inits[wdq.input[1]]).astype(np.float32)
    if codes.dtype != np.int8 or codes.shape != (1024, 1024): continue
    bname = addb.input[1] if addb.input[1] in inits else addb.input[0]; bias = numpy_helper.to_array(inits[bname]).astype(np.float32)
    cosT = numpy_helper.to_array(inits[cos_name]).astype(np.float32); sinT = numpy_helper.to_array(inits[sin_name]).astype(np.float32)   # [Ntok, 64]
    tag = f"lqrope_{n_spliced}"; ascale = float(numpy_helper.to_array(inits[prod[xdq.input[0]].input[1]]))
    plug = helper.make_node("LQGemmRope", [xdq.input[0]], [tag + "_out"], name=tag, domain="lq", ascale=ascale,
                            codes=numpy_helper.from_array(np.ascontiguousarray(codes.T), tag + "_codes"), wscale=numpy_helper.from_array(wscale, tag + "_ws"),
                            bias=numpy_helper.from_array(bias, tag + "_b"), cos=numpy_helper.from_array(cosT, tag + "_cos"), sin=numpy_helper.from_array(sinT, tag + "_sin"))
    resh = helper.make_node("Reshape", [tag + "_out", R.input[1]], [tag + "_r"], name=tag + "_reshape")
    tr = helper.make_node("Transpose", [tag + "_r"], [add_out.output[0]], name=tag + "_transpose")
    for a in T.attribute: tr.attribute.append(a)
    new_nodes += [plug, resh, tr]
    kill |= {n.name for n in (mul_sin, mul_cos, add_out, T, R, addb, mm, TR, RR, addbR, mmR, wdq)}
    # the x DQ node may feed v's MatMul too — keep it; drop the WR DQ
    wdqR = prod.get(mmR.input[1]); kill.add(wdqR.name)
    n_spliced += 1
keep = [n for n in g.node if n.name not in kill]
# drop DQ(x) nodes that no longer have consumers
used = {i for n in keep for i in n.input} | {i for n in new_nodes for i in n.input} | {o.name for o in g.output}
keep = [n for n in keep if not (n.op_type == "DequantizeLinear" and n.output[0] not in used)]
pending = {n.output[0]: n for n in new_nodes}; final = []
for n in keep:
    for i in n.input:
        if i in pending: final.append(pending.pop(i))
        # reshape/transpose chains: emit their producers first
        j = i
        while j in pending: final.append(pending.pop(j)); j = None
    final.append(n)
final += list(pending.values())
# simple topological fix-up (plugin -> reshape -> transpose ordering)
prodf = {o: n for n in final for o in n.output}; ordered = []; done = set()
def visit(n):
    if id(n) in done: return
    done.add(id(n))
    for i in n.input:
        if i in prodf: visit(prodf[i])
    ordered.append(n)
for n in final: visit(n)
del g.node[:]; g.node.extend(ordered)
if not any(o.domain == "lq" for o in m.opset_import): m.opset_import.append(helper.make_opsetid("lq", 1))
print(f"spliced {n_spliced} q/k projections -> LQGemmRope")
if os.path.exists(src + ".data"): onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(dst) + ".data", size_threshold=1024)
else: onnx.save(m, dst)
