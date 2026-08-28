"""K5 splice (block-level MLP): in a quantized Hadamard-basis (or plain-LN) HF graph, replace
   r -> [masked-RMSNorm | LayerNormalization] -> Q -> DQ -> MatMul(fc1) -> Add(b1) -> Gelu -> Q -> DQ -> MatMul(fc2) -> Add(b2) -> Add(r)  = out
by  LQMlp(r) = out.  All INT8 work (LN+quantize, fc1+GELU+requantize, fc2+bias+residual) happens inside the plugin.
usage: splice_mlp_plugin.py in.onnx out.onnx [--gelu erf|tanh]
"""
import sys, os, argparse, onnx, numpy as np
from onnx import numpy_helper, helper
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--gelu", default="erf"); a = ap.parse_args()
m = onnx.load(a.src); g = m.graph; inits = {i.name: i for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}; cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
arr = lambda name: numpy_helper.to_array(inits[name])
def const_in(n): return next((i for i in n.input if i in inits), None)
def only_consumer(t, op): c = [x for x in cons.get(t, []) if x.op_type == op]; return c[0] if len(cons.get(t, [])) == 1 and c else None
n_spliced = 0; kill = set(); new_nodes = []
for mm1 in [n for n in g.node if n.op_type == "MatMul"]:
    wdq1 = prod.get(mm1.input[1]); xdq1 = prod.get(mm1.input[0])
    if wdq1 is None or wdq1.op_type != "DequantizeLinear" or xdq1 is None or xdq1.op_type != "DequantizeLinear": continue
    codes1 = arr(wdq1.input[0])
    if codes1.dtype != np.int8 or codes1.shape != (1024, 4736): continue
    q1 = prod.get(xdq1.input[0])
    if q1 is None or q1.op_type != "QuantizeLinear": continue
    # ---- forward chain: Add(b1) -> Gelu -> Q -> DQ -> MatMul(fc2) -> Add(b2) -> Add(residual) ----
    add1 = only_consumer(mm1.output[0], "Add"); gelu = add1 and only_consumer(add1.output[0], "Gelu")
    chan_mul = gelu and only_consumer(gelu.output[0], "Mul")                       # optional per-channel activation scale (fc2 SmoothQuant): Gelu -> Mul(1/c) -> Q
    q2 = (chan_mul and only_consumer(chan_mul.output[0], "QuantizeLinear")) or (gelu and only_consumer(gelu.output[0], "QuantizeLinear"))
    dq2 = q2 and only_consumer(q2.output[0], "DequantizeLinear"); mm2 = dq2 and only_consumer(dq2.output[0], "MatMul")
    if not mm2: continue
    wdq2 = prod.get(mm2.input[1]); codes2 = arr(wdq2.input[0]) if wdq2 is not None and wdq2.op_type == "DequantizeLinear" else None
    if codes2 is None or codes2.shape != (4736, 1024): continue
    add2 = only_consumer(mm2.output[0], "Add"); addr = add2 and only_consumer(add2.output[0], "Add")
    if not addr: continue
    r = [i for i in addr.input if i != add2.output[0]][0]                       # residual stream tensor
    # ---- backward chain from Q1 input to the norm: masked RMSNorm (Mul(mask) ... Mul) or LayerNormalization ----
    normout = q1.input[0]; nn = prod.get(normout); mode = None; gamma = beta = None; eps = 1e-6; chain = []
    if nn is not None and nn.op_type == "LayerNormalization":
        mode = 0; gamma = arr(nn.input[1]).astype(np.float32); beta = arr(nn.input[2]).astype(np.float32); eps = float(next((at.f for at in nn.attribute if at.name == "epsilon"), 1e-6)); chain = [nn]
        if nn.input[0] != r: continue
    elif nn is not None and nn.op_type == "Mul":                                  # xm * rsqrt(var+eps): inputs (xm, Reciprocal(Sqrt(Add(Div(ReduceSum(Mul(xm,xm)), C), eps))))
        mode = 1; ins = [prod.get(i) for i in nn.input]
        rcp = next((x for x in ins if x is not None and x.op_type == "Reciprocal"), None); xm_node = next((x for x in ins if x is not None and x.op_type == "Mul"), None)
        if rcp is None or xm_node is None: continue
        sq = prod.get(rcp.input[0]); adde = prod.get(sq.input[0]) if sq is not None else None
        if sq is None or sq.op_type != "Sqrt" or adde is None or adde.op_type != "Add": continue
        eps = float(arr(const_in(adde)).reshape(-1)[0]); div = prod.get([i for i in adde.input if i not in inits][0]); rs = prod.get(div.input[0]); sqm = prod.get(rs.input[0])
        if div.op_type != "Div" or rs.op_type != "ReduceSum" or sqm.op_type != "Mul": continue
        if xm_node.input[0] != r and xm_node.input[1] != r: continue                # Mul(r, mask)
        chain = [nn, rcp, sq, adde, div, rs, sqm, xm_node]
    else: continue
    s1 = float(arr(q1.input[1])); s2 = float(arr(q2.input[1]))
    ws1 = arr(wdq1.input[1]).astype(np.float32); ws2 = arr(wdq2.input[1]).astype(np.float32)
    b1 = arr(const_in(add1)).astype(np.float32); b2 = arr(const_in(add2)).astype(np.float32)
    tag = f"lqmlp_{n_spliced}"
    extra = {}
    if chan_mul is not None:
        inv_c = arr(const_in(chan_mul)).astype(np.float32).reshape(-1); assert inv_c.shape == (4736,), inv_c.shape
        extra["chan"] = numpy_helper.from_array((inv_c / s2).astype(np.float32), tag + "_chan"); kill.add(chan_mul.name)
    node = helper.make_node("LQMlp", [r], [addr.output[0]], name=tag, domain="lq", mode=mode, gelu=1 if a.gelu == "tanh" else 0, eps=eps, s1=s1, s2=s2, **extra,
                            gamma=numpy_helper.from_array(gamma if gamma is not None else np.zeros(0, np.float32), tag + "_g"), beta=numpy_helper.from_array(beta if beta is not None else np.zeros(0, np.float32), tag + "_b"),
                            codes1=numpy_helper.from_array(np.ascontiguousarray(codes1.T), tag + "_c1"), ws1=numpy_helper.from_array(ws1, tag + "_ws1"), b1=numpy_helper.from_array(b1, tag + "_b1"),
                            codes2=numpy_helper.from_array(np.ascontiguousarray(codes2.T), tag + "_c2"), ws2=numpy_helper.from_array(ws2, tag + "_ws2"), b2=numpy_helper.from_array(b2, tag + "_b2"))
    new_nodes.append(node)
    kill |= {x.name for x in chain} | {q1.name, xdq1.name, wdq1.name, mm1.name, add1.name, gelu.name, q2.name, dq2.name, wdq2.name, mm2.name, add2.name, addr.name}
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
print(f"spliced {n_spliced} MLP blocks -> LQMlp ({a.gelu})")
if os.path.exists(a.src + ".data"): onnx.save(m, a.dst, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(a.dst) + ".data", size_threshold=1024)
else: onnx.save(m, a.dst)
