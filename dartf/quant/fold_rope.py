"""Fold the RoPE pairwise rotation into the q/k projections (exact graph rewrite, works on FP32/FP16 and Q/DQ INT8 HF graphs).

HF SAM3 RoPE: q_rot = q*cos + rotate_pairwise(q)*sin, rotate_pairwise(x)[2i] = -x[2i+1], [2i+1] = x[2i] (per head).
rotate_pairwise is linear: rot(q) = q @ R with R = kron(I_heads, R64). Since q = x @ W + b, rot(q) = x @ (W R) + b R,
and W R is a signed column permutation of W (so INT8 codes and per-channel scales are permuted exactly).
The pass replaces the Reshape/Slice/Squeeze/Neg/Unsqueeze/Concat/Reshape chain feeding Mul(sin) by a second projection
MatMul(x, W R) (+ DQ for quantized weights) + Add(b R) + the same Reshape/Transpose as q, then Mul(sin).
usage: fold_rope.py in.onnx out.onnx   (external data is preserved via save_as_external_data when in.onnx.data exists)
"""
import sys, os, onnx, numpy as np
from onnx import numpy_helper, helper
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph
inits = {i.name: i for i in g.initializer}; prod = {}; cons = {}
for n in g.node:
    for o in n.output: prod[o] = n
    for i in n.input: cons.setdefault(i, []).append(n)
def is_const(x): return x in inits
def arr(x): return numpy_helper.to_array(inits[x])

# rotary tables: rope_embeddings_cos / _sin initializers [N, 64]
sin_names = set(k for k in inits if k.endswith("rope_embeddings_sin")); assert sin_names
head_dim = int(inits[next(iter(sin_names))].dims[-1])
R64 = np.zeros((head_dim, head_dim), np.float32)
for i in range(head_dim // 2): R64[2 * i + 1, 2 * i] = -1.0; R64[2 * i, 2 * i + 1] = 1.0   # (x @ R)[2i] = -x[2i+1], [2i+1] = x[2i]

def walk_back(t, ops):
    """follow single-input chain backwards while producers are in ops; returns list of nodes (nearest first)"""
    out = []
    while t in prod and prod[t].op_type in ops:
        n = prod[t]; out.append(n); t = n.input[0]
    return out, t

new_nodes = []; kill = set(); n_fold = 0; new_inits = []
for mul_sin in [n for n in g.node if n.op_type == "Mul" and any(i in sin_names for i in n.input)]:
    rot_in = [i for i in mul_sin.input if i not in sin_names][0]
    # rotate chain: Reshape(flatten) <- Concat <- Unsqueeze(x1) / Unsqueeze(Neg(x2)) <- Squeeze <- Slice <- Reshape(view) <- T
    flat = prod[rot_in]; assert flat.op_type == "Reshape", flat.op_type
    cat = prod[flat.input[0]]; assert cat.op_type == "Concat"
    chain = {flat.name, cat.name}; view = None
    for ci in cat.input:
        n = prod[ci]; chain.add(n.name)                       # Unsqueeze
        n = prod[n.input[0]]; chain.add(n.name)
        if n.op_type == "Neg": n = prod[n.input[0]]; chain.add(n.name)
        assert n.op_type == "Squeeze", n.op_type
        n = prod[n.input[0]]; assert n.op_type == "Slice"; chain.add(n.name)
        v = prod[n.input[0]]; assert v.op_type == "Reshape"; chain.add(v.name); view = v
    T = prod[view.input[0]]; assert T.op_type == "Transpose", T.op_type
    # the q (or k) tensor T feeds: Mul(cos) and the view; the projection is Reshape <- Add(bias) <- MatMul
    V = prod[T.input[0]]; assert V.op_type == "Reshape"
    addb = prod[V.input[0]]; assert addb.op_type == "Add"
    mm_out = [i for i in addb.input if not is_const(i)][0]; b_name = [i for i in addb.input if is_const(i)][0]
    mm = prod[mm_out]; assert mm.op_type == "MatMul"
    x_in, w_in = mm.input[0], mm.input[1]
    n_out = int(inits[b_name].dims[0]); heads = n_out // head_dim; Rb = np.kron(np.eye(heads, dtype=np.float32), R64)   # [n_out, n_out]
    tag = f"rope{n_fold}"
    # ---- weight (plain initializer or DequantizeLinear(int8 codes, per-channel scale, axis=1)) ----
    if is_const(w_in):
        W = arr(w_in); WR = (W.astype(np.float32) @ Rb).astype(W.dtype); wr_name = w_in + "_R"; new_inits.append(numpy_helper.from_array(WR, wr_name))
        w_feed = wr_name
    else:
        dq = prod[w_in]; assert dq.op_type == "DequantizeLinear", dq.op_type
        codes = arr(dq.input[0]); sc = arr(dq.input[1]); assert codes.ndim == 2 and codes.shape[1] == n_out
        # column permutation with sign: column j of W R = sum_i W[:, i] Rb[i, j]; Rb has exactly one nonzero per column
        perm = np.argmax(np.abs(Rb), axis=0); sign = Rb[perm, np.arange(n_out)]
        codesR = (codes[:, perm].astype(np.int16) * sign.astype(np.int16)).astype(codes.dtype); scR = sc[perm] if sc.ndim == 1 else sc
        cr, sr = dq.input[0] + "_R", dq.input[1] + "_R"
        new_inits += [numpy_helper.from_array(codesR, cr), numpy_helper.from_array(scR, sr)]
        dqn = helper.make_node("DequantizeLinear", [cr, sr] + list(dq.input[2:]), [w_in + "_R"], name=dq.name + "_R")
        for a in dq.attribute: dqn.attribute.append(a)
        new_nodes.append(dqn); w_feed = w_in + "_R"
    b = arr(b_name); bR = (b.astype(np.float32) @ Rb).astype(b.dtype); new_inits.append(numpy_helper.from_array(bR, b_name + "_R"))
    mmR = helper.make_node("MatMul", [x_in, w_feed], [mm_out + "_R"], name=mm.name + "_R")
    addR = helper.make_node("Add", [mm_out + "_R", b_name + "_R"], [addb.output[0] + "_R"], name=addb.name + "_R")
    VR = helper.make_node("Reshape", [addb.output[0] + "_R", V.input[1]], [V.output[0] + "_R"], name=V.name + "_R")
    TR = helper.make_node("Transpose", [V.output[0] + "_R"], [T.output[0] + "_R"], name=T.name + "_R")
    for a in T.attribute: TR.attribute.append(a)
    new_nodes += [mmR, addR, VR, TR]
    # rewire Mul(sin) to consume the rotated projection directly
    mul_sin.input[list(mul_sin.input).index(rot_in)] = T.output[0] + "_R"
    kill |= chain; n_fold += 1
# remove the dead rotate chains (only if nothing else consumes their outputs)
keep = []
for n in g.node:
    if n.name in kill: continue
    keep.append(n)
g.ClearField("node"); g.node.extend(new_nodes + keep); g.initializer.extend(new_inits)
# topological order: new MatMul/Add/Reshape/Transpose only depend on existing tensors, but Mul(sin) precedes them in the list -> sort
from collections import defaultdict
nodes = list(g.node); produced = {o: n for n in nodes for o in n.output}; ordered = []; done = set(); state = {}
def visit(n):
    if id(n) in done: return
    done.add(id(n))
    for i in n.input:
        if i in produced: visit(produced[i])
    ordered.append(n)
for n in nodes: visit(n)
g.ClearField("node"); g.node.extend(ordered)
print(f"folded {n_fold} RoPE chains ({len(kill)} nodes removed, {len(new_nodes)} added)")
ext = os.path.exists(src + ".data") or any(i.data_location == 1 for i in g.initializer)
if ext: onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(dst) + ".data", size_threshold=1024)
else: onnx.save(m, dst)
