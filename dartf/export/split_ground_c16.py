"""Post-graph for prefix sharing from the K-bucket export (dynamo-style names): cut at add_1 (layer-0 residual out) and transpose_2 (pos).
Uses shape inference (with weights loaded, so Reshape targets resolve) for the layout/dtype of the cut tensors, then adapts the K=1 prefix outputs
(x1_in, pos_in: [1, N, C] float32) with Cast / Transpose / Expand.  usage: split_ground_c16.py ground_cK.onnx goldens_dir out_post.onnx"""
import sys, os, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto
src, gdir, out = sys.argv[1:4]; X1, POS = "add_1", "transpose_2"
m = onnx.load(src); g = m.graph; inits = {i.name for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}; ginputs = {i.name for i in g.input}
K = next(d.dim_value for i in g.input if i.name == "img_feat" for d in i.type.tensor_type.shape.dim)
def deps(targets, stop):
    keep, stack, seen = set(), list(targets), set()
    while stack:
        t = stack.pop()
        if t in seen or t in stop or t in inits: continue
        seen.add(t); n = prod.get(t)
        if n is None: continue
        keep.add(id(n)); stack.extend(n.input)
    return [n for n in g.node if id(n) in keep]
pre_nodes = deps([X1, POS], set()); used = {i for n in pre_nodes for i in n.input}
pre = helper.make_model(helper.make_graph(pre_nodes, "pre", [i for i in g.input if i.name in used], [onnx.ValueInfoProto(name=X1), onnx.ValueInfoProto(name=POS)],
                        initializer=[i for i in g.initializer if i.name in used]), opset_imports=m.opset_import, ir_version=m.ir_version)
vi = {v.name: v for v in onnx.shape_inference.infer_shapes(pre).graph.value_info}; types = {n: vi[n].type.tensor_type.elem_type for n in (X1, POS)}
shapes = {n: [d.dim_value for d in vi[n].type.tensor_type.shape.dim] for n in (X1, POS)}; print("bucket cut tensors (inferred):", shapes, types)
assert all(all(d > 0 for d in sh) and len(sh) == 3 for sh in shapes.values()), "unresolved shapes"
N, C = 5184, 256
post_nodes = deps([o_.name for o_ in g.output], {X1, POS, "text_feats", "text_mask"}); used = {i for n in post_nodes for i in n.input}
adapters, extra_inits = [], []
for name, inp in ((X1, "x1_in"), (POS, "pos_in")):
    sh = shapes[name]; cur = inp; t = f"lq_{inp}"
    if types[name] != TensorProto.FLOAT: adapters.append(helper.make_node("Cast", [cur], [t + "_c"], to=types[name])); cur = t + "_c"
    if sh[0] == N and sh[1] in (K, 1):        # [N, B, C] layout in the bucket export
        adapters.append(helper.make_node("Transpose", [cur], [t + "_t"], perm=[1, 0, 2])); cur = t + "_t"; exp = [1, K, 1]
    elif sh[0] in (K, 1) and sh[1] == N: exp = [K, 1, 1]
    else: raise SystemExit(f"unexpected layout {sh}")
    if sh[0] == K or sh[1] == K:
        c = numpy_helper.from_array(np.array(exp, dtype=np.int64), t + "_exp"); extra_inits.append(c); adapters.append(helper.make_node("Expand", [cur, c.name], [name])); cur = name
    else: adapters.append(helper.make_node("Identity", [cur], [name]))
post = helper.make_graph(adapters + post_nodes, "ground_post",
                         [helper.make_tensor_value_info("x1_in", TensorProto.FLOAT, [1, N, C]), helper.make_tensor_value_info("pos_in", TensorProto.FLOAT, [1, N, C])] + [i for i in g.input if i.name in ("text_feats", "text_mask")],
                         list(g.output), initializer=[i for i in g.initializer if i.name in used] + extra_inits)
mp = helper.make_model(post, opset_imports=m.opset_import, ir_version=m.ir_version)   # (checker skipped: old onnx rejects Split(num_outputs) that TensorRT accepts)
onnx.save(mp, out, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(out) + ".data"); print(f"post: {len(post_nodes)} nodes + {len(adapters)} adapters; K={K}; inputs {[i.name for i in post.input]}")
