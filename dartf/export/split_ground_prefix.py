"""Exact prompt-independent prefix sharing for the SAM 3 grounding head.
Encoder layer 0's self-attention block (LayerNorm, +pos, q/k/v, attention, out-proj, residual) depends only on the image, so for K prompts
it is computed once instead of K times:  pre.onnx : img_feat, img_pos (B=1) -> x1 [N,1,C], pos [N,1,C]
                                          post.onnx: x1 [N,1,C] (Expand -> [N,K,C]), pos [N,1,C] (Expand), text_feats, text_mask -> scores, boxes, presence
usage: split_ground_prefix.py ground_c1.onnx ground_cK.onnx out_pre.onnx out_post.onnx"""
import sys, onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto
c1, cK, out_pre, out_post = sys.argv[1:5]
X1, POS = "/encoder/layers.0/Add_1_output_0", "/encoder/Concat_3_output_0"
def deps(g, targets, stop):
    inits = {i.name for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}; keep = set(); stack = list(targets); seen = set()
    while stack:
        t = stack.pop()
        if t in seen or t in stop or t in inits: continue
        seen.add(t); n = prod.get(t)
        if n is None: continue
        keep.add(id(n)); stack.extend(n.input)
    return [n for n in g.node if id(n) in keep]
# ---- prefix from the K=1 graph
m1 = onnx.load(c1); g1 = m1.graph; pre_nodes = deps(g1, [X1, POS], set())
used = {i for n in pre_nodes for i in n.input}
pre = helper.make_graph(pre_nodes, "ground_pre", [i for i in g1.input if i.name in ("img_feat", "img_pos")], [onnx.ValueInfoProto(name=X1), onnx.ValueInfoProto(name=POS)],
                        initializer=[i for i in g1.initializer if i.name in used])
mp = helper.make_model(pre, opset_imports=m1.opset_import, ir_version=m1.ir_version)
vi = {v.name: v for v in onnx.shape_inference.infer_shapes(mp).graph.value_info}          # types (x1 fp16, pos fp32) and shapes from inference
cut = {n: (vi[n].type.tensor_type.elem_type, [d.dim_value for d in vi[n].type.tensor_type.shape.dim]) for n in (X1, POS)}
del pre.output[:]; pre.output.extend([helper.make_tensor_value_info(n, cut[n][0], cut[n][1]) for n in (X1, POS)])
onnx.checker.check_model(mp); onnx.save(mp, out_pre); print(f"pre: {len(pre_nodes)} nodes, inputs {[i.name for i in pre.input]}, cut {cut}")
# ---- post from the K-bucket graph: cut at X1/POS, feed via Expand from batch 1
mK = onnx.load(cK); gK = mK.graph; K = next(d.dim_value for i in gK.input if i.name == "img_feat" for d in i.type.tensor_type.shape.dim); print("K =", K)
outs = [o.name for o in gK.output]; post_nodes = deps(gK, outs, {X1, POS, "text_feats", "text_mask"})
used = {i for n in post_nodes for i in n.input}
# x1/pos layout in the bucket graph: infer from the K=1 graph's value_info if present, else assume [N, B, C] (seq-major encoder)
shape_c = numpy_helper.from_array(np.array([1, K, 1], dtype=np.int64), "lq_expandK"); 
exp_nodes = [helper.make_node("Expand", ["x1_in", "lq_expandK"], [X1], name="lq_expand_x1"), helper.make_node("Expand", ["pos_in", "lq_expandK"], [POS], name="lq_expand_pos")]
post = helper.make_graph(exp_nodes + post_nodes, "ground_post",
                         [helper.make_tensor_value_info("x1_in", cut[X1][0], cut[X1][1]), helper.make_tensor_value_info("pos_in", cut[POS][0], cut[POS][1])] + [i for i in gK.input if i.name in ("text_feats", "text_mask")],
                         list(gK.output), initializer=[i for i in gK.initializer if i.name in used] + [shape_c])
assert cut[X1][1][1] == 1, "Expand assumes the batch axis at dim 1 ([N, B, C])"
mpost = helper.make_model(post, opset_imports=mK.opset_import, ir_version=mK.ir_version); onnx.checker.check_model(mpost); onnx.save(mpost, out_post); print(f"post: {len(post_nodes)} nodes, inputs {[i.name for i in post.input]}")
