"""Explicit INT8 (Q/DQ) for the mask head's pixel-decoder convolutions: per-channel symmetric weights, per-tensor static activations
calibrated on dumped inputs (p99.99). Everything else stays FP16. usage: python quant_maskhead.py in.onnx calib.npz out.onnx [conv_name ...]"""
import sys, numpy as np, onnx, onnx_graphsurgeon as gs, onnxruntime as ort
import re
src, calib, dst = sys.argv[1], sys.argv[2], sys.argv[3]; pat = sys.argv[4] if len(sys.argv) > 4 else r"pixel_decoder.*Conv|instance_seg_head.*Conv"
m = onnx.load(src); g = gs.import_onnx(m); targets = [n.name for n in g.nodes if n.op == "Conv" and re.search(pat, n.name)]; convs = {n.name: n for n in g.nodes if n.name in targets}; print("quantizing", targets)
act_names = [convs[t].inputs[0].name for t in targets]
# ---- activation statistics with onnxruntime (intermediate tensors exposed as outputs) ----
m2 = onnx.load(src)
for nm in act_names: m2.graph.output.append(onnx.helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, None))
sess = ort.InferenceSession(m2.SerializeToString(), providers=["CPUExecutionProvider"]); z = np.load(calib); n = int(z["n"]); amax = {nm: [] for nm in act_names}
for i in range(n):
    feeds = {k: z[f"{k}_{i}"] for k in ("fpn_0", "fpn_1", "enc", "text_feats", "text_mask", "hs_sel")}
    outs = sess.run(act_names, feeds)
    for nm, o in zip(act_names, outs): amax[nm].append(np.percentile(np.abs(o.astype(np.float32)), 99.99))
scales = {nm: float(np.max(v)) / 127.0 for nm, v in amax.items()}; print("activation scales:", {k: round(v, 5) for k, v in scales.items()})
# ---- insert Q/DQ ----
def qdq(tensor, scale, axis=None, name=""):
    s = gs.Constant(name + "_s", np.array(scale, np.float32)); zp = gs.Constant(name + "_zp", np.zeros(np.shape(scale), np.int8) if axis is not None else np.array(0, np.int8))
    q = gs.Variable(name + "_q", dtype=np.int8); dq = gs.Variable(name + "_dq", dtype=np.float32)
    attrs = {"axis": axis} if axis is not None else {}
    g.nodes.append(gs.Node("QuantizeLinear", name + "_Q", inputs=[tensor, s, zp], outputs=[q], attrs=attrs)); g.nodes.append(gs.Node("DequantizeLinear", name + "_DQ", inputs=[q, s, zp], outputs=[dq], attrs=attrs)); return dq
for t in targets:
    node = convs[t]; x, w = node.inputs[0], node.inputs[1]; W = np.asarray(w.values, np.float32); ws = np.maximum(np.abs(W).reshape(W.shape[0], -1).max(1), 1e-8) / 127.0
    node.inputs[0] = qdq(x, scales[x.name], name=t.replace("/", "_") + "_x"); node.inputs[1] = qdq(w, ws, axis=0, name=t.replace("/", "_") + "_w")
g.cleanup().toposort(); mo = gs.export_onnx(g); mo.opset_import[0].version = max(mo.opset_import[0].version, 13); onnx.save(mo, dst); print("wrote", dst)
