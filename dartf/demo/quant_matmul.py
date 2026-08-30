"""Calibrated Q/DQ on MatMul nodes with initializer weights (per-tensor activations p99.99 from an npz of real inputs, per-channel weights).
usage: python quant_matmul.py in.onnx calib.npz out.onnx "regex over node names"  (calib.npz: input_name -> array, single sample)"""
import sys, re, numpy as np, onnx, onnx_graphsurgeon as gs, onnxruntime as ort
src, calib, dst, pat = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
m = onnx.load(src); g = gs.import_onnx(m)
targets = [n for n in g.nodes if n.op == "MatMul" and re.search(pat, n.name) and isinstance(n.inputs[1], gs.Constant)]; print("quantizing", len(targets), "MatMuls:", [n.name for n in targets][:6], "...")
act_names = sorted({n.inputs[0].name for n in targets})
m2 = onnx.load(src)
for nm in act_names: m2.graph.output.append(onnx.helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, None))
sess = ort.InferenceSession(m2.SerializeToString(), providers=["CPUExecutionProvider"]); z = np.load(calib); names = [i.name for i in sess.get_inputs()]
feeds = {k: z[k] for k in names}; outs = sess.run(act_names, feeds); scales = {nm: float(np.percentile(np.abs(o), 99.99)) / 127.0 for nm, o in zip(act_names, outs)}
def qdq(tensor, scale, axis=None, name=""):
    s = gs.Constant(name + "_s", np.array(scale, np.float32)); zp = gs.Constant(name + "_zp", np.zeros(np.shape(scale), np.int8) if axis is not None else np.array(0, np.int8))
    q = gs.Variable(name + "_q", dtype=np.int8); dq = gs.Variable(name + "_dq", dtype=np.float32); attrs = {"axis": axis} if axis is not None else {}
    g.nodes.append(gs.Node("QuantizeLinear", name + "_Q", inputs=[tensor, s, zp], outputs=[q], attrs=attrs)); g.nodes.append(gs.Node("DequantizeLinear", name + "_DQ", inputs=[q, s, zp], outputs=[dq], attrs=attrs)); return dq
done_act = {}
for n in targets:
    x, w = n.inputs[0], n.inputs[1]; W = np.asarray(w.values, np.float32); ws = np.maximum(np.abs(W).max(0), 1e-8) / 127.0     # W [K, N]: per output column
    if x.name not in done_act: done_act[x.name] = qdq(x, scales[x.name], name=n.name.replace("/", "_") + "_x")
    n.inputs[0] = done_act[x.name]; n.inputs[1] = qdq(w, ws, axis=1, name=n.name.replace("/", "_") + "_w")
g.cleanup().toposort(); mo = gs.export_onnx(g); onnx.save(mo, dst); print("wrote", dst, "| activation scales", {k[:40]: round(v, 5) for k, v in list(scales.items())[:4]})
