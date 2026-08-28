"""Rename HF backbone ONNX I/O to the handoff contract (images / fpn_0 / fpn_1 / fpn_2) and point external data at <out>.data."""
import onnx, sys
src, dst, data_name = sys.argv[1], sys.argv[2], sys.argv[3]
m = onnx.load(src, load_external_data=False); g = m.graph
ren = {g.input[0].name: "images"}
outs = sorted(g.output, key=lambda o: -o.type.tensor_type.shape.dim[-1].dim_value)
for i, o in enumerate(outs): ren[o.name] = f"fpn_{i}"
for n in g.node:
    for k, v in enumerate(n.input):
        if v in ren: n.input[k] = ren[v]
    for k, v in enumerate(n.output):
        if v in ren: n.output[k] = ren[v]
for t in list(g.input) + list(g.output):
    if t.name in ren: t.name = ren[t.name]
for vi in g.value_info:
    if vi.name in ren: vi.name = ren[vi.name]
n_loc = 0
for t in g.initializer:
    for kv in t.external_data:
        if kv.key == "location": kv.value = data_name; n_loc += 1
onnx.save(m, dst); print("renamed", ren, "| external-data locations set:", n_loc)
