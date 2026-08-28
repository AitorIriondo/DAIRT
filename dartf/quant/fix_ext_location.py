"""Point every external-data initializer of <onnx> at <data_name> (no data rewrite). usage: fix_ext_location.py model.onnx model.onnx.data"""
import sys, onnx
from onnx.external_data_helper import ExternalDataInfo
m = onnx.load(sys.argv[1], load_external_data=False); n = 0
for t in m.graph.initializer:
    if t.HasField("data_location") and t.data_location == onnx.TensorProto.EXTERNAL:
        for kv in t.external_data:
            if kv.key == "location": kv.value = sys.argv[2]; n += 1
onnx.save(m, sys.argv[1]); print("external-data locations set:", n)
