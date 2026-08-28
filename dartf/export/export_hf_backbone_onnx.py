"""ONNX-only HF SAM3 backbone export using DART's export_onnx() (no TensorRT on the host).
usage: export_hf_backbone_onnx.py <dart_root> <out_dir> <imgsz> [pruned_checkpoint]"""
import sys, os
dart, out, imgsz = sys.argv[1], sys.argv[2], int(sys.argv[3]); pruned = sys.argv[4] if len(sys.argv)>4 else None
sys.path.insert(0, dart); sys.path.insert(0, os.path.join(dart,"scripts"))
import torch
# minimal CPU shim: only the Meta decoder's torch.arange(device="cuda") and the sine-PE precompute need redirecting
_orig_arange=torch.arange
def _arange(*a,**k):
    if str(k.get("device","")).startswith("cuda"): k["device"]="cpu"
    return _orig_arange(*a,**k)
torch.arange=_arange
try:
    sys.path.insert(0, dart)
    from sam3.model.position_encoding import PositionEmbeddingSine as _PES
    _oi=_PES.__init__
    def _ci(self,*a,**k): k["precompute_resolution"]=None; _oi(self,*a,**k)
    _PES.__init__=_ci
except Exception as e: print("PES patch skipped:", e)
import export_hf_backbone as E
os.makedirs(out, exist_ok=True)
res = E.export_onnx(out, imgsz=imgsz, pruned_checkpoint=pruned)
print("export_onnx returned:", res)
