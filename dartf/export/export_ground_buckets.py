"""Export text_cK / ground_cK ONNX graphs for K in {1,4,8} on CPU using the handoff's own exporter internals
(host_export/source_export.py), bypassing only its frozen K=16 CLI guard. Output: <out>/{text,ground}_cK.onnx (+.data), img_pos_cK.npy"""
import sys, json, argparse
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument("--handoff", required=True); ap.add_argument("--out", required=True); ap.add_argument("--buckets", default="1,4,8"); ap.add_argument("--threads", type=int, default=24); ap.add_argument("--components", default="ground")
ap.add_argument("--spatial", type=int, default=72, help="feature grid (72 for 1008 px, 48 for 672 px)")
a=ap.parse_args(); H=Path(a.handoff); sys.path.insert(0, str(H)); sys.path.insert(0, str(H/"host_export"))
import torch
# CPU shim: the pinned SAM3 source hard-codes device="cuda" in a few constructors; redirect to CPU.
def _cpu(dev): return "cpu" if (dev is not None and str(dev).startswith("cuda")) else dev
for fn in ("arange","zeros","ones","empty","tensor","linspace","full","rand","randn"):
    orig=getattr(torch,fn)
    def make(orig):
        def w(*a,**k):
            if "device" in k: k["device"]=_cpu(k["device"])
            return orig(*a,**k)
        return w
    setattr(torch,fn,make(orig))
torch.Tensor.cuda=lambda self,*a,**k: self; torch.nn.Module.cuda=lambda self,*a,**k: self
_orig_to=torch.Tensor.to
def _to(self,*a,**k):
    a=tuple(_cpu(x) if isinstance(x,(str,torch.device)) else x for x in a)
    if "device" in k: k["device"]=_cpu(k["device"])
    return _orig_to(self,*a,**k)
torch.Tensor.to=_to
_orig_dev=torch.device
_orig_export=torch.onnx.export
def _export(*a,**k):
    k.setdefault("dynamo", False); return _orig_export(*a,**k)
torch.onnx.export=_export
import source_export as se
if a.spatial != 72:
    # re-materialize _export_ground with the requested spatial grid (the pinned exporter hard-codes 72)
    import inspect, re as _re, textwrap
    src = inspect.getsource(se._export_ground); src = _re.sub(r"\b72\b", str(a.spatial), src)
    ns = dict(se.__dict__); exec(textwrap.dedent(src), ns); se._export_ground = ns["_export_ground"]; print("exporter patched to spatial", a.spatial)
torch.set_num_threads(a.threads); torch.manual_seed(314159)
model=se._load_model(H/"third_party"/"sam3", H/"assets"/"sam3.pt")
out=Path(a.out); out.mkdir(parents=True, exist_ok=True); reports=[]
for K in [int(k) for k in a.buckets.split(",")]:
    if "text" in a.components: reports.append(se._export_text(torch, model, out/f"text_c{K}.onnx", K, 17)); print("exported text_c%d"%K, flush=True)
    if "ground" in a.components: reports.append(se._export_ground(torch, model, out/f"ground_c{K}.onnx", K, 17)); print("exported ground_c%d"%K, flush=True)
(out/"export_report.json").write_text(json.dumps({"artifacts":reports}, indent=1, default=str))
