"""Export ground_cKm.onnx (grounding head that also emits the decoder queries `hs` [K,200,256] and encoder states `enc` [5184,K,256])
and maskhead_qQ.onnx (the SAM 3 segmentation head with a dynamic prompt axis K). CPU, a few seconds; needs the DART sam3 package and sam3.pt.
usage: python export_ground_mask.py --dart <DART root> --ckpt sam3.pt --out out/head [--buckets 4,1] [--qsel 32]"""
import sys, os, json, time, argparse
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--dart", required=True, help="DART repository root (its sam3 package is imported)"); ap.add_argument("--ckpt", required=True, help="official sam3.pt")
ap.add_argument("--out", required=True); ap.add_argument("--buckets", default="4,1", help="prompt buckets K to export"); ap.add_argument("--qsel", type=int, default=32, help="queries per prompt fed to the mask head")
ap.add_argument("--threads", type=int, default=16); args = ap.parse_args()
DART = args.dart; CKPT = args.ckpt; OUT = Path(args.out); BUCKETS = [int(k) for k in args.buckets.split(",")]; QSEL = args.qsel
sys.path.insert(0, DART)
import torch
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
torch.set_num_threads(args.threads); torch.manual_seed(314159)
from sam3.model.position_encoding import PositionEmbeddingSine
_oi = PositionEmbeddingSine.__init__
def _ci(self, *a, **k): k["precompute_resolution"] = None; _oi(self, *a, **k)
PositionEmbeddingSine.__init__ = _ci
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import inverse_sigmoid
t0 = time.time()
model = build_sam3_image_model(checkpoint_path=CKPT, device="cpu", eval_mode=True, load_from_HF=False, enable_segmentation=True, enable_inst_interactivity=False)
PositionEmbeddingSine.__init__ = _oi
print("model loaded %.0fs" % (time.time() - t0), flush=True)
seg = model.segmentation_head; print("seg head:", type(seg).__name__, "cross_attend:", seg.cross_attend_prompt is not None, "act_ckpt:", seg.act_ckpt)
seg.act_ckpt = False

class GroundingWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder, scoring):
        super().__init__(); self.encoder = encoder; self.decoder = decoder; self.scoring = scoring
    def forward(self, img_feat, img_pos, text_feats, text_mask):
        img_feat = img_feat.to(torch.float32); img_pos = img_pos.to(torch.float32); text_feats = text_feats.to(torch.float32); tm = text_mask.to(torch.bool)
        img_seq = img_feat.flatten(2).permute(2, 0, 1); pos_seq = img_pos.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src=[img_seq], src_key_padding_mask=None, src_pos=[pos_seq], prompt=text_feats, prompt_key_padding_mask=tm, feat_sizes=[(72, 72)])
        query_embed = self.decoder.query_embed.weight; target = query_embed.unsqueeze(1).expand(-1, img_feat.shape[0], -1)
        hidden, reference_boxes, presence, _ = self.decoder(tgt=target, memory=memory["memory"], memory_key_padding_mask=memory["padding_mask"], pos=memory["pos_embed"],
            reference_boxes=None, level_start_index=memory["level_start_index"], spatial_shapes=memory["spatial_shapes"], valid_ratios=memory["valid_ratios"],
            tgt_mask=None, memory_text=text_feats, text_attention_mask=tm, apply_dac=False)
        hidden = hidden.transpose(1, 2); reference_boxes = reference_boxes.transpose(1, 2); hidden_last = hidden[-1:]
        scores = self.scoring(hidden_last, text_feats, tm)[0]; box_offsets = self.decoder.bbox_embed(hidden_last)[0]
        boxes = (inverse_sigmoid(reference_boxes[-1]) + box_offsets).sigmoid(); presence_last = presence[-1].permute(1, 0)
        return scores.to(torch.float16), boxes.to(torch.float16), presence_last.to(torch.float16), hidden_last[0].to(torch.float16), memory["memory"].to(torch.float16)

class MaskWrapper(torch.nn.Module):
    """fpn_0 [1,256,288,288] fp32, fpn_1 [1,256,144,144] fp32, enc [5184,K,256] fp16, text_feats [32,K,256] fp16, text_mask [K,32] fp32, hs_sel [K,Q,256] fp16 -> masks [K,Q,288,288] fp16"""
    def __init__(self, seg): super().__init__(); self.seg = seg
    def forward(self, fpn_0, fpn_1, enc, text_feats, text_mask, hs_sel):
        enc = enc.to(torch.float32); tf = text_feats.to(torch.float32); tm = text_mask.to(torch.bool); hs = hs_sel.to(torch.float32)
        K = enc.shape[1]; fpn_2 = torch.zeros((1, 256, 72, 72), dtype=torch.float32)
        out = self.seg(backbone_feats=[fpn_0.to(torch.float32), fpn_1.to(torch.float32), fpn_2], obj_queries=hs.unsqueeze(0), image_ids=torch.zeros((K,), dtype=torch.long),
                       encoder_hidden_states=enc, prompt=tf, prompt_mask=tm)
        return out["pred_masks"].to(torch.float16)

decoder = model.transformer.decoder; decoder.compile_mode = None; decoder.compiled = True
ch, cw = decoder._get_coords(72, 72, device="cpu"); decoder.compilable_cord_cache = (ch, cw); decoder.compilable_stored_size = (72, 72)
wrapper = GroundingWrapper(model.transformer.encoder, decoder, model.dot_prod_scoring).cpu().eval()
position_encoding = model.backbone.vision_backbone.position_encoding
OUT.mkdir(parents=True, exist_ok=True)
with torch.inference_mode(): img_pos1 = position_encoding(torch.zeros((1, 256, 72, 72), dtype=torch.float32))
import numpy as np
for K in BUCKETS:
    img_pos = img_pos1.repeat(K, 1, 1, 1).to(torch.float16).contiguous(); np.save(OUT / f"img_pos_c{K}.npy", img_pos.numpy(), allow_pickle=False)
    inputs = (torch.zeros((K, 256, 72, 72), dtype=torch.float16), img_pos, torch.zeros((32, K, 256), dtype=torch.float16), torch.zeros((K, 32), dtype=torch.float32))
    with torch.inference_mode():
        outs = wrapper(*inputs); print("ground_c%d shapes:" % K, [tuple(o.shape) for o in outs], flush=True)
        torch.onnx.export(wrapper, inputs, str(OUT / f"ground_c{K}m.onnx"), opset_version=17, input_names=["img_feat", "img_pos", "text_feats", "text_mask"],
                          output_names=["scores", "boxes", "presence", "hs", "enc"], dynamic_axes=None, do_constant_folding=True, dynamo=False)
    print("exported ground_c%dm %.0fs" % (K, time.time() - t0), flush=True)
mw = MaskWrapper(seg).cpu().eval(); K = 2
minp = (torch.zeros((1, 256, 288, 288)), torch.zeros((1, 256, 144, 144)), torch.zeros((5184, K, 256), dtype=torch.float16), torch.zeros((32, K, 256), dtype=torch.float16), torch.zeros((K, 32)), torch.zeros((K, QSEL, 256), dtype=torch.float16))
with torch.inference_mode():
    m = mw(*minp); print("mask shapes:", tuple(m.shape), flush=True)
    torch.onnx.export(mw, minp, str(OUT / f"maskhead_q{QSEL}.onnx"), opset_version=17, input_names=["fpn_0", "fpn_1", "enc", "text_feats", "text_mask", "hs_sel"], output_names=["masks"],
                      dynamic_axes={"enc": {1: "K"}, "text_feats": {1: "K"}, "text_mask": {0: "K"}, "hs_sel": {0: "K"}, "masks": {0: "K"}}, do_constant_folding=True, dynamo=False)
print("exported maskhead_q%d %.0fs" % (QSEL, time.time() - t0), flush=True)
print("EXPORT_DONE")
