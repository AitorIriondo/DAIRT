"""HF SAM3 backbone export with the window partition/unpartition hoisted out of the trunk (exact rewrite).

Residual/LayerNorm/MLP are per-token, so tokens can stay in window-major order for the whole trunk: partition once after
the patch embedding, run windowed layers with window_size=0 on [B*nw, ws, ws, C], run global layers on a [B, H, W, C]
*view* of the same buffer with their RoPE tables permuted to window-major order, unpartition once before the neck.
usage: export_hf_nowin.py <dart_root> <out_dir> <imgsz> <check_image>
"""
import sys, os
dart, out, imgsz, img = (sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]) if len(sys.argv) > 4 and __name__ == "__main__" else (None, None, 1008, None)
sys.path.insert(0, dart); sys.path.insert(0, os.path.join(dart, "scripts")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
_orig_arange = torch.arange
def _arange(*a, **k):
    if str(k.get("device", "")).startswith("cuda"): k["device"] = "cpu"
    return _orig_arange(*a, **k)
torch.arange = _arange
try:
    from sam3.model.position_encoding import PositionEmbeddingSine as _PES
    _oi = _PES.__init__
    def _ci(self, *a, **k): k["precompute_resolution"] = None; _oi(self, *a, **k)
    _PES.__init__ = _ci
except Exception as e: print("PES patch skipped:", e)
import transformers.models.sam3.modeling_sam3 as M

def prepare(vit):
    """set windowed layers to window_size=0 and permute global layers' RoPE tables (in place); returns the model"""
    cfg = vit.config; ws = cfg.window_size; H = W = imgsz // cfg.patch_size; assert H % ws == 0 and W % ws == 0, (H, ws)
    nh, nw = H // ws, W // ws
    perm = torch.arange(H * W).view(nh, ws, nw, ws).permute(0, 2, 1, 3).reshape(-1)   # window-major position -> raster index
    for layer in vit.layers:
        if layer.window_size > 0: layer.window_size = 0
        else:
            re = layer.rotary_emb; assert re.rope_embeddings_cos.shape[0] == H * W, re.rope_embeddings_cos.shape
            re.rope_embeddings_cos = torch.nn.Buffer(re.rope_embeddings_cos[perm].clone(), persistent=False)
            re.rope_embeddings_sin = torch.nn.Buffer(re.rope_embeddings_sin[perm].clone(), persistent=False)
    vit._nowin = (nh, nw, ws)
    return vit

def forward_nowin(self, pixel_values, **kw):
    h = self.embeddings(pixel_values); B = h.shape[0]; H = pixel_values.shape[-2] // self.config.patch_size; W = pixel_values.shape[-1] // self.config.patch_size; C = h.shape[-1]
    h = self.layer_norm(h.view(B, H, W, C)); nh, nw, ws = self._nowin
    h = h.view(B, nh, ws, nw, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B * nh * nw, ws, ws, C)      # partition once
    for layer in self.layers:
        if layer.rotary_emb.rope_embeddings_cos.shape[0] == ws * ws: h = layer(h, **kw)             # windowed: each item is one window
        else: h = layer(h.view(B, H, W, C), **kw).view(B * nh * nw, ws, ws, C)                      # global: window-major view
    h = h.view(B, nh, nw, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, C)                  # unpartition once
    return M.BaseModelOutput(last_hidden_state=h)

if __name__ == "__main__":
    import export_hf_backbone as E
    # --- numerical check of the rewrite on one image, full FP32 vision model (trunk + neck) ---
    from transformers import Sam3Model
    from preprocess import load_image_tensor
    x, _ = load_image_tensor(img, imgsz); x = torch.from_numpy(x)
    m = Sam3Model.from_pretrained("facebook/sam3", attn_implementation="eager").eval()
    with torch.no_grad():
        ref_last = m.vision_encoder.backbone(pixel_values=x).last_hidden_state
        prepare(m.vision_encoder.backbone); M.Sam3ViTModel.forward = forward_nowin
        new_last = m.vision_encoder.backbone(pixel_values=x).last_hidden_state
    d = (new_last - ref_last).abs().max().item(); rel = ((new_last - ref_last).norm() / ref_last.norm()).item()
    print(f"nowin vs original backbone last_hidden_state: max_abs={d:.3e} rel={rel:.3e}", flush=True)
    assert rel < 1e-4, "rewrite is not exact"
    del m
    # --- export: DART's export_onnx builds its own Sam3Model; hook from_pretrained so the backbone gets prepared ---
    _orig_fp = Sam3Model.from_pretrained.__func__
    @classmethod
    def _fp(cls, *a, **k):
        mm = _orig_fp(cls, *a, **k); prepare(mm.vision_encoder.backbone); return mm
    Sam3Model.from_pretrained = _fp
    os.makedirs(out, exist_ok=True)
    res = E.export_onnx(out, imgsz=imgsz)
    print("export_onnx returned:", res)
