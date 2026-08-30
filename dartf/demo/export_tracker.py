"""SAM 3 native tracker -> three ONNX graphs with a static tensor ABI (the handoff's track_init / track_step / mask_memory gap).
  trk_neck.onnx : trunk [1,1024,72,72] -> feat72 [1,256,72,72], hr0 [1,32,288,288], hr1 [1,64,144,144]   (SAM2 neck clone + decoder conv_s0/conv_s1)
  trk_init.onnx : feat72, hr0, hr1, mask_in [B,1,1008,1008] -> obj_ptr [B,256], maskmem [B,64,72,72]      (mask prompt, no memory attention)
  trk_step.onnx : feat72, hr0, hr1, mem [B,M,64,72,72], tpos_idx [M], ptrs [B,P,256], ptr_pos [P] -> masks [B,1,288,288], obj_ptr, obj_score [B,1], iou [B,1], maskmem
Constants written next to the graphs: pos72.npy (frame PE, 256ch), pe_mem.npy (memory PE, 64ch).
usage: python export_tracker.py --dart <DART root> --ckpt sam3.pt --out out/trk [--check] [--export]"""
import sys, os, time, argparse, math, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--dart", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--check", action="store_true"); ap.add_argument("--v2", action="store_true", help="step v2: gathered memory keys with per-key RoPE, valid-key counts, shared layer-0 queries, LQMemAttn custom op when --plugin"); ap.add_argument("--plugin", action="store_true"); ap.add_argument("--trunk", default=None, help="npz with a real trunk output for the check"); ap.add_argument("--export", action="store_true"); ap.add_argument("--threads", type=int, default=16); a = ap.parse_args()
sys.path.insert(0, a.dart)
import torch, torch.nn as nn, torch.nn.functional as F
def _cpu(dev): return "cpu" if (dev is not None and str(dev).startswith("cuda")) else dev
for fn in ("arange", "zeros", "ones", "empty", "tensor", "linspace", "full", "rand", "randn"):
    orig = getattr(torch, fn)
    def make(orig):
        def w(*x, **k):
            if "device" in k: k["device"] = _cpu(k["device"])
            return orig(*x, **k)
        return w
    setattr(torch, fn, make(orig))
torch.Tensor.cuda = lambda self, *x, **k: self; torch.nn.Module.cuda = lambda self, *x, **k: self
_orig_to = torch.Tensor.to
def _to(self, *x, **k):
    x = tuple(_cpu(v) if isinstance(v, (str, torch.device)) else v for v in x)
    if "device" in k: k["device"] = _cpu(k["device"])
    return _orig_to(self, *x, **k)
torch.Tensor.to = _to; torch.cuda.is_available = lambda: False
torch.set_num_threads(a.threads); torch.manual_seed(0)
from sam3.model.position_encoding import PositionEmbeddingSine
_oi = PositionEmbeddingSine.__init__
def _ci(self, *x, **k): k["precompute_resolution"] = None; _oi(self, *x, **k)
PositionEmbeddingSine.__init__ = _ci
from sam3.model_builder import build_tracker, _create_vision_backbone
from sam3.sam.rope import compute_axial_cis
from sam3.model.sam3_tracker_utils import get_1d_sine_pe
t0 = time.time()
tracker = build_tracker(apply_temporal_disambiguation=True).eval(); vb = _create_vision_backbone().eval()
sd = torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=False); sd = sd.get("model", sd)
r = tracker.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("tracker.")}, strict=True); print("tracker weights:", r)
pre = "detector.backbone.vision_backbone."; r2 = vb.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=False)
print("neck weights: missing", len(r2.missing_keys), "unexpected", len(r2.unexpected_keys), "sam2_convs:", vb.sam2_convs is not None, "%.0fs" % (time.time() - t0))
del sd
for prm in list(tracker.parameters()) + list(vb.parameters()): prm.requires_grad_(False)
H = 72; N = H * H; C = tracker.hidden_dim; MD = tracker.mem_dim; NM = tracker.num_maskmem; MAXPTR = tracker.max_obj_ptrs_in_encoder
print("hidden", C, "mem_dim", MD, "num_maskmem", NM, "max_obj_ptrs", MAXPTR, "use_memory_selection", tracker.use_memory_selection, "max_cond", tracker.max_cond_frames_in_attn)
enc = tracker.transformer.encoder; layer0 = enc.layers[0]
print("encoder: layers", len(enc.layers), "pos_enc_at_input", enc.pos_enc_at_input, "batch_first", enc.batch_first, "pre_norm", getattr(layer0, "pre_norm", None) if hasattr(layer0, "pre_norm") else "?",
      "sa heads", layer0.self_attn.num_heads, "ca kv_in", layer0.cross_attn_image.k_proj.in_features, "rope_k_repeat", layer0.cross_attn_image.rope_k_repeat, "pos_at_attn", layer0.pos_enc_at_attn, "pos_at_ca_q", layer0.pos_enc_at_cross_attn_queries, "pos_at_ca_k", layer0.pos_enc_at_cross_attn_keys, "ca_first", layer0.cross_attention_first)
assert layer0.self_attn.num_heads == 1 and layer0.cross_attn_image.num_heads == 1 and not layer0.pos_enc_at_attn and not layer0.pos_enc_at_cross_attn_queries and layer0.pos_enc_at_cross_attn_keys and enc.pos_enc_at_input and not layer0.cross_attention_first
# rotary tables (real form) for the 72x72 grid, head dim 256
fc = compute_axial_cis(dim=layer0.self_attn.internal_dim // layer0.self_attn.num_heads, end_x=H, end_y=H); FR, FI = fc.real.contiguous(), fc.imag.contiguous()   # [5184, 128]
def rope(x, fr, fi):
    """x [B, T, 256] with T a multiple of 5184 (frames stacked): rotate every 5184-token frame with the same table"""
    xr = x.float().unflatten(-1, (-1, 2)); xre, xim = xr[..., 0], xr[..., 1]                     # [B, T, 128]
    xre = xre.unflatten(1, (-1, N)); xim = xim.unflatten(1, (-1, N))                              # [B, F, 5184, 128]
    out = torch.stack([xre * fr - xim * fi, xre * fi + xim * fr], -1).flatten(1, 2).flatten(-2)   # [B, T, 256]
    return out.type_as(x)
def sdpa(q, k, v): return F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
def memory_attention(src, pos_src, mem_tokens, pos_mem, ptr_tokens, pos_ptr):
    """src [B,5184,256], pos_src [1,5184,256]; mem_tokens [B,M*5184,64], pos_mem [1,M*5184,64]; ptr_tokens [B,4P,64], pos_ptr [1,4P,64]"""
    x = src + 0.1 * pos_src
    for L in enc.layers:
        x2 = L.norm1(x); A = L.self_attn; q = rope(A.q_proj(x2), FR, FI); k = rope(A.k_proj(x2), FR, FI); x = x + A.out_proj(sdpa(q, k, A.v_proj(x2)))
        x2 = L.norm2(x); A = L.cross_attn_image; q = rope(A.q_proj(x2), FR, FI)
        ksp = rope(A.k_proj(mem_tokens + pos_mem), FR, FI); kpt = A.k_proj(ptr_tokens + pos_ptr); k = torch.cat([ksp, kpt], 1); v = torch.cat([A.v_proj(mem_tokens), A.v_proj(ptr_tokens)], 1)
        x = x + A.out_proj(sdpa(q, k, v))
        x2 = L.norm3(x); x = x + L.linear2(F.relu(L.linear1(x2)))
    return enc.norm(x)
# ---- v2 attention: reference (masked sdpa) for tracing/checking, custom op LQMemAttn for the TensorRT export ----
def masked_sdpa(q, k, v, nvalid):
    """q [Bq,Q,256], k/v [B,Kt,256], nvalid [B] int (total valid keys incl. pointers); Bq = 1 broadcasts"""
    B = k.shape[0]; q = q.expand(B, -1, -1) if q.shape[0] == 1 else q; s = (q @ k.transpose(-1, -2)) * (1.0 / 16.0)
    ar = torch.arange(k.shape[1], device=k.device)[None, :]; s = s.masked_fill((ar >= nvalid[:, None])[:, None, :], float("-inf")); return s.softmax(-1) @ v
class LQMemAttnOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, nvalid): return masked_sdpa(q, k, v, nvalid)
    @staticmethod
    def symbolic(g, q, k, v, nvalid): return g.op("LQMemAttn", q, k, v, nvalid, scale_f=1.0 / 16.0, cfg_i=-1)
def attn_v2(q, k, v, nvalid): return LQMemAttnOp.apply(q, k, v, nvalid) if a.plugin else masked_sdpa(q, k, v, nvalid)
def rope_tokens(x, fr, fi):
    """x [B, T, 256] with per-token tables fr/fi [B or 1, T, 128]"""
    xr = x.float().unflatten(-1, (-1, 2)); xre, xim = xr[..., 0], xr[..., 1]; return torch.stack([xre * fr - xim * fi, xre * fi + xim * fr], -1).flatten(-2).type_as(x)
def memory_attention_v2(src, pos_src, mem_tokens, pos_mem, key_fr, key_fi, ptr_tokens, pos_ptr, nvalid):
    """src [1,5184,256] (shared), mem_tokens/pos_mem [B,Kn,64], key_fr/key_fi [B,Kn,128] per-key RoPE tables, ptr_tokens [B,4P,64], pos_ptr [1,4P,64], nvalid [B] (memory keys; pointers appended and always valid)"""
    B = mem_tokens.shape[0]; nv_total = nvalid.to(torch.int32) + ptr_tokens.shape[1]
    x = src + 0.1 * pos_src                                                    # [1, 5184, 256]: layer-0 self attention is shared by every object
    for li, L in enumerate(enc.layers):
        x2 = L.norm1(x); A = L.self_attn; q = rope(A.q_proj(x2), FR, FI); k = rope(A.k_proj(x2), FR, FI); nvs = torch.full((x2.shape[0],), N, dtype=torch.int32)
        x = x + A.out_proj(attn_v2(q, k, A.v_proj(x2), nvs))
        x2 = L.norm2(x); A = L.cross_attn_image; q = rope(A.q_proj(x2), FR, FI)                   # layer 0: q of batch 1, broadcast inside the attention
        ksp = rope_tokens(A.k_proj(mem_tokens + pos_mem), key_fr, key_fi); kpt = A.k_proj(ptr_tokens + pos_ptr); k = torch.cat([ksp, kpt], 1); v = torch.cat([A.v_proj(mem_tokens), A.v_proj(ptr_tokens)], 1)
        x = x + A.out_proj(attn_v2(q, k, v, nv_total))                                            # broadcast add: from here on x is per object
        x2 = L.norm3(x); x = x + L.linear2(F.relu(L.linear1(x2)))
    return enc.norm(x)
class StepV2(nn.Module):
    """feat72 [1,256,72,72], hr0, hr1, memtok [B,Kn,64] fp16, mempos [B,Kn,64] fp16 (memory PE + temporal slot, built on the host), keyidx [B,Kn] int64 (grid index of each key for RoPE),
    nvalid [B] int32, ptrs [B,P,256], ptr_pos [P] -> masks [B,1,288,288], obj_ptr, obj_score, iou, maskmem"""
    def __init__(s): super().__init__(); s.tr = tracker; s.register_buffer("pos72", POS72); s.register_buffer("fr", FR); s.register_buffer("fi", FI)
    def forward(s, feat72, hr0, hr1, memtok, mempos, keyidx, nvalid, ptrs, ptr_pos):
        B = memtok.shape[0]; src = feat72.flatten(2).transpose(1, 2); pos_src = s.pos72.flatten(2).transpose(1, 2)
        key_fr = s.fr[keyidx]; key_fi = s.fi[keyidx]                                              # [B, Kn, 128]
        tp = tracker.obj_ptr_tpos_proj(get_1d_sine_pe(ptr_pos, dim=C)); ptr_tokens = ptrs.unflatten(-1, (C // MD, MD)).flatten(1, 2); pos_ptr = tp.repeat_interleave(C // MD, 0)[None]
        x = memory_attention_v2(src, pos_src, memtok.float(), mempos.float(), key_fr, key_fi, ptr_tokens, pos_ptr, nvalid); pix = x.transpose(1, 2).reshape(B, C, H, H)
        h0 = hr0.expand(B, -1, -1, -1); h1 = hr1.expand(B, -1, -1, -1); fb = feat72.expand(B, -1, -1, -1)
        _, _, ious, low, high, obj_ptr, obj_score = sam_heads(pix, h0, h1, None, MM_TRACK); ious = ious.max(-1, keepdim=True)[0]
        return low, obj_ptr, obj_score, ious, encode_memory(fb, high, obj_score)
class Neck(nn.Module):
    def __init__(s): super().__init__(); s.convs = vb.sam2_convs; s.s0 = tracker.sam_mask_decoder.conv_s0; s.s1 = tracker.sam_mask_decoder.conv_s1
    def forward(s, trunk):
        f0 = s.convs[0](trunk); f1 = s.convs[1](trunk); f2 = s.convs[2](trunk)          # 288, 144, 72 (scalp drops the 4th level)
        return f2, s.s0(f0), s.s1(f1)
MM_TRACK = tracker._use_multimask(is_init_cond_frame=False, point_inputs=None)
print("multimask: in_sam", tracker.multimask_output_in_sam, "for_tracking", tracker.multimask_output_for_tracking, "pts", tracker.multimask_min_pt_num, tracker.multimask_max_pt_num, "-> tracking uses multimask =", MM_TRACK)
def sam_heads(feat_b, hr0_b, hr1_b, mask_inputs, multimask=False):
    return tracker._forward_sam_heads(backbone_features=feat_b, point_inputs=None, mask_inputs=mask_inputs, high_res_features=[hr0_b, hr1_b], multimask_output=multimask)
def encode_memory(feat_b, high_res_masks, obj_score):
    mem, _ = tracker._encode_new_memory(image=None, current_vision_feats=[feat_b.flatten(2).permute(2, 0, 1)], feat_sizes=[(H, H)], pred_masks_high_res=high_res_masks, object_score_logits=obj_score, is_mask_from_pts=False)
    return mem
class Init(nn.Module):
    def __init__(s): super().__init__(); s.tr = tracker
    def forward(s, feat72, hr0, hr1, mask_in):
        B = mask_in.shape[0]; fb = feat72.expand(B, -1, -1, -1); h0 = hr0.expand(B, -1, -1, -1); h1 = hr1.expand(B, -1, -1, -1)
        m = mask_in.float(); high = m * 20.0 - 10.0
        _, _, _, _, _, obj_ptr, _ = sam_heads(fb, h0, h1, tracker.mask_downsample(m))
        appearing = torch.any(m.flatten(1) > 0, dim=1)[:, None].float(); obj_score = 20.0 * appearing - 10.0
        obj_ptr = appearing * obj_ptr + (1 - appearing) * tracker.no_obj_ptr
        return obj_ptr, encode_memory(fb, high, obj_score)
class Step(nn.Module):
    def __init__(s): super().__init__(); s.tr = tracker; s.register_buffer("pos72", POS72); s.register_buffer("pe_mem", PE_MEM)
    def forward(s, feat72, hr0, hr1, mem, tpos_idx, ptrs, ptr_pos):
        B = mem.shape[0]; fb = feat72.expand(B, -1, -1, -1); h0 = hr0.expand(B, -1, -1, -1); h1 = hr1.expand(B, -1, -1, -1)
        src = fb.flatten(2).transpose(1, 2); pos_src = s.pos72.flatten(2).transpose(1, 2)                          # [B,5184,256], [1,5184,256]
        mem_tokens = mem.permute(0, 1, 3, 4, 2).flatten(1, 3)                                                      # [B, M*5184, 64]
        pos_mem = (s.pe_mem.flatten(2).transpose(1, 2)[:, None] + tracker.maskmem_tpos_enc[tpos_idx][None, :, 0]).flatten(1, 2)   # [1, M*5184, 64]
        tp = tracker.obj_ptr_tpos_proj(get_1d_sine_pe(ptr_pos, dim=C))                                              # [P, 64]
        ptr_tokens = ptrs.unflatten(-1, (C // MD, MD)).flatten(1, 2); pos_ptr = tp.repeat_interleave(C // MD, 0)[None]   # [B,4P,64], [1,4P,64]
        x = memory_attention(src, pos_src, mem_tokens, pos_mem, ptr_tokens, pos_ptr); pix = x.transpose(1, 2).reshape(B, C, H, H)
        if getattr(s, "return_pix", False): return pix
        _, _, ious, low, high, obj_ptr, obj_score = sam_heads(pix, h0, h1, None, MM_TRACK); ious = ious.max(-1, keepdim=True)[0]
        return low, obj_ptr, obj_score, ious, encode_memory(fb, high, obj_score)
neck = Neck().eval(); os.makedirs(a.out, exist_ok=True)
with torch.inference_mode():
    POS72 = vb.position_encoding(torch.zeros(1, C, H, H)).float(); PE_MEM = tracker.maskmem_backbone.position_encoding(torch.zeros(1, MD, H, H)).float()
    np.save(f"{a.out}/pos72.npy", POS72.numpy()); np.save(f"{a.out}/pe_mem.npy", PE_MEM.numpy()); print("pos72", tuple(POS72.shape), "pe_mem", tuple(PE_MEM.shape))
step = Step().eval(); init = Init().eval()
# ---- reference check against tracker.track_step with a constructed memory bank ----
if a.check:
    with torch.inference_mode():
        B, M, P = 2, 7, 5; trunk = torch.from_numpy(np.load(a.trunk)["trunk"]).float() if a.trunk else torch.randn(1, 1024, H, H) * 0.5; feat72, hr0, hr1 = neck(trunk)
        masks0 = torch.zeros(B, 1, 1008, 1008); masks0[0, :, 300:600, 350:650] = 1; masks0[1, :, 500:800, 900:1300] = 1     # two rectangle prompts
        optr0, mem0 = init(feat72, hr0, hr1, masks0); print("init ok", tuple(optr0.shape), tuple(mem0.shape), "%.0fs" % (time.time() - t0))
        frame_idx = 8; cond_t = 0; prev_t = list(range(frame_idx - 6, frame_idx))               # cond frame 0, non-cond frames 2..7
        bank = {t: (mem0 + torch.randn(B, MD, H, H) * 0.02, optr0 + torch.randn(B, C) * 0.02, 1.0) for t in prev_t}; bank[cond_t] = (mem0, optr0, 1.0)
        # wrapper inputs: memory order = [cond, oldest .. newest]; tpos index: cond -> NM-1; t_pos=1..6 -> NM-t_pos-1
        mem = torch.stack([bank[cond_t][0]] + [bank[t][0] for t in prev_t], 1); tpos_idx = torch.tensor([NM - 1] + [NM - tp - 1 for tp in range(1, NM)])
        # pointers: cond first (rel pos frame_idx - cond_t) then t_diff = 1 .. 15 over existing frames (most recent first), positions / (max-1)
        # memory selection rule of the reference: pointers = cond frame (rel pos frame_idx - t_cond) + valid non-cond frames by recency index d = 1 .. len(valid) - 1 (the oldest valid frame contributes no pointer)
        valid = sorted(prev_t); pl = [(frame_idx - cond_t, bank[cond_t][1])] + [(d, bank[valid[-d]][1]) for d in range(1, min(MAXPTR, len(valid)))]
        ptrs = torch.stack([p for _, p in pl], 1); ptr_pos = torch.tensor([float(d) for d, _ in pl]) / (min(100, MAXPTR) - 1)
        low, optr, oscore, ious, memn = step(feat72, hr0, hr1, mem, tpos_idx, ptrs, ptr_pos); print("step ok", tuple(low.shape), "%.0fs" % (time.time() - t0))
        # reference
        vf = [hr0.expand(B, -1, -1, -1).flatten(2).permute(2, 0, 1), hr1.expand(B, -1, -1, -1).flatten(2).permute(2, 0, 1), feat72.expand(B, -1, -1, -1).flatten(2).permute(2, 0, 1)]
        vp = [torch.zeros(288 * 288, B, 32), torch.zeros(144 * 144, B, 64), POS72.expand(B, -1, -1, -1).flatten(2).permute(2, 0, 1)]
        od = {"cond_frame_outputs": {cond_t: {"maskmem_features": bank[cond_t][0], "maskmem_pos_enc": [PE_MEM.expand(B, -1, -1, -1)], "obj_ptr": bank[cond_t][1], "eff_iou_score": 1.0}},
              "non_cond_frame_outputs": {t: {"maskmem_features": bank[t][0], "maskmem_pos_enc": [PE_MEM.expand(B, -1, -1, -1)], "obj_ptr": bank[t][1], "eff_iou_score": 1.0} for t in prev_t}}
        ref = tracker.track_step(frame_idx=frame_idx, is_init_cond_frame=False, current_vision_feats=vf, current_vision_pos_embeds=vp, feat_sizes=[(288, 288), (144, 144), (H, H)], image=None, point_inputs=None, mask_inputs=None, output_dict=od, num_frames=100)
        def rel(x, y): return float((x - y).norm() / (y.norm() + 1e-9))
        print("CHECK rel: masks %.2e obj_ptr %.2e obj_score %.2e maskmem %.2e" % (rel(low, ref["pred_masks"]), rel(optr, ref["obj_ptr"]), rel(oscore, ref["object_score_logits"]), rel(memn, ref["maskmem_features"])))
        print("  obj_score wrapper", oscore.flatten().tolist(), "ref", ref["object_score_logits"].flatten().tolist(), "| iou", ious.flatten().tolist(), "| mask area", (low > 0).float().mean((1, 2, 3)).tolist())
        step.return_pix = True; pix_mine = step(feat72, hr0, hr1, mem, tpos_idx, ptrs, ptr_pos); step.return_pix = False
        pix_ref = tracker._prepare_memory_conditioned_features(frame_idx=frame_idx, is_init_cond_frame=False, current_vision_feats=vf[-1:], current_vision_pos_embeds=vp[-1:], feat_sizes=[(H, H)], output_dict=od, num_frames=100)
        print("  pix rel %.2e" % rel(pix_mine, pix_ref))
        np.savez(f"{a.out}/ref_check.npz", trunk=trunk.numpy(), masks0=masks0.numpy(), optr0=optr0.numpy(), mem0=mem0.numpy(), mem=mem.numpy(), tpos_idx=tpos_idx.numpy(), ptrs=ptrs.numpy(), ptr_pos=ptr_pos.numpy(),
                 low=low.numpy(), optr=optr.numpy(), oscore=oscore.numpy(), ious=ious.numpy(), memn=memn.numpy(), feat72=feat72.numpy(), hr0=hr0.numpy(), hr1=hr1.numpy())
        o1 = sam_heads(pix_ref, hr0.expand(B, -1, -1, -1), hr1.expand(B, -1, -1, -1), None, MM_TRACK); o2 = sam_heads(pix_ref, hr0.expand(B, -1, -1, -1), hr1.expand(B, -1, -1, -1), None, MM_TRACK)
        print("  decoder determinism: masks %.2e obj_ptr %.2e" % (rel(o1[3], o2[3]), rel(o1[5], o2[5])), "| ref masks vs decoder on pix_ref: %.2e" % rel(o1[3], ref["pred_masks"]), "| mine vs decoder on pix_ref: %.2e" % rel(low, o1[3]))
if a.export:
    _interp = F.interpolate
    def interp_noaa(x, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=False): return _interp(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=False)
    F.interpolate = interp_noaa    # antialias only appears on upsampling paths inside these graphs (252->288, 1008->1152), where it is a no-op
    with torch.no_grad():
        trunk = torch.randn(1, 1024, H, H); feat72, hr0, hr1 = neck(trunk)
        torch.onnx.export(neck, (trunk,), f"{a.out}/trk_neck.onnx", opset_version=17, input_names=["trunk"], output_names=["feat72", "hr0", "hr1"], dynamo=False, do_constant_folding=True); print("exported trk_neck")
        B = 2; mask_in = (torch.rand(B, 1, 1008, 1008) > 0.7).float()
        torch.onnx.export(init, (feat72, hr0, hr1, mask_in), f"{a.out}/trk_init.onnx", opset_version=17, input_names=["feat72", "hr0", "hr1", "mask_in"], output_names=["obj_ptr", "maskmem"],
                          dynamic_axes={"mask_in": {0: "B"}, "obj_ptr": {0: "B"}, "maskmem": {0: "B"}}, dynamo=False, do_constant_folding=True); print("exported trk_init")
        M, P = 3, 4; mem = torch.randn(B, M, MD, H, H) * 0.3; tpos_idx = torch.tensor([NM - 1, NM - 6, NM - 7 + 6 - 6]); tpos_idx = torch.tensor([6, 1, 5]); ptrs = torch.randn(B, P, C) * 0.3; ptr_pos = torch.tensor([8.0, 1.0, 2.0, 3.0]) / (MAXPTR - 1)
        torch.onnx.export(step, (feat72, hr0, hr1, mem, tpos_idx, ptrs, ptr_pos), f"{a.out}/trk_step.onnx", opset_version=17, input_names=["feat72", "hr0", "hr1", "mem", "tpos_idx", "ptrs", "ptr_pos"], output_names=["masks", "obj_ptr", "obj_score", "iou", "maskmem"],
                          dynamic_axes={"mem": {0: "B", 1: "M"}, "tpos_idx": {0: "M"}, "ptrs": {0: "B", 1: "P"}, "ptr_pos": {0: "P"}, "masks": {0: "B"}, "obj_ptr": {0: "B"}, "obj_score": {0: "B"}, "iou": {0: "B"}, "maskmem": {0: "B"}}, dynamo=False, do_constant_folding=True); print("exported trk_step")
    import onnx
    for n in ("trk_neck", "trk_init", "trk_step"):
        m = onnx.load(f"{a.out}/{n}.onnx"); onnx.checker.check_model(m); print(n, "ok", len(m.graph.node), "nodes", os.path.getsize(f"{a.out}/{n}.onnx") // 1e6, "MB")
if a.v2:
    stepv2 = StepV2().eval()
    def bank_v2(mem, tpos_idx, ptrs, ptr_pos, keep=None):
        """host-side construction of the v2 inputs from the v1 bank (mem [B,M,64,72,72]); keep [B, M*5184] bool selects keys (None = all)"""
        Bn, M = mem.shape[0], mem.shape[1]; tok = mem.permute(0, 1, 3, 4, 2).reshape(Bn, M * N, MD); pos = (PE_MEM.flatten(2).transpose(1, 2)[:, None] + tracker.maskmem_tpos_enc[tpos_idx][None, :, 0]).reshape(1, M * N, MD).expand(Bn, -1, -1)
        idx = torch.arange(N).repeat(M)[None].expand(Bn, -1)
        if keep is None: return tok.half(), pos.half(), idx, torch.full((Bn,), M * N, dtype=torch.int32)
        Kn = int(keep.sum(1).max()); tk = torch.zeros(Bn, Kn, MD); ps = torch.zeros(Bn, Kn, MD); ix = torch.zeros(Bn, Kn, dtype=torch.long); nv = torch.zeros(Bn, dtype=torch.int32)
        for b in range(Bn): sel = keep[b].nonzero()[:, 0]; n = len(sel); tk[b, :n] = tok[b, sel]; ps[b, :n] = pos[b, sel]; ix[b, :n] = idx[b, sel]; nv[b] = n
        return tk.half(), ps.half(), ix, nv
    if a.check:
        with torch.inference_mode():
            tk, ps, ix, nv = bank_v2(mem, tpos_idx, ptrs, ptr_pos); out2 = stepv2(feat72, hr0, hr1, tk, ps, ix, nv, ptrs, ptr_pos)
            print("CHECK v2 (full keys) vs reference: masks %.2e obj_ptr %.2e obj_score %.2e maskmem %.2e" % (rel(out2[0], ref["pred_masks"]), rel(out2[1], ref["obj_ptr"]), rel(out2[2], ref["object_score_logits"]), rel(out2[4], ref["maskmem_features"])))
            keep = torch.rand(B, mem.shape[1] * N) < 0.3; tk, ps, ix, nv = bank_v2(mem, tpos_idx, ptrs, ptr_pos, keep); o3 = stepv2(feat72, hr0, hr1, tk, ps, ix, nv, ptrs, ptr_pos); print("v2 pruned (30 % random keys, padded) ran: nvalid", nv.tolist(), "Kn", tk.shape[1])
            np.savez(f"{a.out}/ref_check_v2.npz", memtok=tk.numpy(), mempos=ps.numpy(), keyidx=ix.numpy(), nvalid=nv.numpy(), low=o3[0].numpy(), optr=o3[1].numpy(), oscore=o3[2].numpy(), ious=o3[3].numpy(), memn=o3[4].numpy(), ptrs=ptrs.numpy(), ptr_pos=ptr_pos.numpy(), feat72=feat72.numpy(), hr0=hr0.numpy(), hr1=hr1.numpy())
    if a.export:
        with torch.no_grad():
            Bn, M, P, Kn = 2, 3, 4, 2 * N + 40; memtok = (torch.randn(Bn, Kn, MD) * 0.3).half(); mempos = (torch.randn(Bn, Kn, MD) * 0.3).half(); keyidx = torch.randint(0, N, (Bn, Kn)); nvalid = torch.tensor([Kn, Kn - 40], dtype=torch.int32)
            ptrs = torch.randn(Bn, P, C) * 0.3; ptr_pos = torch.tensor([8.0, 1.0, 2.0, 3.0]) / (MAXPTR - 1)
            torch.onnx.export(stepv2, (feat72, hr0, hr1, memtok, mempos, keyidx, nvalid, ptrs, ptr_pos), f"{a.out}/trk_step_v2{'_plugin' if a.plugin else ''}.onnx", opset_version=17,
                              input_names=["feat72", "hr0", "hr1", "memtok", "mempos", "keyidx", "nvalid", "ptrs", "ptr_pos"], output_names=["masks", "obj_ptr", "obj_score", "iou", "maskmem"],
                              dynamic_axes={"memtok": {0: "B", 1: "Kn"}, "mempos": {0: "B", 1: "Kn"}, "keyidx": {0: "B", 1: "Kn"}, "nvalid": {0: "B"}, "ptrs": {0: "B", 1: "P"}, "ptr_pos": {0: "P"}, "masks": {0: "B"}, "obj_ptr": {0: "B"}, "obj_score": {0: "B"}, "iou": {0: "B"}, "maskmem": {0: "B"}},
                              dynamo=False, do_constant_folding=True, custom_opsets={"": 17}); print("exported trk_step_v2", "plugin" if a.plugin else "")
print("DONE %.0fs" % (time.time() - t0))
