"""Activation-aware sequential GPTQ for the HF SAM3 trunk on the host CPU.

Differences from the plain GPTQ cache (gptq_a05.npz):
  * blocks are processed in order and every block sees the *fake-quantized* output of the previous quantized blocks
    (weights = current INT8 codes, activations = static per-tensor INT8 at the recipe's sites), so each site's Hessian
    H = X^T X is built from the inputs the deployed engine will actually feed it;
  * the activation quantizer of the site is applied to X before building H (so GPTQ compensates activation rounding);
  * per-output-channel bias correction: after quantizing a site, b <- b - mean_tokens(y_quant - y_fp).
Recipe knobs mirror quantize_hf.py: SmoothQuant alpha (folded into LN affine for qkv/fc1 and into V columns for proj),
weights INT8 per-channel on --blocks, activations on --act-blocks (+ proj always), p99.999 static scales.
Outputs: <out>_cache.npz (block{b}.{fam}::codes [K,N] int8, ::scale [N] float32 — same contract as gptq_a05.npz, on the
SMOOTHED weights, so pass the same --smooth to quantize_hf.py), <out>_act.json ({"act_scales": {...}}) and
<out>_bias.json ({"block{b}.{fam}[:role]": [N] corrected bias}) consumed via --act-override / --bias-override.
usage: gptq_actaware.py --images DIR --ids JSON --out PREFIX [--blocks 0-31] [--act-blocks 1-31] [--smooth 0.5] [--threads 28]
"""
import argparse, json, time, sys, os, math, numpy as np, torch, torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True); ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--blocks", default="0-31"); ap.add_argument("--act-blocks", default="1-31"); ap.add_argument("--smooth", type=float, default=0.5)
ap.add_argument("--threads", type=int, default=28); ap.add_argument("--size", type=int, default=1008); ap.add_argument("--percentile", type=float, default=0.99999)
ap.add_argument("--act-scales", default=None, help="json {act_scales: {block{b}.{fam}: s}} to use as the activation quantizer instead of the percentile"); ap.add_argument("--hadamard", action="store_true", help="quantize the Hadamard-basis trunk (rotation replaces SmoothQuant; use --smooth 0)"); ap.add_argument("--smooth-fams", default="qkv,fc1,proj", help="which families --smooth applies to (subset of qkv,fc1,proj); with --hadamard use proj only"); ap.add_argument("--fc2-chan", type=float, default=0.0, help="per-input-channel activation scales on fc2 (SmoothQuant alpha on the GELU output, folded into fc2 columns; exact; alpha in (0,1])"); ap.add_argument("--token-mix-fams", default="", help="experimental option; default off"); ap.add_argument("--readout-metric", default=None, help="experimental option; default off"); ap.add_argument("--readout-blocks", default="28-31"); ap.add_argument("--stream-match", type=float, default=0.0, help="experimental option; default off"); ap.add_argument("--stream-match-fams", default="qkv,proj,fc1,fc2", help="experimental option; default off"); ap.add_argument("--ridge", type=float, default=1e-2, help="experimental option; default off"); ap.add_argument("--sparse24-fams", default="", help="experimental option; default off"); ap.add_argument("--head-rot", action="store_true", help="with --hadamard: per-head V/o_proj rotation (must match the exported graph; sets LQ_HEAD_ROT=1)"); ap.add_argument("--no-bias-corr", action="store_true"); ap.add_argument("--max-tokens", type=int, default=40000, help="tokens sampled per site for the Hessian")
a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from preprocess import load_image_tensor
from transformers import Sam3Model
m = Sam3Model.from_pretrained("facebook/sam3", attn_implementation="eager").eval(); bb = m.vision_encoder.backbone
if a.hadamard:
    import transformers.models.sam3.modeling_sam3 as M_; from hadamard_basis import prepare as had_prepare, forward_had
    had_prepare(bb, head_rot=a.head_rot); M_.Sam3ViTModel.forward = forward_had; assert a.smooth == 0 or set(a.smooth_fams.split(",")) <= {"proj"}, "with --hadamard only --smooth-fams proj is allowed (LN sites are rotated)"
b0, b1 = [int(v) for v in a.blocks.split("-")]; WBLK = set(range(b0, b1 + 1)); ab0, ab1 = [int(v) for v in a.act_blocks.split("-")]; ABLK = set(range(ab0, ab1 + 1))
RM = None
if a.readout_metric:
    RM = torch.from_numpy(np.load(a.readout_metric)).float(); rb0, rb1 = [int(v) for v in a.readout_blocks.split("-")]; RBLK = set(range(rb0, rb1 + 1))
    PM = RM.t() @ torch.linalg.pinv(RM).t()   # projector onto the row space of M (acts on the output-channel axis): P = M^T (M M^T)^-1 M
    print("read-out metric:", tuple(RM.shape), "rank", torch.linalg.matrix_rank(RM).item())
ids = json.load(open(a.ids)); ACT_FIX = json.load(open(a.act_scales))["act_scales"] if a.act_scales else {}

@torch.no_grad()
def embed_all():
    hs = []
    for iid in ids:
        x, _ = load_image_tensor(f"{a.images}/{iid}.jpg", a.size); x = torch.from_numpy(x)
        h = bb.embeddings(x); B = h.shape[0]; Hh = x.shape[-2] // bb.config.patch_size; Ww = x.shape[-1] // bb.config.patch_size
        if a.hadamard:
            g_, b_ = bb._had_pre; hs.append(torch.nn.functional.layer_norm(h.view(B, Hh, Ww, -1), (h.shape[-1],), g_, b_, bb._had_eps) @ bb._had)
        else: hs.append(bb.layer_norm(h.view(B, Hh, Ww, -1)))
    return hs

def q_act_scale(xs):
    v = torch.cat([t.abs().flatten()[torch.randperm(t.numel())[:500_000]] for t in xs]).float()   # <= 8M elements
    return float(np.percentile(v.numpy(), a.percentile * 100.0)) / 127.0
def fq_act(x, s): return torch.clamp(torch.round(x / s), -127, 127) * s
_H64 = None
def _wht64():
    global _H64
    if _H64 is None:
        H = torch.tensor([[1.]]);
        while H.shape[0] < 64: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _H64 = H / 8.0
    return _H64
def tokmix_perm(x):
    """x [B, Hh, Ww, C] raster -> [B*nw*nh*ws*ws, C] window-major token order (24x24 windows), and the inverse permutation"""
    B, Hh, Ww, C = x.shape; ws = 24; nh, nw = Hh // ws, Ww // ws
    xm = x.view(B, nh, ws, nw, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, C); return xm, (B, nh, nw, ws, C)
def tokmix_unperm(xm, meta):
    B, nh, nw, ws, C = meta; return xm.view(B, nh, nw, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, nh * ws, nw * ws, C)
_SGN = torch.tensor([1.0 if ((j * 40503 + 17) % 97) & 1 else -1.0 for j in range(64)])
def fq_act_mix(x, s):
    """token-mixed fake quantization: T^T Q(T x) with T = H D over 64 consecutive window-major tokens (randomized Hadamard: D = fixed +-1 signs
    per position so that correlated neighbouring tokens do not concentrate into one row; orthogonal => exact for the following GEMM)"""
    H = _wht64().to(x.dtype); D = _SGN.to(x.dtype)[:, None]; xm, meta = tokmix_perm(x); n = xm.shape[0] // 64
    y = (H @ (D * xm.view(n, 64, -1))); y = torch.clamp(torch.round(y / s), -127, 127) * s; y = (D * (H.t() @ y)).view(-1, xm.shape[-1])
    return tokmix_unperm(y, meta)
def mixed_values(xs):
    """the token-mixed activations (for the per-tensor scale)"""
    H = _wht64(); out = []
    for x in xs:
        xm, _ = tokmix_perm(x); n = xm.shape[0] // 64; out.append((H.to(x.dtype) @ (_SGN.to(x.dtype)[:, None] * xm.view(n, 64, -1))).view(-1, xm.shape[-1]))
    return out
def fq_w_rtn(W):  # W [N,K] -> per-output-channel
    s = W.abs().amax(dim=1, keepdim=True) / 127.0; s = torch.where(s > 0, s, torch.ones_like(s)); return torch.clamp(torch.round(W / s), -127, 127) * s, s.squeeze(1)

def gptq(W, Hm, blocksize=128, percdamp=0.01, sparse24=False):
    """W [N,K] (rows = output channels), Hm [K,K]; returns codes [N,K] int8 and scale [N] (symmetric per-channel).
    sparse24: SparseGPT — at every 4-column group boundary pick the 2 least salient (w^2 / Hinv_ii^2) weights per row, force them to 0
    (their error is propagated like a quantization error), then quantize the survivors."""
    W = W.clone().double(); Hm = Hm.clone().double(); K = W.shape[1]
    dead = torch.diag(Hm) == 0; Hm[dead, dead] = 1; W[:, dead] = 0
    Hm += percdamp * torch.mean(torch.diag(Hm)) * torch.eye(K, dtype=Hm.dtype)
    L = torch.linalg.cholesky(Hm); Hinv = torch.cholesky_inverse(L); Hinv = torch.linalg.cholesky(Hinv, upper=True)
    scale = (W.abs().amax(dim=1) / 127.0).clamp(min=1e-12); Q = torch.zeros_like(W)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        mask1 = torch.zeros_like(W1, dtype=torch.bool)
        for i in range(i2 - i1):
            if sparse24 and i % 4 == 0:
                tmp = W1[:, i:i + 4] ** 2 / (torch.diag(Hinv1)[i:i + 4].reshape(1, -1)) ** 2
                mask1.scatter_(1, i + torch.topk(tmp, 2, dim=1, largest=False)[1], True)
            w = W1[:, i]; d = Hinv1[i, i]
            q = torch.clamp(torch.round(w / scale), -127, 127) * scale
            if sparse24: q[mask1[:, i]] = 0
            Q1[:, i] = q
            err = (w - q) / d; W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0); Err1[:, i] = err
        Q[:, i1:i2] = Q1; W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    codes = torch.clamp(torch.round(Q / scale.unsqueeze(1)), -127, 127).to(torch.int8)
    return codes, scale.float()

class Site:
    def __init__(self, block, fam, lins): self.block, self.fam, self.lins = block, fam, lins

cache = {}; act_over = {}; bias_over = {}; report = {}; smooth_over = {}
hs = embed_all(); hs_fp = [h.clone() for h in hs] if a.stream_match > 0 else None; t0 = time.time()
for b, layer in enumerate(bb.layers):
    with torch.no_grad():
        ln1, ln2 = layer.layer_norm1, layer.layer_norm2; att = layer.attention; mlp = layer.mlp
        # ---- SmoothQuant folds (alpha) into LN affine (qkv, fc1) and into V columns (proj), computed on the current inputs ----
        def site_inputs(fn_pre):
            """collect the inputs of q_proj/o_proj/fc1/fc2 for all images through the current (partially quantized) layer"""
            caught = {k: [] for k in ("qkv", "proj", "fc1", "fc2", "r1")}
            hooks = []
            for fam_, mods_ in (("qkv", [att.q_proj, att.k_proj, att.v_proj]), ("proj", [att.o_proj]), ("fc1", [mlp.fc1]), ("fc2", [mlp.fc2])):   # this block's already-fixed activation quantizers
                if f"block{b}.{fam_}" in act_over:
                    s__ = act_over[f"block{b}.{fam_}"]; fq__ = fq_act_mix if fam_ in a.token_mix_fams.split(",") else fq_act
                    for md in mods_: hooks.append(md.register_forward_pre_hook(lambda m_, i, s__=s__, fq__=fq__: (fq__(i[0], s__),)))
            hooks += [layer.layer_norm2.register_forward_pre_hook(lambda m_, i: caught["r1"].append(i[0].detach())),
                     att.q_proj.register_forward_pre_hook(lambda m_, i: caught["qkv"].append(i[0].detach())),
                     att.o_proj.register_forward_pre_hook(lambda m_, i: caught["proj"].append(i[0].detach())),
                     mlp.fc1.register_forward_pre_hook(lambda m_, i: caught["fc1"].append(i[0].detach())),
                     mlp.fc2.register_forward_pre_hook(lambda m_, i: caught["fc2"].append(i[0].detach()))]
            for h in hs: layer(h)
            for hk in hooks: hk.remove()
            return caught
        Yfp = None
        if a.stream_match > 0:   # FP32 stream targets: outputs of the six linears + LN2 input on the FP32 stream (block weights still exact here)
            Yfp = {k: [] for k in ("q", "k", "v", "proj", "fc1", "fc2", "r1")}
            hk = [att.q_proj.register_forward_hook(lambda m_, i, o: Yfp["q"].append(o.detach().half())), att.k_proj.register_forward_hook(lambda m_, i, o: Yfp["k"].append(o.detach().half())),
                  att.v_proj.register_forward_hook(lambda m_, i, o: Yfp["v"].append(o.detach().half())), att.o_proj.register_forward_hook(lambda m_, i, o: Yfp["proj"].append(o.detach().half())),
                  mlp.fc1.register_forward_hook(lambda m_, i, o: Yfp["fc1"].append(o.detach().half())), mlp.fc2.register_forward_hook(lambda m_, i, o: Yfp["fc2"].append(o.detach().half())),
                  layer.layer_norm2.register_forward_pre_hook(lambda m_, i: Yfp["r1"].append(i[0].detach()))]
            hs_fp_next = [layer(h) for h in hs_fp]
            for h_ in hk: h_.remove()
        X = site_inputs(None)
        SF = set(a.smooth_fams.split(","))
        if a.smooth > 0 and b in WBLK:
            for fam, lnm, lins in [t_ for t_ in (("qkv", ln1, [att.q_proj, att.k_proj, att.v_proj]), ("fc1", ln2, [mlp.fc1])) if t_[0] in SF]:
                xmax = torch.stack([t.abs().flatten(0, -2).amax(0) for t in X[fam]]).amax(0).clamp(min=1e-5)
                wmax = torch.cat([l.weight.abs().amax(0, keepdim=True) for l in lins]).amax(0).clamp(min=1e-5)   # per input channel
                sv = (xmax ** a.smooth) / (wmax ** (1 - a.smooth))
                lnm.weight.div_(sv); lnm.bias.div_(sv); smooth_over[f"block{b}.{fam}"] = sv.tolist()
                for l in lins: l.weight.mul_(sv[None, :])
            # proj: fold into V columns (v_proj output channels) — attention output channel i = sum_j p_ij v_j[i], so scaling v's output
            # channel i by 1/s_i scales proj's input channel i by 1/s_i; compensate in proj's input columns.
            if "proj" in SF:
                xmax = torch.stack([t.abs().flatten(0, -2).amax(0) for t in X["proj"]]).amax(0).clamp(min=1e-5)
                wmax = att.o_proj.weight.abs().amax(0).clamp(min=1e-5); sv = (xmax ** a.smooth) / (wmax ** (1 - a.smooth))
                att.v_proj.weight.div_(sv[:, None]); att.v_proj.bias.div_(sv); att.o_proj.weight.mul_(sv[None, :]); smooth_over[f"block{b}.proj"] = sv.tolist()
            X = site_inputs(None)   # inputs changed by the folds
        if a.fc2_chan > 0 and b in WBLK:
            # per-input-channel scale on fc2's input (GELU output): x_k / c_k with W2[:, k] * c_k  (exact; the plugin applies 1/c_k in fc1's requant epilogue)
            xmax = torch.stack([t.abs().flatten(0, -2).amax(0) for t in X["fc2"]]).amax(0).clamp(min=1e-5)
            wmax = mlp.fc2.weight.abs().amax(0).clamp(min=1e-5); cv = (xmax ** a.fc2_chan) / (wmax ** (1 - a.fc2_chan)); cv = cv / cv.mean()
            mlp.fc2.weight.mul_(cv[None, :]); smooth_over[f"block{b}.fc2"] = cv.tolist()
            mlp.fc2._lq_fc2_hook = mlp.fc2.register_forward_pre_hook(lambda m_, i, cv=cv: (i[0] / cv,))   # the model now sees the scaled input
            X = site_inputs(None)
        # ---- per site: activation scale (if A8 here), Hessian from (fake-quantized) inputs, GPTQ, bias correction ----
        for fam, lins in (("qkv", [att.q_proj, att.k_proj, att.v_proj]), ("proj", [att.o_proj]), ("fc1", [mlp.fc1]), ("fc2", [mlp.fc2])):
            a8 = (b in ABLK) or fam == "proj"
            if a.stream_match > 0 and fam != "qkv": X = site_inputs(None)          # inputs through the already-quantized earlier families of this block
            TM = fam in a.token_mix_fams.split(",")
            xs = [t.reshape(-1, t.shape[-1]) for t in X[fam]]
            sa = (ACT_FIX.get(f"block{b}.{fam}") or q_act_scale(mixed_values(X[fam]) if TM else xs)) if a8 else None
            if a8 and RM is not None and b in RBLK and fam in ("proj", "fc2") and not ACT_FIX.get(f"block{b}.{fam}"):
                Xs = torch.cat(xs)[torch.randperm(sum(x.shape[0] for x in xs))[:8000]]; Wm = (RM @ lins[0].weight.detach()).t()          # [K, R]: site error -> read-out features
                best = None
                for mult in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3):
                    s_ = sa * mult; err = ((fq_act(Xs, s_) - Xs) @ Wm).pow(2).sum().item()
                    if best is None or err < best[0]: best = (err, s_)
                print(f"  read-out clipping block{b}.{fam}: scale x{best[1] / sa:.2f}", flush=True); sa = best[1]
            if a8: act_over[f"block{b}.{fam}"] = sa
            if b not in WBLK: continue
            # Hessian on the inputs the engine will see
            n_tok = sum(x.shape[0] for x in xs); keep = min(a.max_tokens, n_tok); idx = torch.randperm(n_tok)[:keep]
            Xall = torch.cat(xs)[idx]; Xq = ((torch.cat([fq_act_mix(t, sa).reshape(-1, t.shape[-1]) for t in X[fam]])[idx]) if TM else fq_act(Xall, sa)) if a8 else Xall
            Hm = (Xq.double().t() @ Xq.double()) * (2.0 / keep)
            if a.stream_match > 0:   # accumulated residual-stream error the writers must cancel: e = stream_fp - stream_q at the writer's residual add
                if fam == "proj": e_add = (torch.cat([h.reshape(-1, h.shape[-1]) for h in hs_fp]) - torch.cat([h.reshape(-1, h.shape[-1]) for h in hs]))[idx]
                elif fam == "fc2": e_add = (torch.cat([t.reshape(-1, t.shape[-1]) for t in Yfp["r1"]]) - torch.cat([t.reshape(-1, t.shape[-1]) for t in X["r1"]]))[idx]
                else: e_add = None
                XtX = (Xq.double().t() @ Xq.double()); XtX += a.ridge * torch.diag(XtX).mean() * torch.eye(XtX.shape[0], dtype=XtX.dtype); XtX_inv = torch.linalg.inv(XtX)
            for role, lin in zip(("q", "k", "v") if fam == "qkv" else (fam,), lins):
                W = lin.weight.detach()                     # [N,K]
                T = None
                if a.stream_match > 0 and fam in a.stream_match_fams.split(","):
                    T = torch.cat([t.reshape(-1, t.shape[-1]) for t in Yfp[role]])[idx].float() - lin.bias.detach()[None, :]   # FP32 output (bias removed)
                    if e_add is not None: T = T + e_add
                    Wls = (XtX_inv @ (Xq.double().t() @ T.double())).t().float()                                         # [N,K] least-squares weight on the quantized inputs
                    dW = a.stream_match * (Wls - W)
                    if RM is not None and b == max(RBLK) and fam in ("proj", "fc2"): dW = PM @ dW                        # last block: only the components the neck can see
                    W = W + dW
                codes, sc = gptq(W, Hm, sparse24=(fam in a.sparse24_fams.split(",")))   # [N,K] int8, [N]
                Wq = codes.float() * sc[:, None]
                if fam in a.sparse24_fams.split(","): print(f"  block{b}.{fam}: zero fraction {(codes == 0).float().mean().item():.3f}", flush=True)
                if not a.no_bias_corr:                      # b <- b - E[(Xq Wq^T) - target]   (target = X W^T, or the FP32-stream target when stream matching)
                    dy = (Xq @ Wq.t() - (T if T is not None else Xall @ W.t())).mean(0); lin.bias.sub_(dy)
                    bias_over[f"block{b}.{fam}" + (":" + role if fam == "qkv" else "")] = lin.bias.detach().clone().numpy().tolist()
                lin.weight.copy_(Wq)                         # the layer now runs with quantized weights (propagation)
                key = f"block{b}.{fam}"
                if fam == "qkv":
                    cache.setdefault(key + "::codes", []).append(codes.t().contiguous().numpy()); cache.setdefault(key + "::scale", []).append(sc.numpy())
                else: cache[key + "::codes"] = codes.t().contiguous().numpy(); cache[key + "::scale"] = sc.numpy()
            if fam == "qkv": cache[f"block{b}.qkv::codes"] = np.concatenate(cache[f"block{b}.qkv::codes"], 1); cache[f"block{b}.qkv::scale"] = np.concatenate(cache[f"block{b}.qkv::scale"])
        # ---- fake-quantize this block's activations for propagation, then run the block on all images ----
        hooks = []
        if b in ABLK or True:
            for fam, mods in (("qkv", [att.q_proj, att.k_proj, att.v_proj]), ("proj", [att.o_proj]), ("fc1", [mlp.fc1]), ("fc2", [mlp.fc2])):
                if f"block{b}.{fam}" in act_over:
                    s_ = act_over[f"block{b}.{fam}"]; fq_ = fq_act_mix if fam in a.token_mix_fams.split(",") else fq_act
                    for md in mods: hooks.append(md.register_forward_pre_hook(lambda m_, i, s_=s_, fq_=fq_: (fq_(i[0], s_),)))
        hs = [layer(h) for h in hs]
        for hk in hooks: hk.remove()
        if a.stream_match > 0: hs_fp = hs_fp_next; Yfp = None
    print(f"block {b} done ({time.time()-t0:.0f}s)", flush=True)
    np.savez(a.out + "_cache.npz", **{k: v for k, v in cache.items() if not isinstance(v, list)})
    json.dump({"act_scales": act_over}, open(a.out + "_act.json", "w"), indent=1); json.dump(bias_over, open(a.out + "_bias.json", "w")); json.dump(smooth_over, open(a.out + "_smooth.json", "w"))
print("done ->", a.out + "_cache.npz")
