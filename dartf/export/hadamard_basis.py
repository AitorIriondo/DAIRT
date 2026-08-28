"""Hadamard-basis rewrite of the HF SAM3 ViT trunk (exact). See export_hf_hadamard.py for the derivation."""
import math, os, torch, torch.nn as nn
import transformers.models.sam3.modeling_sam3 as M

def hadamard(n):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)

class MaskedRMSNorm(nn.Module):
    """LayerNorm expressed in the Hadamard basis: zero coordinate 0, divide by sqrt((sum_{i>=1} x_i^2)/C + eps)."""
    def __init__(self, C, eps):
        super().__init__(); self.C = C; self.eps = eps
        self.register_buffer("mask", torch.ones(C)); self.mask[0] = 0.0
    def forward(self, x):
        xm = x * self.mask
        var = (xm * xm).sum(-1, keepdim=True) / self.C
        return xm * torch.rsqrt(var + self.eps)

@torch.no_grad()
def prepare(vit, neck_first_linear=None, Q=None, head_rot=None):
    """Q: optional orthogonal [C,C] rotation whose ones-direction is coordinate 0 (default: Hadamard).
    head_rot: also rotate each head's V subspace by a d x d Hadamard (exact: a = P V  ->  a R = P (V R)); folds into v_proj (rows) and
    o_proj (columns) and lowers the crest factor of the o_proj INPUT, which no residual-stream rotation can reach. Default from env LQ_HEAD_ROT."""
    if head_rot is None: head_rot = os.environ.get("LQ_HEAD_ROT", "0") == "1"
    C = vit.config.hidden_size; H = (hadamard(C).to(torch.float32) if Q is None else torch.as_tensor(Q, dtype=torch.float32)); Ht = H.t().contiguous()
    assert (torch.ones(C) @ H)[1:].abs().max() < 1e-3, 'rotation must map the ones-direction to coordinate 0'
    eps = vit.config.layer_norm_eps
    # pre-trunk LayerNorm (vit.layer_norm) -> masked RMSNorm; its affine folds into every consumer of the residual
    # stream... it feeds the residual directly (x0 = LN_pre(embed)), so fold gamma/beta as: x0 = n * g + b -> x0~ = n (diag(g) H) + b H.
    # We keep this as an explicit affine in the rotated basis: x0~ = (n H) (H^T diag(g) H) + b H.  (H^T diag(g) H is dense but
    # applied once per frame; cheaper: apply diag(g), add b, then rotate.)  We rotate explicitly once after the pre-LN.
    ln = vit.layer_norm; vit._had = H; vit._had_pre = (ln.weight.detach().clone(), ln.bias.detach().clone()); vit._had_eps = eps
    for layer in vit.layers:
        for lnname, lins in (("layer_norm1", [layer.attention.q_proj, layer.attention.k_proj, layer.attention.v_proj]), ("layer_norm2", [layer.mlp.fc1])):
            lnm = getattr(layer, lnname); g, b = lnm.weight.detach(), lnm.bias.detach()
            for lin in lins:                       # y = LN(x) g W^T + b W^T + bias   (torch Linear: y = x W^T + bias, W [out,in])
                W = lin.weight.detach()            # [out, in]
                Wg = W * g[None, :]                # W diag(g) in the "x W^T" convention: (x g) W^T = x (diag(g) W^T) -> rows of W^T scaled -> W[:, i] * g[i]
                lin.bias.copy_(lin.bias.detach() + W @ b)   # beta W^T folded into the bias
                lin.weight.copy_(Wg @ H)           # z = LN(x) H  ->  z (H^T diag(g) W^T) = z (Wg @ H)^T  since (H^T Wg^T)^T = Wg H
            setattr(layer, lnname, MaskedRMSNorm(C, eps))
        for lin in (layer.attention.o_proj, layer.mlp.fc2):   # output added to rotated residual: y~ = y H = x W^T H + b H
            W = lin.weight.detach(); lin.weight.copy_(Ht @ W); lin.bias.copy_(lin.bias.detach() @ H)
        if head_rot:                                           # per-head V rotation: v' = v (I (x) R), o_proj consumes a' = a (I (x) R)
            d = C // layer.attention.num_attention_heads if hasattr(layer.attention, "num_attention_heads") else C // vit.config.num_attention_heads
            R = hadamard(d).to(torch.float32); nh = C // d; IR = torch.block_diag(*([R] * nh))          # [C, C]
            v = layer.attention.v_proj; v.weight.copy_(IR.t() @ v.weight.detach()); v.bias.copy_(v.bias.detach() @ IR)   # y' = y (I(x)R) = x (W^T (I(x)R)) + b (I(x)R)
            o = layer.attention.o_proj; o.weight.copy_(o.weight.detach() @ IR)                                          # y = a W^T = a' (I(x)R)^T W^T = a' (W (I(x)R))^T
    return vit

def forward_had(self, pixel_values, **kw):
    h = self.embeddings(pixel_values); B = h.shape[0]; Hh = pixel_values.shape[-2] // self.config.patch_size; Ww = pixel_values.shape[-1] // self.config.patch_size; C = h.shape[-1]
    h = h.view(B, Hh, Ww, C); g, b = self._had_pre
    h = torch.nn.functional.layer_norm(h, (C,), g, b, self._had_eps) @ self._had          # rotate once
    for layer in self.layers: h = layer(h, **kw)
    h = h @ self._had.t()                                                                   # un-rotate once
    return M.BaseModelOutput(last_hidden_state=h.view(B, Hh * Ww, C))

