"""DARTF explainer for a CVPR / NeurIPS audience: slide set (PDF) and video (MP4) rendered from one scene script.
Black background, Lato text, Computer Modern equations (matplotlib mathtext), measured flow layout (blocks stack by their
rendered height, so nothing can overlap), staggered fade ins and simple figure animations.
usage: python make_media.py <out_dir> [--video] [--fps 30] [--preview i,j,k]"""
import sys, os, re, math, random, subprocess, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["mathtext.fontset"] = "cm"
from matplotlib import mathtext
W, H = 1920, 1080; MARGIN = 120; COLW = W - 2 * MARGIN
FONTS = {"r": "/usr/share/fonts/truetype/lato/Lato-Regular.ttf", "m": "/usr/share/fonts/truetype/lato/Lato-Medium.ttf", "s": "/usr/share/fonts/truetype/lato/Lato-Semibold.ttf", "b": "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"}
BLUE, YELLOW, GREEN, RED, WHITE, GREY, TEAL, ORANGE = (88, 196, 221), (247, 214, 98), (131, 193, 103), (252, 98, 85), (236, 236, 236), (150, 150, 150), (94, 210, 188), (245, 160, 80)
ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:\n")
def check(t):
    bad = sorted(set(c for c in t if c not in ALLOWED))
    if bad: raise SystemExit(f"disallowed characters {bad} in prose: {t!r}")
    return t
_fonts = {}
def font(size, w="r"):
    k = (size, w)
    if k not in _fonts: _fonts[k] = ImageFont.truetype(FONTS[w], size)
    return _fonts[k]
def col(c, a): return tuple(int(v * max(0.0, min(1.0, a))) for v in c)
def ease(x): x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)
def hexc(c): return "#%02x%02x%02x" % c
# ---------------------------------------------------------------- equations (mathtext, cached) ----------------------------------------------
_eqs = {}
def eq_image(tex, px, color):
    """render $tex$ so that a lowercase x is about 0.5*px tall; returns RGBA PIL image"""
    k = (tex, px, color)
    if k in _eqs: return _eqs[k]
    buf = io.BytesIO(); mathtext.math_to_image(f"${tex}$", buf, dpi=72 * 4, format="png", color="black"); buf.seek(0)
    lum = np.asarray(Image.open(buf).convert("L")).astype(np.float32); alpha = np.clip((255 - lum) / 255.0, 0, 1)   # black ink on white -> alpha
    rgba = np.zeros(lum.shape + (4,), np.uint8); rgba[..., 0], rgba[..., 1], rgba[..., 2] = color; rgba[..., 3] = (alpha * 255).astype(np.uint8)
    img = Image.fromarray(rgba); ys, xs = np.where(rgba[..., 3] > 8)
    if len(ys): img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    scale = (px / 10.0) / 4.0 * 1.18     # math_to_image uses 10 pt; 4x oversampling; CM glyphs run small next to Lato
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS); _eqs[k] = img; return img
# ---------------------------------------------------------------- blocks ------------------------------------------------------------------
def P(text, size=34, color=WHITE, w="r", t0=0.0, gap=14, indent=0): return dict(kind="p", s=check(text), size=size, color=color, w=w, t0=t0, gap=gap, indent=indent)
def T(text, size=58): return dict(kind="p", s=check(text), size=size, color=YELLOW, w="b", t0=0.0, gap=26, indent=0)
def E(tex, size=40, color=BLUE, t0=0.0, gap=18, indent=60): return dict(kind="eq", tex=tex, size=size, color=color, t0=t0, gap=gap, indent=indent)
def G(px): return dict(kind="gap", h=px)
def F(kind, h, t0=0.0, dur=2.0, gap=18, **kw): return dict(kind="fig", fig=kind, h=h, t0=t0, dur=dur, gap=gap, **kw)
def wrap(s, f, maxw):
    lines, cur = [], ""
    for wd in s.split(" "):
        c = (cur + " " + wd).strip()
        if f.getlength(c) <= maxw or not cur: cur = c
        else: lines.append(cur); cur = wd
    lines.append(cur); return lines
def eq_fit(b):
    im = eq_image(b["tex"], b["size"], b["color"]); maxw = COLW - b["indent"]
    if im.width > maxw: im = im.resize((maxw, max(1, int(im.height * maxw / im.width))), Image.LANCZOS)
    return im
def block_height(b):
    if b["kind"] == "p": f = font(b["size"], b["w"]); return len(wrap(b["s"], f, COLW - b["indent"])) * int(b["size"] * 1.32)
    if b["kind"] == "eq": return eq_fit(b).height + 6
    if b["kind"] == "gap": return b["h"]
    return b["h"]
def layout(blocks):
    y = 84; out = []
    for b in blocks:
        h = block_height(b); out.append((y, b)); y += h + b.get("gap", 0)
    return out, y
# ---------------------------------------------------------------- figures -------------------------------------------------------------------
def draw_fig(d, img, b, y, a):
    x0 = MARGIN; k = b["fig"]
    if k == "bars":
        items = b["items"]; n = len(items); bh = min(60, (b["h"] - 10) // n - 12); labw = 420; bw = COLW - labw - 260
        for i, (label, frac, color, value) in enumerate(items):
            ai = ease((a * n * 1.15 - i) ); yy = y + i * (bh + 14); wfill = int(bw * frac * ai)
            d.text((x0 + labw - 24, yy + bh // 2), check(label), font=font(30, "m"), fill=col(WHITE, min(1, ai * 2)), anchor="rm")
            d.rectangle([x0 + labw, yy, x0 + labw + max(wfill, 1), yy + bh], fill=col(color, 0.92))
            if ai >= 0.999: d.text((x0 + labw + wfill + 18, yy + bh // 2), check(value), font=font(30, "s"), fill=col(color, 1), anchor="lm")
    elif k == "spikes":
        n = 40; rng = random.Random(3); bw = COLW / n
        for i in range(n):
            base = 0.12 + 0.08 * rng.random(); spike = 1.0 if i == 11 else base; hgt = (spike * (1 - a) + (0.24 + 0.08 * rng.random()) * a) * (b["h"] - 30)
            d.rectangle([x0 + i * bw + 3, y + b["h"] - 30 - hgt, x0 + (i + 1) * bw - 3, y + b["h"] - 30], fill=col(RED if (i == 11 and a < 0.5) else BLUE, 0.9))
        d.text((x0, y + b["h"] - 24), check("channels of one token: before the rotation one channel carries most of the range; after it the profile is flat") if a < 0.5 else check("after the rotation: crest factor 4.07, the value of a Gaussian vector"), font=font(24), fill=col(GREY, 1), anchor="la")
    elif k == "cloud":
        cx, cy, r = x0 + COLW // 2, y + b["h"] // 2, b["h"] // 2 - 20; rng = random.Random(7); pts = [(rng.gauss(0, 1.0), rng.gauss(0, 0.12)) for _ in range(160)]; th = a * math.pi / 4
        d.line([cx - r * 1.4, cy, cx + r * 1.4, cy], fill=col(GREY, 0.5), width=2); d.line([cx, cy - r * 1.1, cx, cy + r * 1.1], fill=col(GREY, 0.5), width=2)
        for (px, py) in pts:
            qx, qy = px * math.cos(th) - py * math.sin(th), px * math.sin(th) + py * math.cos(th); X, Y = cx + max(-1.4, min(1.4, qx)) * r * 0.7, cy - max(-0.95, min(0.95, qy)) * r; d.ellipse([X - 5, Y - 5, X + 5, Y + 5], fill=col(BLUE, 0.9))
        d.text((cx + r * 1.5, cy), check("the same vectors in the rotated basis") if a > 0.5 else check("activations with one dominant channel"), font=font(26), fill=col(YELLOW, 1), anchor="lm")
    elif k == "perm":
        n = 8; sp = 84; yy = y + 10
        for i in range(n):
            j = (i + 1) if i % 2 == 0 else (i - 1); xi = x0 + 200 + i * sp; xj = x0 + 200 + j * sp; xx = xi + (xj - xi) * a; sign = -1 if (i % 2 == 1 and a > 0.5) else 1
            d.rectangle([xx, yy, xx + 60, yy + b["h"] - 60], fill=col(RED if sign < 0 else BLUE, 0.9)); d.text((xx + 30, yy + b["h"] - 40), check("w" + str(i + 1)), font=font(24), fill=col(WHITE, 1), anchor="ma")
        d.text((x0 + 200 + n * sp + 60, yy + (b["h"] - 60) // 2), check("pairs swap and one of each pair changes sign: exact on INT8 codes") if a > 0.5 else check("columns of the query projection weight"), font=font(26), fill=col(YELLOW, 1), anchor="lm")
    elif k == "lanes":
        lw = COLW - 500; lh = 44; y1, y2 = y + 10, y + 10 + lh + 30; xs = x0 + 260
        d.text((xs - 24, y1 + lh // 2), check("tensor pipe"), font=font(26, "m"), fill=col(WHITE, 1), anchor="rm"); d.text((xs - 24, y2 + lh // 2), check("FP32 softmax pipe"), font=font(26, "m"), fill=col(WHITE, 1), anchor="rm")
        seq = a < 0.5; p = ease(a * 2) if seq else 1.0; q = ease((a - 0.5) * 2)
        if seq or q < 1:
            for i in range(4): xx = xs + i * 270; d.rectangle([xx, y1, xx + 120 * p, y1 + lh], fill=col(BLUE, 0.9)); d.rectangle([xx + 130, y2, xx + 130 + 130 * p, y2 + lh], fill=col(YELLOW, 0.9))
            cap = "measured: the two pipes take turns; their times add"
        if not seq:
            for i in range(4): xx = xs + i * 150; d.rectangle([xx, y1, xx + 120 * q, y1 + lh], fill=col(BLUE, 0.9)); d.rectangle([xx + 20, y2, xx + 20 + 130 * q, y2 + lh], fill=col(YELLOW, 0.9))
            cap = "pipelined: the score tile of step t plus 1 is issued before the softmax of step t"
        d.text((x0, y + b["h"] - 24), check(cap), font=font(24), fill=col(GREY, 1), anchor="la")
    elif k == "blocks":
        names = ["masked RMSNorm", "quantize", "q k v GEMMs, RoPE", "attention", "proj GEMM, residual", "masked RMSNorm", "quantize", "fc1 GEMM, GELU", "fc2 GEMM, residual"]; k2 = int(a * len(names) + 0.999)
        for i, nm in enumerate(names[:k2]):
            xx = x0 + (i % 5) * 330; yy = y + (i // 5) * 100; c = GREEN if i < 5 else TEAL
            d.rectangle([xx, yy, xx + 300, yy + 70], outline=col(c, 1), width=3); d.text((xx + 150, yy + 35), check(nm), font=font(24), fill=col(WHITE, 1), anchor="mm")
        d.text((x0, y + b["h"] - 24), check("three plugins per block; INT8 tensors never cross a plugin boundary, only the fp16 residual stream does"), font=font(24), fill=col(GREY, 1), anchor="la")
    elif k == "grid":
        n = 6; s = 32; k2 = int(a * n * n); gx0 = x0 + COLW // 2 - 260
        for i in range(n * n):
            r, c = i // n, i % n; win = (r // 3) * 2 + (c // 3); cc = [BLUE, YELLOW, GREEN, TEAL][win]
            if i < k2: X, Y = gx0 + 300 + (win % 2) * 120 + (c % 3) * s, y + (win // 2) * 120 + (r % 3) * s
            else: X, Y = gx0 + c * s, y + r * s
            d.rectangle([X, Y, X + s - 5, Y + s - 5], fill=col(cc, 0.9))
        d.text((x0, y + b["h"] - 24), check("tokens are partitioned into windows once; every block then sees the same layout"), font=font(24), fill=col(GREY, 1), anchor="la")
def render(blocks, t):
    img = Image.new("RGB", (W, H), (0, 0, 0)); d = ImageDraw.Draw(img); placed, total = layout(blocks)
    for y, b in placed:
        a = ease((t - b.get("t0", 0.0)) / 0.7) if b["kind"] in ("p", "eq") else ease((t - b.get("t0", 0.0)) / b.get("dur", 2.0))
        if a <= 0 or b["kind"] == "gap": continue
        if b["kind"] == "p":
            f = font(b["size"], b["w"]); lh = int(b["size"] * 1.32)
            for i, ln in enumerate(wrap(b["s"], f, COLW - b["indent"])): d.text((MARGIN + b["indent"], y + i * lh), ln, font=f, fill=col(b["color"], a))
        elif b["kind"] == "eq":
            im = eq_fit(b)
            if a < 1: im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * a)))
            img.paste(im, (MARGIN + b["indent"], y + 3), im)
        elif b["kind"] == "fig": draw_fig(d, img, b, y, a)
    return img, total
# ---------------------------------------------------------------- credits from the verified bib -------------------------------------------
def credits_from_bib(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(path))); import check_references as cr
    labels = {aid: label for aid, _, label in cr.REFS}; labels["2603.11441"] = "DART"; out = []
    for entry in re.split(r"\n(?=@)", open(path).read()):
        a = re.search(r"author\s*=\s*\{(.*?)\}\s*,", entry, re.S); y = re.search(r"year\s*=\s*\{?(\d{4})", entry); e = re.search(r"eprint\s*=\s*\{?([\d.]+)", entry)
        if not (a and y and e): continue
        first = a.group(1).split(" and ")[0].strip(); last = first.split(",")[0].strip() if "," in first else first.split()[-1]
        out.append((re.sub(r"[^A-Za-z0-9 ]", " ", labels.get(e.group(1), "")).strip(), re.sub(r"[^A-Za-z ]", "", last) + (" et al." if " and " in a.group(1) else ""), y.group(1)))
    return out
# ---------------------------------------------------------------- the script ---------------------------------------------------------------
def scenes(bib):
    S = []
    S.append((8.0, [G(300), P("DARTF", 150, YELLOW, "b", 0.0, 10), P("Detect Anything in Real Time Faster", 60, WHITE, "m", 0.8, 24),
                    P("W8A8 deployment of the SAM 3 ViT H detector on a Jetson AGX Orin with exact reparametrizations, activation aware GPTQ and block level TensorRT plugins.", 34, BLUE, "r", 1.8, 40),
                    P("Mehmet Kerem Turkcan, 2026", 30, GREY, "r", 2.8)]))
    S.append((16.0, [T("Setting and result"),
                     P("SAM 3 is an open vocabulary detector. Its ViT H backbone has 32 transformer blocks of width 1024 and 5184 tokens per 1008 by 1008 frame, with 4 global and 28 windowed attention blocks.", 34, WHITE, "r", 0.6),
                     P("On a Jetson AGX Orin the FP16 TensorRT engine needs 275 ms per frame. The goal is 8 bit weights and 8 bit activations with detection quality equal to FP32.", 34, GREY, "r", 2.0, 30),
                     F("bars", 240, 3.5, 3.0, items=[("FP16 engine", 1.0, BLUE, "275 ms; 13.2 J"), ("DARTF W8A8", 158 / 275, GREEN, "158 ms; 7.7 J")]),
                     P("All 32 blocks run in INT8, including block 0, which consumes the raw patch embedding and which no earlier recipe could quantize; keeping it in FP16 costs 6 ms and buys nothing measurable.", 30, GREY, "r", 7.0),
                     P("On the 5000 image COCO val set the INT8 engine scores 56.0 AP against 56.1 for FP32, with every size and recall metric within 0.2 points; on 100 images every detection level count matches within one.", 32, WHITE, "r", 8.5),
                     P("43 percent less time at FP32 quality, with no change to the model, the resolution or the runtime.", 36, GREEN, "s", 10.5)]))
    S.append((16.0, [T("Background: uniform quantization and where its error comes from"),
                     P("A symmetric 8 bit quantizer with a single static scale per tensor, which is what TensorRT executes at full speed:", 34, WHITE, "r", 0.5),
                     E(r"Q_s(x) = s\,\mathrm{clip}\!\left(\mathrm{round}(x/s),\,-127,\,127\right),\qquad s = \max|x| / 127", 42, BLUE, 1.2),
                     P("Rounding error is uniform with variance one twelfth of the step, so the relative error of a tensor is set by its crest factor, the ratio of its largest value to its root mean square:", 34, WHITE, "r", 3.0),
                     E(r"\frac{\|x - Q_s(x)\|}{\|x\|} \;\approx\; \frac{s/\sqrt{12}}{\mathrm{rms}(x)} \;=\; \frac{\max|x|}{127\sqrt{3}\;\mathrm{rms}(x)} \;\approx\; \frac{\mathrm{crest}(x)}{220}", 42, BLUE, 4.0),
                     P("Per token scales would track the range of each token but are not available inside the fused TensorRT kernels; the whole problem is therefore to lower the crest factor of every quantized tensor by exact transformations of the network.", 34, GREY, "r", 7.0),
                     P("Measured on SAM 3 the crest factor at the LayerNorm fed sites is about 28, close to the square root of the width: a single channel per token carries the whole range, so per tensor INT8 is hopeless there without a change of basis. The proj and fc2 inputs, behind the softmax and the GELU, carry token wise heavy tails instead, which a change of basis cannot fix.", 34, WHITE, "r", 10.0)]))
    S.append((18.0, [T("Background: the methods this work builds on"),
                     P("GPTQ rounds weights with second order information. With inputs X it solves column by column, propagating each rounding error to the remaining columns through the inverse Hessian:", 32, WHITE, "r", 0.5),
                     E(r"\min_{\hat W}\ \|(W - \hat W)\,X\|_F^2,\qquad H = X X^{\top}", 40, BLUE, 1.5),
                     P("SmoothQuant migrates difficulty between an activation and its weight with a per channel scale, which is a diagonal reparametrization:", 32, WHITE, "r", 4.0),
                     E(r"X W = (X S^{-1})(S W),\qquad S = \mathrm{diag}\!\left(\max|X_{:,k}|^{\alpha}\,/\,\max|W_{k,:}|^{1-\alpha}\right)", 40, BLUE, 5.0),
                     P("QuaRot and SpinQuant use a full orthogonal transform instead, folded into the weights, so that outliers are spread over all channels. This is exact only because RMSNorm commutes with rotations:", 32, WHITE, "r", 8.0),
                     E(r"X W = (X Q)(Q^{\top} W),\qquad \mathrm{RMSNorm}(xQ) = \mathrm{RMSNorm}(x)\,Q", 40, BLUE, 9.0),
                     P("Vision transformers use LayerNorm, which subtracts the mean, and the identity above fails. The vision PTQ line, PTQ4ViT, FQ ViT and RepQ ViT, therefore worked with twin uniform quantizers, log quantizers and per channel reparametrizations of the LayerNorm affine, which are all diagonal and cannot mix channels.", 32, GREY, "r", 12.0)]))
    S.append((20.0, [T("What is different here: an exact rotation for LayerNorm"),
                     P("Take any orthogonal Q whose first column is the normalized ones vector, and write y for xQ. Then the mean of x is the first rotated coordinate and the centered norm is the norm of the remaining coordinates:", 32, WHITE, "r", 0.5),
                     E(r"Q^{\top}\mathbf{1} = \sqrt{C}\,e_0,\qquad y_0 = \sqrt{C}\,\mu,\qquad \|x - \mu\mathbf{1}\|^2 = \|y\|^2 - y_0^2", 40, BLUE, 1.8),
                     P("LayerNorm in the rotated basis is therefore an RMSNorm over coordinates 1 to C minus 1 with coordinate 0 set to zero, and the affine parameters fold into every consumer:", 32, WHITE, "r", 4.5),
                     E(r"\mathrm{LN}(x)\,Q = \frac{y - y_0 e_0}{\sigma},\qquad \sigma^2 = \frac{\|y\|^2 - y_0^2}{C} + \epsilon", 38, BLUE, 5.8, 10),
                     E(r"(\mathrm{LN}(x)\odot\gamma + \beta)\,W = \frac{y - y_0 e_0}{\sigma}\,\left(Q^{\top}\mathrm{diag}(\gamma)\,W\right) + \beta W", 38, BLUE, 6.4),
                     P("The residual stream runs rotated end to end: the patch embedding is rotated once, proj and fc2 outputs are rotated on the way in, the neck input is rotated back once. Nothing is computed online at inference; the rewritten network is exact and was verified against PyTorch.", 32, WHITE, "r", 9.5),
                     P("With a Walsh Hadamard Q the crest factor at the LayerNorm fed sites drops from 28 to 4, the value of a Gaussian vector, and block 0, which no rescaling could quantize, becomes an ordinary block. RepQ ViT can only rescale channels; QuaRot and SpinQuant need RMSNorm; this identity needs neither an online operation nor a change of normalization.", 32, GREEN, "r", 12.5),
                     ]))
    S.append((16.0, [T("What is different here: three more exact rewrites"),
                     P("The attention output is linear in V, so a per head orthogonal R folds into the value projection on the right and into the output projection on the left, as QuaRot does online for language models; here it is folded offline and it lowers the crest factor of the proj input, which no residual stream rotation can reach:", 32, WHITE, "r", 0.5),
                     E(r"a = P V,\qquad a\,(I_H\otimes R) = P\,\left(V (I_H\otimes R)\right):\quad W_v \leftarrow W_v (I_H\otimes R),\quad W_o \leftarrow (I_H\otimes R)^{\top} W_o", 38, BLUE, 2.0),
                     P("The rotary embedding pairs coordinates and rotates each pair by a position dependent angle. The pairwise rotation of the query is a signed column permutation of the projection weight, which is exact even on INT8 codes, so the rotated branch is just a second projection:", 32, WHITE, "r", 5.5),
                     E(r"\mathrm{RoPE}(q) = \cos\theta \odot q + \sin\theta \odot (q R_{\pi/2}),\qquad q R_{\pi/2} = x\,(W_q R_{\pi/2})", 36, BLUE, 7.0, 10),
                     E(r"R_{\pi/2} = I_{32}\otimes J,\qquad J = \left[\,0\ \ 1;\ -1\ \ 0\,\right],\qquad W_q R_{\pi/2}\ \mathrm{is\ a\ signed\ column\ permutation\ of}\ W_q", 36, BLUE, 7.6),
                     F("perm", 150, 9.5, 3.0),
                     P("Tokens are partitioned into windows once after the patch embedding and un partitioned once before the neck, with permuted RoPE tables in the global blocks. All 32 blocks become structurally identical, which is what allows one plugin design for the whole trunk.", 32, GREY, "r", 12.5)]))
    S.append((17.0, [T("What is different here: activation aware GPTQ"),
                     P("Standard GPTQ forms its Hessian from full precision inputs and quantizes weights only. Here blocks are quantized in order, each block sees the fake quantized outputs of the quantized prefix, and the activation quantizer of the site is applied before the Hessian is formed, so the weight rounding compensates the activation rounding it will meet:", 32, WHITE, "r", 0.5),
                     E(r"\hat X = Q_{s_a}(\tilde X),\qquad \hat W = \mathrm{GPTQ}\left(W,\ H = \hat X \hat X^{\top}\right),\qquad b \leftarrow b - \mathbb{E}_{t}\!\left[\hat X \hat W^{\top} - \tilde X W^{\top}\right]", 40, BLUE, 3.5),
                     P("The tilde marks inputs propagated through the quantized prefix; the expectation runs over calibration tokens. Scales, codes and biases are consumed verbatim by the ONNX quantizer, so the deployed graph is the one the Hessians saw.", 32, GREY, "r", 6.0),
                     P("The one site neither rotation reaches is the fc2 input behind the GELU. There the SmoothQuant migration is exact and free, because the plugin applies the per channel factor inside the fc1 requantization epilogue:", 32, WHITE, "r", 8.5),
                     E(r"x_k / c_k,\quad W_2[k,:]\,c_k,\qquad c_k = \max|x_k|^{1/2}\,/\,\max|W_2[k,:]|^{1/2}", 40, BLUE, 10.0),
                     F("bars", 190, 11.5, 3.0, items=[("plain GPTQ", 1.0, GREY, "100"), ("activation aware GPTQ", 0.129 / 0.146, BLUE, "88"), ("with the exact rotations", 0.113 / 0.146, TEAL, "77"), ("with fc2 channel scales", 0.105 / 0.146, GREEN, "72")]),
                     P("Feature error relative to plain GPTQ, in percent. Each step is free at inference: the same engine, the same latency.", 28, GREY, "r", 14.5)]))
    S.append((14.0, [T("The crest factor argument, measured"),
                     F("spikes", 220, 0.8, 3.0),
                     P("A learned rotation, trained on a crest factor surrogate, lowers the crest factor further but leaves the network error unchanged: once any rotation is applied, the LayerNorm fed sites have left the error budget. The Hadamard is enough.", 32, WHITE, "r", 5.0),
                     P("Per site ablations agree: after the rotations most of the remaining error sits behind the softmax and the GELU, where no residual stream rotation reaches. That is what the per head V rotation and the fc2 channel scales address.", 32, GREY, "r", 8.5),
                     P("Feature error stops predicting detection quality once it is small: engines that differ by a third in feature error are indistinguishable at detection level. Quality has to be judged on detections, against a measured noise floor.", 32, YELLOW, "r", 11.0)]))
    S.append((15.0, [T("Making TensorRT execute it: block level plugins"),
                     P("TensorRT explicit quantization has two rules that shape the design. A plugin is not handed the scale of an INT8 input that a QuantizeLinear node produces, and a plugin may not produce the INT8 tensor that a DequantizeLinear node consumes.", 32, WHITE, "r", 0.5),
                     P("A plugin that replaces a single fused kernel therefore becomes an island: the surrounding quantize and dequantize nodes lose their fusions. Only plugins that remove work or internalize whole subgraphs pay.", 32, RED, "r", 3.0),
                     F("blocks", 250, 5.0, 4.0),
                     P("Each block became three plugins on fp16 residual stream edges, with CUTLASS INT8 GEMMs, RoPE and GELU in the epilogues and the residual add fused. TensorRT keeps the graph and the scheduling; the plugins own everything between two residual adds. That alone took the trunk from 187 to 166 ms.", 32, WHITE, "r", 10.0)]))
    S.append((18.0, [T("A pipelined flash attention kernel for Ampere"),
                     P("Flash attention keeps a running maximum and normalizer per row and never materializes the score matrix:", 32, WHITE, "r", 0.5),
                     E(r"S_t = Q K_t^{\top},\qquad m_t = \max(m_{t-1}, \mathrm{rowmax}\,S_t),\qquad P_t = e^{S_t - m_t}", 36, BLUE, 1.5, 10),
                     E(r"\ell_t = e^{m_{t-1} - m_t}\,\ell_{t-1} + \mathrm{rowsum}\,P_t,\qquad O_t = e^{m_{t-1} - m_t}\,O_{t-1} + P_t V_t,\qquad O = O_T / \ell_T", 36, BLUE, 2.1),
                     P("The first kernel ran at less than half of the tensor pipe floor. Two measurements explained the gap: an ablation that compiled out one component at a time showed that the tensor work and the FP32 softmax work simply add, and a SASS histogram showed the loop dominated by address arithmetic rather than tensor instructions.", 32, WHITE, "r", 4.5),
                     F("lanes", 170, 8.0, 5.0),
                     P("The fix issues the score tile of step t plus 1 on the tensor pipe before the softmax of step t runs on the FP32 pipe, with the scores double buffered in registers, and hoists the shared memory addressing out of the loop. Global attention got 19 percent faster and windowed attention 12 percent, ahead of TensorRT on the same GPU, exact to fp16 rounding.", 32, GREEN, "r", 13.5)]))
    S.append((12.0, [T("The grounding head: exact sharing across prompts"),
                     P("Scoring a prompt is linear in the number of prompts, and half of it is global attention over the image tokens. Kernel fusion does not help there; TensorRT is already at parity. The lever is structure, not kernels.", 32, WHITE, "r", 0.5),
                     P("The first encoder layer depends only on the image. It is computed once per image by a prefix engine and broadcast to every prompt of a class bucket by the post engine; CUDA graph replay removes the launch overhead of the rest.", 32, WHITE, "r", 3.0),
                     F("bars", 150, 5.5, 3.0, items=[("monolithic engine", 1.0, GREY, ""), ("prefix engine plus post engine", 351 / 387, GREEN, "10 percent less per prompt")]),
                     P("The outputs are bit identical: the head is faster without a single approximation.", 32, YELLOW, "s", 9.0)]))
    S.append((16.0, [T("What is new, in five sentences"),
                     P("A closed form rotation for LayerNorm networks: any orthogonal transform that sends the ones direction to one coordinate turns LayerNorm into a masked RMSNorm, so rotation based quantization applies to vision transformers without online operations and without changing the normalization.", 32, WHITE, "r", 0.5),
                     P("A per head rotation of the value subspace and a signed permutation form of the rotary embedding, both folded into the weights, extend the exactness to the attention output and to the rotary branch.", 32, WHITE, "r", 3.5),
                     P("An activation aware sequential GPTQ with bias correction whose Hessians come from the quantized inputs the deployed graph will see, plus a per channel migration at the one site rotation cannot reach.", 32, WHITE, "r", 6.0),
                     P("A deployment recipe for explicit quantization in TensorRT: whole block plugins with INT8 kept inside, and a token major flash attention kernel that is faster than the vendor kernel on Ampere by overlapping its tensor and softmax pipes.", 32, WHITE, "r", 8.5),
                     P("A validation methodology at detection level with a measured noise floor, which shows that feature level error stops predicting detection quality once it is small, so the last rounding decisions must be judged on detections.", 32, WHITE, "r", 11.0)]))
    cr = credits_from_bib(bib); el = [T("Built on")]; rows = [f"{s}: {a} {y}" for s, a, y in cr]
    for i in range(0, len(rows), 2): el.append(P(";  ".join(rows[i:i + 2]), 28, WHITE if (i // 2) % 2 == 0 else GREY, "r", 0.6 + i * 0.18, 4))
    S.append((12.0, el))
    S.append((12.0, [T("Summary"),
                     P("Exact rewrites: rotated residual stream with a masked RMSNorm, per head V rotation, RoPE fold and window major layout.", 34, WHITE, "r", 0.8),
                     P("Activation aware GPTQ with bias correction and per channel fc2 scales.", 34, WHITE, "r", 2.0),
                     P("Block level TensorRT plugins and a pipelined attention kernel; exact prefix sharing in the head.", 34, WHITE, "r", 3.2),
                     G(30), P("275 ms to 158 ms per frame and 42 percent less energy, at FP32 detection quality.", 42, GREEN, "s", 4.6, 30),
                     P("Code and the full pipeline: the dartf folder of the DART repository.", 32, BLUE, "r", 6.0)]))
    return S
def write_video(S, mp4, fps):
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", mp4], stdin=subprocess.PIPE)
    n = 0
    for dur, bl in S:
        for f in range(int(dur * fps)):
            t = f / fps; a = min(1.0, t / 0.5); b = 1.0 if t < dur - 0.6 else max(0.0, (dur - t) / 0.6); arr = np.asarray(render(bl, t)[0])
            if a < 1 or b < 1: arr = (arr.astype(np.float32) * min(a, b)).astype(np.uint8)
            ff.stdin.write(arr.tobytes()); n += 1
    ff.stdin.close(); ff.wait(); return n
def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); fps = int(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 30
    bib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "references.bib"); S = scenes(bib)
    for i, (dur, blocks) in enumerate(S):
        _, total = layout(blocks)
        if total > H - 40: print(f"WARNING scene {i} overflows: {total} px")
    if "--preview" in sys.argv:
        ids = [int(v) for v in sys.argv[sys.argv.index("--preview") + 1].split(",")]
        for i in ids: render(S[i][1], S[i][0] + 10)[0].resize((960, 540)).save(os.path.join(out, f"preview_{i}.png"))
        print("previews", ids); return
    pages = [render(bl, dur + 10)[0].convert("RGB") for dur, bl in S]; pages[0].save(os.path.join(out, "dartf_slides.pdf"), save_all=True, append_images=pages[1:], resolution=100.0)
    print("slides:", os.path.join(out, "dartf_slides.pdf"), len(pages), "pages")
    if "--video" in sys.argv:
        mp4 = os.path.join(out, "dartf_explainer.mp4"); n = write_video(S, mp4, fps); print("video:", mp4, n, "frames", f"{n / fps:.0f} s")
if __name__ == "__main__": main()
