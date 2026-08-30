"""dump mask-head inputs (fpn_0, fpn_1, enc, text_feats, text_mask, hs_sel) for calibration / validation from video frames
usage: python dump_maskhead_calib.py --assets <dir> out.npz video1:prompts:frames video2:prompts:frames   (frames = comma list; <dir> as for run_video.py)"""
import sys, os, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtime"))
from trt_util import load_engine, Runner
from tokenize_classes import Sam3Tokenizer, CONTEXT_LENGTH, BUCKET
import onnxruntime as ort
assert sys.argv[1] == "--assets"; A = sys.argv[2]; out = sys.argv[3]; specs = sys.argv[4:]
vision = Runner(load_engine(f"{A}/vision_int8.plan")); ground = Runner(load_engine(f"{A}/ground_c4m_fp16.plan")); img_pos = np.load(f"{A}/img_pos_c4.npy")
tok = Sam3Tokenizer(f"{A}/bpe_simple_vocab_16e6.txt.gz"); sess = ort.InferenceSession(f"{A}/text_c16.onnx", providers=["CPUExecutionProvider"])
def sigmoid(x): return 1 / (1 + np.exp(-x.astype(np.float64)))
samples = []
for spec in specs:
    video, prompts, frames = spec.split(":"); prompts = prompts.split(","); K = len(prompts)
    ids = np.zeros((BUCKET, CONTEXT_LENGTH), np.int32); ids[:, 0] = 49406; ids[:, 1] = 49407
    for i, p in enumerate(prompts): e = tok.encode(p); ids[i, :] = 0; ids[i, :len(e)] = e
    tf, tm = sess.run(None, {"token_ids": ids}); tf4 = np.ascontiguousarray(tf[:, :4]).astype(np.float16); tm4 = np.ascontiguousarray(tm[:4]).astype(np.float32)
    for i in range(K, 4): tm4[i] = tm[BUCKET - 1]; tf4[:, i] = tf[:, BUCKET - 1]
    for f in frames.split(","):
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-vf", "select=eq(n\\,%s),scale=1008:1008:flags=bilinear" % f, "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        x = np.frombuffer(p.stdout.read(1008 * 1008 * 3), np.uint8).reshape(1008, 1008, 3).transpose(2, 0, 1).astype(np.float32) * (1 / 127.5) - 1.0
        v = vision({"images": x[None]}); g = ground({"img_feat": np.repeat(v["fpn_2"].astype(np.float16), 4, 0), "img_pos": img_pos, "text_feats": tf4, "text_mask": tm4})
        probs = sigmoid(g["scores"][:K, :, 0]) * sigmoid(g["presence"][:K, 0])[:, None]; active = [k for k in range(K) if (probs[k] > 0.3).any()]
        if not active: continue
        sel = np.stack([np.argsort(-probs[k])[:32] for k in active]); hs = np.stack([g["hs"][k, sel[i]] for i, k in enumerate(active)]).astype(np.float16)
        samples.append({"fpn_0": v["fpn_0"], "fpn_1": v["fpn_1"], "enc": np.ascontiguousarray(g["enc"][:, active]), "text_feats": np.ascontiguousarray(tf4[:, :K][:, active]), "text_mask": np.ascontiguousarray(tm4[:K][active]), "hs_sel": hs})
        print(video, f, "active", active, flush=True)
np.savez(out, n=len(samples), **{f"{k}_{i}": s[k] for i, s in enumerate(samples) for k in s}); print("saved", len(samples), "samples")
