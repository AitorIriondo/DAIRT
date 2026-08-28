"""Fetch and verify the bibliography programmatically: every entry is downloaded from arXiv's bibtex export endpoint
(https://arxiv.org/bibtex/<id>) and its title is checked against the expected words. Nothing is hand written; failures are printed.
usage: python check_references.py [out.bib]"""
import sys, re, time, urllib.request
REFS = [  # (arxiv id, words that must appear in the title, short label)
    ("2210.17323", ["gptq", "post-training", "quantization"], "GPTQ"),
    ("2211.10438", ["smoothquant"], "SmoothQuant"),
    ("2404.00456", ["quarot"], "QuaRot"),
    ("2405.16406", ["spinquant"], "SpinQuant"),
    ("2307.08691", ["flashattention-2"], "FlashAttention-2"),
    ("2205.14135", ["flashattention"], "FlashAttention"),
    ("2102.05426", ["brecq"], "BRECQ"),
    ("2004.10568", ["up or down", "adaptive rounding"], "AdaRound"),
    ("2212.08254", ["repq-vit"], "RepQ-ViT"),
    ("2111.13824", ["fq-vit"], "FQ-ViT"),
    ("2111.12293", ["ptq4vit"], "PTQ4ViT"),
    ("2410.09426", ["flatquant"], "FlatQuant"),
    ("2406.01721", ["duquant"], "DuQuant"),
    ("2402.17762", ["massive activations"], "Massive activations"),
    ("2306.12929", ["quantizable transformers"], "Quantizable Transformers"),
    ("2208.07339", ["llm.int8"], "LLM.int8"),
    ("2408.00714", ["sam 2"], "SAM 2"),
    ("2304.02643", ["segment anything"], "Segment Anything"),
    ("2010.11929", ["image is worth 16x16 words"], "ViT"),
    ("2104.09864", ["roformer"], "RoPE"),
    ("2203.16527", ["exploring plain vision transformer backbones"], "ViTDet"),
    ("2511.16719", ["sam 3"], "SAM 3"),
]
def fetch(aid):
    req = urllib.request.Request(f"https://arxiv.org/bibtex/{aid}", headers={"User-Agent": "dartf-refcheck"}); return urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
def main():
    out = []; ok = 0
    for aid, words, label in REFS:
        try: bib = fetch(aid)
        except Exception as e: print(f"FAIL  {label:24s} {aid}: fetch error {e}"); continue
        m = re.search(r"title\s*=\s*\{(.*?)\}\s*,", bib, re.S); title = re.sub(r"\s+", " ", m.group(1)) if m else ""
        good = all(any(w.lower() in title.lower() for w in [word]) for word in words) if isinstance(words, list) else False
        good = all(w.lower() in title.lower() for w in words)
        print(f"{'ok  ' if good else 'MISMATCH'}  {label:24s} {aid}: {title[:90]}")
        if good: out.append(bib.strip()); ok += 1
        time.sleep(1.0)
    open(sys.argv[1] if len(sys.argv) > 1 else "references.bib", "w").write("\n\n".join(out) + "\n")
    print(f"{ok}/{len(REFS)} verified entries written")
if __name__ == "__main__":
    main()
