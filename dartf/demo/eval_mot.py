"""MOT17 train-half evaluation of the DARTF video pipeline (HOTA / MOTA / IDF1 via TrackEval).
Protocol (ByteTrack / OC-SORT validation split): the 7 MOT17 FRCNN training sequences, second half of each (frames > N/2), pedestrians (prompt "person").
usage: python eval_mot.py --mot /data/datasets/MOT17 --trackeval <TrackEval root> --out <dir> --python <py> --runner run_video.py
       --configs "boxes_none:--heads boxes --tracker none" "masks_light:--tracker light" ... [--runner-args "..."] [--seqs MOT17-02-FRCNN,...] [--eval-only]
Writes <out>/half/gt (derived GT), <out>/half/trackers/<cfg>/data/<seq>.txt, <out>/results.json (metrics + ms per frame)."""
import os, sys, json, argparse, subprocess, configparser, time, shutil
ap = argparse.ArgumentParser(); ap.add_argument("--mot", required=True); ap.add_argument("--trackeval", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--python", default=sys.executable); ap.add_argument("--runner", default="run_video.py"); ap.add_argument("--runner-args", default=""); ap.add_argument("--configs", nargs="+", required=True)
ap.add_argument("--seqs", default=None); ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--prompt", default="person"); ap.add_argument("--split", default="train", help="train: derive the second half of each training sequence; ablation: use the mirror's ready-made half split as is"); a = ap.parse_args()
seqs = a.seqs.split(",") if a.seqs else sorted(d for d in os.listdir(f"{a.mot}/{a.split}") if d.endswith("-FRCNN"))
half = f"{a.out}/half"; gt_root = f"{half}/gt/MOT17-train"; trk_root = f"{half}/trackers/MOT17-train"; os.makedirs(gt_root, exist_ok=True); os.makedirs(trk_root, exist_ok=True)
# ---- derived half-split GT: frames > N/2 renumbered from 1, seqinfo with the half length ----
info = {}
for s in seqs:
    cp = configparser.ConfigParser(); cp.read(f"{a.mot}/{a.split}/{s}/seqinfo.ini"); N = int(cp["Sequence"]["seqLength"]); W, H, fps = int(cp["Sequence"]["imWidth"]), int(cp["Sequence"]["imHeight"]), float(cp["Sequence"]["frameRate"])
    start = N // 2 + 1 if a.split == "train" else 1; n_half = N - start + 1; info[s] = dict(N=N, start=start, n=n_half, W=W, H=H, fps=fps)
    d = f"{gt_root}/{s}"; os.makedirs(f"{d}/gt", exist_ok=True)
    with open(f"{a.mot}/{a.split}/{s}/gt/gt.txt") as f, open(f"{d}/gt/gt.txt", "w") as g:
        for line in f:
            p = line.strip().split(",")
            if int(p[0]) >= start: p[0] = str(int(p[0]) - start + 1); g.write(",".join(p) + "\n")
    cp["Sequence"]["seqLength"] = str(n_half); cp["Sequence"]["imDir"] = "img1"
    with open(f"{d}/seqinfo.ini", "w") as g: cp.write(g)
os.makedirs(f"{half}/gt/seqmaps", exist_ok=True)
with open(f"{half}/gt/seqmaps/MOT17-train.txt", "w") as g: g.write("name\n" + "".join(s + "\n" for s in seqs))
# ---- run the pipeline per config and sequence (headless, MOT output at the source resolution) ----
results = {}
for cfg in a.configs:
    name, args = cfg.split(":", 1); tdir = f"{trk_root}/{name}/data"; os.makedirs(tdir, exist_ok=True); ms = {}
    for s in seqs:
        out_txt = f"{tdir}/{s}.txt"; stats = f"{a.out}/{name}_{s}.json"
        if not a.eval_only or not os.path.exists(out_txt):
            i = info[s]; pattern = f"{a.mot}/{a.split}/{s}/img1/%06d.jpg"
            cmd = [a.python, a.runner, pattern, f"{a.out}/{name}_{s}.mp4", a.prompt, "--no-render", "--mot-out", out_txt + ".tmp", "--fps-in", str(i["fps"]), "--src-size", f"{i['W']}x{i['H']}", "--stats", stats] + args.split() + a.runner_args.split()
            # ffmpeg image2 demuxer: start at the first frame of the half via -start_number handled by renaming inside the pattern is not possible; run all frames and drop the first half below
            t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0: print(name, s, "FAILED\n", r.stderr[-2000:]); sys.exit(1)
            with open(out_txt + ".tmp") as f, open(out_txt, "w") as g:
                for line in f:
                    p = line.split(","); fr = int(p[0])
                    if fr >= i["start"]: p[0] = str(fr - i["start"] + 1); g.write(",".join(p))
            os.remove(out_txt + ".tmp"); print(f"{name} {s}: {time.time() - t0:.0f}s", flush=True)
        if os.path.exists(stats): d = json.load(open(stats)); ms[s] = {k: d[k] for k in ("backbone_ms", "head_ms", "mask_ms") if k in d} | ({"sam3": d["sam3_tracker_ms_per_frame"]} if "sam3_tracker_ms_per_frame" in d else {})
    results[name] = {"ms": ms}
# ---- TrackEval ----
for name in results:
    r = subprocess.run([a.python, f"{a.trackeval}/scripts/run_mot_challenge.py", "--GT_FOLDER", f"{half}/gt", "--TRACKERS_FOLDER", f"{half}/trackers", "--BENCHMARK", "MOT17", "--SPLIT_TO_EVAL", "train",
                        "--TRACKERS_TO_EVAL", name, "--METRICS", "HOTA", "CLEAR", "Identity", "--USE_PARALLEL", "False", "--PRINT_CONFIG", "False", "--PRINT_RESULTS", "False", "--OUTPUT_SUMMARY", "True", "--OUTPUT_DETAILED", "False", "--PLOT_CURVES", "False"], capture_output=True, text=True)
    summ = f"{trk_root}/{name}/pedestrian_summary.txt"
    if r.returncode != 0 or not os.path.exists(summ): print("TrackEval failed for", name, r.stdout[-1500:], r.stderr[-1500:]); continue
    keys, vals = [l.split() for l in open(summ).read().strip().splitlines()[:2]]; m = dict(zip(keys, [float(v) for v in vals]))
    results[name]["metrics"] = {k: m[k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "MT", "ML", "FP", "FN") if k in m}
    print(f"{name}: HOTA {m['HOTA']:.1f}  MOTA {m['MOTA']:.1f}  IDF1 {m['IDF1']:.1f}  IDSW {int(m['IDSW'])}  DetA {m['DetA']:.1f}  AssA {m['AssA']:.1f}")
json.dump(results, open(f"{a.out}/results.json", "w"), indent=1); print("wrote", f"{a.out}/results.json")
