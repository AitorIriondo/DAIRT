"""assemble the MOT17 half-split table: metrics from TrackEval summaries + mean ms per frame from the per-sequence stats. usage: python mot_table.py <out_dir> [label]"""
import sys, os, json, glob
out = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else out; trk = f"{out}/half/trackers/MOT17-train"; rows = []
for name in sorted(os.listdir(trk)):
    summ = f"{trk}/{name}/pedestrian_summary.txt"
    if not os.path.exists(summ): continue
    k, v = [l.split() for l in open(summ).read().strip().splitlines()[:2]]; m = dict(zip(k, map(float, v)))
    st = [json.load(open(f)) for f in glob.glob(f"{out}/{name}_MOT17-*.json")]; fr = sum(s["frames"] for s in st) or 1
    ms = {key: sum(s.get(key, 0) * s["frames"] for s in st) / fr for key in ("backbone_ms", "head_ms", "mask_ms")}; sam = sum(sum(s.get("sam3_tracker_ms_per_frame", {}).values()) * s["frames"] for s in st) / fr
    rows.append((name, m["HOTA"], m["DetA"], m["AssA"], m["MOTA"], m["IDF1"], int(m["IDSW"]), ms["backbone_ms"], ms["head_ms"], ms["mask_ms"], sam, ms["backbone_ms"] + ms["head_ms"] + ms["mask_ms"] + sam))
print(f"{label}\n| config | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | backbone | head | masks | SAM3 | total ms/frame |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows: print(f"| {r[0]} | {r[1]:.1f} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.1f} | {r[5]:.1f} | {r[6]} | {r[7]:.1f} | {r[8]:.1f} | {r[9]:.1f} | {r[10]:.1f} | **{r[11]:.0f}** |")
