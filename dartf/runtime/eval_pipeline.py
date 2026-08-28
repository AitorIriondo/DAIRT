"""Quality harness: candidate vision engine vs reference vision engine.
 feature-level: fpn_0/1/2 rel-L2 + cosine on feature-validation images
 task-level: text->ground on task images; official postprocess (sigmoid(score)*sigmoid(presence) > thr);
             candidate-vs-reference agreement and recall/FP vs COCO GT."""
import sys, json, numpy as np, argparse, os, time
from trt_util import load_engine, Runner
from preprocess import load_image_tensor
ap=argparse.ArgumentParser()
ap.add_argument("--cand", required=True); ap.add_argument("--ref", required=True)
ap.add_argument("--text", required=True); ap.add_argument("--ground", required=True)
ap.add_argument("--handoff", default=os.path.expanduser("~/sam3_orin_agx64_jp72_handoff"))
ap.add_argument("--gt", default=os.path.expanduser("~/w8a8/coco_gt_dev100.json"))
ap.add_argument("--feature-images", default="feature_validation_image_ids")
ap.add_argument("--task-only", action="store_true"); ap.add_argument("--features-only", action="store_true")
ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--out", required=True)
ap.add_argument("--bench", action="store_true")
ap.add_argument("--dev-images", type=int, default=0, help="also evaluate N dev images with COCO-derived positive/absent prompts (deterministic)")
ap.add_argument("--size", type=int, default=1008, help="vision input resolution")
ap.add_argument("--img-pos", default=None, help="img_pos_cK.npy matching the ground engine (default: handoff assets/img_pos_c16.npy)")
ap.add_argument("--ref-size", type=int, default=None, help="resolution of the reference engine (default: --size)")
ap.add_argument("--ground-ref", default=None, help="ground engine for the reference path (default: same as --ground)")
ap.add_argument("--ground-pre", default=None, help="prefix engine (img_feat, img_pos @B=1 -> x1, pos) when --ground is a post engine with inputs x1_in/pos_in (exact layer-0 prefix sharing)")
a=ap.parse_args()
H=a.handoff; sys.path.insert(0, f"{H}/client")
from tokenize_classes import Sam3Tokenizer, CONTEXT_LENGTH, BUCKET
split=json.load(open(f"{H}/manifests/data_split.json"))
TASKS=[("000000332276","person","dog"),("000000132517","vase","airplane"),("000000152898","bear","toaster"),("000000218215","bear","carrot")]
if a.dev_images:
    import random; gtall=json.load(open(a.gt)); cats=gtall["categories"]; rnd=random.Random(20260826)
    pool=[i for i in split["train_image_ids"]+split["feature_validation_image_ids"] if i not in {t[0] for t in TASKS}]
    for iid in pool[:a.dev_images]:
        present={}
        for b in gtall["images"][iid]["boxes"]:
            if not b["crowd"]: present[b["cat"]]=present.get(b["cat"],0)+1
        if not present: continue
        pos=max(present,key=present.get); absent=rnd.choice([c for c in cats if c not in present]); TASKS.append((iid,pos,absent))
    print("task images:",len(TASKS))
ref=Runner(load_engine(a.ref)); cand=Runner(load_engine(a.cand))
res={"cand":a.cand,"ref":a.ref,"features":{},"tasks":{}}
def rel(x,y): return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-12))
def cos(x,y): return float((x*y).sum()/max(np.linalg.norm(x)*np.linalg.norm(y),1e-12))
if not a.task_only:
    ids=split[a.feature_images] if a.feature_images in split else a.feature_images.split(",")
    agg={k:[] for k in("fpn_0","fpn_1","fpn_2")}
    for iid in ids:
        x,_=load_image_tensor(f"{H}/data/images/{iid}.jpg", a.size); xr,_=load_image_tensor(f"{H}/data/images/{iid}.jpg", a.ref_size or a.size); r=ref({"images":xr}, want=["fpn_0","fpn_1","fpn_2"]); c=cand({"images":x}, want=["fpn_0","fpn_1","fpn_2"])
        if any(c[k].shape!=r[k].shape for k in ("fpn_0","fpn_1","fpn_2")): res["features"][iid]={"note":"resolution differs; no feature comparison"}; continue
        res["features"][iid]={k:{"rel_l2":rel(c[k].astype(np.float32),r[k].astype(np.float32)),"cos":cos(c[k].ravel().astype(np.float64),r[k].ravel().astype(np.float64))} for k in agg}
        for k in agg: agg[k].append(res["features"][iid][k]["rel_l2"])
    res["features_summary"]={k:{"rel_l2_mean":float(np.mean(v)) if v else None,"rel_l2_max":float(np.max(v)) if v else None} for k,v in agg.items()}
    print("features:", json.dumps(res["features_summary"]))
if not a.features_only:
    tok=Sam3Tokenizer(f"{H}/assets/bpe_simple_vocab_16e6.txt.gz")
    text=Runner(load_engine(a.text)); ground=Runner(load_engine(a.ground))
    img_pos=np.load(a.img_pos or f"{H}/assets/img_pos_c16.npy"); gt=json.load(open(a.gt))["images"]
    KB=int(ground.bufs["text_mask"][1][0]) if "x1_in" in ground.inputs else int(ground.bufs[ground.inputs[0]][1][0]); assert KB>=2, "ground bucket must hold positive+absent prompts"
    img_pos_ref=np.load(f"{H}/assets/img_pos_c16.npy"); ground_ref=Runner(load_engine(a.ground_ref)) if a.ground_ref else ground
    def box_iou(A,B):
        if len(A)==0 or len(B)==0: return np.zeros((len(A),len(B)))
        A=np.array(A,dtype=np.float64); B=np.array(B,dtype=np.float64)
        lt=np.maximum(A[:,None,:2],B[None,:,:2]); rb=np.minimum(A[:,None,2:],B[None,:,2:]); wh=np.clip(rb-lt,0,None); inter=wh[...,0]*wh[...,1]
        ar=(A[:,2]-A[:,0])*(A[:,3]-A[:,1]); br=(B[:,2]-B[:,0])*(B[:,3]-B[:,1]); return inter/np.maximum(ar[:,None]+br[None,:]-inter,1e-9)
    def detections(scores,boxes,presence,cls_idx,W,Hh):
        p=1/(1+np.exp(-scores[cls_idx,:,0].astype(np.float64)))*(1/(1+np.exp(-float(presence[cls_idx,0]))))
        b=boxes[cls_idx].astype(np.float64); xyxy=np.stack([(b[:,0]-b[:,2]/2)*W,(b[:,1]-b[:,3]/2)*Hh,(b[:,0]+b[:,2]/2)*W,(b[:,1]+b[:,3]/2)*Hh],1)
        keep=p>a.thr; return p, xyxy, keep
    ground_pre=Runner(load_engine(a.ground_pre)) if a.ground_pre else None
    def run_ground(fpn2, tf, tm, eng=None, pos=None):
        eng=eng or ground; pos=img_pos if pos is None else pos; K=int(eng.bufs[eng.inputs[-1]][1][0]) if "x1_in" in eng.inputs else int(eng.bufs[eng.inputs[0]][1][0])
        if "x1_in" in eng.inputs:      # post engine: prefix once per image (B=1), broadcast to the K prompts inside the post engine
            assert ground_pre is not None, "--ground-pre required for a post engine"
            o=ground_pre({"img_feat":fpn2.astype(np.float16)[:1],"img_pos":pos[:1]})
            return eng({"x1_in":o[ground_pre.outputs[0]],"pos_in":o[ground_pre.outputs[1]],"text_feats":np.ascontiguousarray(tf[:,:K,:]),"text_mask":np.ascontiguousarray(tm[:K])})
        feat=np.repeat(fpn2.astype(np.float16),K,axis=0)
        return eng({"img_feat":feat,"img_pos":pos[:K],"text_feats":np.ascontiguousarray(tf[:,:K,:]),"text_mask":np.ascontiguousarray(tm[:K])})
    summ={"pos_ref_count":0,"pos_cand_count":0,"count_equal":0,"absent_fp_ref":0,"absent_fp_cand":0,"absent_max_prob_cand":0.0,"top20_overlap_min":1.0,"matched_iou_cand_vs_ref":[], "gt_recall50":[0,0],"gt_recall75":[0,0],"gt_recall50_ref":[0,0],"gt_recall75_ref":[0,0],"gt_mean_matched_iou":[[],[]]}
    for iid,pos,absent in TASKS:
        x,(W,Hh)=load_image_tensor(f"{H}/data/images/{iid}.jpg", a.size); xr,_=load_image_tensor(f"{H}/data/images/{iid}.jpg", a.ref_size or a.size)
        rows=[]; 
        for t in (pos,absent):
            ids_=tok.encode(t); rows.extend(ids_+[0]*(CONTEXT_LENGTH-len(ids_)))
        rows.extend([0]*((BUCKET-2)*CONTEXT_LENGTH)); tokens=np.array(rows,dtype=np.int32).reshape(BUCKET,CONTEXT_LENGTH)
        to=text({"token_ids":tokens}); tf,tm=to["text_feats"],to["text_mask"]
        r=ref({"images":xr}, want=["fpn_2"]); c=cand({"images":x}, want=["fpn_2"])
        gr=run_ground(r["fpn_2"],tf,tm,ground_ref,img_pos_ref); gc=run_ground(c["fpn_2"],tf,tm)
        entry={"fpn_2_rel_l2":rel(c["fpn_2"],r["fpn_2"]) if c["fpn_2"].shape==r["fpn_2"].shape else None}
        for ci,(name,kind) in enumerate(((pos,"positive"),(absent,"absent"))):
            pr,br,kr=detections(gr["scores"],gr["boxes"],gr["presence"],ci,W,Hh); pc,bc,kc=detections(gc["scores"],gc["boxes"],gc["presence"],ci,W,Hh)
            top_r=set(np.argsort(-pr)[:20]); top_c=set(np.argsort(-pc)[:20]); ov=len(top_r&top_c)/20
            e={"kind":kind,"ref_count":int(kr.sum()),"cand_count":int(kc.sum()),"top20_overlap":ov,"logit_max_abs_diff":float(np.abs(gc["scores"][ci]-gr["scores"][ci]).max()),
               "presence_ref":float(gr["presence"][ci,0]),"presence_cand":float(gc["presence"][ci,0]),"max_prob_ref":float(pr.max()),"max_prob_cand":float(pc.max())}
            summ["top20_overlap_min"]=min(summ["top20_overlap_min"],ov)
            if kind=="positive":
                summ["pos_ref_count"]+=e["ref_count"]; summ["pos_cand_count"]+=e["cand_count"]; summ["count_equal"]+=int(e["ref_count"]==e["cand_count"])
                iou=box_iou(bc[kc],br[kr]); 
                if iou.size: m=iou.max(1); e["cand_vs_ref_matched_iou_mean"]=float(m.mean()); summ["matched_iou_cand_vs_ref"]+=m.tolist()
                gtb=[g["xyxy"] for g in gt[iid]["boxes"] if g["cat"]==name and not g["crowd"]]
                e["gt_count"]=len(gtb)
                for lab,(bb,kk) in (("cand",(bc,kc)),("ref",(br,kr))):
                    iou_g=box_iou(gtb,bb[kk]); best=iou_g.max(1) if iou_g.size else np.zeros(len(gtb))
                    e[f"{lab}_recall50"]=float((best>=0.5).mean()) if len(gtb) else None; e[f"{lab}_recall75"]=float((best>=0.75).mean()) if len(gtb) else None
                    e[f"{lab}_mean_matched_iou"]=float(best[best>0].mean()) if (best>0).any() else 0.0
                    key="gt_recall50" if lab=="cand" else "gt_recall50_ref"; summ[key][0]+=int((best>=0.5).sum()); summ[key][1]+=len(gtb)
                    key="gt_recall75" if lab=="cand" else "gt_recall75_ref"; summ[key][0]+=int((best>=0.75).sum()); summ[key][1]+=len(gtb)
                    summ["gt_mean_matched_iou"][0 if lab=="cand" else 1]+=best[best>0].tolist()
            else:
                summ["absent_fp_ref"]+=e["ref_count"]; summ["absent_fp_cand"]+=e["cand_count"]; summ["absent_max_prob_cand"]=max(summ["absent_max_prob_cand"],e["max_prob_cand"])
            entry[name]=e
        res["tasks"][iid]=entry; print(iid, json.dumps(entry)[:400])
    summ["matched_iou_cand_vs_ref_mean"]=float(np.mean(summ["matched_iou_cand_vs_ref"])) if summ["matched_iou_cand_vs_ref"] else None
    summ["matched_iou_cand_vs_ref_min"]=float(np.min(summ["matched_iou_cand_vs_ref"])) if summ["matched_iou_cand_vs_ref"] else None
    summ["gt_mean_matched_iou_cand"]=float(np.mean(summ["gt_mean_matched_iou"][0])) if summ["gt_mean_matched_iou"][0] else None
    summ["gt_mean_matched_iou_ref"]=float(np.mean(summ["gt_mean_matched_iou"][1])) if summ["gt_mean_matched_iou"][1] else None
    del summ["matched_iou_cand_vs_ref"], summ["gt_mean_matched_iou"]
    res["task_summary"]=summ; print("tasks:", json.dumps(summ))
if a.bench:
    x,_=load_image_tensor(f"{H}/data/images/{TASKS[0][0]}.jpg")
    res["bench_ms"]={"cand":cand.time({"images":x}),"ref":ref.time({"images":x})}; print("bench:",res["bench_ms"])
json.dump(res, open(a.out,"w"), indent=1); print("wrote",a.out)
