"""One-time: pull COCO GT boxes (xyxy, non-crowd) + category names for the 100 dev images."""
import json, sys, os
ann_path, image_dir, out = sys.argv[1:4]
ids={f[:-4] for f in os.listdir(image_dir) if f.endswith(".jpg")}
d=json.load(open(ann_path))
cats={c["id"]:c["name"] for c in d["categories"]}
imgs={im["id"]:im for im in d["images"] if f"{im['id']:012d}" in ids}
gt={f"{i:012d}":{"width":im["width"],"height":im["height"],"boxes":[]} for i,im in imgs.items()}
for a in d["annotations"]:
    if a["image_id"] in imgs:
        x,y,w,h=a["bbox"]; gt[f"{a['image_id']:012d}"]["boxes"].append({"cat":cats[a["category_id"]],"xyxy":[x,y,x+w,y+h],"crowd":int(a.get("iscrowd",0))})
json.dump({"categories":sorted(cats.values()),"images":gt}, open(out,"w")); print("images:",len(gt))
