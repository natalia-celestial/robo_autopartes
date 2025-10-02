# tools/yolo_txt_to_coco.py
import json, os, argparse, glob
from pathlib import Path
from PIL import Image

def load_classes(names_file):
    if names_file is None: 
        return ["autoparts_theft"]
    with open(names_file) as f:
        return [l.strip() for l in f if l.strip()]

def yolo_to_coco(img_dir, lbl_dir, out_json, names=None):
    classes = load_classes(names)
    categories = [{"id": i+1, "name": n} for i, n in enumerate(classes)]
    images, annotations = [], []
    ann_id, img_id = 1, 1
    for ip in sorted(glob.glob(str(Path(img_dir)/"*.jpg"))):
        w, h = Image.open(ip).size
        images.append({"id": img_id, "file_name": Path(ip).name, "width": w, "height": h})
        lp = Path(lbl_dir) / (Path(ip).stem + ".txt")
        if lp.exists():
            with open(lp) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5: 
                        continue
                    cid, xc, yc, bw, bh = map(float, parts)
                    cid = int(cid)
                    x = (xc - bw/2) * w
                    y = (yc - bh/2) * h
                    ww = bw * w
                    hh = bh * h
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cid + 1,
                        "bbox": [x, y, ww, hh],
                        "area": ww*hh,
                        "iscrowd": 0
                    })
                    ann_id += 1
        img_id += 1
    coco = {"images": images, "annotations": annotations, "categories": categories}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco, f)
    print(f"Wrote {out_json} | images={len(images)} anns={len(annotations)} classes={classes}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--lbl_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--names", default=None, help="optional classes.txt")
    args = ap.parse_args()
    yolo_to_coco(args.img_dir, args.lbl_dir, args.out_json, args.names)
