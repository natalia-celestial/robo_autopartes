import argparse, glob, os, json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

COCO_NAMES = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
  "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
  "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
  "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
  "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
  "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
  "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
  "hair drier","toothbrush"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_list", required=True)      # e.g., candidates_train.txt
    ap.add_argument("--pred_json", default="preds_pc.jsonl") # optional sidecar
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)         # cuda:0 or cpu
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    model = YOLO("yolo11n.pt")
    imgs = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")))
    assert imgs, f"No images in {args.img_dir}"

    keep = []
    with open(args.pred_json, "w") as jf:
        for i in tqdm(range(0, len(imgs), args.batch)):
            chunk = imgs[i:i+args.batch]
            results = model.predict(chunk, conf=args.conf, device=args.device, verbose=False)
            for ip, r in zip(chunk, results):
                names = [COCO_NAMES[int(c)] for c in (r.boxes.cls.tolist() if r.boxes is not None else [])]
                has_person = "person" in names
                has_car = "car" in names or "truck" in names
                if has_person and has_car:
                    keep.append(ip)
                # write minimal jsonl per image (optional)
                boxes = r.boxes.xyxy.tolist() if r.boxes is not None else []
                cls   = [COCO_NAMES[int(c)] for c in (r.boxes.cls.tolist() if r.boxes is not None else [])]
                conf  = r.boxes.conf.tolist() if r.boxes is not None else []
                jf.write(json.dumps({"image": ip, "boxes": boxes, "labels": cls, "scores": conf}) + "\n")

    Path(args.out_list).write_text("\n".join(keep))
    print(f"wrote {len(keep)} candidates to {args.out_list}; preds in {args.pred_json}")

if __name__ == "__main__":
    main()
