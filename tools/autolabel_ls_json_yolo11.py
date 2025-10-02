# tools/autolabel_ls_json_yolo11.py
import argparse, json, glob, os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Clases COCO (YOLO11 preentrenado)
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

def to_ls_rect(xyxy, w, h, label):
    x0,y0,x1,y1 = map(float, xyxy)
    return {
        "from_name": "label",
        "to_name": "img",
        "type": "rectanglelabels",
        "value": {
            "x": x0 / w * 100.0,
            "y": y0 / h * 100.0,
            "width": (x1 - x0) / w * 100.0,
            "height": (y1 - y0) / h * 100.0,
            "rotation": 0,
            "rectanglelabels": [label]
        }
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="carpeta con .jpg")
    ap.add_argument("--out_json", required=True, help="Label Studio tasks+predictions JSON")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--labels", default="person,car", help="clases a predecir, coma-sep")
    ap.add_argument("--device", default=None, help="cuda:0 | cpu (auto si None)")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    want = set([s.strip() for s in args.labels.split(",") if s.strip()])
    model = YOLO(args.model)
    img_paths = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")))
    assert img_paths, f"No hay .jpg en {args.img_dir}"

    # Predicción en lotes (más rápido)
    items = []
    for i in range(0, len(img_paths), args.batch):
        chunk = img_paths[i:i+args.batch]
        results = model.predict(chunk, conf=args.conf, device=args.device, verbose=False)
        for ip, res in zip(chunk, results):
            W, H = Image.open(ip).size
            rects = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist()):
                    cls_idx = int(c)
                    name = COCO_NAMES[cls_idx] if 0 <= cls_idx < len(COCO_NAMES) else f"id_{cls_idx}"
                    if name in want:
                        rects.append(to_ls_rect(b, W, H, name))
            items.append({
                "data": { "image": "file://" + str(Path(ip).resolve()) },
                "predictions": [{
                    "model_version": f"{Path(args.model).stem}-coco",
                    "result": rects
                }]
            })

    with open(args.out_json, "w") as f:
        json.dump(items, f)
    print(f"Listo: {args.out_json} con {len(items)} imágenes.")

if __name__ == "__main__":
    main()
