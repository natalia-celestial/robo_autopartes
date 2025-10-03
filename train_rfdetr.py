# train_rfdetr.py
"""
Minimal fine-tune loop for RF-DETR on a 1-class dataset in COCO format.
Assumes you have an RF-DETR implementation exposing:
  - build_model(num_classes, pretrained=True)
  - load_coco_weights(model, path)
  - collate_fn for DETR-style targets
Replace imports below to match your repo (e.g., IDEA-Research RF-DETR).
"""
import os, argparse, math, time, json, random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection

from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rfdetr_glue import build_model, load_coco_weights, detr_collate_fn, postprocess_outputs


#from rfdetr_glue import build_model, load_coco_weights, detr_collate_fn, postprocess_outputs


def seed_everything(s=123):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ToTargets:
    """Map COCO annots -> DETR target dicts (boxes in [0,1], labels in 0..C-1)."""
    def __init__(self, width_key="width", height_key="height"):
        self.width_key = width_key; self.height_key = height_key
    def __call__(self, img, target):
        w, h = img.size
        boxes, labels = [], []
        if isinstance(target, list):
            for ann in target:
                if "bbox" not in ann: continue
                x,y,bw,bh = ann["bbox"]
                cx = (x + bw/2) / w; cy = (y + bh/2) / h
                nw = bw / w; nh = bh / h
                # DETR expects [x0,y0,x1,y1] or normalized cxcywh depending on repo;
                # adapt here to your model’s expected format:
                boxes.append([x/w, y/h, (x+bw)/w, (y+bh)/h])
                labels.append(ann["category_id"]-1)  # categories started at 1 in converter
        out = {}
        out["boxes"]  = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4),dtype=torch.float32)
        out["labels"] = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,),dtype=torch.int64)
        return img, out

def get_datasets(img_root_tr, ann_tr, img_root_va, ann_va):
    tform = transforms.Compose([
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
        transforms.ToTensor(),
    ])
    tform_val = transforms.ToTensor()
    train = CocoDetection(img_root_tr, ann_tr, transforms.Compose([tform, ToTargets()]))
    val   = CocoDetection(img_root_va, ann_va, transforms.Compose([tform_val, ToTargets()]))
    return train, val

def build_optimizer(model, lr=2e-4, lr_backbone=2e-5, wd=1e-4):
    param_dicts = [
        {"params": [p for n,p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": lr},
        {"params": [p for n,p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": lr_backbone},
    ]
    return torch.optim.AdamW(param_dicts, lr=lr, weight_decay=wd)

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    tot, det, gt = 0, 0, 0
    for imgs, targets in data_loader:
        imgs = [im.to(device) for im in imgs]
        outs = model(imgs)
        # Simple precision proxy: count frames with any >0.4 detection vs any GT
        for i, out in enumerate(outs):
            conf = out["scores"].detach().cpu() if "scores" in out else torch.tensor([])
            det += int((conf>0.4).any().item())
            gt  += int((targets[i]["labels"].numel()>0))
            tot += 1
    prec = det/max(tot,1); rec = det/max(gt,1) if gt>0 else 0.0
    return {"frame_precision": prec, "frame_recall": rec}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_images", default="dataset/images/train")
    ap.add_argument("--train_ann",    default="dataset/annotations_train.json")
    ap.add_argument("--val_images",   default="dataset/images/val")
    ap.add_argument("--val_ann",      default="dataset/annotations_val.json")
    ap.add_argument("--num_classes",  type=int, default=1)
    ap.add_argument("--epochs",       type=int, default=60)
    ap.add_argument("--batch",        type=int, default=8)
    ap.add_argument("--accum",        type=int, default=1)
    ap.add_argument("--lr",           type=float, default=2e-4)
    ap.add_argument("--lr_backbone",  type=float, default=2e-5)
    ap.add_argument("--warmup",       type=int, default=1000)
    ap.add_argument("--device",       default="cuda")
    ap.add_argument("--ckpt",         default="rfdetr_coco_pretrained.pth")
    ap.add_argument("--out",          default="runs/rfdetr")
    args = ap.parse_args()

    seed_everything(1337)
    os.makedirs(args.out, exist_ok=True)

    # Data
    train_set, val_set = get_datasets(args.train_images, args.train_ann, args.val_images, args.val_ann)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=4,
                              collate_fn=detr_collate_fn)
    val_loader   = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=4,
                              collate_fn=detr_collate_fn)

    # Model
    model = build_model(num_classes=args.num_classes, pretrained=True)   # creates RF-DETR
    if args.ckpt and os.path.exists(args.ckpt):
        load_coco_weights(model, args.ckpt)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer / scaler
    optimizer = build_optimizer(model, args.lr, args.lr_backbone)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_metric, patience, bad = 0.0, 10, 0
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for imgs, targets in train_loader:
            imgs = [im.to(device) for im in imgs]
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                out = model(imgs, targets)       # expect dict with 'loss' / components
                loss = out["loss"] if isinstance(out, dict) else out

            scaler.scale(loss).step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # warmup lr
            if global_step < args.warmup:
                for g in optimizer.param_groups:
                    g["lr"] = (global_step+1)/args.warmup * (args.lr if "backbone" not in g.get("name","") else args.lr_backbone)
            global_step += 1

        metrics = evaluate(model, val_loader, device)
        score = (metrics["frame_precision"] + metrics["frame_recall"]) / 2.0
        print(f"Epoch {epoch}: {metrics}  score={score:.3f}")

        # save best
        if score > best_metric:
            best_metric, bad = score, 0
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))

if __name__ == "__main__":
    main()
