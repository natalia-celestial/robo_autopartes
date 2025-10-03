# rfdetr_glue.py  (usa la API del paquete que ya tienes)
from typing import List, Any, Dict
import torch
import torch.nn.functional as F

# Usamos el builder del paquete, con sus defaults completos
from rfdetr.main import build_model as _rf_build_model  # devuelve (model, criterion, postprocessors) o model
from rfdetr.main import populate_args                   # rellena todos los args con defaults

def build_model(num_classes: int = 1, pretrained: bool = True):
    args = populate_args([])              # defaults del paquete
    # normalizamos campos de clases/pretrain que existan en tu versión
    for k in ("num_classes", "nc", "classes"): 
        if hasattr(args, k): setattr(args, k, num_classes)
    for k in ("pretrained", "pretrain"): 
        if hasattr(args, k): setattr(args, k, pretrained)
    if not hasattr(args, "device"): 
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    out = _rf_build_model(args)
    # Algunas versiones devuelven (model, criterion, postprocessors)
    return out[0] if isinstance(out, tuple) else out

def load_coco_weights(model, path):
    if path:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=False)

def detr_collate_fn(batch: List[Any]):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

def _cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx-0.5*w, cy-0.5*h, cx+0.5*w, cy+0.5*h], dim=-1)

@torch.no_grad()
def postprocess_outputs(raw, conf_th: float = 0.4) -> List[Dict[str, torch.Tensor]]:
    # ya en formato listo
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "boxes" in raw[0]:
        return raw
    # salida estilo DETR
    if isinstance(raw, dict) and "pred_logits" in raw and "pred_boxes" in raw:
        logits, boxes = raw["pred_logits"], raw["pred_boxes"]   # (B,Q,C), (B,Q,4 cxcywh)
        B, Q, C = logits.shape
        probs = logits.sigmoid() if C == 1 else F.softmax(logits, -1)[..., :-1]  # sin bg
        outs = []
        for b in range(B):
            pb = probs[b]
            if pb.ndim == 1 or (pb.dim()==2 and pb.shape[1]==1):
                scores = pb.squeeze(-1); labels = torch.zeros(Q, dtype=torch.long)
            else:
                scores, labels = pb.max(-1)
            keep = scores >= conf_th
            xyxy = _cxcywh_to_xyxy(boxes[b])[keep].clamp(0,1).cpu()
            outs.append({"boxes": xyxy, "scores": scores[keep].cpu(), "labels": labels[keep].cpu()})
        return outs
    raise ValueError("Unknown RF-DETR output format")


