# tools/yolo_txt_to_coco.py
import json, os, argparse, glob, sys
from pathlib import Path

# Opcional: soporte AVIF si está el plugin instalado
try:
    from PIL import Image  # noqa
    try:
        import pillow_avif  # noqa: F401
    except Exception:
        pass
except Exception as e:
    print(f"[WARN] PIL no disponible: {e}", file=sys.stderr)
    sys.exit(1)

from PIL import Image


IMG_PATTERNS = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.avif"]


def load_classes(names_file):
    if names_file is None:
        # Fallback (no recomendado): una sola clase
        return ["autoparts_theft"]
    with open(names_file, "r", encoding="utf-8") as f:
        classes = [l.strip() for l in f if l.strip()]
    if not classes:
        raise ValueError("El archivo de clases está vacío.")
    return classes


def list_images(img_dir):
    img_dir = Path(img_dir)
    paths = []
    for pat in IMG_PATTERNS:
        paths.extend(sorted(img_dir.glob(pat)))
    return paths


def yolo_to_coco_split(img_dir, lbl_dir, classes, category_base=0, start_img_id=1, start_ann_id=1):
    """
    Convierte un split (img_dir + lbl_dir) a estructuras COCO (images, annotations).
    - category_base=0 → categories id: 0..N-1 (recomendado)
    - category_base=1 → categories id: 1..N   (COCO clásico)
    Devuelve: (images_list, annotations_list, next_img_id, next_ann_id)
    """
    images, annotations = [], []
    img_id = start_img_id
    ann_id = start_ann_id

    image_paths = list_images(img_dir)

    for ip in image_paths:
        try:
            with Image.open(ip) as im:
                w, h = im.size
        except Exception as e:
            print(f"[WARN] No pude abrir imagen {ip}: {e}", file=sys.stderr)
            continue

        images.append({
            "id": img_id,
            "file_name": ip.name,  # solo el nombre de archivo
            "width": w,
            "height": h
        })

        # etiqueta con el mismo stem
        lp = Path(lbl_dir) / (ip.stem + ".txt")
        if lp.exists():
            try:
                with open(lp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            # formato YOLO esperado: cls cx cy w h (normalizado 0..1)
                            continue
                        try:
                            cid, xc, yc, bw, bh = map(float, parts)
                        except Exception:
                            continue
                        cid = int(cid)

                        # Coordenadas COCO en píxeles
                        x = (xc - bw / 2.0) * w
                        y = (yc - bh / 2.0) * h
                        ww = bw * w
                        hh = bh * h

                        # (Opcional) validar rangos (no forzamos clip; sólo descartamos bboxes inválidos)
                        if ww <= 1 or hh <= 1:
                            continue

                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cid + category_base,
                            "bbox": [x, y, ww, hh],
                            "area": ww * hh,
                            "iscrowd": 0
                        })
                        ann_id += 1
            except Exception as e:
                print(f"[WARN] Leyendo labels {lp}: {e}", file=sys.stderr)

        img_id += 1

    return images, annotations, img_id, ann_id


def build_categories(classes, category_base=0):
    """
    Crea la lista COCO 'categories' con ids contiguos.
    Si category_base=0 → ids 0..N-1 (recomendado).
    Si category_base=1 → ids 1..N   (COCO clásico).
    """
    categories = []
    for i, name in enumerate(classes):
        categories.append({"id": i + category_base, "name": name})
    return categories


def save_coco(path, images, annotations, categories):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": [],
        "info": {}
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    print(f"[OK] Wrote {path} | images={len(images)} anns={len(annotations)} cats={len(categories)}")


def main():
    ap = argparse.ArgumentParser()
    # Nuevo modo: soporta train+val en una sola corrida
    ap.add_argument("--images-train", dest="images_train", default=None)
    ap.add_argument("--labels-train", dest="labels_train", default=None)
    ap.add_argument("--images-val", dest="images_val", default=None)
    ap.add_argument("--labels-val", dest="labels_val", default=None)
    ap.add_argument("--out-train", dest="out_train", default=None)
    ap.add_argument("--out-val", dest="out_val", default=None)

    # Modo retrocompatible (un solo split)
    ap.add_argument("--img_dir", default=None)
    ap.add_argument("--lbl_dir", default=None)
    ap.add_argument("--out_json", default=None)

    # Clases y base de categorías
    ap.add_argument("--classes-file", dest="classes_file", default=None,
                    help="Ruta a dataset/classes.txt (uno por línea).")
    ap.add_argument("--category-base", dest="category_base", type=int, default=0,
                    choices=[0, 1], help="0: ids 0..N-1 (recomendado). 1: ids 1..N (COCO clásico).")

    args = ap.parse_args()
    classes = load_classes(args.classes_file)

    # Si nos pasaron train+val:
    if args.images_train and args.labels_train and args.out_train:
        categories = build_categories(classes, category_base=args.category_base)

        # Train
        im_tr, ann_tr, next_img_id, next_ann_id = yolo_to_coco_split(
            args.images_train, args.labels_train, classes,
            category_base=args.category_base,
            start_img_id=1, start_ann_id=1
        )
        save_coco(args.out_train, im_tr, ann_tr, categories)

        # Val (opcional)
        if args.images_val and args.labels_val and args.out_val:
            im_val, ann_val, _, _ = yolo_to_coco_split(
                args.images_val, args.labels_val, classes,
                category_base=args.category_base,
                start_img_id=1, start_ann_id=1
            )
            save_coco(args.out_val, im_val, ann_val, categories)
        return

    # Modo retrocompatible (un solo split)
    if args.img_dir and args.lbl_dir and args.out_json:
        categories = build_categories(classes, category_base=args.category_base)
        im, ann, _, _ = yolo_to_coco_split(
            args.img_dir, args.lbl_dir, classes,
            category_base=args.category_base,
            start_img_id=1, start_ann_id=1
        )
        save_coco(args.out_json, im, ann, categories)
        return

    print("[ERROR] Debes pasar train+val (recomendado) o bien --img_dir/--lbl_dir/--out_json.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
