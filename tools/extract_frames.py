# tools/extract_frames.py
import os, cv2, argparse
from glob import glob
from pathlib import Path
import random

def extract_frames(video_path: str, out_dir: str, fps: int = 8) -> int:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(src_fps / fps)), 1)
    vid = Path(video_path).stem
    saved, i = 0, 0
    while True:
        grabbed = cap.grab()
        if not grabbed:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok: break
            fname = f"{vid}_{saved:06d}.jpg"
            cv2.imwrite(str(Path(out_dir) / fname), frame)
            saved += 1
        i += 1
    cap.release()
    return saved

def split_by_video(img_dir: str, train_ratio: float = 0.8, seed: int = 42):
    imgs = sorted(glob(str(Path(img_dir) / "*.jpg")))
    per_vid = {}
    for p in imgs:
        vid = Path(p).name.split("_")[0]
        per_vid.setdefault(vid, []).append(p)
    vids = list(per_vid.keys())
    random.Random(seed).shuffle(vids)
    ntr = int(len(vids)*train_ratio)
    train_vids, val_vids = set(vids[:ntr]), set(vids[ntr:])
    train_imgs = [p for v in train_vids for p in per_vid[v]]
    val_imgs   = [p for v in val_vids   for p in per_vid[v]]
    return train_imgs, val_imgs

def move_imgs(paths, dst_dir):
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    for p in paths:
        src = Path(p); dst = Path(dst_dir) / src.name
        os.replace(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=str, required=True, help="folder with .mp4/.mov/.avi")
    ap.add_argument("--out", type=str, default="dataset", help="dataset root")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    images_all = Path(args.out) / "images" / "all"
    images_all.mkdir(parents=True, exist_ok=True)

    # 1) extract all frames to images/all
    exts = ("*.mp4", "*.mov", "*.avi", "*.mkv")
    vids = []
    for ext in exts:
        vids += glob(str(Path(args.videos) / ext))
    assert vids, f"No videos found in {args.videos}"
    total = 0
    for v in sorted(vids):
        n = extract_frames(v, str(images_all), fps=args.fps)
        print(f"{Path(v).name} -> {n} frames")
        total += n

    # 2) split by video and move to train/val
    train, val = split_by_video(str(images_all), train_ratio=0.8)
    move_imgs(train, Path(args.out) / "images/train")
    move_imgs(val,   Path(args.out) / "images/val")

    # 3) create empty label dirs
    (Path(args.out)/"labels/train").mkdir(parents=True, exist_ok=True)
    (Path(args.out)/"labels/val").mkdir(parents=True, exist_ok=True)

    print(f"Done. Frames total: {total}. Train: {len(train)}, Val: {len(val)}")
    print(f"Images: {Path(args.out)/'images'} | Labels: {Path(args.out)/'labels'}")

if __name__ == "__main__":
    main()
