import argparse, os, cv2, numpy as np
from pathlib import Path
from glob import glob

HELP = "Mouse: drag to add box | Keys: s=save  d=del-last  c=clear  ←/→=prev/next  q=quit"

def write_yolo(txt_path, boxes, W, H, cls=0):
    with open(txt_path, "w") as f:
        for (x0,y0,x1,y1) in boxes:
            x0,y0,x1,y1 = map(float, (x0,y0,x1,y1))
            cx = ((x0+x1)/2.0)/W; cy = ((y0+y1)/2.0)/H
            w  = abs(x1-x0)/W;    h  = abs(y1-y0)/H
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_lbl_dir", required=True)    # e.g., dataset/labels/train
    ap.add_argument("--list", default=None, help="optional text file with subset of images to label")
    args = ap.parse_args()

    os.makedirs(args.out_lbl_dir, exist_ok=True)
    if args.list:
        imgs = [l.strip() for l in open(args.list) if l.strip()]
    else:
        imgs = sorted(glob(os.path.join(args.img_dir, "*.jpg")))
    assert imgs, "No images to annotate"

    idx = 0
    boxes = []
    start = None
    cv2.namedWindow("annot", cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal start, boxes, frame_disp
        if event == cv2.EVENT_LBUTTONDOWN:
            start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and start is not None:
            img2 = frame_disp.copy()
            cv2.rectangle(img2, start, (x,y), (0,255,0), 2)
            cv2.imshow("annot", img2)
        elif event == cv2.EVENT_LBUTTONUP and start is not None:
            x0,y0 = start; x1,y1 = x,y
            if x1<x0: x0,x1 = x1,x0
            if y1<y0: y0,y1 = y1,y0
            boxes.append((x0,y0,x1,y1))
            start = None

    while 0 <= idx < len(imgs):
        ip = imgs[idx]
        frame = cv2.imread(ip)
        if frame is None:
            print("Cannot read:", ip); idx += 1; continue
        H, W = frame.shape[:2]
        frame_disp = frame.copy()
        cv2.putText(frame_disp, HELP, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # load existing boxes if present
        txt = Path(args.out_lbl_dir) / (Path(ip).stem + ".txt")
        boxes = []
        if txt.exists():
            for line in open(txt):
                cls, cx, cy, w, h = map(float, line.strip().split())
                x0 = (cx - w/2) * W; y0 = (cy - h/2) * H
                x1 = (cx + w/2) * W; y1 = (cy + h/2) * H
                boxes.append((x0,y0,x1,y1))

        cv2.setMouseCallback("annot", on_mouse)
        while True:
            # draw current boxes
            disp = frame_disp.copy()
            for (x0,y0,x1,y1) in boxes:
                cv2.rectangle(disp, (int(x0),int(y0)), (int(x1),int(y1)), (0,255,0), 2)
            cv2.imshow("annot", disp)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'): cv2.destroyAllWindows(); return
            if k == ord('d') and boxes: boxes.pop()
            if k == ord('c'): boxes = []
            if k == ord('s'):
                write_yolo(str(txt), boxes, W, H, cls=0)
                print("saved:", txt)
            if k == 83 or k == ord(']'):   # right arrow
                idx += 1; break
            if k == 81 or k == ord('['):   # left arrow
                idx = max(0, idx-1); break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
