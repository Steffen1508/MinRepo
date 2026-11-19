"""
Kør en trænet model på en video og lav en ny video med bounding box om skateren
og det forudsagte trick med confidence.

    python predict.py klip.mov --checkpoint runs/r21d_partial_skater/best.pt
    python predict.py lang_video.mp4 --checkpoint runs/r21d_partial_skater/best.pt --mode sliding

Modes:
  clip     (standard) Ét trick per video, som under træning: vinduet omkring det
           mest bevægelsesrige øjeblik klassificeres. Passer til korte klip.
  sliding  Glidende vindue hen over hele videoen. Til længere optagelser med
           flere tricks. Labels vises kun når confidence > --thr. Modellen har
           ingen "intet trick"-klasse, så i rolige passager vil den stadig gætte.

Output: <video>_pred.mp4 (afspilles i --out-fps, standard 30, så 120 fps-klip
bliver slowmotion) og <video>_pred.csv med sandsynligheder per vindue.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from dataset import CLASSES, _letterbox, motion_center, read_video
from detect import (apply_crop, crop_boxes, detect_frames, load_detector,
                    pick_skater, smooth_boxes)
from models import build_model

COLORS = {"Ollie": (80, 200, 255), "Shuvit": (255, 170, 60), "FrontShuvit": (255, 90, 160),
          "Kickflip": (90, 230, 90), "Heelflip": (90, 120, 255)}  # BGR


def window_tensor(frames, boxes, lo, hi, a) -> torch.Tensor:
    """Byg modelinput for frames[lo:hi] præcis som under træning."""
    idx = np.linspace(lo, hi - 1, a["n_cache"]).round().astype(int)
    sel = frames[idx]
    if a["crop_mode"] == "skater":
        cxy = crop_boxes(boxes[idx], sel.shape[1:3])
        arr = np.stack([apply_crop(f, b, 128) for f, b in zip(sel, cxy)])
    else:
        arr = np.stack([_letterbox(f, 128) for f in sel])
    t = np.linspace(0, len(arr) - 1, a["num_frames"]).round().astype(int)
    x = torch.from_numpy(arr[t]).permute(0, 3, 1, 2).float() / 255
    x = torch.nn.functional.interpolate(x, size=(a["crop"], a["crop"]), mode="bilinear",
                                        align_corners=False)
    return x.permute(1, 0, 2, 3).unsqueeze(0)  # (1,3,T,H,W)


def draw(frame_rgb, box, probs, show_label, in_window):
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    k = int(np.argmax(probs))
    name, conf = CLASSES[k], float(probs[k])
    col = COLORS[name]
    # Sandsynlighedspanel øverst til venstre
    bw = max(50, int(W * 0.15))
    px2, py2 = 100 + bw + 44, 14 + 20 * len(CLASSES)
    cv2.rectangle(img, (6, 6), (px2, py2), (0, 0, 0), -1)
    for i, c in enumerate(CLASSES):
        y = 22 + 20 * i
        cv2.putText(img, c, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(img, (100, y - 10), (100 + int(bw * probs[i]), y - 1), COLORS[c], -1)
        cv2.putText(img, f"{probs[i]:.2f}", (106 + bw, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
    if box is not None and not np.isnan(box[0]):
        x1, y1, x2, y2 = np.asarray(box).round().astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), col if show_label else (200, 200, 200), 2)
        if show_label:
            txt = f"{name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            lx = int(np.clip(x1, 0, W - tw - 8))
            ty = y1  # label over boksen ...
            if ty - th - 8 < 0 or (ty - th - 8 < py2 and lx < px2):
                ty = min(H, y2 + th + 8)  # ... eller under, hvis den rammer kanten/panelet
            cv2.rectangle(img, (lx, ty - th - 8), (lx + tw + 8, ty), col, -1)
            cv2.putText(img, txt, (lx + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    if in_window:
        cv2.circle(img, (W - 16, 16), 7, (0, 0, 255), -1)  # rød prik = vinduet modellen ser
    return img


def main():
    p = argparse.ArgumentParser(description="Trick-detektion på en video med boks og confidence")
    p.add_argument("video", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--mode", choices=["clip", "sliding"], default="clip")
    p.add_argument("--stride-sec", type=float, default=0.25)
    p.add_argument("--thr", type=float, default=0.5, help="Min. confidence for at vise label (sliding)")
    p.add_argument("--det-size", type=int, default=640, help="Længste side ved detektion og output")
    p.add_argument("--det-fps", type=float, default=30, help="Detektioner per sekund (interpoleres)")
    p.add_argument("--out-fps", type=float, default=30)
    p.add_argument("--detector", choices=["mobilenet", "resnet50"])
    a_cli = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a_cli.checkpoint, map_location="cpu", weights_only=False)
    a = {k: ck["args"][k] for k in ("model", "num_frames", "crop", "n_cache", "window_sec")}
    a["crop_mode"] = ck["args"].get("crop_mode", "full")
    model = build_model(a["model"], len(CLASSES), pretrained=False).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    print(f"Model: {a['model']}, crop_mode={a['crop_mode']}, {a['num_frames']} frames")

    frames, fps = read_video(str(a_cli.video), a_cli.det_size, letterbox=False)
    N, hw = len(frames), frames.shape[1:3]
    print(f"{N} frames, {fps:.0f} fps, {hw[1]}x{hw[0]}")

    # Detektion på et udsnit af frames, interpoleret til alle
    step = max(1, int(round(fps / a_cli.det_fps)))
    det_idx = np.arange(0, N, step)
    detector = load_detector(a_cli.detector or ck["args"].get("detector", "resnet50"), device)
    raw_sub = pick_skater(detect_frames(frames[det_idx], detector), hw)
    raw = np.full((N, 4), np.nan)
    raw[det_idx] = raw_sub
    boxes = smooth_boxes(raw, k=5)
    found = float((~np.isnan(raw_sub[:, 0])).mean())
    print(f"Skater fundet i {found:.0%} af de detekterede frames")
    del detector

    # Vinduer
    win = int(round((a["window_sec"] or N / fps) * fps))
    win = max(min(win, N), 2)
    if a_cli.mode == "clip":
        c = motion_center(frames, fps)
        lo = int(np.clip(c - win // 2, 0, N - win))
        windows = [(lo, lo + win)]
    else:
        stride = max(1, int(round(a_cli.stride_sec * fps)))
        windows = [(s, s + win) for s in range(0, max(1, N - win + 1), stride)]

    rows, probs_w = [], []
    with torch.no_grad():
        for lo, hi in windows:
            pr = model(window_tensor(frames, boxes, lo, hi, a).to(device)).softmax(1)[0].cpu().numpy()
            probs_w.append(pr)
            rows.append({"start_s": lo / fps, "end_s": hi / fps, "pred": CLASSES[pr.argmax()],
                         "conf": float(pr.max())} | {f"p_{c}": float(pr[i]) for i, c in enumerate(CLASSES)})
    probs_w = np.array(probs_w)
    df = pd.DataFrame(rows)

    # Output-video
    out = a_cli.out or a_cli.video.with_name(a_cli.video.stem + "_pred.mp4")
    H, W = hw
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a_cli.out_fps, (W, H))
    centers = np.array([(lo + hi) / 2 for lo, hi in windows])
    for i in range(N):
        if a_cli.mode == "clip":
            pr = probs_w[0]
            in_win = windows[0][0] <= i < windows[0][1]
            show = in_win
        else:
            j = int(np.abs(centers - i).argmin())
            pr = probs_w[j]
            in_win = True
            show = pr.max() >= a_cli.thr
        vw.write(draw(frames[i], boxes[i], pr, show, in_win))
    vw.release()
    df.to_csv(out.with_suffix(".csv"), index=False)

    if a_cli.mode == "clip":
        top = np.argsort(-probs_w[0])[:3]
        print("Forudsigelse: " + ", ".join(f"{CLASSES[k]} {probs_w[0][k]:.0%}" for k in top))
    else:
        print(df[df.conf >= a_cli.thr][["start_s", "end_s", "pred", "conf"]].round(2).to_string(index=False))
    print(f"Gemt: {out}")


if __name__ == "__main__":
    main()
