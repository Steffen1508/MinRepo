"""
Skater-detektion og beskæring.

Bruger en Faster R-CNN fra torchvision, pretrained på COCO, hvor "person" (1)
og "skateboard" (41) er klasser. For hvert frame vælges den person, der hører
til skateboardet, og boksen for person + board slås sammen. Over et klip
glattes boksene, og der laves en stabiliseret, kvadratisk crop med fast
størrelse, der følger skaterens centrum.
"""

import numpy as np
import torch
from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                          FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_mobilenet_v3_large_fpn,
                                          fasterrcnn_resnet50_fpn_v2)

PERSON, SKATEBOARD = 1, 41


def load_detector(name: str = "mobilenet", device=None, pretrained: bool = True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if name == "mobilenet":
        m = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None,
            weights_backbone=None, box_score_thresh=0.2)
    elif name == "resnet50":
        m = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
            weights_backbone=None, box_score_thresh=0.2)
    else:
        raise ValueError(name)
    return m.eval().to(device)


@torch.no_grad()
def detect_frames(frames: np.ndarray, detector, batch: int = 16) -> list[dict]:
    """frames: (N, H, W, 3) uint8 RGB. Returnerer per frame dict med
    'person' og 'board': arrays af [x1, y1, x2, y2, score]."""
    device = next(detector.parameters()).device
    out = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(frames[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255
        for d in detector(list(x)):
            b = torch.cat([d["boxes"], d["scores"][:, None]], 1).cpu().numpy()
            lab = d["labels"].cpu().numpy()
            out.append({"person": b[(lab == PERSON) & (b[:, 4] > 0.5)],
                        "board": b[(lab == SKATEBOARD) & (b[:, 4] > 0.3)]})
    return out


def _union(a, b):
    return np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])])


def _center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def pick_skater(dets: list[dict], frame_hw) -> np.ndarray:
    """Én boks [x1,y1,x2,y2] per frame (NaN hvis intet fundet).

    Med skateboard i billedet: personen hvis bund er tættest på boardet.
    Uden: personen tættest på sidste kendte skaterposition, ellers den største.
    Det håndterer tilskuere og filmere i baggrunden nogenlunde.
    """
    H, W = frame_hw
    diag = np.hypot(H, W)
    boxes = np.full((len(dets), 4), np.nan)
    last = None
    for i, d in enumerate(dets):
        P, B = d["person"], d["board"]
        best = None
        if len(B):
            board = B[B[:, 4].argmax(), :4]
            if len(P):
                foot = np.stack([(P[:, 0] + P[:, 2]) / 2, P[:, 3]], 1)
                dist = np.linalg.norm(foot - _center(board), axis=1)
                j = dist.argmin()
                best = _union(P[j, :4], board) if dist[j] < 0.25 * diag else board
            else:
                best = board
        elif len(P):
            if last is not None:
                j = np.linalg.norm(np.stack([_center(p) for p in P[:, :4]]) - _center(last), axis=1).argmin()
            else:
                j = ((P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1])).argmax()
            best = P[j, :4]
        if best is not None:
            boxes[i] = best
            last = best
    return boxes


def smooth_boxes(boxes: np.ndarray, k: int = 5) -> np.ndarray:
    """Interpolér huller og glat med glidende median."""
    out = boxes.copy()
    idx = np.arange(len(boxes))
    ok = ~np.isnan(boxes[:, 0])
    if ok.sum() == 0:
        return out
    for c in range(4):
        out[:, c] = np.interp(idx, idx[ok], boxes[ok, c])
    if k > 1 and len(out) >= k:
        pad = np.pad(out, ((k // 2, k // 2), (0, 0)), mode="edge")
        out = np.stack([np.median(pad[i:i + k], 0) for i in range(len(out))])
    return out


def crop_boxes(boxes: np.ndarray, frame_hw, margin: float = 1.4) -> np.ndarray:
    """Kvadratiske crops med fast side for hele klippet (så skalaen er den samme
    gennem trickket) og centrum der følger skateren. Side = margin * største
    boks-side i klippet. Returnerer (N, 4) heltal [x1, y1, x2, y2], må gå
    uden for billedet (der paddes ved beskæring)."""
    H, W = frame_hw
    if np.isnan(boxes[:, 0]).all():
        s = max(H, W)
        c = np.array([W / 2, H / 2])
        return np.tile(np.r_[c - s / 2, c + s / 2], (len(boxes), 1)).round().astype(int)
    wh = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
    side = min(max(H, W), margin * np.nanmax(wh))
    side = max(side, 0.25 * min(H, W))  # ikke absurd små crops
    c = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    return np.concatenate([c - side / 2, c + side / 2], 1).round().astype(int)


def apply_crop(frame: np.ndarray, box, size: int) -> np.ndarray:
    import cv2
    x1, y1, x2, y2 = box
    H, W = frame.shape[:2]
    pad = max(0, -x1, -y1, x2 - W, y2 - H)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    return cv2.resize(frame[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


def skater_crops(frames: np.ndarray, detector, size: int = 128, margin: float = 1.4):
    """frames (N,H,W,3) -> crops (N,size,size,3), skaterbokse (N,4), crop-bokse (N,4)."""
    hw = frames.shape[1:3]
    raw = pick_skater(detect_frames(frames, detector), hw)
    found = float((~np.isnan(raw[:, 0])).mean())
    boxes = smooth_boxes(raw)
    crops_xy = crop_boxes(boxes, hw, margin)
    crops = np.stack([apply_crop(f, b, size) for f, b in zip(frames, crops_xy)])
    return crops, boxes, crops_xy, found
