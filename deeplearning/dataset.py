"""
Data-del af skateboard trick-detektoren.

Tre trin:
  1. find_clips()   finder videoer og deres klasse ud fra mappenavne
  2. build_cache()  læser hver video én gang, finder trickket via bevægelse,
                    gemmer N frames som uint8 .npy (128x128) + index.csv
  3. TrickClips     PyTorch Dataset der sampler T frames fra cachen med augmentation

Kan også køres alene for at bygge cachen og se et par eksempler:
    python dataset.py --data-root Dataset_Kaggle --cache-dir cache/Dataset_Kaggle
"""

import argparse
import hashlib
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLASSES = ["Ollie", "Shuvit", "FrontShuvit", "Kickflip", "Heelflip"]
VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".m4v"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# Mappenavne der skal tolkes som en af vores klasser. Alt andet ignoreres
# (fx Backside180, Frontside180, PressureFlip i sensui-datasættet).
_ALIASES = {
    "ollie": "Ollie", "ollies": "Ollie",
    "shuvit": "Shuvit", "shoveit": "Shuvit", "shuveit": "Shuvit", "bsshuvit": "Shuvit",
    "backsideshuvit": "Shuvit",
    "frontshuvit": "FrontShuvit", "frontsideshuvit": "FrontShuvit", "fsshuvit": "FrontShuvit",
    "frontshoveit": "FrontShuvit", "frontsideshoveit": "FrontShuvit",
    "kickflip": "Kickflip", "kickflips": "Kickflip",
    "heelflip": "Heelflip", "heelflips": "Heelflip",
}


def label_from_path(path: Path, root: Path) -> str | None:
    """Klassen er navnet på den nærmeste overmappe der matcher en kendt klasse."""
    for part in reversed(path.relative_to(root).parts[:-1]):
        if _norm(part) in _ALIASES:
            return _ALIASES[_norm(part)]
    return None


def find_clips(root: Path) -> pd.DataFrame:
    root = Path(root).resolve()
    rows, skipped = [], {}
    for f in sorted(root.rglob("*")):
        if not (f.is_file() and f.suffix.lower() in VIDEO_EXT):
            continue
        lab = label_from_path(f, root)
        if lab is None:
            key = f.parent.name
            skipped[key] = skipped.get(key, 0) + 1
            continue
        rows.append({"path": str(f), "label": lab, "rel": str(f.relative_to(root))})
    if skipped:
        print("Ignorerede mapper (ikke en af de 5 klasser):",
              ", ".join(f"{k} ({v})" for k, v in sorted(skipped.items())))
    df = pd.DataFrame(rows, columns=["path", "label", "rel"])
    if len(df):
        print(df["label"].value_counts().reindex(CLASSES, fill_value=0).to_string())
    return df


# ---------------------------------------------------------------- video -> frames

def _letterbox(frame: np.ndarray, size: int) -> np.ndarray:
    """Pad til kvadrat med sort og skaler til size x size. Virker for både
    portræt- og landskabsvideo uden at forvrænge skateren."""
    h, w = frame.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(frame, top, s - h - top, left, s - w - left,
                            cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def _fit(frame: np.ndarray, longest: int) -> np.ndarray:
    """Skaler så længste side = longest, uden padding (til detektion)."""
    h, w = frame.shape[:2]
    s = longest / max(h, w)
    return cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def read_video(path: str, size: int = 128, letterbox: bool = True) -> tuple[np.ndarray, float]:
    """letterbox=True: (N, size, size, 3) med sort padding.
    letterbox=False: længste side = size, original aspekt bevaret."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(_letterbox(fr, size) if letterbox else _fit(fr, size))
    cap.release()
    if not frames:
        raise RuntimeError(f"Kunne ikke læse frames fra {path}")
    return np.stack(frames), float(fps)


def draw_preview(frame: np.ndarray, skater_box, crop_box, label: str = "") -> np.ndarray:
    """Frame med skaterboks (grøn) og crop-område (gul). RGB ind, BGR ud."""
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
    if skater_box is not None and not np.isnan(skater_box[0]):
        x1, y1, x2, y2 = np.asarray(skater_box).round().astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
    if crop_box is not None:
        x1, y1, x2, y2 = np.asarray(crop_box).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 1)
    if label:
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def motion_center(frames: np.ndarray, fps: float) -> int:
    """Index for det mest bevægelsesrige øjeblik. Forskellen mellem nabo-frames,
    glattet over ~0.25 s. Kamerapanorering kan snyde den, så tjek eksemplerne."""
    if len(frames) < 3:
        return len(frames) // 2
    gray = frames.mean(axis=3).astype(np.float32)
    energy = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    k = max(1, int(round(fps * 0.25)))
    smooth = np.convolve(energy, np.ones(k) / k, mode="same")
    return int(np.argmax(smooth)) + 1


def select_frames(frames: np.ndarray, fps: float, n: int, window_sec: float | None):
    """Vælg n frames jævnt fordelt, enten over hele klippet eller i et vindue
    omkring bevægelsestoppen."""
    total = len(frames)
    center = motion_center(frames, fps)
    if window_sec:
        half = int(window_sec * fps / 2)
        lo, hi = max(0, center - half), min(total, center + half)
        if hi - lo < n:  # for kort klip: brug det hele
            lo, hi = 0, total
    else:
        lo, hi = 0, total
    idx = np.linspace(lo, hi - 1, n).round().astype(int)
    return frames[idx], center, lo, hi


def build_cache(clips: pd.DataFrame, cache_dir: Path, n_cache: int = 32,
                size: int = 128, window_sec: float | None = 1.5,
                crop_mode: str = "full", detector=None, det_size: int = 480) -> pd.DataFrame:
    """Gemmer et (n_cache, size, size, 3) uint8-array per klip. Springer klip over
    der allerede er cachet med samme indstillinger.

    crop_mode="full":   hele billedet, letterboxet.
    crop_mode="skater": videoen læses i det_size, skateren detekteres i de valgte
                        frames, og der beskæres omkring den. Der gemmes også et
                        preview-billede med boksene (<hash>_preview.jpg).
    """
    from detect import skater_crops

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"n{n_cache}_s{size}_w{window_sec}"
    index_path = cache_dir / "index.csv"
    if crop_mode == "skater":
        tag += f"_skater{det_size}"
        index_path = cache_dir / "index_skater.csv"
        if detector is None:
            raise ValueError("crop_mode='skater' kræver en detector")
    old = pd.read_csv(index_path) if index_path.exists() else None

    rows = []
    for i, r in enumerate(clips.itertuples(), 1):
        h = hashlib.md5(f"{r.path}|{tag}".encode()).hexdigest()[:16]
        out = cache_dir / f"{h}.npy"
        if out.exists() and old is not None and (old["cache"] == str(out)).any():
            rows.append(old[old["cache"] == str(out)].iloc[0].to_dict()
                        | {"label": r.label, "rel": r.rel, "path": r.path})
            continue
        try:
            if crop_mode == "skater":
                frames, fps = read_video(r.path, det_size, letterbox=False)
                sel, center, lo, hi = select_frames(frames, fps, n_cache, window_sec)
                crops, boxes, crop_xy, found = skater_crops(sel, detector, size)
                mid = len(sel) // 2
                cv2.imwrite(str(cache_dir / f"{h}_preview.jpg"),
                            draw_preview(sel[mid], boxes[mid], crop_xy[mid], r.label))
                np.save(cache_dir / f"{h}_boxes.npy", boxes)
                sel = crops
            else:
                frames, fps = read_video(r.path, size)
                sel, center, lo, hi = select_frames(frames, fps, n_cache, window_sec)
                found = np.nan
        except Exception as e:
            print(f"  [{i}/{len(clips)}] SPRINGER OVER {r.path}: {e}")
            continue
        np.save(out, sel)
        rows.append({"path": r.path, "rel": r.rel, "label": r.label, "cache": str(out),
                     "fps": fps, "n_frames": len(frames), "center": center,
                     "win_lo": lo, "win_hi": hi, "det_found": found})
        if i % 20 == 0 or i == len(clips):
            print(f"  cachet {i}/{len(clips)}")
    df = pd.DataFrame(rows)
    df.to_csv(index_path, index=False)
    if crop_mode == "skater" and len(df):
        print(f"  Skater fundet i {df['det_found'].mean():.0%} af de valgte frames i gennemsnit; "
              f"{(df['det_found'] == 0).sum()} klip uden nogen detektion (bruger hele billedet)")
    return df


def filter_by_list(clips: pd.DataFrame, list_file: Path) -> pd.DataFrame:
    """Behold kun klip hvis filnavn står i en liste (fx SkateboardML's
    trainlist02.txt / testlist02.txt). Matcher på filnavn, så stiformatet i
    listen er ligegyldigt."""
    def key(p: Path, with_dir: bool) -> str:
        return (f"{p.parent.name}/{p.name}" if with_dir else p.name).lower()

    entries = [Path(l.split()[0].replace("\\", "/"))
               for l in Path(list_file).read_text().splitlines() if l.strip()]
    # Med mappe i listen (fx "Ollie/abc.mov") matches mappe + filnavn, så ens
    # filnavne i forskellige klassemapper ikke blandes sammen.
    with_dir = bool(entries) and all(len(e.parts) > 1 for e in entries)
    names = {key(e, with_dir) for e in entries}
    keep = clips[clips["path"].map(lambda p: key(Path(p), with_dir) in names)]
    print(f"  {list_file.name}: {len(keep)} af {len(clips)} klip matcher listen")
    return keep


# ---------------------------------------------------------------- splits

def make_splits(index: pd.DataFrame, val: float = 0.15, test: float = 0.15,
                seed: int = 0, groups: pd.Series | None = None) -> pd.Series:
    """Stratificeret split per klasse. Med groups (fx spot eller session per klip)
    holdes hver gruppe samlet i ét split, så modellen ikke kan genkende baggrunden
    fra træningen. Returnerer en Series med 'train'/'val'/'test'."""
    rng = random.Random(seed)
    split = pd.Series("train", index=index.index)
    for lab, sub in index.groupby("label"):
        if groups is None:
            units = [[i] for i in sub.index]
        else:
            units = [list(g.index) for _, g in sub.groupby(groups.loc[sub.index])]
        rng.shuffle(units)
        n = len(sub)
        n_test, n_val = max(1, round(n * test)), max(1, round(n * val))
        got = {"test": 0, "val": 0}
        for u in units[:-1]:  # sidste enhed går altid til train
            if got["test"] < n_test:
                s = "test"
            elif got["val"] < n_val:
                s = "val"
            else:
                break
            split[u] = s
            got[s] += len(u)
        if 0 in got.values():
            print(f"  ADVARSEL: {lab} har for få klip/grupper til både val og test "
                  f"({got['val']} val, {got['test']} test)")
    return split


# ---------------------------------------------------------------- Dataset

class TrickClips(Dataset):
    """Returnerer (video, label) hvor video er float (3, T, H, W) i [0, 1].
    Normalisering sker inde i modellen, så hver backbone kan bruge sine egne tal."""

    def __init__(self, index: pd.DataFrame, num_frames: int = 16, crop: int = 112,
                 train: bool = False, classes=CLASSES):
        self.df = index.reset_index(drop=True)
        self.T, self.crop, self.train = num_frames, crop, train
        self.cls2idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.df)

    def _temporal(self, n: int) -> np.ndarray:
        if not self.train:
            return np.linspace(0, n - 1, self.T).round().astype(int)
        # Temporal jitter: tilfældigt udsnit på 70-100 % af cachen
        span = random.uniform(0.7, 1.0) * (n - 1)
        start = random.uniform(0, (n - 1) - span)
        return np.linspace(start, start + span, self.T).round().astype(int)

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, 3, H, W)
        H, W = x.shape[-2:]
        if self.train:
            scale = random.uniform(0.6, 1.0)
            ch = cw = int(min(H, W) * scale)
            top, left = random.randint(0, H - ch), random.randint(0, W - cw)
            x = x[..., top:top + ch, left:left + cw]
            if random.random() < 0.5:
                # Spejling gør regular til goofy. Kickflip/heelflip og frontside/backside
                # er defineret ud fra skaterens stance, så labels bevares.
                x = x.flip(-1)
            b = random.uniform(0.8, 1.2)
            c = random.uniform(0.8, 1.2)
            m = x.mean()
            x = ((x - m) * c + m) * b
        else:
            ch = cw = min(H, W)
            top, left = (H - ch) // 2, (W - cw) // 2
            x = x[..., top:top + ch, left:left + cw]
        x = torch.nn.functional.interpolate(x, size=(self.crop, self.crop),
                                            mode="bilinear", align_corners=False)
        return x.clamp(0, 1)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        arr = np.load(r["cache"])                       # (N, H, W, 3) uint8
        arr = arr[self._temporal(len(arr))]
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0  # (T,3,H,W)
        x = self._spatial(x).permute(1, 0, 2, 3).contiguous()          # (3,T,H,W)
        y = self.cls2idx.get(r["label"], -1)
        return x, y


def save_examples(ds: TrickClips, out: Path, n: int = 6):
    """Gemmer en PNG-strimmel per eksempel, så man kan se om bevægelsesvinduet
    rammer trickket."""
    out.mkdir(parents=True, exist_ok=True)
    for i in random.sample(range(len(ds)), min(n, len(ds))):
        x, _ = ds[i]
        strip = (x.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)  # (T,H,W,3)
        img = np.concatenate(list(strip), axis=1)
        lab = ds.df.iloc[i]["label"]
        cv2.imwrite(str(out / f"{i:04d}_{lab}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Eksempler gemt i {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--n-cache", type=int, default=32)
    p.add_argument("--window-sec", type=float, default=1.5,
                   help="Sekunder omkring bevægelsestoppen. 0 = hele klippet")
    p.add_argument("--crop-mode", choices=["full", "skater"], default="full")
    p.add_argument("--detector", choices=["mobilenet", "resnet50"], default="resnet50")
    a = p.parse_args()
    clips = find_clips(a.data_root)
    det = None
    if a.crop_mode == "skater":
        from detect import load_detector
        det = load_detector(a.detector)
    idx = build_cache(clips, a.cache_dir, a.n_cache, window_sec=a.window_sec or None,
                      crop_mode=a.crop_mode, detector=det)
    save_examples(TrickClips(idx, train=False), a.cache_dir / f"examples_{a.crop_mode}")
