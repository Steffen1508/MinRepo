"""
Træning og evaluering af trick-klassifikatoren.

    python train.py --model frame --name baseline_frame
    python train.py --model r2plus1d --freeze partial --crop-mode skater --name r21d_partial_skater
    python train.py --eval-only runs/r21d_partial_skater/best.pt \
        --external-root SkateboardML-1.0/Tricks --external-list SkateboardML-1.0/testlist02.txt

Alle flag: python train.py --help. Hver kørsel gemmer config, split, log, checkpoints,
metrikker, confusion matrices og forudsigelser per klip i runs/<navn>/.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CLASSES, TrickClips, build_cache, filter_by_list, find_clips, make_splits
from models import build_model, count_params


# ---------------------------------------------------------------- metrics

def confusion(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def metrics_from_cm(cm, classes):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    present = cm.sum(1) > 0  # macro over klasser der findes i sættet
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]),
                          "f1": float(f1[i]), "support": int(cm[i].sum())}
                      for i, c in enumerate(classes)},
    }


def plot_confusion(cm, classes, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Forudsagt")
    ax.set_ylabel("Sand")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- loops

def run_epoch(model, loader, device, criterion=None, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, ys, ps, probs = 0.0, [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y) if criterion is not None else torch.zeros((), device=device)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(y)
        ys.append(y.cpu())
        ps.append(logits.argmax(1).cpu())
        probs.append(logits.float().softmax(1).detach().cpu())
    ys, ps, probs = torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy()
    return total_loss / max(len(ys), 1), ys, ps, probs


def evaluate(model, index, args, device, out_dir, tag):
    ds = TrickClips(index, args.num_frames, args.crop, train=False)
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    _, ys, ps, probs = run_epoch(model, dl, device)
    cm = confusion(ys, ps, len(CLASSES))
    m = metrics_from_cm(cm, CLASSES)
    m["confusion"] = cm.tolist()
    # Fair metrik for sæt der kun har nogle af klasserne (fx SkateboardML: Ollie/Kickflip):
    # modellen vælger kun blandt de klasser der faktisk findes i sættet.
    present = sorted(set(ys.tolist()))
    if len(present) < len(CLASSES):
        sub = np.array(present)[probs[:, present].argmax(1)]
        cm_p = confusion(ys, sub, len(CLASSES))
        mp = metrics_from_cm(cm_p, CLASSES)
        m["present_classes"] = [CLASSES[i] for i in present]
        m["accuracy_present"] = mp["accuracy"]
        m["macro_f1_present"] = mp["macro_f1"]
        print(f"[{tag}] kun blandt {m['present_classes']}: accuracy {mp['accuracy']:.3f}  "
              f"macro F1 {mp['macro_f1']:.3f}")
        plot_confusion(cm_p, CLASSES, out_dir / f"confusion_{tag}_present.png",
                       f"{tag} (kun {', '.join(m['present_classes'])}): acc {mp['accuracy']:.2f}")
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(m, indent=2))
    plot_confusion(cm, CLASSES, out_dir / f"confusion_{tag}.png",
                   f"{tag}: acc {m['accuracy']:.2f}, macro F1 {m['macro_f1']:.2f}")
    pred = index[["rel", "label"]].reset_index(drop=True).copy()
    pred["pred"] = [CLASSES[p] for p in ps]
    for i, c in enumerate(CLASSES):
        pred[f"p_{c}"] = probs[:, i].round(4)
    pred.to_csv(out_dir / f"predictions_{tag}.csv", index=False)
    print(f"\n[{tag}] accuracy {m['accuracy']:.3f}  macro F1 {m['macro_f1']:.3f}")
    for c, v in m["per_class"].items():
        if v["support"]:
            print(f"  {c:<12} P {v['precision']:.2f}  R {v['recall']:.2f}  F1 {v['f1']:.2f}  n={v['support']}")
    return m


_DETECTOR = None


def load_index(root, args, list_file=None):
    """Find klip under root, byg/genbrug cachen og filtrér evt. efter en liste.
    Cachen bygges altid for hele roden, så train- og testlisten deler den."""
    global _DETECTOR
    root = Path(root)
    clips = find_clips(root)
    if clips.empty:
        raise SystemExit(f"Ingen klip med kendte klasser under {root}")
    if args.crop_mode == "skater" and _DETECTOR is None:
        from detect import load_detector
        print(f"Indlæser detektor ({args.detector}) ...")
        _DETECTOR = load_detector(args.detector)
    idx = build_cache(clips, args.cache_root / root.name, args.n_cache,
                      window_sec=args.window_sec or None,
                      crop_mode=args.crop_mode, detector=_DETECTOR)
    if list_file:
        idx = filter_by_list(idx, Path(list_file))
    return idx.reset_index(drop=True)


def ext_tag(root: Path) -> str:
    """'external' for SkateboardML, ellers fx 'external_egne' for egne optagelser."""
    name = Path(root).name
    return "external" if name in ("Tricks", "SkateboardML", "SkateboardML-1.0") else f"external_{name}"


def free_detector():
    global _DETECTOR
    _DETECTOR = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Træn skateboard trick-klassifikator")
    p.add_argument("--data-root", type=Path, default=Path("Dataset_Kaggle"))
    p.add_argument("--cache-root", type=Path, default=Path("cache"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--name", default=None, help="Navn på kørslen (standard: model_tidspunkt)")
    p.add_argument("--model", choices=["frame", "r2plus1d", "r3d"], default="frame")
    p.add_argument("--freeze", choices=["none", "partial", "backbone"], default="none")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--n-cache", type=int, default=32, help="Frames gemt per klip i cachen")
    p.add_argument("--window-sec", type=float, default=1.5,
                   help="Sekunder omkring bevægelsestoppen. 0 = hele klippet")
    p.add_argument("--crop", type=int, default=112)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--class-weights", action="store_true", help="Vægt loss efter klassefrekvens")
    p.add_argument("--patience", type=int, default=8, help="Early stopping på val macro F1")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--groups-csv", type=Path, help="CSV med kolonnerne rel,group til gruppe-split")
    p.add_argument("--splits-csv", type=Path, help="Genbrug et tidligere splits.csv")
    p.add_argument("--external-root", type=Path, help="Eksternt testsæt, fx SkateboardML-1.0/Tricks")
    p.add_argument("--external-list", type=Path,
                   help="Brug kun klip fra denne liste i det eksterne testsæt, fx testlist02.txt")
    p.add_argument("--extra-train-root", type=Path,
                   help="Ekstra træningsdata (alle klip går i train), fx SkateboardML-1.0/Tricks")
    p.add_argument("--extra-train-list", type=Path, help="Fx SkateboardML-1.0/trainlist02.txt")
    p.add_argument("--crop-mode", choices=["full", "skater"], default="full",
                   help="skater: detektér skateren og beskær omkring den")
    p.add_argument("--detector", choices=["mobilenet", "resnet50"], default="resnet50")
    p.add_argument("--eval-only", type=Path, help="Sti til checkpoint: spring træning over")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    # ---- eval-only: indstillinger hentes fra checkpointet
    if args.eval_only:
        ck = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        for k in ("model", "num_frames", "crop", "n_cache", "window_sec"):
            setattr(args, k, ck["args"][k])
        args.crop_mode = ck["args"].get("crop_mode", "full")
        args.detector = ck["args"].get("detector", args.detector)
        model = build_model(args.model, len(CLASSES), pretrained=False).to(device)
        model.load_state_dict(ck["state_dict"])
        out_dir = args.eval_only.parent
        if args.external_root:
            idx = load_index(args.external_root, args, args.external_list)
            free_detector()
            evaluate(model, idx, args, device, out_dir, ext_tag(args.external_root))
        else:
            splits = pd.read_csv(out_dir / "splits.csv")
            evaluate(model, splits[splits.split == "test"], args, device, out_dir, "test")
        return

    name = args.name or f"{args.model}_{time.strftime('%m%d-%H%M')}"
    out_dir = args.runs_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    # ---- data (alle caches bygges først, så detektoren kan smides ud før træning)
    index = load_index(args.data_root, args)
    extra = load_index(args.extra_train_root, args, args.extra_train_list) if args.extra_train_root else None
    ext = load_index(args.external_root, args, args.external_list) if args.external_root else None
    free_detector()
    if extra is not None and ext is not None:
        overlap = set(extra["path"]) & set(ext["path"])
        if overlap:
            raise SystemExit(f"{len(overlap)} klip er både i ekstra træningsdata og eksternt testsæt. "
                             "Brug --extra-train-list og --external-list.")
    if args.splits_csv:
        s = pd.read_csv(args.splits_csv)[["rel", "split"]]
        index = index.merge(s, on="rel", how="inner")
    else:
        groups = None
        if args.groups_csv:
            g = pd.read_csv(args.groups_csv).set_index("rel")["group"]
            groups = index["rel"].map(g).fillna(index["rel"])
            print(f"Gruppe-split med {groups.nunique()} grupper")
        index["split"] = make_splits(index, args.val, args.test, args.seed, groups)
    if extra is not None:
        extra = extra.assign(split="train", source="extra")
        index = pd.concat([index.assign(source="main"), extra], ignore_index=True)
        print(f"Tilføjet {len(extra)} ekstra træningsklip")
    index.to_csv(out_dir / "splits.csv", index=False)
    print(index.groupby(["label", "split"]).size().unstack(fill_value=0).reindex(CLASSES))

    tr, va, te = (index[index.split == s] for s in ("train", "val", "test"))
    if va.empty or te.empty:
        raise SystemExit("Val- eller testsættet er tomt. Brug flere grupper eller juster --val/--test.")
    dl_tr = DataLoader(TrickClips(tr, args.num_frames, args.crop, train=True), batch_size=args.bs,
                       shuffle=True, num_workers=args.workers, pin_memory=torch.cuda.is_available(), drop_last=len(tr) > args.bs)
    dl_va = DataLoader(TrickClips(va, args.num_frames, args.crop, train=False), batch_size=args.bs,
                       num_workers=args.workers, pin_memory=torch.cuda.is_available())

    # ---- model
    model = build_model(args.model, len(CLASSES), not args.no_pretrained, args.freeze).to(device)
    tot, trn = count_params(model)
    print(f"Model {args.model} (freeze={args.freeze}): {tot/1e6:.1f}M parametre, {trn/1e6:.2f}M trænes")

    weight = None
    if args.class_weights:
        cnt = tr["label"].value_counts().reindex(CLASSES, fill_value=1).values.astype(float)
        weight = torch.tensor(cnt.sum() / (len(cnt) * cnt), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    warmup = max(1, args.epochs // 10)
    sched = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, warmup),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, args.epochs - warmup)),
    ], milestones=[warmup])
    scaler = torch.amp.GradScaler() if use_amp else None

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out_dir / "tb")
    except Exception:
        pass

    # ---- træning
    best_f1, best_ep, log = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, ys, ps, _ = run_epoch(model, dl_tr, device, criterion, optimizer, scaler)
        tr_acc = float((ys == ps).mean())
        va_loss, ys, ps, _ = run_epoch(model, dl_va, device, criterion)
        vm = metrics_from_cm(confusion(ys, ps, len(CLASSES)), CLASSES)
        sched.step()
        row = {"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
               "val_acc": vm["accuracy"], "val_f1": vm["macro_f1"],
               "lr": optimizer.param_groups[0]["lr"], "sec": time.time() - t0}
        log.append(row)
        pd.DataFrame(log).to_csv(out_dir / "log.csv", index=False)
        if writer:
            for k, v in row.items():
                if k != "epoch":
                    writer.add_scalar(k, v, ep)
        flag = ""
        ck = {"state_dict": model.state_dict(), "args": vars(args) | {"data_root": str(args.data_root)},
              "classes": CLASSES, "epoch": ep}
        torch.save(ck, out_dir / "last.pt")
        if vm["macro_f1"] > best_f1:
            best_f1, best_ep, flag = vm["macro_f1"], ep, "  *"
            torch.save(ck, out_dir / "best.pt")
        print(f"ep {ep:3d}  train {tr_loss:.3f}/{tr_acc:.2f}  val {va_loss:.3f}/{vm['accuracy']:.2f}"
              f"  F1 {vm['macro_f1']:.3f}  {row['sec']:.0f}s{flag}")
        if ep - best_ep >= args.patience:
            print(f"Early stopping: ingen forbedring i {args.patience} epoker")
            break

    if writer:
        writer.close()
    plot_curves(pd.DataFrame(log), out_dir / "curves.png")

    # ---- test med bedste model
    print(f"\nBedste val macro F1 {best_f1:.3f} i epoke {best_ep}")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=False)["state_dict"])
    evaluate(model, va, args, device, out_dir, "val")
    evaluate(model, te, args, device, out_dir, "test")
    if ext is not None:
        evaluate(model, ext, args, device, out_dir, ext_tag(args.external_root))


def plot_curves(log, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    a.plot(log.epoch, log.train_loss, label="train")
    a.plot(log.epoch, log.val_loss, label="val")
    a.set_title("Loss"); a.set_xlabel("Epoke"); a.legend()
    b.plot(log.epoch, log.train_acc, label="train acc")
    b.plot(log.epoch, log.val_acc, label="val acc")
    b.plot(log.epoch, log.val_f1, label="val macro F1")
    b.set_title("Metrikker"); b.set_xlabel("Epoke"); b.set_ylim(0, 1); b.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
