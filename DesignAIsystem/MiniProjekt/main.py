"""
Kingdomino AI — fuld evalueringsrapport.

Kør:  python main.py

Genererer i reports/:
  terrain_confusion_matrix.png
  terrain_class_metrics.png
  terrain_roc_curves.png
  terrain_class_distribution.png
  crown_confusion_matrix.png
  crown_score_histogram.png
  crown_roc_curve.png
  crown_threshold_sweep.png

Printer metrics til terminal.
"""

import json
import os
import pickle

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

from features import extract_features
from matching import count_crowns, max_match_score

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE          = r"C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt"
DATASET_DIR   = os.path.join(BASE, "dataset")
GT_TERRAIN    = os.path.join(BASE, "ground_truth_terrain.json")
TEMPLATE_PATH = os.path.join(BASE, "krone_transparent.png")
CROWN_DIR     = os.path.join(BASE, "annotations", "crown")
NO_CROWN_DIR  = os.path.join(BASE, "annotations", "no_crown")
CROWN_COUNTS  = os.path.join(BASE, "crown_counts.json")
REPORTS_DIR   = os.path.join(BASE, "reports")

TILE_SIZE       = 100
CV_FOLDS        = 5
CROWN_THRESHOLD = 0.70

os.makedirs(REPORTS_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(REPORTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gemt: reports/{name}")


# ════════════════════════════════════════════════════════════════════════════════
# 1. TERRAIN SVM
# ════════════════════════════════════════════════════════════════════════════════

def load_terrain_dataset():
    with open(GT_TERRAIN, encoding="utf-8") as f:
        labels = json.load(f)

    X, y = [], []
    cached = {}

    for tile_id, terrain in labels.items():
        parts    = tile_id.rsplit("_x", 1)
        image_id = parts[0]
        col_str, row_str = parts[1].split("_y")
        x_pos, y_pos = int(col_str), int(row_str)

        if image_id not in cached:
            for ext in (".jpg", ".png"):
                img = cv.imread(os.path.join(DATASET_DIR, image_id + ext))
                if img is not None:
                    break
            cached[image_id] = img

        img = cached[image_id]
        if img is None:
            continue

        tile = img[y_pos * TILE_SIZE:(y_pos + 1) * TILE_SIZE,
                   x_pos * TILE_SIZE:(x_pos + 1) * TILE_SIZE]
        if tile.size == 0:
            continue

        X.append(extract_features(tile))
        y.append(terrain)

    X  = np.array(X, dtype=np.float32)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
    ])


def evaluate_terrain(X, y_enc, le):
    print("\n" + "═" * 60)
    print("  TERRAIN SVM")
    print("═" * 60)

    cv_split = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    pipeline = build_pipeline()

    y_pred = cross_val_predict(pipeline, X, y_enc, cv=cv_split)
    y_prob = cross_val_predict(pipeline, X, y_enc, cv=cv_split, method="predict_proba")

    classes   = le.classes_
    n_classes = len(classes)

    acc      = (y_pred == y_enc).mean()
    macro_f1 = f1_score(y_enc, y_pred, average="macro")
    roc_auc  = roc_auc_score(y_enc, y_prob, multi_class="ovr", average="macro")

    print(f"\n  Accuracy     : {acc*100:.2f}%")
    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Macro ROC AUC: {roc_auc:.4f}")
    print(f"\n  Classification report ({CV_FOLDS}-fold CV):")
    print(classification_report(y_enc, y_pred, target_names=classes, zero_division=0))

    # ── Plot 1: Confusion matrices ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm_raw  = confusion_matrix(y_enc, y_pred)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

    ConfusionMatrixDisplay(cm_raw,  display_labels=classes).plot(
        ax=axes[0], colorbar=False, xticks_rotation="vertical")
    axes[0].set_title("Confusion matrix — antal")

    ConfusionMatrixDisplay(cm_norm, display_labels=classes).plot(
        ax=axes[1], colorbar=False, xticks_rotation="vertical")
    axes[1].set_title("Confusion matrix — normaliseret")

    plt.suptitle(f"Terrain SVM — Confusion matrices ({CV_FOLDS}-fold CV)", fontsize=13)
    plt.tight_layout()
    savefig("terrain_confusion_matrix.png")

    # ── Plot 2: Precision / Recall / F1 per klasse ───────────────────────────
    prec = precision_score(y_enc, y_pred, average=None, zero_division=0)
    rec  = recall_score   (y_enc, y_pred, average=None, zero_division=0)
    f1   = f1_score       (y_enc, y_pred, average=None, zero_division=0)

    x = np.arange(n_classes)
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, prec, w, label="Precision", color="#4C72B0")
    ax.bar(x,     rec,  w, label="Recall",    color="#DD8452")
    ax.bar(x + w, f1,   w, label="F1",        color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Terrain SVM — Precision / Recall / F1 per klasse ({CV_FOLDS}-fold CV)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    savefig("terrain_class_metrics.png")

    # ── Plot 3: ROC kurver (one-vs-rest) ──────────────────────────────────────
    y_bin  = label_binarize(y_enc, classes=np.arange(n_classes))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (cls, col) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{cls}  (AUC={auc(fpr,tpr):.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Terrain SVM — ROC kurver per klasse (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig("terrain_roc_curves.png")

    # ── Plot 4: Klassefordeling ───────────────────────────────────────────────
    counts = np.bincount(y_enc)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(classes, counts, color="#4C72B0")
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Antal tiles")
    ax.set_title("Terrain — Klassefordeling i datasættet")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    savefig("terrain_class_distribution.png")


# ════════════════════════════════════════════════════════════════════════════════
# 2. CROWN TEMPLATE MATCHING
# ════════════════════════════════════════════════════════════════════════════════

def load_crown_scores(template):
    """Returner (scores, true_binary) for alle annoterede tiles."""
    with open(CROWN_COUNTS, encoding="utf-8") as f:
        gt_counts = json.load(f)

    scores, true_binary = [], []

    for fname in sorted(os.listdir(CROWN_DIR)):
        if not fname.endswith(".png"):
            continue
        tile_id = os.path.splitext(fname)[0]
        if gt_counts.get(tile_id) is None:
            continue
        tile = cv.imread(os.path.join(CROWN_DIR, fname))
        scores.append(max_match_score(tile, template))
        true_binary.append(1)

    for fname in sorted(os.listdir(NO_CROWN_DIR)):
        if not fname.endswith(".png"):
            continue
        tile = cv.imread(os.path.join(NO_CROWN_DIR, fname))
        scores.append(max_match_score(tile, template))
        true_binary.append(0)

    return np.array(scores), np.array(true_binary)


def evaluate_crown(template):
    print("\n" + "═" * 60)
    print("  KRONE TEMPLATE MATCHING")
    print("═" * 60)

    scores, true_binary = load_crown_scores(template)
    pred_binary = (scores >= CROWN_THRESHOLD).astype(int)

    total   = len(scores)
    bin_acc = (pred_binary == true_binary).mean()
    prec    = precision_score(true_binary, pred_binary, zero_division=0)
    rec     = recall_score   (true_binary, pred_binary, zero_division=0)
    f1      = f1_score       (true_binary, pred_binary, zero_division=0)
    roc_auc = roc_auc_score  (true_binary, scores)
    fp      = int(((pred_binary == 1) & (true_binary == 0)).sum())
    fn      = int(((pred_binary == 0) & (true_binary == 1)).sum())

    print(f"\n  Threshold        : {CROWN_THRESHOLD}")
    print(f"  Total tiles      : {total}  ({true_binary.sum()} med krone, {(true_binary==0).sum()} uden)")
    print(f"\n  Binær nøjagtighed: {bin_acc*100:.2f}%")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall           : {rec:.4f}")
    print(f"  F1               : {f1:.4f}")
    print(f"  ROC AUC          : {roc_auc:.4f}")
    print(f"  False Positives  : {fp}")
    print(f"  False Negatives  : {fn}")

    # ── Plot 1: Confusion matrix ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(true_binary, pred_binary),
        display_labels=["Ingen krone", "Krone"],
    ).plot(ax=ax, colorbar=False)
    ax.set_title(f"Krone confusion matrix (threshold={CROWN_THRESHOLD})")
    plt.tight_layout()
    savefig("crown_confusion_matrix.png")

    # ── Plot 2: Score histogram ───────────────────────────────────────────────
    bins = np.linspace(0, 1, 40)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores[true_binary == 0], bins=bins, alpha=0.7, label="Ingen krone", color="#DD8452")
    ax.hist(scores[true_binary == 1], bins=bins, alpha=0.7, label="Krone",       color="#4C72B0")
    ax.axvline(CROWN_THRESHOLD, color="red", linestyle="--", lw=1.5,
               label=f"Threshold = {CROWN_THRESHOLD}")
    ax.set_xlabel("Max match score")
    ax.set_ylabel("Antal tiles")
    ax.set_title("Fordeling af template match scores (krone vs. ingen krone)")
    ax.legend()
    ax.grid(alpha=0.4)
    plt.tight_layout()
    savefig("crown_score_histogram.png")

    # ── Plot 3: ROC kurve ─────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(true_binary, scores)
    roc_area    = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#4C72B0", label=f"AUC = {roc_area:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Krone detection — ROC kurve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig("crown_roc_curve.png")

    # ── Plot 4: Threshold sweep ───────────────────────────────────────────────
    thresholds = np.arange(0.50, 0.96, 0.05)
    bin_accs, sweep_fps, sweep_fns = [], [], []

    for t in thresholds:
        pb = (scores >= t).astype(int)
        bin_accs.append((pb == true_binary).mean() * 100)
        sweep_fps.append(int(((pb == 1) & (true_binary == 0)).sum()))
        sweep_fns.append(int(((pb == 0) & (true_binary == 1)).sum()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(thresholds, bin_accs, "o-", color="#4C72B0")
    ax1.axvline(CROWN_THRESHOLD, color="red", linestyle="--", lw=1.2,
                label=f"Valgt threshold = {CROWN_THRESHOLD}")
    ax1.set_ylabel("Binær nøjagtighed (%)")
    ax1.set_title("Krone detection — Threshold sweep")
    ax1.legend()
    ax1.grid(alpha=0.4)

    ax2.plot(thresholds, sweep_fps, "o-", color="#DD8452", label="False Positives")
    ax2.plot(thresholds, sweep_fns, "s-", color="#55A868", label="False Negatives")
    ax2.axvline(CROWN_THRESHOLD, color="red", linestyle="--", lw=1.2)
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Antal fejl")
    ax2.legend()
    ax2.grid(alpha=0.4)

    plt.tight_layout()
    savefig("crown_threshold_sweep.png")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Kingdomino AI — Evalueringsrapport")
    print(f"Gemmer plots i: {REPORTS_DIR}\n")

    print("Indlæser terrain-datasæt …")
    X, y_enc, le = load_terrain_dataset()
    evaluate_terrain(X, y_enc, le)

    template = cv.imread(TEMPLATE_PATH, cv.IMREAD_UNCHANGED)
    if template is None:
        print(f"Fejl: Kan ikke læse template: {TEMPLATE_PATH}")
    else:
        evaluate_crown(template)

    print("\n" + "═" * 60)
    print("  Rapport færdig — alle plots gemt i reports/")
    print("═" * 60)
