"""
Evaluér template matching mod crown_counts.json ground truth.

Tester på alle tiles i annotations/crown/ (tiles MED kroner)
samt annotations/no_crown/ (tiles UDEN kroner).

Rapporterer:
  - Threshold sweep (0.50 → 0.95)
  - Præcis count-nøjagtighed (forudsagt antal == faktisk antal)
  - Binær nøjagtighed (krone fundet / ikke fundet)
  - Fejlliste ved valgt threshold

Usage:
  python evaluate_crown.py
"""

import json
import os

import cv2 as cv
import numpy as np

from matching import count_crowns

# ── Config ─────────────────────────────────────────────────────────────────────
CROWN_DIR      = r"C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt\annotations\crown"
NO_CROWN_DIR   = r"C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt\annotations\no_crown"
TEMPLATE_PATH  = r"C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt\krone_transparent.png"
CROWN_COUNTS   = r"C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt\crown_counts.json"
THRESHOLD      = 0.7
# ───────────────────────────────────────────────────────────────────────────────


def load_ground_truth():
    with open(CROWN_COUNTS, encoding="utf-8") as f:
        return json.load(f)


def evaluate(template, ground_truth, threshold=THRESHOLD):
    results = []

    # ── Tiles med kroner ───────────────────────────────────────────────────────
    for filename in sorted(os.listdir(CROWN_DIR)):
        if not filename.endswith(".png"):
            continue
        tile_id    = os.path.splitext(filename)[0]
        true_count = ground_truth.get(tile_id)

        if true_count is None:
            print(f"  Advarsel: {tile_id} mangler i crown_counts.json — springer over")
            continue

        tile       = cv.imread(os.path.join(CROWN_DIR, filename))
        pred_count = count_crowns(tile, template, threshold=threshold)
        results.append((tile_id, true_count, pred_count))

    # ── Tiles uden kroner ──────────────────────────────────────────────────────
    for filename in sorted(os.listdir(NO_CROWN_DIR)):
        if not filename.endswith(".png"):
            continue
        tile_id    = os.path.splitext(filename)[0]
        tile       = cv.imread(os.path.join(NO_CROWN_DIR, filename))
        pred_count = count_crowns(tile, template, threshold=threshold)
        results.append((tile_id, 0, pred_count))

    return results


def print_report(results):
    total          = len(results)
    exact_correct  = sum(1 for _, true, pred in results if pred == true)
    binary_correct = sum(1 for _, true, pred in results if (pred > 0) == (true > 0))

    print(f"\n{'='*50}")
    print(f"  Total tiles evalueret    : {total}")
    print(f"  Præcis count-nøjagtighed : {exact_correct}/{total}  ({exact_correct/total*100:.1f}%)")
    print(f"  Binær nøjagtighed        : {binary_correct}/{total}  ({binary_correct/total*100:.1f}%)")
    print(f"{'='*50}")

    errors = [(tid, true, pred) for tid, true, pred in results if pred != true]
    if errors:
        print(f"\nFejl ({len(errors)} stk):")
        print(f"  {'Tile ID':<30} {'Sand':>5} {'Forudsagt':>10}")
        print(f"  {'-'*47}")
        for tile_id, true, pred in errors:
            print(f"  {tile_id:<30} {true:>5} {pred:>10}")
    else:
        print("\nIngen fejl — perfekt score!")

    false_positives = sum(1 for _, true, pred in results if true == 0 and pred > 0)
    false_negatives = sum(1 for _, true, pred in results if true > 0 and pred == 0)
    wrong_count     = sum(1 for _, true, pred in results if true > 0 and pred > 0 and pred != true)

    print(f"\nFejltyper:")
    print(f"  False positives (fandt krone hvor der ingen er) : {false_positives}")
    print(f"  False negatives (missede krone)                 : {false_negatives}")
    print(f"  Forkert antal (fandt krone men forkert antal)   : {wrong_count}")


def threshold_sweep(template, ground_truth):
    print("\nThreshold sweep:")
    print(f"  {'Thresh':>7} {'Binær %':>8} {'Præcis %':>9} {'FP':>5} {'FN':>5} {'Forkert antal':>14}")
    print(f"  {'-'*50}")

    for t in np.arange(0.50, 0.96, 0.05):
        t = round(t, 2)
        results = evaluate(template, ground_truth, threshold=t)
        total  = len(results)
        fp     = sum(1 for _, true, pred in results if true == 0 and pred > 0)
        fn     = sum(1 for _, true, pred in results if true > 0 and pred == 0)
        wc     = sum(1 for _, true, pred in results if true > 0 and pred > 0 and pred != true)
        bin_ok = sum(1 for _, true, pred in results if (pred > 0) == (true > 0))
        ex_ok  = sum(1 for _, true, pred in results if pred == true)
        print(f"  {t:>7.2f} {bin_ok/total*100:>7.1f}% {ex_ok/total*100:>8.1f}% {fp:>5} {fn:>5} {wc:>14}")


if __name__ == "__main__":
    template = cv.imread(TEMPLATE_PATH, cv.IMREAD_UNCHANGED)
    if template is None:
        print(f"Fejl: Kan ikke læse template: {TEMPLATE_PATH}")
        raise SystemExit(1)

    ground_truth = load_ground_truth()
    print(f"Ground truth loaded: {len(ground_truth)} tiles med kroner")

    threshold_sweep(template, ground_truth)

    print(f"\nDetaljeret rapport ved threshold={THRESHOLD}:")
    results = evaluate(template, ground_truth, threshold=THRESHOLD)
    print_report(results)
