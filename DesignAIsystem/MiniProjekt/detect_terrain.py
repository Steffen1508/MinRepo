"""
Automatic terrain detection for Kingdomino tiles using HSV color ranges.

Usage:
  python detect_terrain.py                  — demo on image 1 + evaluate against ground truth
  python detect_terrain.py <image_path>     — detect a single image and print the board
"""

import os
import sys
import json
import cv2 as cv
import numpy as np

# ── config ─────────────────────────────────────────────────────────────────────
tile_size = 100
grid_size = 5

# HSV ranges per terrain: list of (lower, upper) pairs
hsv_ranges = {
    'wheat':  [(np.array([15,  60, 100]), np.array([35, 255, 255]))],
    'forest': [(np.array([35,  50,  20]), np.array([85, 255, 150]))],
    'water':  [(np.array([90,  60,  60]), np.array([140, 255, 255]))],
    'grass':  [(np.array([35,  40, 100]), np.array([85,  255, 255]))],
    'swamp':  [(np.array([10,  30,  30]), np.array([40,  150, 130]))],
    'mine':   [(np.array([0,    0,  30]), np.array([180,  40, 130]))],
}

abbrev = {
    'wheat':   'WHE',
    'forest':  'FOR',
    'water':   'WAT',
    'grass':   'GRS',
    'swamp':   'SWP',
    'mine':    'MIN',
    'castle':  'CST',
    'unknown': '???',
}
# ───────────────────────────────────────────────────────────────────────────────


def classify_tile(tile):
    """Return (terrain_label, pixel_score) for one 100x100 tile."""
    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    best_label = 'unknown'
    best_score = 0

    for label, ranges in hsv_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv.inRange(hsv, lower, upper)
        score = int(np.sum(mask > 0))
        if score > best_score:
            best_score = score
            best_label = label

    # Castle / home tile: very low saturation and high brightness
    sat = float(hsv[:, :, 1].mean())
    val = float(hsv[:, :, 2].mean())
    if sat < 30 and val > 150:
        best_label = 'castle'

    return best_label, best_score


def detect_board(image):
    """Return a 5x5 list of terrain labels for a 500x500 board image."""
    board = []
    for y in range(grid_size):
        row = []
        for x in range(grid_size):
            tile = image[y * tile_size:(y + 1) * tile_size,
                         x * tile_size:(x + 1) * tile_size]
            label, _ = classify_tile(tile)
            row.append(label)
        board.append(row)
    return board


def print_board(board):
    print("     " + "  ".join(f"x{x}" for x in range(grid_size)))
    for y, row in enumerate(board):
        cells = "  ".join(abbrev.get(t, '???') for t in row)
        print(f"  y{y}  {cells}")


def evaluate(image_folder, ground_truth_path):
    """Compare automatic predictions against a ground truth JSON file."""
    if not os.path.exists(ground_truth_path):
        print(f"No ground truth found at: {ground_truth_path}")
        return

    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)

    correct = 0
    total = 0
    errors = []

    for filename in sorted(os.listdir(image_folder)):
        if not filename.lower().endswith(('.jpg', '.png')):
            continue
        image_id = os.path.splitext(filename)[0]
        img = cv.imread(os.path.join(image_folder, filename))
        if img is None:
            continue

        for y in range(grid_size):
            for x in range(grid_size):
                tile_id = f"{image_id}_x{x}_y{y}"
                if tile_id not in gt:
                    continue
                tile = img[y * tile_size:(y + 1) * tile_size,
                           x * tile_size:(x + 1) * tile_size]
                predicted, _ = classify_tile(tile)
                expected = gt[tile_id]
                total += 1
                if predicted == expected:
                    correct += 1
                else:
                    errors.append((tile_id, expected, predicted))

    if total == 0:
        print("No annotated tiles found in ground truth.")
        return

    accuracy = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for tile_id, exp, pred in errors[:20]:
            print(f"  {tile_id}: expected={exp}, predicted={pred}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    image_folder  = r"C:\Users\110492\Github\MinRepo\DesignAIsystem\MiniProjekt\dataset"
    ground_truth  = r"C:\Users\110492\Github\MinRepo\DesignAIsystem\MiniProjekt\ground_truth_terrain.json"

    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(image_folder, "1.jpg")
    img = cv.imread(target)
    if img is not None:
        print(f"=== Detection on {os.path.basename(target)} ===")
        print_board(detect_board(img))

    print("\n=== Evaluation against ground truth ===")
    evaluate(image_folder, ground_truth)
