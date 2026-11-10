"""
Kingdomino point-beregner.

Pipeline for ét bræt-billede:
  1. Opdel 500x500 billede i 5x5 tiles
  2. SVM forudsiger terrain-type for hvert felt
  3. Template matching tæller kroner per felt
  4. BFS finder sammenhængende regioner af samme terrain
  5. Point = sum over alle regioner: (antal felter × antal kroner i regionen)

Usage:
  python score.py <sti-til-billede>

Eksempel:
  python score.py dataset/9.jpg
"""

import os
import pickle
import sys
from collections import deque

import cv2 as cv
import numpy as np

from features import extract_features
from matching import count_crowns, rotate_template

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = r"C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt\terrain_svm.pkl"
TEMPLATE_PATH = r"C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt\krone_transparent.png"
TILE_SIZE     = 100
GRID_SIZE     = 5
# ───────────────────────────────────────────────────────────────────────────────


def load_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["pipeline"], data["label_encoder"]


def predict_terrain(img, pipeline, le) -> list[list[str]]:
    """Return a 5x5 grid of terrain labels (row-major)."""
    grid = []
    for row in range(GRID_SIZE):
        grid_row = []
        for col in range(GRID_SIZE):
            tile = img[row * TILE_SIZE:(row + 1) * TILE_SIZE,
                       col * TILE_SIZE:(col + 1) * TILE_SIZE]
            feat    = extract_features(tile).reshape(1, -1)
            enc_idx = pipeline.predict(feat)[0]
            grid_row.append(le.classes_[enc_idx])
        grid.append(grid_row)
    return grid


def count_all_crowns(img, template) -> list[list[int]]:
    """Return a 5x5 grid of crown counts per tile."""
    grid = []
    for row in range(GRID_SIZE):
        grid_row = []
        for col in range(GRID_SIZE):
            tile   = img[row * TILE_SIZE:(row + 1) * TILE_SIZE,
                         col * TILE_SIZE:(col + 1) * TILE_SIZE]
            crowns = count_crowns(tile, template)
            grid_row.append(crowns)
        grid.append(grid_row)
    return grid


def calculate_score(terrain_grid: list[list[str]],
                    crown_grid:   list[list[int]]) -> tuple[int, list[dict]]:
    """
    BFS over the 5x5 grid.

    Returns (total_score, regions) where each region is:
      { terrain, size, crowns, points }
    Castle/unknown tiles score 0 and are excluded from regions.
    """
    visited = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
    total_score = 0
    regions = []

    for start_row in range(GRID_SIZE):
        for start_col in range(GRID_SIZE):
            if visited[start_row][start_col]:
                continue

            terrain = terrain_grid[start_row][start_col]
            if terrain in ("castle", "unknown"):
                visited[start_row][start_col] = True
                continue

            # BFS
            queue         = deque([(start_row, start_col)])
            visited[start_row][start_col] = True
            region_size   = 0
            region_crowns = 0

            while queue:
                r, c = queue.popleft()
                region_size   += 1
                region_crowns += crown_grid[r][c]

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        if not visited[nr][nc] and terrain_grid[nr][nc] == terrain:
                            visited[nr][nc] = True
                            queue.append((nr, nc))

            points = region_size * region_crowns
            total_score += points
            if region_crowns > 0:
                regions.append({
                    "terrain": terrain,
                    "size":    region_size,
                    "crowns":  region_crowns,
                    "points":  points,
                })

    regions.sort(key=lambda r: r["points"], reverse=True)
    return total_score, regions


def print_board(terrain_grid, crown_grid, total_score, regions):
    """Print terrain grid and per-region point breakdown."""
    print("\nTerrain grid (med kroner):")
    for row in range(GRID_SIZE):
        line = ""
        for col in range(GRID_SIZE):
            t = terrain_grid[row][col][:5].ljust(5)
            c = crown_grid[row][col]
            crown_str = f"♛{c}" if c > 0 else "  "
            line += f" {t}{crown_str} |"
        print(line)

    print(f"\n{'Terrain':<10} {'Felter':>7} {'Kroner':>7} {'Point':>7}")
    print(f"  {'-'*33}")
    for r in regions:
        print(f"  {r['terrain']:<10} {r['size']:>5}  ×  {r['crowns']:>3}  =  {r['points']:>4}")

    print(f"  {'─'*33}")
    print(f"  {'TOTAL':<10} {'':>7} {'':>7} {total_score:>7}")


def score_image(image_path: str):
    img = cv.imread(image_path)
    if img is None:
        print(f"Fejl: Kan ikke læse billede: {image_path}")
        return

    template = cv.imread(TEMPLATE_PATH, cv.IMREAD_UNCHANGED)
    if template is None:
        print(f"Fejl: Kan ikke læse template: {TEMPLATE_PATH}")
        return

    pipeline, le = load_model()

    terrain_grid        = predict_terrain(img, pipeline, le)
    crown_grid          = count_all_crowns(img, template)
    total_score, regions = calculate_score(terrain_grid, crown_grid)

    print_board(terrain_grid, crown_grid, total_score, regions)
    return total_score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Brug: python score.py <sti-til-billede>")
        print("Eksempel: python score.py dataset/9.jpg")
        sys.exit(1)

    score_image(sys.argv[1])
