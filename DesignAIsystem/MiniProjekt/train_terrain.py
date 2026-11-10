"""
Train a terrain-classification SVM on the annotated Kingdomino boards.

Input:
  - ground_truth_terrain.json   (tile_id → terrain label)
  - dataset/                    (board images, 500x500, named <id>.jpg)

Output:
  - terrain_svm.pkl             (trained sklearn pipeline: scaler + SVC)

Usage:
  python train_terrain.py
"""

import json
import os
import pickle

import cv2 as cv
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from features import extract_features

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_FOLDER   = r"C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt\dataset"
GROUND_TRUTH     = r"C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt\ground_truth_terrain.json"
MODEL_OUTPUT     = r"C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt\terrain_svm.pkl"
TILE_SIZE        = 100
GRID_SIZE        = 5
# ───────────────────────────────────────────────────────────────────────────────


def load_dataset():
    """Load all labelled tiles and return (X, y, label_names)."""
    with open(GROUND_TRUTH, encoding="utf-8") as f:
        labels = json.load(f)

    X, y = [], []
    missing = 0

    # Cache board images so we don't reload the same file repeatedly
    cached_boards = {}

    for tile_id, terrain in labels.items():
        # tile_id format: <image_id>_x<col>_y<row>
        parts = tile_id.rsplit("_x", 1)
        image_id = parts[0]
        coords   = parts[1]          # "col_yrow"
        col_str, row_str = coords.split("_y")
        x, y_pos = int(col_str), int(row_str)

        if image_id not in cached_boards:
            path = os.path.join(DATASET_FOLDER, f"{image_id}.jpg")
            img  = cv.imread(path)
            if img is None:
                path = os.path.join(DATASET_FOLDER, f"{image_id}.png")
                img  = cv.imread(path)
            cached_boards[image_id] = img

        img = cached_boards[image_id]
        if img is None:
            missing += 1
            continue

        tile = img[y_pos * TILE_SIZE:(y_pos + 1) * TILE_SIZE,
                   x     * TILE_SIZE:(x      + 1) * TILE_SIZE]
        if tile.size == 0:
            missing += 1
            continue

        X.append(extract_features(tile))
        y.append(terrain)

    if missing:
        print(f"  Advarsel: {missing} tiles kunne ikke loades")

    X = np.array(X, dtype=np.float32)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"Dataset: {len(X)} tiles, {len(le.classes_)} klasser: {list(le.classes_)}")
    return X, y_enc, le


def grid_search(X, y_enc):
    """Find bedste C og gamma via 5-fold GridSearchCV."""
    print("\nGrid search (dette tager lidt tid) …")
    param_grid = {
        "svm__C":     [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.01, 0.1],
    }
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", probability=True)),
    ])
    gs = GridSearchCV(
        pipeline, param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy", n_jobs=-1, verbose=1,
    )
    gs.fit(X, y_enc)
    print(f"\n  Bedste parametre : {gs.best_params_}")
    print(f"  Bedste CV score  : {gs.best_score_*100:.2f}%")
    return gs.best_params_["svm__C"], gs.best_params_["svm__gamma"]


def train(X, y_enc, le):
    """Find bedste hyperparametre, evaluer med CV og gem model."""
    best_C, best_gamma = grid_search(X, y_enc)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=best_C, gamma=best_gamma, probability=True)),
    ])

    print(f"\nKrydsvalidering (5-fold) med C={best_C}, gamma={best_gamma} …")
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv   = cross_val_predict(pipeline, X, y_enc, cv=cv_splitter)

    print("\nClassification report (CV):")
    print(classification_report(y_enc, y_pred_cv,
                                target_names=le.classes_, zero_division=0))

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_enc, y_pred_cv,
        display_labels=le.classes_,
        ax=ax, colorbar=False,
        xticks_rotation="vertical",
    )
    ax.set_title("Confusion matrix — 5-fold CV")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    print("Confusion matrix gemt som confusion_matrix.png")
    plt.show()

    # Fit on all data for the final saved model
    print("\nTræner på hele datasættet …")
    pipeline.fit(X, y_enc)
    return pipeline


def save_model(pipeline, le):
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump({"pipeline": pipeline, "label_encoder": le}, f)
    print(f"Model gemt: {MODEL_OUTPUT}")


if __name__ == "__main__":
    X, y_enc, le = load_dataset()
    pipeline      = train(X, y_enc, le)
    save_model(pipeline, le)
