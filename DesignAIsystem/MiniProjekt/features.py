"""
Feature extraction for Kingdomino tiles.

Extracts a flattened HSV colour histogram from a 100x100 BGR tile image.
The vector is used as input to the terrain SVM.
"""

import cv2 as cv
import numpy as np

# Bins per HSV channel — coarse enough to generalise, fine enough to separate classes
H_BINS = 16
S_BINS = 8
V_BINS = 8
FEATURE_DIM = H_BINS + S_BINS + V_BINS   # 32 features


def extract_features(tile_bgr: np.ndarray) -> np.ndarray:
    """
    Return a 1-D float32 feature vector for one tile.

    Each channel histogram is normalised to sum to 1 so that
    partially-cropped tiles (edges) are still comparable.
    """
    hsv = cv.cvtColor(tile_bgr, cv.COLOR_BGR2HSV)

    h_hist = cv.calcHist([hsv], [0], None, [H_BINS], [0, 180]).flatten()
    s_hist = cv.calcHist([hsv], [1], None, [S_BINS], [0, 256]).flatten()
    v_hist = cv.calcHist([hsv], [2], None, [V_BINS], [0, 256]).flatten()

    # Normalise each channel independently
    def _norm(h):
        total = h.sum()
        return h / total if total > 0 else h

    features = np.concatenate([_norm(h_hist), _norm(s_hist), _norm(v_hist)])
    return features.astype(np.float32)
