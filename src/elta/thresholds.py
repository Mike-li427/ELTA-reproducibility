from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Select the threshold with the best F1 on calibration data."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    idx = int(np.nanargmax(f1))
    threshold = float(thresholds[idx]) if idx < len(thresholds) else float(scores.max() + 1e-6)
    return {
        "threshold": threshold,
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(f1[idx]),
    }
