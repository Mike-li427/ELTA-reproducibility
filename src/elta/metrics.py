from __future__ import annotations

import numpy as np


def tecr_at_threshold(
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
) -> float | None:
    """Tail-Emerging Confusion Rate for image-level emerging scores.

    TECR is the false emerging-label activation rate on images that contain at
    least one tail-known label and no emerging-label positive.
    """
    pred_any = scores >= threshold
    return tecr_from_predictions(pred_any, labels, tail_idx, emerging_idx)


def tecr_from_predictions(
    pred_any: np.ndarray,
    labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
) -> float | None:
    """TECR from binary image-level emerging predictions."""
    risk_population = (
        (labels[:, tail_idx].max(axis=1) > 0)
        & (labels[:, emerging_idx].max(axis=1) == 0)
    )
    if risk_population.sum() == 0:
        return None
    return float((pred_any.astype(bool) & risk_population).sum() / risk_population.sum())
