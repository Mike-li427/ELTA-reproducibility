from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def confidence_gate_scores(
    scores: np.ndarray,
    known_confidence: np.ndarray,
    residual_power: float,
    confidence_cutoff: float,
    confidence_temperature: float,
) -> np.ndarray:
    """Apply the held-out confidence gate to emerging-label scores.

    Args:
        scores: Matrix of emerging-label scores with shape [num_images, num_emerging].
        known_confidence: Known-label confidence for each image, shape [num_images].
        residual_power: Power p in residual term max(0, 1-a_i)^p.
        confidence_cutoff: Cutoff b for the emerging-score confidence term.
        confidence_temperature: Temperature tau for the emerging-score confidence term.

    Returns:
        Gated emerging-label scores with the same shape as ``scores``.
    """
    residual_gate = np.maximum(0.0, 1.0 - known_confidence) ** residual_power
    confidence = sigmoid((scores - confidence_cutoff) / max(confidence_temperature, 1e-6))
    gate = confidence + (1.0 - confidence) * residual_gate[:, None]
    return scores * gate
