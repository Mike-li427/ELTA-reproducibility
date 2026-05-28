from __future__ import annotations


def select_gate_setting(
    calibration_rows: list[dict],
    baseline: dict,
    ap_tolerance: float,
    f1_tolerance: float,
) -> dict:
    """Select the gate setting with lowest calibration TECR under AP/F1 constraints."""
    candidates = [
        row
        for row in calibration_rows
        if row["method"] == "elta_confidence"
        and row["average_precision"] is not None
        and row["best_f1"] is not None
        and row["tecr"] is not None
        and row["average_precision"] >= baseline["average_precision"] * (1.0 - ap_tolerance)
        and row["best_f1"] >= baseline["best_f1"] * (1.0 - f1_tolerance)
    ]
    if not candidates:
        return {
            "method": "clip_knn",
            "residual_power": 0.0,
            "confidence_cutoff": 0.0,
            "confidence_temperature": 1.0,
            "selection_status": "fallback_baseline",
        }
    best = min(candidates, key=lambda row: (row["tecr"], -row["best_f1"], -row["average_precision"]))
    best = dict(best)
    best["selection_status"] = "selected_under_constraints"
    return best
