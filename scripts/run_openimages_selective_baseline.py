from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import yaml

from run_openimages_heldout_calibration import (
    aggregate_scores,
    best_f1_threshold,
    build_splits,
    metrics_with_calibrated_threshold,
    pct_change,
    pct_reduction,
    split_retrieval_calibration_eval,
)
from run_openimages_margin_gate import load_cached_arrays
from run_openimages_pilot import knn_score_matrix


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_entropy(scores: np.ndarray) -> np.ndarray:
    probs = np.maximum(scores, 0.0)
    denom = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, np.maximum(denom, 1e-12))
    ent = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    return ent / max(np.log(scores.shape[1]), 1e-12)


def selective_reject_scores(scores: np.ndarray, entropy: np.ndarray, cutoff: float) -> np.ndarray:
    out = np.array(scores, copy=True)
    out[entropy >= cutoff] = 0.0
    return out


def select_cutoff(
    rows: list[dict],
    baseline: dict,
    ap_tolerance: float,
    f1_tolerance: float,
) -> dict:
    candidates = [
        row for row in rows
        if row["average_precision"] is not None
        and row["best_f1"] is not None
        and row["tecr"] is not None
        and row["average_precision"] >= baseline["average_precision"] * (1.0 - ap_tolerance)
        and row["best_f1"] >= baseline["best_f1"] * (1.0 - f1_tolerance)
    ]
    if not candidates:
        return {"selection_status": "fallback_baseline", "entropy_cutoff": 1.01}
    best = min(candidates, key=lambda row: (row["tecr"], -row["best_f1"], -row["average_precision"]))
    best = dict(best)
    best["selection_status"] = "selected_under_constraints"
    return best


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    out = []
    for method, group in grouped.items():
        item = {"method": method, "num_splits": len(group)}
        for key in ["average_precision", "auroc", "best_f1", "tecr"]:
            vals = [row[key] for row in group if row[key] is not None]
            item[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    return sorted(out, key=lambda row: row["method"])


def run_one(config_path: Path, output_dir: Path, seed_override: int | None = None) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_seed = int(seed_override) if seed_override is not None else int(cfg["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))
    ap_tolerance = float(calibration_cfg.get("ap_tolerance", 0.0025))
    f1_tolerance = float(calibration_cfg.get("f1_tolerance", 0.0025))

    features, labels, _text_features, class_names = load_cached_arrays(cfg, Path(cfg["output_dir"]))
    (
        retrieval_features,
        retrieval_labels,
        calibration_features,
        calibration_labels,
        eval_features,
        eval_labels,
    ) = split_retrieval_calibration_eval(
        features,
        labels,
        run_seed,
        retrieval_fraction,
        calibration_fraction,
    )
    splits = build_splits(class_names, retrieval_labels, calibration_labels, cfg["protocol"], run_seed)

    eval_rows: list[dict] = []
    grid_rows: list[dict] = []
    selected_rows: list[dict] = []
    cutoffs = np.linspace(0.0, 1.0, 51).tolist() + [1.01]

    for split in splits:
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]

        calibration_knn = knn_score_matrix(retrieval_features, retrieval_labels, calibration_features, emerging_idx, cfg["knn"])
        eval_knn = knn_score_matrix(retrieval_features, retrieval_labels, eval_features, emerging_idx, cfg["knn"])

        y_cal = calibration_labels[:, emerging_idx].max(axis=1).astype(int)
        y_eval = eval_labels[:, emerging_idx].max(axis=1).astype(int)

        cal_base_scores = aggregate_scores(calibration_knn)
        eval_base_scores = aggregate_scores(eval_knn)
        cal_threshold = best_f1_threshold(y_cal, cal_base_scores)
        cal_baseline = metrics_with_calibrated_threshold(
            y_cal, cal_base_scores, cal_threshold["threshold"], calibration_labels, tail_idx, emerging_idx
        )
        eval_baseline = metrics_with_calibrated_threshold(
            y_eval, eval_base_scores, cal_threshold["threshold"], eval_labels, tail_idx, emerging_idx
        )
        eval_baseline.update({
            "method": "clip_knn",
            "split": split["name"],
            "selection_status": "baseline",
            "entropy_cutoff": None,
        })
        eval_rows.append(eval_baseline)

        cal_entropy = normalized_entropy(calibration_knn)
        eval_entropy = normalized_entropy(eval_knn)
        cal_candidates = []
        eval_candidates = {}
        for cutoff in cutoffs:
            cutoff = float(cutoff)
            cal_scores = selective_reject_scores(cal_base_scores, cal_entropy, cutoff)
            threshold = best_f1_threshold(y_cal, cal_scores)
            cal_metrics = metrics_with_calibrated_threshold(
                y_cal, cal_scores, threshold["threshold"], calibration_labels, tail_idx, emerging_idx
            )
            cal_row = {
                **cal_metrics,
                "method": "entropy_reject",
                "split": split["name"],
                "entropy_cutoff": cutoff,
            }
            cal_candidates.append(cal_row)
            grid_rows.append(cal_row)

            eval_scores = selective_reject_scores(eval_base_scores, eval_entropy, cutoff)
            eval_metrics = metrics_with_calibrated_threshold(
                y_eval, eval_scores, threshold["threshold"], eval_labels, tail_idx, emerging_idx
            )
            eval_candidates[cutoff] = {
                **eval_metrics,
                "method": "entropy_reject",
                "split": split["name"],
                "entropy_cutoff": cutoff,
            }

        selected = select_cutoff(cal_candidates, cal_baseline, ap_tolerance, f1_tolerance)
        cutoff = float(selected["entropy_cutoff"])
        eval_selected = eval_candidates.get(cutoff, dict(eval_baseline))
        eval_selected["selection_status"] = selected["selection_status"]
        eval_rows.append(eval_selected)
        selected_rows.append({
            "split": split["name"],
            "selection_status": selected["selection_status"],
            "entropy_cutoff": cutoff,
            "calibration_ap": selected.get("average_precision"),
            "calibration_f1": selected.get("best_f1"),
            "calibration_tecr": selected.get("tecr"),
            "baseline_calibration_ap": cal_baseline["average_precision"],
            "baseline_calibration_f1": cal_baseline["best_f1"],
            "baseline_calibration_tecr": cal_baseline["tecr"],
        })

    summary = summarize(eval_rows)
    baseline_summary = next(row for row in summary if row["method"] == "clip_knn")
    for row in summary:
        if row["method"] == "clip_knn":
            row["ap_delta_pct"] = 0.0
            row["f1_delta_pct"] = 0.0
            row["tecr_reduction_pct"] = 0.0
        else:
            row["ap_delta_pct"] = pct_change(baseline_summary["average_precision_mean"], row["average_precision_mean"])
            row["f1_delta_pct"] = pct_change(baseline_summary["best_f1_mean"], row["best_f1_mean"])
            row["tecr_reduction_pct"] = pct_reduction(baseline_summary["tecr_mean"], row["tecr_mean"])

    eval_fields = [
        "method", "split", "selection_status", "entropy_cutoff",
        "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    summary_fields = [
        "method", "num_splits", "average_precision_mean", "average_precision_std",
        "auroc_mean", "auroc_std", "best_f1_mean", "best_f1_std", "tecr_mean", "tecr_std",
        "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct",
    ]
    selection_fields = [
        "split", "selection_status", "entropy_cutoff",
        "calibration_ap", "calibration_f1", "calibration_tecr",
        "baseline_calibration_ap", "baseline_calibration_f1", "baseline_calibration_tecr",
    ]
    grid_fields = [
        "method", "split", "entropy_cutoff",
        "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    write_csv(output_dir / "selective_eval_rows.csv", eval_rows, eval_fields)
    write_csv(output_dir / "selective_summary.csv", summary, summary_fields)
    write_csv(output_dir / "selective_selected_settings.csv", selected_rows, selection_fields)
    write_csv(output_dir / "selective_calibration_grid.csv", grid_rows, grid_fields)

    report = [
        "# Entropy-Based Selective Prediction Baseline",
        "",
        f"Config: `{config_path}`",
        f"Seed: `{run_seed}`",
        "",
        "The baseline uses only the emerging-label score distribution. Images with normalized score entropy above a calibration-selected cutoff are rejected by setting their image-level emerging score to zero. The cutoff is selected only on the calibration split under the same AP/F1 preservation constraints as the confidence gate.",
        "",
        "| Method | AP | F1 | TECR | AP delta | F1 delta | TECR reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['method']} | {row['average_precision_mean']:.4f} | {row['best_f1_mean']:.4f} | "
            f"{row['tecr_mean']:.4f} | {row['ap_delta_pct']:.2f}% | {row['f1_delta_pct']:.2f}% | "
            f"{row['tecr_reduction_pct']:.1f}% |"
        )
    (output_dir / "selective_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "status": "selective_baseline_complete",
        "config": str(config_path),
        "seed": run_seed,
        "output_dir": str(output_dir),
        "summary": summary,
    }
    (output_dir / "selective_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-override", type=int)
    args = parser.parse_args()
    run_one(Path(args.config), Path(args.output_dir), args.seed_override)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
