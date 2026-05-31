from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from run_openimages_heldout_calibration import (
    aggregate_scores,
    build_splits,
    confidence_gate_scores,
    metrics_with_calibrated_threshold,
    pct_change,
    pct_reduction,
    select_setting,
    split_retrieval_calibration_eval,
)
from run_openimages_margin_gate import load_cached_arrays
from run_openimages_pilot import knn_score_matrix


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import precision_recall_curve

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


def pure_residual_scores(knn_scores: np.ndarray, known_explanation: np.ndarray, residual_power: float) -> np.ndarray:
    residual_gate = np.maximum(0.0, 1.0 - known_explanation) ** float(residual_power)
    return knn_scores * residual_gate[:, None]


def summarize(rows: list[dict]) -> list[dict]:
    grouped = {}
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
    baseline = next((row for row in out if row["method"] == "clip_knn"), None)
    if baseline:
        for row in out:
            if row["method"] == "clip_knn":
                row["ap_delta_pct"] = 0.0
                row["f1_delta_pct"] = 0.0
                row["tecr_reduction_pct"] = 0.0
            else:
                row["ap_delta_pct"] = pct_change(baseline["average_precision_mean"], row["average_precision_mean"])
                row["f1_delta_pct"] = pct_change(baseline["best_f1_mean"], row["best_f1_mean"])
                row["tecr_reduction_pct"] = pct_reduction(baseline["tecr_mean"], row["tecr_mean"])
    order = {
        "clip_knn": 0,
        "pure_residual_heldout": 1,
        "full_gate_tol_0.00pct": 2,
        "full_gate_tol_0.25pct": 3,
        "full_gate_tol_0.50pct": 4,
        "full_gate_tol_1.00pct": 5,
        "full_gate_eval_oracle": 6,
    }
    return sorted(out, key=lambda row: (order.get(row["method"], 99), row["method"]))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/openimages_10k_heldout_ultrastrict.yaml")
    parser.add_argument("--output-dir", default="outputs/openimages_10k_gate_ablation")
    parser.add_argument("--seed-override", type=int)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_seed = int(args.seed_override) if args.seed_override is not None else int(cfg["seed"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))
    residual_powers = [float(x) for x in calibration_cfg.get("residual_powers", [0.0, 0.1, 0.2, 0.3, 0.4])]
    cutoffs = [float(x) for x in calibration_cfg.get("confidence_cutoffs", [0.2, 0.3, 0.4, 0.5, 0.6])]
    temperatures = [float(x) for x in calibration_cfg.get("confidence_temperatures", [0.01, 0.02, 0.05, 0.1])]
    tolerance_grid = [0.0, 0.0025, 0.005, 0.01]

    features, labels, text_features, class_names = load_cached_arrays(cfg, Path(cfg["output_dir"]))
    (
        retrieval_features,
        retrieval_labels,
        calibration_features,
        calibration_labels,
        eval_features,
        eval_labels,
    ) = split_retrieval_calibration_eval(features, labels, run_seed, retrieval_fraction, calibration_fraction)
    splits = build_splits(class_names, retrieval_labels, calibration_labels, cfg["protocol"], run_seed)

    retrieval_logits_raw = retrieval_features @ text_features.T
    logit_mean = retrieval_logits_raw.mean(axis=0, keepdims=True)
    logit_std = retrieval_logits_raw.std(axis=0, keepdims=True) + 1e-8
    calibration_logits = (calibration_features @ text_features.T - logit_mean) / logit_std
    eval_logits = (eval_features @ text_features.T - logit_mean) / logit_std

    rows = []
    selected_rows = []
    for split in splits:
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
        known_idx = [i for i, name in enumerate(class_names) if name not in split["emerging_labels"]]

        calibration_known = 1.0 / (1.0 + np.exp(-calibration_logits[:, known_idx].max(axis=1)))
        eval_known = 1.0 / (1.0 + np.exp(-eval_logits[:, known_idx].max(axis=1)))
        calibration_knn = knn_score_matrix(retrieval_features, retrieval_labels, calibration_features, emerging_idx, cfg["knn"])
        eval_knn = knn_score_matrix(retrieval_features, retrieval_labels, eval_features, emerging_idx, cfg["knn"])

        y_cal = calibration_labels[:, emerging_idx].max(axis=1).astype(int)
        y_eval = eval_labels[:, emerging_idx].max(axis=1).astype(int)

        cal_base_scores = aggregate_scores(calibration_knn)
        eval_base_scores = aggregate_scores(eval_knn)
        base_threshold = best_f1_threshold(y_cal, cal_base_scores)["threshold"]
        cal_baseline = metrics_with_calibrated_threshold(y_cal, cal_base_scores, base_threshold, calibration_labels, tail_idx, emerging_idx)
        eval_baseline = metrics_with_calibrated_threshold(y_eval, eval_base_scores, base_threshold, eval_labels, tail_idx, emerging_idx)
        rows.append({**eval_baseline, "method": "clip_knn", "split": split["name"], "selection_status": "baseline"})

        residual_cal_rows = []
        residual_eval_rows = []
        full_cal_rows = []
        full_eval_rows = []
        for residual_power in residual_powers:
            cal_scores = aggregate_scores(pure_residual_scores(calibration_knn, calibration_known, residual_power))
            eval_scores = aggregate_scores(pure_residual_scores(eval_knn, eval_known, residual_power))
            threshold = best_f1_threshold(y_cal, cal_scores)["threshold"]
            cal_metrics = metrics_with_calibrated_threshold(y_cal, cal_scores, threshold, calibration_labels, tail_idx, emerging_idx)
            eval_metrics = metrics_with_calibrated_threshold(y_eval, eval_scores, threshold, eval_labels, tail_idx, emerging_idx)
            residual_cal_rows.append({
                **cal_metrics,
                "method": "elta_confidence",
                "split": split["name"],
                "residual_power": residual_power,
                "confidence_cutoff": -1.0,
                "confidence_temperature": -1.0,
            })
            residual_eval_rows.append({
                **eval_metrics,
                "method": "pure_residual_pool",
                "split": split["name"],
                "residual_power": residual_power,
                "confidence_cutoff": -1.0,
                "confidence_temperature": -1.0,
            })
            for cutoff in cutoffs:
                for temperature in temperatures:
                    cal_scores = aggregate_scores(confidence_gate_scores(calibration_knn, calibration_known, residual_power, cutoff, temperature))
                    eval_scores = aggregate_scores(confidence_gate_scores(eval_knn, eval_known, residual_power, cutoff, temperature))
                    threshold = best_f1_threshold(y_cal, cal_scores)["threshold"]
                    cal_metrics = metrics_with_calibrated_threshold(y_cal, cal_scores, threshold, calibration_labels, tail_idx, emerging_idx)
                    eval_metrics = metrics_with_calibrated_threshold(y_eval, eval_scores, threshold, eval_labels, tail_idx, emerging_idx)
                    full_cal_rows.append({
                        **cal_metrics,
                        "method": "elta_confidence",
                        "split": split["name"],
                        "residual_power": residual_power,
                        "confidence_cutoff": cutoff,
                        "confidence_temperature": temperature,
                    })
                    full_eval_rows.append({
                        **eval_metrics,
                        "method": "full_gate_pool",
                        "split": split["name"],
                        "residual_power": residual_power,
                        "confidence_cutoff": cutoff,
                        "confidence_temperature": temperature,
                    })

        selected_residual = select_setting(residual_cal_rows, cal_baseline, 0.0025, 0.0025)
        if selected_residual["method"] == "clip_knn":
            residual_eval = dict(eval_baseline)
        else:
            residual_eval = next(
                row for row in residual_eval_rows
                if row["residual_power"] == selected_residual["residual_power"]
            )
        rows.append({**residual_eval, "method": "pure_residual_heldout", "split": split["name"], "selection_status": selected_residual["selection_status"]})
        selected_rows.append({
            "split": split["name"],
            "ablation": "pure_residual_heldout",
            "selection_status": selected_residual["selection_status"],
            "residual_power": selected_residual["residual_power"],
            "confidence_cutoff": selected_residual["confidence_cutoff"],
            "confidence_temperature": selected_residual["confidence_temperature"],
        })

        for tolerance in tolerance_grid:
            selected = select_setting(full_cal_rows, cal_baseline, tolerance, tolerance)
            if selected["method"] == "clip_knn":
                selected_eval = dict(eval_baseline)
            else:
                selected_eval = next(
                    row for row in full_eval_rows
                    if row["residual_power"] == selected["residual_power"]
                    and row["confidence_cutoff"] == selected["confidence_cutoff"]
                    and row["confidence_temperature"] == selected["confidence_temperature"]
                )
            method = f"full_gate_tol_{tolerance * 100:.2f}pct"
            rows.append({**selected_eval, "method": method, "split": split["name"], "selection_status": selected["selection_status"]})
            selected_rows.append({
                "split": split["name"],
                "ablation": method,
                "selection_status": selected["selection_status"],
                "residual_power": selected["residual_power"],
                "confidence_cutoff": selected["confidence_cutoff"],
                "confidence_temperature": selected["confidence_temperature"],
            })

        eval_candidates = [
            row for row in full_eval_rows
            if row["average_precision"] is not None
            and row["best_f1"] is not None
            and row["tecr"] is not None
            and row["average_precision"] >= eval_baseline["average_precision"] * (1.0 - 0.0025)
            and row["best_f1"] >= eval_baseline["best_f1"] * (1.0 - 0.0025)
        ]
        if eval_candidates:
            oracle = min(eval_candidates, key=lambda row: (row["tecr"], -row["best_f1"], -row["average_precision"]))
            rows.append({**oracle, "method": "full_gate_eval_oracle", "split": split["name"], "selection_status": "eval_oracle_upper_bound"})

    summary = summarize(rows)
    row_fields = [
        "method", "split", "selection_status", "residual_power", "confidence_cutoff", "confidence_temperature",
        "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    summary_fields = [
        "method", "num_splits", "average_precision_mean", "average_precision_std",
        "auroc_mean", "auroc_std", "best_f1_mean", "best_f1_std", "tecr_mean", "tecr_std",
        "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct",
    ]
    selected_fields = [
        "split", "ablation", "selection_status", "residual_power", "confidence_cutoff", "confidence_temperature",
    ]
    write_csv(output_dir / "gate_ablation_rows.csv", rows, row_fields)
    write_csv(output_dir / "gate_ablation_summary.csv", summary, summary_fields)
    write_csv(output_dir / "gate_ablation_selected_settings.csv", selected_rows, selected_fields)

    report = [
        "# Open Images Gate Ablation",
        "",
        "Date: 2026-05-23",
        "",
        f"Seed: `{run_seed}`.",
        "",
        "| Method | AP | F1 | TECR | AP delta | F1 delta | TECR reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['method']} | {row['average_precision_mean']:.4f} | "
            f"{row['best_f1_mean']:.4f} | {row['tecr_mean']:.4f} | "
            f"{row['ap_delta_pct']:.2f}% | {row['f1_delta_pct']:.2f}% | "
            f"{row['tecr_reduction_pct']:.1f}% |"
        )
    report.extend([
        "",
        "Notes:",
        "",
        "- `pure_residual_heldout` removes score-confidence preservation and applies only the known-label residual gate.",
        "- `full_gate_tol_*` selects full gate parameters on calibration under the listed AP/F1 loss tolerance.",
        "- `full_gate_eval_oracle` selects on evaluation and is an upper bound, not a fair method.",
        "",
    ])
    (output_dir / "gate_ablation_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "status": "gate_ablation_complete",
        "seed": run_seed,
        "output_dir": str(output_dir),
        "summary": summary,
    }
    (output_dir / "gate_ablation_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
