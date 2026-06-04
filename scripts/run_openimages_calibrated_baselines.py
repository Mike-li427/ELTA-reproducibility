from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml

from run_openimages_heldout_calibration import (
    aggregate_scores,
    best_f1_threshold,
    build_splits,
    confidence_gate_scores,
    metrics_with_calibrated_threshold,
    split_retrieval_calibration_eval,
)
from run_openimages_margin_gate import load_cached_arrays
from run_openimages_pilot import knn_score_matrix


def read_selected_settings(path: Path) -> dict[str, dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        out[row["split"]] = {
            "residual_power": float(row["residual_power"]),
            "confidence_cutoff": float(row["confidence_cutoff"]),
            "confidence_temperature": float(row["confidence_temperature"]),
        }
    return out


def class_thresholds(calibration_labels: np.ndarray, calibration_scores: np.ndarray) -> np.ndarray:
    thresholds = []
    for j in range(calibration_scores.shape[1]):
        y = calibration_labels[:, j].astype(int)
        scores = calibration_scores[:, j]
        if len(np.unique(y)) < 2:
            thresholds.append(float(scores.max() + 1e-6))
        else:
            thresholds.append(best_f1_threshold(y, scores)["threshold"])
    return np.asarray(thresholds, dtype=np.float32)


def platt_calibrate_scores(
    calibration_labels: np.ndarray,
    calibration_scores: np.ndarray,
    eval_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    calibration_out = np.zeros_like(calibration_scores, dtype=np.float32)
    eval_out = np.zeros_like(eval_scores, dtype=np.float32)
    for j in range(calibration_scores.shape[1]):
        y = calibration_labels[:, j].astype(int)
        if len(np.unique(y)) < 2:
            fill = float(y.mean())
            calibration_out[:, j] = fill
            eval_out[:, j] = fill
            continue
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(calibration_scores[:, j].reshape(-1, 1), y)
        calibration_out[:, j] = model.predict_proba(calibration_scores[:, j].reshape(-1, 1))[:, 1]
        eval_out[:, j] = model.predict_proba(eval_scores[:, j].reshape(-1, 1))[:, 1]
    return calibration_out, eval_out


def isotonic_calibrate_scores(
    calibration_labels: np.ndarray,
    calibration_scores: np.ndarray,
    eval_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    calibration_out = np.zeros_like(calibration_scores, dtype=np.float32)
    eval_out = np.zeros_like(eval_scores, dtype=np.float32)
    for j in range(calibration_scores.shape[1]):
        y = calibration_labels[:, j].astype(int)
        if len(np.unique(y)) < 2:
            fill = float(y.mean())
            calibration_out[:, j] = fill
            eval_out[:, j] = fill
            continue
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(calibration_scores[:, j], y)
        calibration_out[:, j] = model.predict(calibration_scores[:, j])
        eval_out[:, j] = model.predict(eval_scores[:, j])
    return calibration_out, eval_out


def tecr_from_predictions(pred_any: np.ndarray, labels: np.ndarray, tail_idx: list[int], emerging_idx: list[int]) -> float | None:
    eligible = (labels[:, tail_idx].max(axis=1) > 0) & (labels[:, emerging_idx].max(axis=1) == 0)
    if eligible.sum() == 0:
        return None
    return float((pred_any & eligible).sum() / eligible.sum())


def aggregate_metrics_from_predictions(
    y_true: np.ndarray,
    aggregate_scores_values: np.ndarray,
    pred_any: np.ndarray,
    labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
) -> dict:
    if len(np.unique(y_true)) < 2:
        ap = None
        auroc = None
    else:
        ap = float(average_precision_score(y_true, aggregate_scores_values))
        auroc = float(roc_auc_score(y_true, aggregate_scores_values))
    tp = float(((pred_any == 1) & (y_true == 1)).sum())
    fp = float(((pred_any == 1) & (y_true == 0)).sum())
    fn = float(((pred_any == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "average_precision": ap,
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "best_f1": f1,
        "tecr": tecr_from_predictions(pred_any.astype(bool), labels, tail_idx, emerging_idx),
    }


def add_matrix_calibration_rows(
    rows: list[dict],
    method_prefix: str,
    calibration_labels: np.ndarray,
    eval_labels: np.ndarray,
    calibration_scores: np.ndarray,
    eval_scores: np.ndarray,
    y_cal: np.ndarray,
    y_eval: np.ndarray,
    full_eval_labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
    split_name: str,
    run_seed: int,
) -> None:
    cal_agg = aggregate_scores(calibration_scores)
    eval_agg = aggregate_scores(eval_scores)
    threshold = best_f1_threshold(y_cal, cal_agg)["threshold"]
    global_metrics = metrics_with_calibrated_threshold(
        y_eval,
        eval_agg,
        threshold,
        full_eval_labels,
        tail_idx,
        emerging_idx,
    )
    rows.append({**global_metrics, "method": f"{method_prefix}_global_threshold", "split": split_name, "seed": run_seed})

    thresholds = class_thresholds(calibration_labels, calibration_scores)
    pred = (eval_scores >= thresholds[None, :]).max(axis=1)
    class_metrics = aggregate_metrics_from_predictions(y_eval, eval_agg, pred, full_eval_labels, tail_idx, emerging_idx)
    rows.append({**class_metrics, "method": f"{method_prefix}_class_thresholds", "split": split_name, "seed": run_seed})


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
    baseline = next((row for row in out if row["method"] == "clip_knn_global_threshold"), None)
    if baseline:
        for row in out:
            row["ap_delta_pct"] = 0.0 if row is baseline else 100.0 * (row["average_precision_mean"] - baseline["average_precision_mean"]) / baseline["average_precision_mean"]
            row["f1_delta_pct"] = 0.0 if row is baseline else 100.0 * (row["best_f1_mean"] - baseline["best_f1_mean"]) / baseline["best_f1_mean"]
            row["tecr_reduction_pct"] = 0.0 if row is baseline else 100.0 * (baseline["tecr_mean"] - row["tecr_mean"]) / baseline["tecr_mean"]
    return sorted(out, key=lambda row: row["method"])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/openimages_10k_heldout_ultrastrict.yaml")
    parser.add_argument("--heldout-dir", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    heldout_dir = Path(args.heldout_dir)
    output_dir = Path(args.output_dir) if args.output_dir else heldout_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads((heldout_dir / "heldout_calibration_result.json").read_text(encoding="utf-8"))
    run_seed = int(result.get("seed", cfg["seed"]))
    selected = read_selected_settings(heldout_dir / "heldout_selected_settings.csv")

    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))

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
    for split in splits:
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
        known_idx = [i for i, name in enumerate(class_names) if name not in split["emerging_labels"]]
        setting = selected[split["name"]]

        calibration_known = calibration_logits[:, known_idx].max(axis=1)
        calibration_known = 1.0 / (1.0 + np.exp(-calibration_known))
        eval_known = eval_logits[:, known_idx].max(axis=1)
        eval_known = 1.0 / (1.0 + np.exp(-eval_known))

        calibration_knn = knn_score_matrix(retrieval_features, retrieval_labels, calibration_features, emerging_idx, cfg["knn"])
        eval_knn = knn_score_matrix(retrieval_features, retrieval_labels, eval_features, emerging_idx, cfg["knn"])
        calibration_held = confidence_gate_scores(
            calibration_knn,
            calibration_known,
            setting["residual_power"],
            setting["confidence_cutoff"],
            setting["confidence_temperature"],
        )
        eval_held = confidence_gate_scores(
            eval_knn,
            eval_known,
            setting["residual_power"],
            setting["confidence_cutoff"],
            setting["confidence_temperature"],
        )

        y_cal = calibration_labels[:, emerging_idx].max(axis=1).astype(int)
        y_eval = eval_labels[:, emerging_idx].max(axis=1).astype(int)

        calibration_clip = 1.0 / (1.0 + np.exp(-calibration_logits[:, emerging_idx]))
        eval_clip = 1.0 / (1.0 + np.exp(-eval_logits[:, emerging_idx]))
        add_matrix_calibration_rows(
            rows,
            "clip_zeroshot",
            calibration_labels[:, emerging_idx],
            eval_labels[:, emerging_idx],
            calibration_clip,
            eval_clip,
            y_cal,
            y_eval,
            eval_labels,
            tail_idx,
            emerging_idx,
            split["name"],
            run_seed,
        )

        cal_base_agg = aggregate_scores(calibration_knn)
        eval_base_agg = aggregate_scores(eval_knn)
        base_global_threshold = best_f1_threshold(y_cal, cal_base_agg)["threshold"]
        base_global = metrics_with_calibrated_threshold(y_eval, eval_base_agg, base_global_threshold, eval_labels, tail_idx, emerging_idx)
        rows.append({**base_global, "method": "clip_knn_global_threshold", "split": split["name"], "seed": run_seed})

        base_class_thresholds = class_thresholds(calibration_labels[:, emerging_idx], calibration_knn)
        base_class_pred = (eval_knn >= base_class_thresholds[None, :]).max(axis=1)
        base_class = aggregate_metrics_from_predictions(y_eval, eval_base_agg, base_class_pred, eval_labels, tail_idx, emerging_idx)
        rows.append({**base_class, "method": "clip_knn_class_thresholds", "split": split["name"], "seed": run_seed})

        platt_cal, platt_eval = platt_calibrate_scores(calibration_labels[:, emerging_idx], calibration_knn, eval_knn)
        add_matrix_calibration_rows(
            rows,
            "clip_knn_platt",
            calibration_labels[:, emerging_idx],
            eval_labels[:, emerging_idx],
            platt_cal,
            platt_eval,
            y_cal,
            y_eval,
            eval_labels,
            tail_idx,
            emerging_idx,
            split["name"],
            run_seed,
        )

        isotonic_cal, isotonic_eval = isotonic_calibrate_scores(calibration_labels[:, emerging_idx], calibration_knn, eval_knn)
        add_matrix_calibration_rows(
            rows,
            "clip_knn_isotonic",
            calibration_labels[:, emerging_idx],
            eval_labels[:, emerging_idx],
            isotonic_cal,
            isotonic_eval,
            y_cal,
            y_eval,
            eval_labels,
            tail_idx,
            emerging_idx,
            split["name"],
            run_seed,
        )

        cal_held_agg = aggregate_scores(calibration_held)
        eval_held_agg = aggregate_scores(eval_held)
        held_global_threshold = best_f1_threshold(y_cal, cal_held_agg)["threshold"]
        held_global = metrics_with_calibrated_threshold(y_eval, eval_held_agg, held_global_threshold, eval_labels, tail_idx, emerging_idx)
        rows.append({**held_global, "method": "heldout_gate_global_threshold", "split": split["name"], "seed": run_seed})

        held_class_thresholds = class_thresholds(calibration_labels[:, emerging_idx], calibration_held)
        held_class_pred = (eval_held >= held_class_thresholds[None, :]).max(axis=1)
        held_class = aggregate_metrics_from_predictions(y_eval, eval_held_agg, held_class_pred, eval_labels, tail_idx, emerging_idx)
        rows.append({**held_class, "method": "heldout_gate_class_thresholds", "split": split["name"], "seed": run_seed})

    summary = summarize(rows)
    row_fields = [
        "method", "split", "seed", "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    summary_fields = [
        "method", "num_splits", "average_precision_mean", "average_precision_std",
        "auroc_mean", "auroc_std", "best_f1_mean", "best_f1_std", "tecr_mean", "tecr_std",
        "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct",
    ]
    write_csv(output_dir / "calibrated_baseline_rows.csv", rows, row_fields)
    write_csv(output_dir / "calibrated_baseline_summary.csv", summary, summary_fields)

    report = [
        "# Calibrated Baseline Comparison",
        "",
        "Date: 2026-05-22",
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
            f"{row.get('ap_delta_pct', 0.0):.2f}% | {row.get('f1_delta_pct', 0.0):.2f}% | "
            f"{row.get('tecr_reduction_pct', 0.0):.1f}% |"
        )
    report.extend([
        "",
        "Notes:",
        "",
        "- Class-threshold baselines choose one calibration threshold per emerging label.",
        "- Platt and isotonic baselines fit one post-hoc score calibrator per emerging label on the same calibration split.",
        "- These baselines test whether ordinary per-class score calibration explains the held-out gate's gains.",
        "",
    ])
    (output_dir / "calibrated_baseline_report.md").write_text("\n".join(report), encoding="utf-8")
    result_out = {
        "status": "calibrated_baselines_complete",
        "seed": run_seed,
        "heldout_dir": str(heldout_dir),
        "summary": summary,
    }
    (output_dir / "calibrated_baseline_result.json").write_text(json.dumps(result_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result_out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
