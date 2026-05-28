from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import yaml

from run_openimages_margin_gate import load_cached_arrays, sigmoid
from run_openimages_pilot import knn_score_matrix


def split_retrieval_calibration_eval(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    retrieval_fraction: float,
    calibration_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    n_retrieval = int(round(len(indices) * retrieval_fraction))
    n_calibration = int(round(len(indices) * calibration_fraction))
    retrieval_idx = np.sort(indices[:n_retrieval])
    calibration_idx = np.sort(indices[n_retrieval:n_retrieval + n_calibration])
    eval_idx = np.sort(indices[n_retrieval + n_calibration:])
    return (
        features[retrieval_idx],
        labels[retrieval_idx],
        features[calibration_idx],
        labels[calibration_idx],
        features[eval_idx],
        labels[eval_idx],
    )


def build_splits(
    class_names: list[str],
    retrieval_labels: np.ndarray,
    calibration_labels: np.ndarray,
    protocol_cfg: dict,
    seed: int,
) -> list[dict]:
    retrieval_counts = retrieval_labels.sum(axis=0)
    calibration_counts = calibration_labels.sum(axis=0)
    eligible = [
        class_names[i]
        for i in range(len(class_names))
        if retrieval_counts[i] >= int(protocol_cfg.get("min_retrieval_positives", 5))
        and calibration_counts[i] >= int(protocol_cfg.get("min_calibration_positives", 3))
    ]
    split_seeds = protocol_cfg.get("split_seeds") or [seed + i for i in range(5)]
    emerging_count = min(int(protocol_cfg.get("emerging_count", 15)), len(eligible))
    tail_count = int(protocol_cfg.get("tail_count", 10))
    tail_pool = sorted(eligible, key=lambda name: (retrieval_counts[class_names.index(name)], name.lower()))
    splits = []
    for i, split_seed in enumerate(split_seeds):
        rng = random.Random(int(split_seed))
        shuffled = list(eligible)
        rng.shuffle(shuffled)
        emerging = sorted(shuffled[:emerging_count])
        tail_candidates = [name for name in tail_pool if name not in emerging]
        tail = sorted(tail_candidates[:tail_count])
        splits.append({
            "name": f"split_{i}",
            "seed": int(split_seed),
            "emerging_labels": emerging,
            "tail_known_labels": tail,
        })
    return splits


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
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


def tecr_at_threshold(
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
) -> float | None:
    eligible = (labels[:, tail_idx].max(axis=1) > 0) & (labels[:, emerging_idx].max(axis=1) == 0)
    if eligible.sum() == 0:
        return None
    return float(((scores >= threshold) & eligible).sum() / eligible.sum())


def metrics_with_calibrated_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
) -> dict:
    if len(np.unique(y_true)) < 2:
        ap = None
        auroc = None
    else:
        ap = float(average_precision_score(y_true, scores))
        auroc = float(roc_auc_score(y_true, scores))
    pred = scores >= threshold
    tp = float(((pred == 1) & (y_true == 1)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "average_precision": ap,
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "best_f1": f1,
        "threshold": threshold,
        "tecr": tecr_at_threshold(scores, threshold, labels, tail_idx, emerging_idx),
    }


def aggregate_scores(score_matrix: np.ndarray) -> np.ndarray:
    return score_matrix.max(axis=1)


def confidence_gate_scores(
    knn_scores: np.ndarray,
    known_explanation: np.ndarray,
    residual_power: float,
    confidence_cutoff: float,
    confidence_temperature: float,
) -> np.ndarray:
    residual_gate = np.maximum(0.0, 1.0 - known_explanation) ** residual_power
    confidence = sigmoid((knn_scores - confidence_cutoff) / max(confidence_temperature, 1e-6))
    gate = confidence + (1.0 - confidence) * residual_gate[:, None]
    return knn_scores * gate


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
    return sorted(out, key=lambda row: row["method"])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct_change(old: float, new: float) -> float:
    return 100.0 * (new - old) / old


def pct_reduction(old: float, new: float) -> float:
    return 100.0 * (old - new) / old


def select_setting(
    calibration_rows: list[dict],
    baseline: dict,
    ap_tolerance: float,
    f1_tolerance: float,
) -> dict:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/openimages_5k_confidence.yaml")
    parser.add_argument("--output-dir", default="outputs/openimages_5k_heldout_calibration")
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
    ap_tolerance = float(calibration_cfg.get("ap_tolerance", 0.01))
    f1_tolerance = float(calibration_cfg.get("f1_tolerance", 0.01))
    residual_powers = calibration_cfg.get("residual_powers", [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6])
    cutoffs = calibration_cfg.get("confidence_cutoffs", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    temperatures = calibration_cfg.get("confidence_temperatures", [0.01, 0.02, 0.05, 0.1])

    features, labels, text_features, class_names = load_cached_arrays(cfg, Path(cfg["output_dir"]))
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

    retrieval_logits_raw = retrieval_features @ text_features.T
    logit_mean = retrieval_logits_raw.mean(axis=0, keepdims=True)
    logit_std = retrieval_logits_raw.std(axis=0, keepdims=True) + 1e-8
    calibration_logits = (calibration_features @ text_features.T - logit_mean) / logit_std
    eval_logits = (eval_features @ text_features.T - logit_mean) / logit_std

    eval_rows = []
    calibration_selection_rows = []
    selected_rows = []
    for split in splits:
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
        known_idx = [i for i, name in enumerate(class_names) if name not in split["emerging_labels"]]

        calibration_known = calibration_logits[:, known_idx].max(axis=1)
        calibration_known = 1.0 / (1.0 + np.exp(-calibration_known))
        eval_known = eval_logits[:, known_idx].max(axis=1)
        eval_known = 1.0 / (1.0 + np.exp(-eval_known))

        calibration_knn = knn_score_matrix(retrieval_features, retrieval_labels, calibration_features, emerging_idx, cfg["knn"])
        eval_knn = knn_score_matrix(retrieval_features, retrieval_labels, eval_features, emerging_idx, cfg["knn"])

        y_cal = calibration_labels[:, emerging_idx].max(axis=1).astype(int)
        y_eval = eval_labels[:, emerging_idx].max(axis=1).astype(int)

        cal_base_scores = aggregate_scores(calibration_knn)
        eval_base_scores = aggregate_scores(eval_knn)
        cal_threshold = best_f1_threshold(y_cal, cal_base_scores)
        cal_baseline = metrics_with_calibrated_threshold(y_cal, cal_base_scores, cal_threshold["threshold"], calibration_labels, tail_idx, emerging_idx)
        cal_baseline.update({
            "method": "clip_knn",
            "split": split["name"],
            "residual_power": None,
            "confidence_cutoff": None,
            "confidence_temperature": None,
        })
        eval_baseline = metrics_with_calibrated_threshold(y_eval, eval_base_scores, cal_threshold["threshold"], eval_labels, tail_idx, emerging_idx)
        eval_baseline.update({
            "method": "clip_knn",
            "split": split["name"],
            "residual_power": None,
            "confidence_cutoff": None,
            "confidence_temperature": None,
            "selection_status": "baseline",
        })
        eval_rows.append(eval_baseline)

        cal_grid_rows = []
        eval_grid_rows = []
        for residual_power in residual_powers:
            for cutoff in cutoffs:
                for temperature in temperatures:
                    residual_power = float(residual_power)
                    cutoff = float(cutoff)
                    temperature = float(temperature)
                    cal_scores = aggregate_scores(confidence_gate_scores(calibration_knn, calibration_known, residual_power, cutoff, temperature))
                    eval_scores = aggregate_scores(confidence_gate_scores(eval_knn, eval_known, residual_power, cutoff, temperature))
                    threshold = best_f1_threshold(y_cal, cal_scores)
                    cal_metrics = metrics_with_calibrated_threshold(y_cal, cal_scores, threshold["threshold"], calibration_labels, tail_idx, emerging_idx)
                    cal_row = {
                        **cal_metrics,
                        "method": "elta_confidence",
                        "split": split["name"],
                        "residual_power": residual_power,
                        "confidence_cutoff": cutoff,
                        "confidence_temperature": temperature,
                    }
                    eval_metrics = metrics_with_calibrated_threshold(y_eval, eval_scores, threshold["threshold"], eval_labels, tail_idx, emerging_idx)
                    eval_row = {
                        **eval_metrics,
                        "method": "elta_confidence_oracle_pool",
                        "split": split["name"],
                        "residual_power": residual_power,
                        "confidence_cutoff": cutoff,
                        "confidence_temperature": temperature,
                    }
                    cal_grid_rows.append(cal_row)
                    eval_grid_rows.append(eval_row)

        selected = select_setting(cal_grid_rows, cal_baseline, ap_tolerance, f1_tolerance)
        selected_rows.append({
            "split": split["name"],
            "selection_status": selected["selection_status"],
            "residual_power": selected["residual_power"],
            "confidence_cutoff": selected["confidence_cutoff"],
            "confidence_temperature": selected["confidence_temperature"],
            "calibration_ap": selected.get("average_precision"),
            "calibration_f1": selected.get("best_f1"),
            "calibration_tecr": selected.get("tecr"),
            "baseline_calibration_ap": cal_baseline["average_precision"],
            "baseline_calibration_f1": cal_baseline["best_f1"],
            "baseline_calibration_tecr": cal_baseline["tecr"],
        })
        calibration_selection_rows.extend(cal_grid_rows)

        if selected["method"] == "clip_knn":
            heldout_eval = dict(eval_baseline)
            heldout_eval["method"] = "elta_confidence_heldout"
            heldout_eval["selection_status"] = selected["selection_status"]
        else:
            selected_eval = next(
                row for row in eval_grid_rows
                if row["residual_power"] == selected["residual_power"]
                and row["confidence_cutoff"] == selected["confidence_cutoff"]
                and row["confidence_temperature"] == selected["confidence_temperature"]
            )
            heldout_eval = dict(selected_eval)
            heldout_eval["method"] = "elta_confidence_heldout"
            heldout_eval["selection_status"] = selected["selection_status"]
        eval_rows.append(heldout_eval)

        eval_candidates = [
            row for row in eval_grid_rows
            if row["average_precision"] is not None
            and row["best_f1"] is not None
            and row["tecr"] is not None
            and row["average_precision"] >= eval_baseline["average_precision"] * (1.0 - ap_tolerance)
            and row["best_f1"] >= eval_baseline["best_f1"] * (1.0 - f1_tolerance)
        ]
        if eval_candidates:
            oracle = min(eval_candidates, key=lambda row: (row["tecr"], -row["best_f1"], -row["average_precision"]))
            oracle = dict(oracle)
            oracle["method"] = "elta_confidence_eval_oracle"
            oracle["selection_status"] = "eval_oracle_upper_bound"
            eval_rows.append(oracle)

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
        "method", "split", "selection_status", "residual_power", "confidence_cutoff", "confidence_temperature",
        "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    summary_fields = [
        "method", "num_splits", "average_precision_mean", "average_precision_std",
        "auroc_mean", "auroc_std", "best_f1_mean", "best_f1_std", "tecr_mean", "tecr_std",
        "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct",
    ]
    selection_fields = [
        "split", "selection_status", "residual_power", "confidence_cutoff", "confidence_temperature",
        "calibration_ap", "calibration_f1", "calibration_tecr",
        "baseline_calibration_ap", "baseline_calibration_f1", "baseline_calibration_tecr",
    ]
    grid_fields = [
        "method", "split", "residual_power", "confidence_cutoff", "confidence_temperature",
        "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    write_csv(output_dir / "heldout_eval_rows.csv", eval_rows, eval_fields)
    write_csv(output_dir / "heldout_summary.csv", summary, summary_fields)
    write_csv(output_dir / "heldout_selected_settings.csv", selected_rows, selection_fields)
    write_csv(output_dir / "heldout_calibration_grid.csv", calibration_selection_rows, grid_fields)

    report = [
        "# Open Images Held-Out Calibration",
        "",
        "Date: 2026-05-22",
        "",
        f"Images: {labels.shape[0]} total; retrieval={retrieval_features.shape[0]}, calibration={calibration_features.shape[0]}, eval={eval_features.shape[0]}.",
        "Task labels and tail ordering are defined from retrieval/calibration labels only; eval labels are used only for final metrics.",
        f"Selection constraints: AP loss <= {ap_tolerance * 100:.1f}%, F1 loss <= {f1_tolerance * 100:.1f}% on calibration.",
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
        "Interpretation:",
        "",
        "- `elta_confidence_heldout` is the fair method: parameters are selected on calibration only.",
        "- `elta_confidence_eval_oracle` is an upper bound that selects on eval and must not be reported as the real method.",
        "",
    ])
    (output_dir / "heldout_calibration_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "status": "heldout_calibration_complete",
        "output_dir": str(output_dir),
        "seed": run_seed,
        "num_images": int(labels.shape[0]),
        "num_retrieval": int(retrieval_features.shape[0]),
        "num_calibration": int(calibration_features.shape[0]),
        "num_eval": int(eval_features.shape[0]),
        "task_definition": "class eligibility and tail ordering use retrieval/calibration labels only; eval labels are final reporting only",
        "summary": summary,
        "selected_settings": selected_rows,
    }
    (output_dir / "heldout_calibration_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
