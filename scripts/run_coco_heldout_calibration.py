from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import clip
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch
import yaml

from run_coco_pilot import COCO_CLASSES, load_or_compute_features, set_seed
from run_openimages_heldout_calibration import (
    aggregate_scores,
    confidence_gate_scores,
    metrics_with_calibrated_threshold,
    select_setting,
    split_retrieval_calibration_eval,
)
from run_openimages_pilot import knn_score_matrix


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
        and calibration_counts[i] >= int(protocol_cfg.get("min_calibration_positives", 5))
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


def pct_change(old: float, new: float) -> float:
    return 100.0 * (new - old) / old


def pct_reduction(old: float, new: float) -> float:
    return 100.0 * (old - new) / old


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def platt_calibrate_scores(calibration_labels: np.ndarray, calibration_scores: np.ndarray, eval_scores: np.ndarray):
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


def isotonic_calibrate_scores(calibration_labels: np.ndarray, calibration_scores: np.ndarray, eval_scores: np.ndarray):
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


def add_matrix_calibration_rows(
    rows: list[dict],
    method_prefix: str,
    calibration_labels: np.ndarray,
    calibration_scores: np.ndarray,
    eval_scores: np.ndarray,
    y_cal: np.ndarray,
    y_eval: np.ndarray,
    eval_labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
    split_name: str,
    run_seed: int,
) -> None:
    cal_agg = aggregate_scores(calibration_scores)
    eval_agg = aggregate_scores(eval_scores)
    threshold = best_f1_threshold(y_cal, cal_agg)["threshold"]
    global_metrics = metrics_with_calibrated_threshold(y_eval, eval_agg, threshold, eval_labels, tail_idx, emerging_idx)
    rows.append({**global_metrics, "method": f"{method_prefix}_global_threshold", "split": split_name, "seed": run_seed})

    thresholds = class_thresholds(calibration_labels, calibration_scores)
    pred = (eval_scores >= thresholds[None, :]).max(axis=1)
    class_metrics = aggregate_metrics_from_predictions(y_eval, eval_agg, pred, eval_labels, tail_idx, emerging_idx)
    rows.append({**class_metrics, "method": f"{method_prefix}_class_thresholds", "split": split_name, "seed": run_seed})


def frequency_groups(reference_counts: np.ndarray, class_names: list[str]) -> dict[str, str]:
    order = sorted(range(len(class_names)), key=lambda i: (-reference_counts[i], class_names[i].lower()))
    groups = {}
    n = len(order)
    for rank, idx in enumerate(order):
        if rank < n / 3:
            group = "head"
        elif rank < 2 * n / 3:
            group = "mid"
        else:
            group = "tail"
        groups[class_names[idx]] = group
    return groups


def mean(values: list[float | None]) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


def frequency_summary(rows: list[dict]) -> list[dict]:
    out = []
    for group in ["head", "mid", "tail"]:
        base_rows = [row for row in rows if row["frequency_group"] == group and row["method"] == "clip_knn"]
        held_rows = [row for row in rows if row["frequency_group"] == group and row["method"] == "elta_confidence_heldout"]
        by_split_base = {row["split"]: row for row in base_rows}
        by_split_held = {row["split"]: row for row in held_rows}
        common = sorted(set(by_split_base) & set(by_split_held))
        base_ap = mean([by_split_base[s]["average_precision"] for s in common])
        held_ap = mean([by_split_held[s]["average_precision"] for s in common])
        base_f1 = mean([by_split_base[s]["best_f1"] for s in common])
        held_f1 = mean([by_split_held[s]["best_f1"] for s in common])
        base_tecr = mean([by_split_base[s]["tecr"] for s in common])
        held_tecr = mean([by_split_held[s]["tecr"] for s in common])
        out.append({
            "frequency_group": group,
            "num_splits": len(common),
            "num_emerging_labels_mean": mean([by_split_base[s]["num_emerging_labels"] for s in common]),
            "clip_knn_ap_mean": base_ap,
            "heldout_ap_mean": held_ap,
            "ap_delta": None if base_ap is None or held_ap is None else held_ap - base_ap,
            "clip_knn_f1_mean": base_f1,
            "heldout_f1_mean": held_f1,
            "f1_delta": None if base_f1 is None or held_f1 is None else held_f1 - base_f1,
            "clip_knn_tecr_mean": base_tecr,
            "heldout_tecr_mean": held_tecr,
            "tecr_reduction_pct": None if base_tecr in (None, 0) or held_tecr is None else 100.0 * (base_tecr - held_tecr) / base_tecr,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/coco_heldout_ultrastrict.yaml")
    parser.add_argument("--output-dir", default="outputs/coco_heldout_ultrastrict")
    parser.add_argument("--seed-override", type=int)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_seed = int(args.seed_override) if args.seed_override is not None else int(cfg["seed"])
    set_seed(run_seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model, preprocess = clip.load(cfg["clip"]["model"], device=device)
    model.eval()
    features, labels, text_features, cache_hit = load_or_compute_features(cfg, model, preprocess, device, Path(cfg["output_dir"]))
    class_names = list(COCO_CLASSES)

    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))
    ap_tolerance = float(calibration_cfg.get("ap_tolerance", 0.01))
    f1_tolerance = float(calibration_cfg.get("f1_tolerance", 0.01))
    residual_powers = calibration_cfg.get("residual_powers", [0.0, 0.1, 0.2, 0.3, 0.4])
    cutoffs = calibration_cfg.get("confidence_cutoffs", [0.2, 0.3, 0.4, 0.5, 0.6])
    temperatures = calibration_cfg.get("confidence_temperatures", [0.01, 0.02, 0.05, 0.1])

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

    eval_rows = []
    calibration_selection_rows = []
    selected_rows = []
    baseline_rows = []
    frequency_rows = []
    reference_counts = retrieval_labels.sum(axis=0) + calibration_labels.sum(axis=0)
    label_groups = frequency_groups(reference_counts, class_names)

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
            calibration_held = calibration_knn
            eval_held = eval_knn
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
            calibration_held = confidence_gate_scores(
                calibration_knn,
                calibration_known,
                selected["residual_power"],
                selected["confidence_cutoff"],
                selected["confidence_temperature"],
            )
            eval_held = confidence_gate_scores(
                eval_knn,
                eval_known,
                selected["residual_power"],
                selected["confidence_cutoff"],
                selected["confidence_temperature"],
            )
        eval_rows.append(heldout_eval)

        # Baseline suite.
        add_matrix_calibration_rows(baseline_rows, "clip_knn", calibration_labels[:, emerging_idx], calibration_knn, eval_knn, y_cal, y_eval, eval_labels, tail_idx, emerging_idx, split["name"], run_seed)
        calibration_clip = 1.0 / (1.0 + np.exp(-calibration_logits[:, emerging_idx]))
        eval_clip = 1.0 / (1.0 + np.exp(-eval_logits[:, emerging_idx]))
        add_matrix_calibration_rows(baseline_rows, "clip_zeroshot", calibration_labels[:, emerging_idx], calibration_clip, eval_clip, y_cal, y_eval, eval_labels, tail_idx, emerging_idx, split["name"], run_seed)
        platt_cal, platt_eval = platt_calibrate_scores(calibration_labels[:, emerging_idx], calibration_knn, eval_knn)
        add_matrix_calibration_rows(baseline_rows, "clip_knn_platt", calibration_labels[:, emerging_idx], platt_cal, platt_eval, y_cal, y_eval, eval_labels, tail_idx, emerging_idx, split["name"], run_seed)
        isotonic_cal, isotonic_eval = isotonic_calibrate_scores(calibration_labels[:, emerging_idx], calibration_knn, eval_knn)
        add_matrix_calibration_rows(baseline_rows, "clip_knn_isotonic", calibration_labels[:, emerging_idx], isotonic_cal, isotonic_eval, y_cal, y_eval, eval_labels, tail_idx, emerging_idx, split["name"], run_seed)
        add_matrix_calibration_rows(baseline_rows, "heldout_gate", calibration_labels[:, emerging_idx], calibration_held, eval_held, y_cal, y_eval, eval_labels, tail_idx, emerging_idx, split["name"], run_seed)

        for group in ["head", "mid", "tail"]:
            local_indices = [
                i for i, class_idx in enumerate(emerging_idx)
                if label_groups[class_names[class_idx]] == group
            ]
            if not local_indices:
                continue
            group_idx = [emerging_idx[i] for i in local_indices]
            group_y_cal = calibration_labels[:, group_idx].max(axis=1).astype(int)
            group_y_eval = eval_labels[:, group_idx].max(axis=1).astype(int)
            cal_group_base = aggregate_scores(calibration_knn[:, local_indices])
            eval_group_base = aggregate_scores(eval_knn[:, local_indices])
            base_threshold = best_f1_threshold(group_y_cal, cal_group_base)["threshold"]
            base_metrics = metrics_with_calibrated_threshold(group_y_eval, eval_group_base, base_threshold, eval_labels, tail_idx, group_idx)
            frequency_rows.append({
                **base_metrics,
                "split": split["name"],
                "seed": run_seed,
                "frequency_group": group,
                "method": "clip_knn",
                "num_emerging_labels": len(local_indices),
            })
            cal_group_held = aggregate_scores(calibration_held[:, local_indices])
            eval_group_held = aggregate_scores(eval_held[:, local_indices])
            held_threshold = best_f1_threshold(group_y_cal, cal_group_held)["threshold"]
            held_metrics = metrics_with_calibrated_threshold(group_y_eval, eval_group_held, held_threshold, eval_labels, tail_idx, group_idx)
            frequency_rows.append({
                **held_metrics,
                "split": split["name"],
                "seed": run_seed,
                "frequency_group": group,
                "method": "elta_confidence_heldout",
                "num_emerging_labels": len(local_indices),
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

    baseline_summary_rows = summarize(baseline_rows)
    global_base = next(row for row in baseline_summary_rows if row["method"] == "clip_knn_global_threshold")
    for row in baseline_summary_rows:
        if row["method"] == "clip_knn_global_threshold":
            row["ap_delta_pct"] = 0.0
            row["f1_delta_pct"] = 0.0
            row["tecr_reduction_pct"] = 0.0
        else:
            row["ap_delta_pct"] = pct_change(global_base["average_precision_mean"], row["average_precision_mean"])
            row["f1_delta_pct"] = pct_change(global_base["best_f1_mean"], row["best_f1_mean"])
            row["tecr_reduction_pct"] = pct_reduction(global_base["tecr_mean"], row["tecr_mean"])

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
    baseline_row_fields = [
        "method", "split", "seed", "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    frequency_fields = [
        "method", "split", "seed", "frequency_group", "num_emerging_labels",
        "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    frequency_summary_fields = [
        "frequency_group", "num_splits", "num_emerging_labels_mean",
        "clip_knn_ap_mean", "heldout_ap_mean", "ap_delta",
        "clip_knn_f1_mean", "heldout_f1_mean", "f1_delta",
        "clip_knn_tecr_mean", "heldout_tecr_mean", "tecr_reduction_pct",
    ]

    write_csv(output_dir / "heldout_eval_rows.csv", eval_rows, eval_fields)
    write_csv(output_dir / "heldout_summary.csv", summary, summary_fields)
    write_csv(output_dir / "heldout_selected_settings.csv", selected_rows, selection_fields)
    write_csv(output_dir / "heldout_calibration_grid.csv", calibration_selection_rows, grid_fields)
    write_csv(output_dir / "calibrated_baseline_rows.csv", baseline_rows, baseline_row_fields)
    write_csv(output_dir / "calibrated_baseline_summary.csv", baseline_summary_rows, summary_fields)
    write_csv(output_dir / "heldout_frequency_group_rows.csv", frequency_rows, frequency_fields)
    write_csv(output_dir / "heldout_frequency_group_summary.csv", frequency_summary(frequency_rows), frequency_summary_fields)

    report = [
        "# COCO Held-Out Calibration",
        "",
        "Date: 2026-05-23",
        "",
        f"Images: {labels.shape[0]} total; retrieval={retrieval_features.shape[0]}, calibration={calibration_features.shape[0]}, eval={eval_features.shape[0]}.",
        f"Cache hit: `{cache_hit}`.",
        "Task labels and tail ordering are defined from retrieval/calibration labels only; eval labels are used only for final metrics.",
        f"Selection constraints: AP loss <= {ap_tolerance * 100:.2f}%, F1 loss <= {f1_tolerance * 100:.2f}% on calibration.",
        "",
        "| Method | AP | F1 | TECR | AP delta | F1 delta | TECR reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in baseline_summary_rows:
        report.append(
            f"| {row['method']} | {row['average_precision_mean']:.4f} | "
            f"{row['best_f1_mean']:.4f} | {row['tecr_mean']:.4f} | "
            f"{row['ap_delta_pct']:.2f}% | {row['f1_delta_pct']:.2f}% | "
            f"{row['tecr_reduction_pct']:.1f}% |"
        )
    report.extend(["", "Held-out selection summary:", ""])
    for row in summary:
        report.append(
            f"- {row['method']}: AP={row['average_precision_mean']:.4f}, "
            f"F1={row['best_f1_mean']:.4f}, TECR={row['tecr_mean']:.4f}, "
            f"TECR reduction={row.get('tecr_reduction_pct', 0.0):.1f}%"
        )
    (output_dir / "coco_heldout_report.md").write_text("\n".join(report), encoding="utf-8")

    result = {
        "status": "coco_heldout_complete",
        "output_dir": str(output_dir),
        "seed": run_seed,
        "num_images": int(labels.shape[0]),
        "num_retrieval": int(retrieval_features.shape[0]),
        "num_calibration": int(calibration_features.shape[0]),
        "num_eval": int(eval_features.shape[0]),
        "task_definition": "class eligibility and tail ordering use retrieval/calibration labels only; eval labels are final reporting only",
        "summary": summary,
        "calibrated_baseline_summary": baseline_summary_rows,
        "selected_settings": selected_rows,
    }
    (output_dir / "coco_heldout_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
