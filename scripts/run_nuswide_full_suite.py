from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import clip
import numpy as np
import torch
import yaml

from nuswide_common import load_or_compute_nuswide_features
from run_coco_heldout_calibration import (
    add_matrix_calibration_rows,
    frequency_groups,
    frequency_summary,
    isotonic_calibrate_scores,
)
from run_openimages_heldout_calibration import (
    aggregate_scores,
    build_splits,
    confidence_gate_scores,
    metrics_with_calibrated_threshold,
    select_setting,
    split_retrieval_calibration_eval,
)
from run_openimages_pilot import knn_score_matrix, set_seed
from run_openimages_training_baselines import (
    add_global_and_class_rows,
    predict_scores,
    summarize as summarize_training_rows,
    train_head,
)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    from run_openimages_heldout_calibration import best_f1_threshold as _best

    return _best(y_true, scores)


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


def add_deltas(summary_rows: list[dict], baseline_method: str) -> None:
    baseline = next(row for row in summary_rows if row["method"] == baseline_method)
    for row in summary_rows:
        if row["method"] == baseline_method:
            row["ap_delta_pct"] = 0.0
            row["f1_delta_pct"] = 0.0
            row["tecr_reduction_pct"] = 0.0
        else:
            row["ap_delta_pct"] = pct_change(baseline["average_precision_mean"], row["average_precision_mean"])
            row["f1_delta_pct"] = pct_change(baseline["best_f1_mean"], row["best_f1_mean"])
            row["tecr_reduction_pct"] = pct_reduction(baseline["tecr_mean"], row["tecr_mean"])


def residual_only_scores(knn_scores: np.ndarray, known: np.ndarray, residual_power: float) -> np.ndarray:
    residual_gate = np.maximum(0.0, 1.0 - known) ** residual_power
    return knn_scores * residual_gate[:, None]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nuswide_heldout_ultrastrict.yaml")
    parser.add_argument("--output-dir", default="outputs/nuswide_heldout_ultrastrict")
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    args = parser.parse_args()

    start = time.time()
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
    features, labels, text_features, class_names, image_names, cache_hit = load_or_compute_nuswide_features(
        cfg, model, preprocess, device
    )

    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))
    ap_tolerance = float(calibration_cfg.get("ap_tolerance", 0.0025))
    f1_tolerance = float(calibration_cfg.get("f1_tolerance", 0.0025))
    residual_powers = [float(v) for v in calibration_cfg.get("residual_powers", [0.0, 0.1, 0.2, 0.3, 0.4])]
    cutoffs = [float(v) for v in calibration_cfg.get("confidence_cutoffs", [0.2, 0.3, 0.4, 0.5, 0.6])]
    temperatures = [float(v) for v in calibration_cfg.get("confidence_temperatures", [0.01, 0.02, 0.05, 0.1])]

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
    selected_rows = []
    calibration_grid = []
    baseline_rows = []
    training_rows = []
    asl_gate_rows = []
    ablation_rows = []
    frequency_rows = []
    reference_counts = retrieval_labels.sum(axis=0) + calibration_labels.sum(axis=0)
    label_groups = frequency_groups(reference_counts, class_names)

    for split_idx, split in enumerate(splits):
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
        known_idx = [i for i, name in enumerate(class_names) if name not in split["emerging_labels"]]
        y_cal = calibration_labels[:, emerging_idx].max(axis=1).astype(int)
        y_eval = eval_labels[:, emerging_idx].max(axis=1).astype(int)

        calibration_known = 1.0 / (1.0 + np.exp(-calibration_logits[:, known_idx].max(axis=1)))
        eval_known = 1.0 / (1.0 + np.exp(-eval_logits[:, known_idx].max(axis=1)))
        calibration_knn = knn_score_matrix(retrieval_features, retrieval_labels, calibration_features, emerging_idx, cfg["knn"])
        eval_knn = knn_score_matrix(retrieval_features, retrieval_labels, eval_features, emerging_idx, cfg["knn"])

        cal_base_scores = aggregate_scores(calibration_knn)
        eval_base_scores = aggregate_scores(eval_knn)
        cal_threshold = best_f1_threshold(y_cal, cal_base_scores)
        cal_baseline = metrics_with_calibrated_threshold(y_cal, cal_base_scores, cal_threshold["threshold"], calibration_labels, tail_idx, emerging_idx)
        eval_baseline = metrics_with_calibrated_threshold(y_eval, eval_base_scores, cal_threshold["threshold"], eval_labels, tail_idx, emerging_idx)
        eval_rows.append({**eval_baseline, "method": "clip_knn", "split": split["name"], "selection_status": "baseline"})

        add_matrix_calibration_rows(baseline_rows, "clip_knn", calibration_labels[:, emerging_idx], calibration_knn, eval_knn, y_cal, y_eval, eval_labels, tail_idx, emerging_idx, split["name"], run_seed)
        iso_cal, iso_eval = isotonic_calibrate_scores(calibration_labels[:, emerging_idx], calibration_knn, eval_knn)
        add_matrix_calibration_rows(baseline_rows, "clip_knn_isotonic", calibration_labels[:, emerging_idx], iso_cal, iso_eval, y_cal, y_eval, eval_labels, tail_idx, emerging_idx, split["name"], run_seed)

        cal_grid_rows = []
        eval_grid_rows = []
        residual_grid_rows = []
        for residual_power in residual_powers:
            cal_residual = aggregate_scores(residual_only_scores(calibration_knn, calibration_known, residual_power))
            eval_residual_matrix = residual_only_scores(eval_knn, eval_known, residual_power)
            res_threshold = best_f1_threshold(y_cal, cal_residual)["threshold"]
            res_metrics = metrics_with_calibrated_threshold(y_eval, aggregate_scores(eval_residual_matrix), res_threshold, eval_labels, tail_idx, emerging_idx)
            residual_grid_rows.append({**res_metrics, "method": "residual_only_gate", "split": split["name"], "residual_power": residual_power})
            for cutoff in cutoffs:
                for temperature in temperatures:
                    cal_scores_matrix = confidence_gate_scores(calibration_knn, calibration_known, residual_power, cutoff, temperature)
                    eval_scores_matrix = confidence_gate_scores(eval_knn, eval_known, residual_power, cutoff, temperature)
                    cal_scores = aggregate_scores(cal_scores_matrix)
                    threshold = best_f1_threshold(y_cal, cal_scores)
                    cal_metrics = metrics_with_calibrated_threshold(y_cal, cal_scores, threshold["threshold"], calibration_labels, tail_idx, emerging_idx)
                    eval_metrics = metrics_with_calibrated_threshold(y_eval, aggregate_scores(eval_scores_matrix), threshold["threshold"], eval_labels, tail_idx, emerging_idx)
                    cal_row = {
                        **cal_metrics,
                        "method": "elta_confidence",
                        "split": split["name"],
                        "residual_power": residual_power,
                        "confidence_cutoff": cutoff,
                        "confidence_temperature": temperature,
                    }
                    eval_row = {
                        **eval_metrics,
                        "method": "elta_confidence_oracle_pool",
                        "split": split["name"],
                        "residual_power": residual_power,
                        "confidence_cutoff": cutoff,
                        "confidence_temperature": temperature,
                        "score_matrix": eval_scores_matrix,
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
        calibration_grid.extend(cal_grid_rows)

        if selected["method"] == "clip_knn":
            calibration_held = calibration_knn
            eval_held = eval_knn
            heldout_eval = dict(eval_baseline)
            heldout_eval.update({"method": "elta_confidence_heldout", "split": split["name"], "selection_status": selected["selection_status"]})
        else:
            selected_eval = next(
                row for row in eval_grid_rows
                if row["residual_power"] == selected["residual_power"]
                and row["confidence_cutoff"] == selected["confidence_cutoff"]
                and row["confidence_temperature"] == selected["confidence_temperature"]
            )
            eval_held = selected_eval.pop("score_matrix")
            calibration_held = confidence_gate_scores(
                calibration_knn,
                calibration_known,
                selected["residual_power"],
                selected["confidence_cutoff"],
                selected["confidence_temperature"],
            )
            heldout_eval = {k: v for k, v in selected_eval.items() if k != "method"}
            heldout_eval.update({"method": "elta_confidence_heldout", "split": split["name"], "selection_status": selected["selection_status"]})
        eval_rows.append(heldout_eval)
        add_matrix_calibration_rows(baseline_rows, "heldout_gate", calibration_labels[:, emerging_idx], calibration_held, eval_held, y_cal, y_eval, eval_labels, tail_idx, emerging_idx, split["name"], run_seed)

        best_residual = min([r for r in residual_grid_rows if r["tecr"] is not None], key=lambda r: (r["tecr"], -r["best_f1"]))
        ablation_rows.append({**eval_baseline, "method": "clip_knn", "split": split["name"]})
        ablation_rows.append(best_residual)
        ablation_rows.append({**heldout_eval, "method": "full_gate", "split": split["name"]})

        for method in ["asl", "db_loss"]:
            train_y = retrieval_labels[:, emerging_idx]
            head = train_head(retrieval_features, train_y, method, device, run_seed + split_idx * 100 + len(method), args.epochs, args.lr, args.weight_decay, args.mixup_alpha)
            cal_scores = predict_scores(head, calibration_features, device)
            eval_scores = predict_scores(head, eval_features, device)
            add_global_and_class_rows(training_rows, method, calibration_labels[:, emerging_idx], cal_scores, eval_scores, y_cal, y_eval, eval_labels, tail_idx, emerging_idx, split["name"], run_seed)
            if method == "asl":
                add_global_and_class_rows(asl_gate_rows, "asl", calibration_labels[:, emerging_idx], cal_scores, eval_scores, y_cal, y_eval, eval_labels, tail_idx, emerging_idx, split["name"], run_seed)
                cal_asl_gate = confidence_gate_scores(cal_scores, calibration_known, selected["residual_power"], selected["confidence_cutoff"], selected["confidence_temperature"])
                eval_asl_gate = confidence_gate_scores(eval_scores, eval_known, selected["residual_power"], selected["confidence_cutoff"], selected["confidence_temperature"])
                add_global_and_class_rows(asl_gate_rows, "asl_gate", calibration_labels[:, emerging_idx], cal_asl_gate, eval_asl_gate, y_cal, y_eval, eval_labels, tail_idx, emerging_idx, split["name"], run_seed)

        for group in ["head", "mid", "tail"]:
            local = [i for i, class_idx in enumerate(emerging_idx) if label_groups[class_names[class_idx]] == group]
            if not local:
                continue
            group_idx = [emerging_idx[i] for i in local]
            group_y_cal = calibration_labels[:, group_idx].max(axis=1).astype(int)
            group_y_eval = eval_labels[:, group_idx].max(axis=1).astype(int)
            for method, cal_matrix, eval_matrix in [
                ("clip_knn", calibration_knn, eval_knn),
                ("elta_confidence_heldout", calibration_held, eval_held),
            ]:
                cal_group = aggregate_scores(cal_matrix[:, local])
                eval_group = aggregate_scores(eval_matrix[:, local])
                threshold = best_f1_threshold(group_y_cal, cal_group)["threshold"]
                metrics = metrics_with_calibrated_threshold(group_y_eval, eval_group, threshold, eval_labels, tail_idx, group_idx)
                frequency_rows.append({
                    **metrics,
                    "method": method,
                    "split": split["name"],
                    "seed": run_seed,
                    "frequency_group": group,
                    "num_emerging_labels": len(local),
                })

    main_summary = summarize(eval_rows)
    add_deltas(main_summary, "clip_knn")
    baseline_summary = summarize(baseline_rows)
    add_deltas(baseline_summary, "clip_knn_global_threshold")
    training_summary = summarize_training_rows(training_rows)
    asl_gate_summary = summarize_training_rows(asl_gate_rows)
    ablation_summary = summarize(ablation_rows)
    add_deltas(ablation_summary, "clip_knn")

    eval_fields = ["method", "split", "selection_status", "residual_power", "confidence_cutoff", "confidence_temperature", "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr"]
    summary_fields = ["method", "num_splits", "average_precision_mean", "average_precision_std", "auroc_mean", "auroc_std", "best_f1_mean", "best_f1_std", "tecr_mean", "tecr_std", "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct"]
    row_fields = ["method", "split", "seed", "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr"]
    selection_fields = ["split", "selection_status", "residual_power", "confidence_cutoff", "confidence_temperature", "calibration_ap", "calibration_f1", "calibration_tecr", "baseline_calibration_ap", "baseline_calibration_f1", "baseline_calibration_tecr"]
    grid_fields = ["method", "split", "residual_power", "confidence_cutoff", "confidence_temperature", "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr"]
    freq_fields = ["method", "split", "seed", "frequency_group", "num_emerging_labels", "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr"]
    freq_summary_fields = ["frequency_group", "num_splits", "num_emerging_labels_mean", "clip_knn_ap_mean", "heldout_ap_mean", "ap_delta", "clip_knn_f1_mean", "heldout_f1_mean", "f1_delta", "clip_knn_tecr_mean", "heldout_tecr_mean", "tecr_reduction_pct"]

    write_csv(output_dir / "heldout_eval_rows.csv", eval_rows, eval_fields)
    write_csv(output_dir / "heldout_summary.csv", main_summary, summary_fields)
    write_csv(output_dir / "heldout_selected_settings.csv", selected_rows, selection_fields)
    write_csv(output_dir / "heldout_calibration_grid.csv", calibration_grid, grid_fields)
    write_csv(output_dir / "calibrated_baseline_rows.csv", baseline_rows, row_fields)
    write_csv(output_dir / "calibrated_baseline_summary.csv", baseline_summary, summary_fields)
    write_csv(output_dir / "training_baseline_rows.csv", training_rows, row_fields)
    write_csv(output_dir / "training_baseline_summary.csv", training_summary, summary_fields)
    write_csv(output_dir / "asl_gate_rows.csv", asl_gate_rows, row_fields)
    write_csv(output_dir / "asl_gate_summary.csv", asl_gate_summary, summary_fields)
    write_csv(output_dir / "gate_ablation_rows.csv", ablation_rows, row_fields + ["residual_power"])
    write_csv(output_dir / "gate_ablation_summary.csv", ablation_summary, summary_fields)
    write_csv(output_dir / "heldout_frequency_group_rows.csv", frequency_rows, freq_fields)
    write_csv(output_dir / "heldout_frequency_group_summary.csv", frequency_summary(frequency_rows), freq_summary_fields)

    report = [
        "# NUS-WIDE Full Suite",
        "",
        f"Images: {labels.shape[0]}; classes: {len(class_names)}; cache_hit={cache_hit}.",
        "",
        "## Main",
        "",
        "| Method | AP | F1 | TECR | TECR reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in baseline_summary:
        if row["method"] in {"clip_knn_global_threshold", "clip_knn_class_thresholds", "clip_knn_isotonic_global_threshold", "heldout_gate_global_threshold", "heldout_gate_class_thresholds"}:
            report.append(f"| {row['method']} | {row['average_precision_mean']:.4f} | {row['best_f1_mean']:.4f} | {row['tecr_mean']:.4f} | {row['tecr_reduction_pct']:.1f}% |")
    report.extend(["", "## Training Baselines", "", "| Method | AP | F1 | TECR |", "|---|---:|---:|---:|"])
    for row in training_summary:
        if row["method"] in {"asl_class_thresholds", "db_loss_class_thresholds"}:
            report.append(f"| {row['method']} | {row['average_precision_mean']:.4f} | {row['best_f1_mean']:.4f} | {row['tecr_mean']:.4f} |")
    report.extend(["", "## ASL+Gate", "", "| Method | AP | F1 | TECR |", "|---|---:|---:|---:|"])
    for row in asl_gate_summary:
        report.append(f"| {row['method']} | {row['average_precision_mean']:.4f} | {row['best_f1_mean']:.4f} | {row['tecr_mean']:.4f} |")
    (output_dir / "nuswide_full_suite_report.md").write_text("\n".join(report), encoding="utf-8")

    result = {
        "status": "nuswide_full_suite_complete",
        "time_seconds": round(time.time() - start, 3),
        "output_dir": str(output_dir),
        "seed": run_seed,
        "num_images": int(labels.shape[0]),
        "num_classes": len(class_names),
        "cache_hit": cache_hit,
        "main_summary": main_summary,
        "baseline_summary": baseline_summary,
        "training_summary": training_summary,
        "asl_gate_summary": asl_gate_summary,
        "ablation_summary": ablation_summary,
    }
    (output_dir / "nuswide_full_suite_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
