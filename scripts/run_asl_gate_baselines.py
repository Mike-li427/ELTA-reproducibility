from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from run_coco_heldout_calibration import build_splits as build_coco_splits
from run_coco_pilot import COCO_CLASSES
from run_coco_training_baselines import load_cached_coco_arrays
from run_openimages_heldout_calibration import (
    aggregate_scores,
    build_splits as build_openimages_splits,
    confidence_gate_scores,
    select_setting,
    split_retrieval_calibration_eval,
)
from run_openimages_margin_gate import load_cached_arrays
from run_openimages_training_baselines import (
    add_global_and_class_rows,
    best_f1_threshold,
    metrics_with_threshold,
    predict_scores,
    summarize,
    train_head,
)


def load_dataset_arrays(dataset: str, cfg: dict, output_dir: Path):
    if dataset == "openimages":
        return load_cached_arrays(cfg, Path(cfg["output_dir"]))
    if dataset == "coco":
        return load_cached_coco_arrays(cfg, Path(cfg["output_dir"]))
    raise ValueError(f"Unknown dataset: {dataset}")


def build_dataset_splits(
    dataset: str,
    class_names: list[str],
    retrieval_labels: np.ndarray,
    calibration_labels: np.ndarray,
    protocol_cfg: dict,
    seed: int,
) -> list[dict]:
    if dataset == "openimages":
        return build_openimages_splits(class_names, retrieval_labels, calibration_labels, protocol_cfg, seed)
    if dataset == "coco":
        return build_coco_splits(class_names, retrieval_labels, calibration_labels, protocol_cfg, seed)
    raise ValueError(f"Unknown dataset: {dataset}")


def summarize_asl_rows(rows: list[dict]) -> list[dict]:
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
    baseline = next((row for row in out if row["method"] == "asl_global_threshold"), None)
    if baseline is not None:
        for row in out:
            if row["method"] == "asl_global_threshold":
                row["ap_delta_pct"] = 0.0
                row["f1_delta_pct"] = 0.0
                row["tecr_reduction_pct"] = 0.0
            else:
                row["ap_delta_pct"] = 100.0 * (row["average_precision_mean"] - baseline["average_precision_mean"]) / baseline["average_precision_mean"]
                row["f1_delta_pct"] = 100.0 * (row["best_f1_mean"] - baseline["best_f1_mean"]) / baseline["best_f1_mean"]
                row["tecr_reduction_pct"] = 100.0 * (baseline["tecr_mean"] - row["tecr_mean"]) / baseline["tecr_mean"]
    order = {
        "asl_global_threshold": 0,
        "asl_class_thresholds": 1,
        "asl_gate_global_threshold": 2,
        "asl_gate_class_thresholds": 3,
    }
    return sorted(out, key=lambda row: (order.get(row["method"], 99), row["method"]))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["openimages", "coco"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    args = parser.parse_args()

    start = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_seed = int(args.seed_override) if args.seed_override is not None else int(cfg["seed"])
    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))
    ap_tolerance = float(calibration_cfg.get("ap_tolerance", 0.01))
    f1_tolerance = float(calibration_cfg.get("f1_tolerance", 0.01))
    residual_powers = [float(v) for v in calibration_cfg.get("residual_powers", [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6])]
    cutoffs = [float(v) for v in calibration_cfg.get("confidence_cutoffs", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])]
    temperatures = [float(v) for v in calibration_cfg.get("confidence_temperatures", [0.01, 0.02, 0.05, 0.1])]

    features, labels, text_features, class_names = load_dataset_arrays(args.dataset, cfg, Path(cfg["output_dir"]))
    (
        retrieval_features,
        retrieval_labels,
        calibration_features,
        calibration_labels,
        eval_features,
        eval_labels,
    ) = split_retrieval_calibration_eval(features, labels, run_seed, retrieval_fraction, calibration_fraction)
    splits = build_dataset_splits(args.dataset, class_names, retrieval_labels, calibration_labels, cfg["protocol"], run_seed)

    retrieval_logits_raw = retrieval_features @ text_features.T
    logit_mean = retrieval_logits_raw.mean(axis=0, keepdims=True)
    logit_std = retrieval_logits_raw.std(axis=0, keepdims=True) + 1e-8
    calibration_logits = (calibration_features @ text_features.T - logit_mean) / logit_std
    eval_logits = (eval_features @ text_features.T - logit_mean) / logit_std

    rows = []
    selected_rows = []

    for split_idx, split in enumerate(splits):
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
        known_idx = [i for i, name in enumerate(class_names) if name not in split["emerging_labels"]]
        y_cal = calibration_labels[:, emerging_idx].max(axis=1).astype(int)
        y_eval = eval_labels[:, emerging_idx].max(axis=1).astype(int)

        model = train_head(
            retrieval_features,
            retrieval_labels[:, emerging_idx],
            "asl",
            device,
            run_seed + split_idx * 100 + 3,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.mixup_alpha,
            init_weights=None,
        )
        cal_scores = predict_scores(model, calibration_features, device)
        eval_scores = predict_scores(model, eval_features, device)

        add_global_and_class_rows(
            rows,
            "asl",
            calibration_labels[:, emerging_idx],
            cal_scores,
            eval_scores,
            y_cal,
            y_eval,
            eval_labels,
            tail_idx,
            emerging_idx,
            split["name"],
            run_seed,
        )

        calibration_known = 1.0 / (1.0 + np.exp(-calibration_logits[:, known_idx].max(axis=1)))
        eval_known = 1.0 / (1.0 + np.exp(-eval_logits[:, known_idx].max(axis=1)))

        cal_base_agg = aggregate_scores(cal_scores)
        eval_base_agg = aggregate_scores(eval_scores)
        base_threshold = best_f1_threshold(y_cal, cal_base_agg)["threshold"]
        base_cal_metrics = metrics_with_threshold(y_cal, cal_base_agg, base_threshold, calibration_labels, tail_idx, emerging_idx)

        candidates = []
        for residual_power in residual_powers:
            for cutoff in cutoffs:
                for temperature in temperatures:
                    cal_gate = confidence_gate_scores(cal_scores, calibration_known, residual_power, cutoff, temperature)
                    eval_gate = confidence_gate_scores(eval_scores, eval_known, residual_power, cutoff, temperature)
                    cal_gate_agg = aggregate_scores(cal_gate)
                    threshold = best_f1_threshold(y_cal, cal_gate_agg)["threshold"]
                    cal_metrics = metrics_with_threshold(y_cal, cal_gate_agg, threshold, calibration_labels, tail_idx, emerging_idx)
                    candidates.append({
                        "method": "elta_confidence",
                        "residual_power": residual_power,
                        "confidence_cutoff": cutoff,
                        "confidence_temperature": temperature,
                        "average_precision": cal_metrics["average_precision"],
                        "best_f1": cal_metrics["best_f1"],
                        "tecr": cal_metrics["tecr"],
                        "calibration_scores": cal_gate,
                        "eval_scores": eval_gate,
                    })

        selected = select_setting(candidates, base_cal_metrics, ap_tolerance, f1_tolerance)
        if selected["selection_status"] == "fallback_baseline":
            selected_cal_scores = cal_scores
            selected_eval_scores = eval_scores
        else:
            selected_cal_scores = next(
                item["calibration_scores"]
                for item in candidates
                if item["residual_power"] == selected["residual_power"]
                and item["confidence_cutoff"] == selected["confidence_cutoff"]
                and item["confidence_temperature"] == selected["confidence_temperature"]
            )
            selected_eval_scores = next(
                item["eval_scores"]
                for item in candidates
                if item["residual_power"] == selected["residual_power"]
                and item["confidence_cutoff"] == selected["confidence_cutoff"]
                and item["confidence_temperature"] == selected["confidence_temperature"]
            )

        selected_rows.append({
            "split": split["name"],
            "seed": run_seed,
            "selection_status": selected["selection_status"],
            "residual_power": selected["residual_power"],
            "confidence_cutoff": selected["confidence_cutoff"],
            "confidence_temperature": selected["confidence_temperature"],
            "calibration_ap": selected.get("average_precision"),
            "calibration_f1": selected.get("best_f1"),
            "calibration_tecr": selected.get("tecr"),
            "baseline_calibration_ap": base_cal_metrics["average_precision"],
            "baseline_calibration_f1": base_cal_metrics["best_f1"],
            "baseline_calibration_tecr": base_cal_metrics["tecr"],
        })

        add_global_and_class_rows(
            rows,
            "asl_gate",
            calibration_labels[:, emerging_idx],
            selected_cal_scores,
            selected_eval_scores,
            y_cal,
            y_eval,
            eval_labels,
            tail_idx,
            emerging_idx,
            split["name"],
            run_seed,
        )

    summary = summarize_asl_rows(rows)
    row_fields = [
        "method",
        "split",
        "seed",
        "average_precision",
        "auroc",
        "precision",
        "recall",
        "best_f1",
        "threshold",
        "tecr",
    ]
    summary_fields = [
        "method",
        "num_splits",
        "average_precision_mean",
        "average_precision_std",
        "auroc_mean",
        "auroc_std",
        "best_f1_mean",
        "best_f1_std",
        "tecr_mean",
        "tecr_std",
        "ap_delta_pct",
        "f1_delta_pct",
        "tecr_reduction_pct",
    ]
    write_csv(output_dir / "asl_gate_rows.csv", rows, row_fields)
    write_csv(output_dir / "asl_gate_summary.csv", summary, summary_fields)
    write_csv(
        output_dir / "asl_gate_selected_settings.csv",
        selected_rows,
        [
            "split",
            "seed",
            "selection_status",
            "residual_power",
            "confidence_cutoff",
            "confidence_temperature",
            "calibration_ap",
            "calibration_f1",
            "calibration_tecr",
            "baseline_calibration_ap",
            "baseline_calibration_f1",
            "baseline_calibration_tecr",
        ],
    )

    baseline = next(row for row in summary if row["method"] == "asl_global_threshold")
    report = [
        "# ASL + Gate Baselines",
        "",
        "Date: 2026-05-27",
        "",
        f"Dataset: `{args.dataset}`.",
        f"Seed: `{run_seed}`.",
        f"Device: `{device}`.",
        "",
        "| Method | AP | F1 | TECR | AP delta | F1 delta | TECR reduction vs ASL |",
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
        "- ASL is trained on the retrieval split only, then evaluated under the same calibration/evaluation protocol.",
        "- Gate parameters are selected on calibration rows only and then applied unchanged to evaluation rows.",
        "- Known-label confidence for the gate comes from CLIP zero-shot logits on the same split.",
        "",
    ])
    (output_dir / "asl_gate_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "status": "asl_gate_baselines_complete",
        "dataset": args.dataset,
        "seed": run_seed,
        "time_seconds": round(time.time() - start, 3),
        "output_dir": str(output_dir),
        "baseline": baseline,
        "summary": summary,
    }
    (output_dir / "asl_gate_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TIME_ESTIMATE:", max(1, int(time.time() - start)))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
