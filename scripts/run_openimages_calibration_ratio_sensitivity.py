from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from run_openimages_heldout_calibration import (
    aggregate_scores,
    best_f1_threshold,
    build_splits,
    confidence_gate_scores,
    metrics_with_calibrated_threshold,
    pct_change,
    pct_reduction,
    select_setting,
)
from run_openimages_margin_gate import load_cached_arrays
from run_openimages_pilot import knn_score_matrix


def split_with_fixed_eval(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    retrieval_fraction: float,
    max_calibration_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    n_retrieval = int(round(len(indices) * retrieval_fraction))
    n_calibration = int(round(len(indices) * max_calibration_fraction))
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


def deterministic_subset(n: int, size: int, seed: int) -> np.ndarray:
    if size >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=size, replace=False))


def summarize(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["calibration_dataset_fraction"], row["method"]), []).append(row)
    out = []
    for (fraction, method), group in grouped.items():
        item = {
            "calibration_dataset_fraction": fraction,
            "method": method,
            "num_splits": len(group),
            "num_calibration_images_used_mean": float(np.mean([row["num_calibration_images_used"] for row in group])),
        }
        for key in ["average_precision", "auroc", "best_f1", "tecr"]:
            vals = [row[key] for row in group if row[key] is not None]
            item[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)

    by_fraction = {}
    for row in out:
        by_fraction.setdefault(row["calibration_dataset_fraction"], {})[row["method"]] = row
    for methods in by_fraction.values():
        baseline = methods.get("clip_knn")
        if not baseline:
            continue
        for row in methods.values():
            if row["method"] == "clip_knn":
                row["ap_delta_pct"] = 0.0
                row["f1_delta_pct"] = 0.0
                row["tecr_reduction_pct"] = 0.0
            else:
                row["ap_delta_pct"] = pct_change(baseline["average_precision_mean"], row["average_precision_mean"])
                row["f1_delta_pct"] = pct_change(baseline["best_f1_mean"], row["best_f1_mean"])
                row["tecr_reduction_pct"] = pct_reduction(baseline["tecr_mean"], row["tecr_mean"])
    order = {"clip_knn": 0, "heldout_gate": 1}
    return sorted(out, key=lambda row: (float(row["calibration_dataset_fraction"]), order.get(row["method"], 99)))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/openimages_10k_heldout_ultrastrict.yaml")
    parser.add_argument("--output-dir", default="outputs/openimages_10k_calibration_ratio")
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--ratios", default="0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--max-calibration-fraction", type=float, default=0.30)
    args = parser.parse_args()

    start = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_seed = int(args.seed_override) if args.seed_override is not None else int(cfg["seed"])
    ratios = [float(item.strip()) for item in args.ratios.split(",") if item.strip()]
    if any(r <= 0 or r > args.max_calibration_fraction for r in ratios):
        raise ValueError("All ratios must be in (0, max_calibration_fraction].")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    ap_tolerance = float(calibration_cfg.get("ap_tolerance", 0.0025))
    f1_tolerance = float(calibration_cfg.get("f1_tolerance", 0.0025))
    residual_powers = [float(x) for x in calibration_cfg.get("residual_powers", [0.0, 0.1, 0.2, 0.3, 0.4])]
    cutoffs = [float(x) for x in calibration_cfg.get("confidence_cutoffs", [0.2, 0.3, 0.4, 0.5, 0.6])]
    temperatures = [float(x) for x in calibration_cfg.get("confidence_temperatures", [0.01, 0.02, 0.05, 0.1])]

    features, labels, text_features, class_names = load_cached_arrays(cfg, Path(cfg["output_dir"]))
    (
        retrieval_features,
        retrieval_labels,
        calibration_pool_features,
        calibration_pool_labels,
        eval_features,
        eval_labels,
    ) = split_with_fixed_eval(features, labels, run_seed, retrieval_fraction, args.max_calibration_fraction)

    splits = build_splits(class_names, retrieval_labels, calibration_pool_labels, cfg["protocol"], run_seed)
    retrieval_logits_raw = retrieval_features @ text_features.T
    logit_mean = retrieval_logits_raw.mean(axis=0, keepdims=True)
    logit_std = retrieval_logits_raw.std(axis=0, keepdims=True) + 1e-8
    calibration_pool_logits = (calibration_pool_features @ text_features.T - logit_mean) / logit_std
    eval_logits = (eval_features @ text_features.T - logit_mean) / logit_std

    rows = []
    selected_rows = []
    total_images = features.shape[0]
    for split_idx, split in enumerate(splits):
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
        known_idx = [i for i, name in enumerate(class_names) if name not in split["emerging_labels"]]

        calibration_pool_known = 1.0 / (1.0 + np.exp(-calibration_pool_logits[:, known_idx].max(axis=1)))
        eval_known = 1.0 / (1.0 + np.exp(-eval_logits[:, known_idx].max(axis=1)))
        calibration_pool_knn = knn_score_matrix(
            retrieval_features,
            retrieval_labels,
            calibration_pool_features,
            emerging_idx,
            cfg["knn"],
        )
        eval_knn = knn_score_matrix(retrieval_features, retrieval_labels, eval_features, emerging_idx, cfg["knn"])
        y_eval = eval_labels[:, emerging_idx].max(axis=1).astype(int)
        eval_base_scores = aggregate_scores(eval_knn)

        for ratio in ratios:
            target_size = int(round(total_images * ratio))
            if target_size > calibration_pool_features.shape[0]:
                raise ValueError(f"Ratio {ratio} exceeds the fixed calibration pool.")
            subset_seed = run_seed + 1000 * split_idx + int(round(ratio * 10000))
            sub_idx = deterministic_subset(calibration_pool_features.shape[0], target_size, subset_seed)
            sub_labels_full = calibration_pool_labels[sub_idx]
            sub_known = calibration_pool_known[sub_idx]
            sub_knn = calibration_pool_knn[sub_idx]
            y_cal = sub_labels_full[:, emerging_idx].max(axis=1).astype(int)

            cal_base_scores = aggregate_scores(sub_knn)
            cal_threshold = best_f1_threshold(y_cal, cal_base_scores)["threshold"]
            cal_baseline = metrics_with_calibrated_threshold(
                y_cal,
                cal_base_scores,
                cal_threshold,
                sub_labels_full,
                tail_idx,
                emerging_idx,
            )
            eval_baseline = metrics_with_calibrated_threshold(
                y_eval,
                eval_base_scores,
                cal_threshold,
                eval_labels,
                tail_idx,
                emerging_idx,
            )
            rows.append({
                **eval_baseline,
                "method": "clip_knn",
                "split": split["name"],
                "seed": run_seed,
                "calibration_dataset_fraction": ratio,
                "num_calibration_images_used": int(sub_idx.size),
                "selection_status": "baseline",
                "residual_power": None,
                "confidence_cutoff": None,
                "confidence_temperature": None,
            })

            cal_grid_rows = []
            eval_grid_rows = []
            for residual_power in residual_powers:
                for cutoff in cutoffs:
                    for temperature in temperatures:
                        cal_scores_matrix = confidence_gate_scores(sub_knn, sub_known, residual_power, cutoff, temperature)
                        eval_scores_matrix = confidence_gate_scores(eval_knn, eval_known, residual_power, cutoff, temperature)
                        cal_scores = aggregate_scores(cal_scores_matrix)
                        eval_scores = aggregate_scores(eval_scores_matrix)
                        threshold = best_f1_threshold(y_cal, cal_scores)["threshold"]
                        cal_metrics = metrics_with_calibrated_threshold(
                            y_cal,
                            cal_scores,
                            threshold,
                            sub_labels_full,
                            tail_idx,
                            emerging_idx,
                        )
                        eval_metrics = metrics_with_calibrated_threshold(
                            y_eval,
                            eval_scores,
                            threshold,
                            eval_labels,
                            tail_idx,
                            emerging_idx,
                        )
                        cal_grid_rows.append({
                            **cal_metrics,
                            "method": "elta_confidence",
                            "split": split["name"],
                            "residual_power": residual_power,
                            "confidence_cutoff": cutoff,
                            "confidence_temperature": temperature,
                        })
                        eval_grid_rows.append({
                            **eval_metrics,
                            "method": "heldout_gate_pool",
                            "split": split["name"],
                            "residual_power": residual_power,
                            "confidence_cutoff": cutoff,
                            "confidence_temperature": temperature,
                        })

            selected = select_setting(cal_grid_rows, cal_baseline, ap_tolerance, f1_tolerance)
            selected_rows.append({
                "split": split["name"],
                "seed": run_seed,
                "calibration_dataset_fraction": ratio,
                "num_calibration_images_used": int(sub_idx.size),
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

            if selected["method"] == "clip_knn":
                heldout_eval = dict(eval_baseline)
            else:
                heldout_eval = next(
                    row for row in eval_grid_rows
                    if row["residual_power"] == selected["residual_power"]
                    and row["confidence_cutoff"] == selected["confidence_cutoff"]
                    and row["confidence_temperature"] == selected["confidence_temperature"]
                )
            rows.append({
                **heldout_eval,
                "method": "heldout_gate",
                "split": split["name"],
                "seed": run_seed,
                "calibration_dataset_fraction": ratio,
                "num_calibration_images_used": int(sub_idx.size),
                "selection_status": selected["selection_status"],
            })

    summary = summarize(rows)
    row_fields = [
        "method", "split", "seed", "calibration_dataset_fraction", "num_calibration_images_used",
        "selection_status", "residual_power", "confidence_cutoff", "confidence_temperature",
        "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    summary_fields = [
        "calibration_dataset_fraction", "method", "num_splits", "num_calibration_images_used_mean",
        "average_precision_mean", "average_precision_std",
        "auroc_mean", "auroc_std", "best_f1_mean", "best_f1_std", "tecr_mean", "tecr_std",
        "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct",
    ]
    selection_fields = [
        "split", "seed", "calibration_dataset_fraction", "num_calibration_images_used",
        "selection_status", "residual_power", "confidence_cutoff", "confidence_temperature",
        "calibration_ap", "calibration_f1", "calibration_tecr",
        "baseline_calibration_ap", "baseline_calibration_f1", "baseline_calibration_tecr",
    ]
    write_csv(output_dir / "calibration_ratio_rows.csv", rows, row_fields)
    write_csv(output_dir / "calibration_ratio_summary.csv", summary, summary_fields)
    write_csv(output_dir / "calibration_ratio_selected_settings.csv", selected_rows, selection_fields)

    report = [
        "# Open Images Calibration-Ratio Sensitivity",
        "",
        "Retrieval and evaluation splits are fixed using a 40% retrieval / 30% maximum calibration / 30% evaluation split. The task definition uses only retrieval plus the maximum calibration pool; evaluation labels are held out for final reporting only. Gate and threshold selection use 10%, 15%, 20%, 25%, or 30% of all images as the calibration subset.",
        "",
        "| Calibration ratio | Method | AP | F1 | TECR | TECR reduction |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {float(row['calibration_dataset_fraction']) * 100:.0f}% | {row['method']} | "
            f"{row['average_precision_mean']:.4f} | {row['best_f1_mean']:.4f} | "
            f"{row['tecr_mean']:.4f} | {row.get('tecr_reduction_pct', 0.0):.1f}% |"
        )
    (output_dir / "calibration_ratio_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    result = {
        "status": "calibration_ratio_sensitivity_complete",
        "seed": run_seed,
        "time_seconds": round(time.time() - start, 3),
        "output_dir": str(output_dir),
        "retrieval_fraction": retrieval_fraction,
        "max_calibration_fraction": args.max_calibration_fraction,
        "eval_fraction": 1.0 - retrieval_fraction - args.max_calibration_fraction,
        "ratios": ratios,
        "summary": summary,
    }
    (output_dir / "calibration_ratio_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
