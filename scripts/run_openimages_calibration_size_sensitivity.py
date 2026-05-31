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
    split_retrieval_calibration_eval,
)
from run_openimages_margin_gate import load_cached_arrays
from run_openimages_pilot import knn_score_matrix


def deterministic_subset(n: int, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    size = max(1, int(round(n * fraction)))
    return np.sort(rng.choice(n, size=size, replace=False))


def summarize(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["calibration_fraction_used"], row["method"]), []).append(row)
    out = []
    for (fraction, method), group in grouped.items():
        item = {"calibration_fraction_used": fraction, "method": method, "num_splits": len(group)}
        for key in ["average_precision", "auroc", "best_f1", "tecr"]:
            vals = [row[key] for row in group if row[key] is not None]
            item[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)

    by_fraction = {}
    for row in out:
        by_fraction.setdefault(row["calibration_fraction_used"], {})[row["method"]] = row
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
    return sorted(out, key=lambda row: (float(row["calibration_fraction_used"]), order.get(row["method"], 99), row["method"]))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/openimages_10k_heldout_ultrastrict.yaml")
    parser.add_argument("--output-dir", default="outputs/openimages_10k_calibration_size")
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--fractions", default="0.125,0.25,0.5,1.0")
    args = parser.parse_args()

    start = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_seed = int(args.seed_override) if args.seed_override is not None else int(cfg["seed"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    fractions = [float(item.strip()) for item in args.fractions.split(",") if item.strip()]
    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))
    ap_tolerance = float(calibration_cfg.get("ap_tolerance", 0.0025))
    f1_tolerance = float(calibration_cfg.get("f1_tolerance", 0.0025))
    residual_powers = [float(x) for x in calibration_cfg.get("residual_powers", [0.0, 0.1, 0.2, 0.3, 0.4])]
    cutoffs = [float(x) for x in calibration_cfg.get("confidence_cutoffs", [0.2, 0.3, 0.4, 0.5, 0.6])]
    temperatures = [float(x) for x in calibration_cfg.get("confidence_temperatures", [0.01, 0.02, 0.05, 0.1])]

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
    for split_idx, split in enumerate(splits):
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
        known_idx = [i for i, name in enumerate(class_names) if name not in split["emerging_labels"]]

        calibration_known = 1.0 / (1.0 + np.exp(-calibration_logits[:, known_idx].max(axis=1)))
        eval_known = 1.0 / (1.0 + np.exp(-eval_logits[:, known_idx].max(axis=1)))
        calibration_knn = knn_score_matrix(retrieval_features, retrieval_labels, calibration_features, emerging_idx, cfg["knn"])
        eval_knn = knn_score_matrix(retrieval_features, retrieval_labels, eval_features, emerging_idx, cfg["knn"])
        y_eval = eval_labels[:, emerging_idx].max(axis=1).astype(int)
        eval_base_scores = aggregate_scores(eval_knn)

        for fraction in fractions:
            subset_seed = run_seed + 1000 * split_idx + int(round(fraction * 10000))
            sub_idx = deterministic_subset(calibration_features.shape[0], fraction, subset_seed)
            sub_labels_full = calibration_labels[sub_idx]
            sub_known = calibration_known[sub_idx]
            sub_knn = calibration_knn[sub_idx]
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
                "calibration_fraction_used": fraction,
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
                "calibration_fraction_used": fraction,
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
                "calibration_fraction_used": fraction,
                "num_calibration_images_used": int(sub_idx.size),
                "selection_status": selected["selection_status"],
            })

    summary = summarize(rows)
    row_fields = [
        "method", "split", "seed", "calibration_fraction_used", "num_calibration_images_used",
        "selection_status", "residual_power", "confidence_cutoff", "confidence_temperature",
        "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    summary_fields = [
        "calibration_fraction_used", "method", "num_splits",
        "average_precision_mean", "average_precision_std",
        "auroc_mean", "auroc_std", "best_f1_mean", "best_f1_std", "tecr_mean", "tecr_std",
        "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct",
    ]
    selection_fields = [
        "split", "seed", "calibration_fraction_used", "num_calibration_images_used",
        "selection_status", "residual_power", "confidence_cutoff", "confidence_temperature",
        "calibration_ap", "calibration_f1", "calibration_tecr",
        "baseline_calibration_ap", "baseline_calibration_f1", "baseline_calibration_tecr",
    ]
    write_csv(output_dir / "calibration_size_rows.csv", rows, row_fields)
    write_csv(output_dir / "calibration_size_summary.csv", summary, summary_fields)
    write_csv(output_dir / "calibration_size_selected_settings.csv", selected_rows, selection_fields)

    report = [
        "# Open Images Calibration-Size Sensitivity",
        "",
        "Date: 2026-05-23",
        "",
        "Retrieval/eval splits are fixed. Only the calibration subset used for threshold and gate selection changes.",
        "",
        "| Calibration pool used | Method | AP | F1 | TECR | TECR reduction |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {float(row['calibration_fraction_used']) * 100:.1f}% | {row['method']} | "
            f"{row['average_precision_mean']:.4f} | {row['best_f1_mean']:.4f} | "
            f"{row['tecr_mean']:.4f} | {row.get('tecr_reduction_pct', 0.0):.1f}% |"
        )
    report.extend(["", f"TIME_ESTIMATE: {max(1, int(time.time() - start))} seconds", ""])
    (output_dir / "calibration_size_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "status": "calibration_size_sensitivity_complete",
        "seed": run_seed,
        "time_seconds": round(time.time() - start, 3),
        "output_dir": str(output_dir),
        "summary": summary,
    }
    (output_dir / "calibration_size_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TIME_ESTIMATE:", max(1, int(time.time() - start)))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
