from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import yaml

from run_openimages_heldout_calibration import (
    aggregate_scores,
    confidence_gate_scores,
    metrics_with_calibrated_threshold,
    select_setting,
    split_retrieval_calibration_eval,
)
from run_openimages_margin_gate import load_cached_arrays
from run_openimages_pilot import knn_score_matrix


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    from run_openimages_heldout_calibration import best_f1_threshold as _best

    return _best(y_true, scores)


def build_variant_splits(
    class_names: list[str],
    retrieval_labels: np.ndarray,
    calibration_labels: np.ndarray,
    protocol_cfg: dict,
    seed: int,
    emerging_count: int,
    tail_fraction: float,
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
    emerging_count = min(int(emerging_count), len(eligible))
    tail_count = max(1, int(round(len(eligible) * tail_fraction)))
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
            "tail_count": len(tail),
            "eligible_count": len(eligible),
        })
    return splits


def summarize(rows: list[dict]) -> list[dict]:
    by_variant = {}
    for row in rows:
        key = (row["variant"], row["tail_fraction"], row["emerging_count"], row["method"])
        by_variant.setdefault(key, []).append(row)
    out = []
    for (variant, tail_fraction, emerging_count, method), group in by_variant.items():
        item = {
            "variant": variant,
            "tail_fraction": tail_fraction,
            "emerging_count": emerging_count,
            "method": method,
            "num_splits": len(group),
            "tail_count_mean": float(np.mean([row["tail_count"] for row in group])),
        }
        for metric in ["average_precision", "best_f1", "tecr"]:
            vals = [row[metric] for row in group if row[metric] is not None]
            item[f"{metric}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    baseline_by_variant = {
        (row["variant"], row["tail_fraction"], row["emerging_count"]): row
        for row in out
        if row["method"] == "clip_knn"
    }
    for row in out:
        base = baseline_by_variant[(row["variant"], row["tail_fraction"], row["emerging_count"])]
        row["tecr_reduction_pct"] = 0.0 if row["method"] == "clip_knn" else 100.0 * (base["tecr_mean"] - row["tecr_mean"]) / base["tecr_mean"]
    return sorted(out, key=lambda row: (row["variant"], row["method"]))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_variant(
    cfg: dict,
    features: np.ndarray,
    labels: np.ndarray,
    text_features: np.ndarray,
    class_names: list[str],
    run_seed: int,
    tail_fraction: float,
    emerging_count: int,
) -> list[dict]:
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
    splits = build_variant_splits(class_names, retrieval_labels, calibration_labels, cfg["protocol"], run_seed, emerging_count, tail_fraction)
    retrieval_logits_raw = retrieval_features @ text_features.T
    logit_mean = retrieval_logits_raw.mean(axis=0, keepdims=True)
    logit_std = retrieval_logits_raw.std(axis=0, keepdims=True) + 1e-8
    calibration_logits = (calibration_features @ text_features.T - logit_mean) / logit_std
    eval_logits = (eval_features @ text_features.T - logit_mean) / logit_std

    rows = []
    variant = f"tail{int(round(tail_fraction * 100))}_emerging{emerging_count}"
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
        threshold = best_f1_threshold(y_cal, cal_base_scores)["threshold"]
        cal_baseline = metrics_with_calibrated_threshold(y_cal, cal_base_scores, threshold, calibration_labels, tail_idx, emerging_idx)
        eval_baseline = metrics_with_calibrated_threshold(y_eval, eval_base_scores, threshold, eval_labels, tail_idx, emerging_idx)
        rows.append({
            **eval_baseline,
            "variant": variant,
            "tail_fraction": tail_fraction,
            "emerging_count": emerging_count,
            "method": "clip_knn",
            "split": split["name"],
            "tail_count": split["tail_count"],
        })

        cal_grid = []
        eval_grid = []
        for residual_power in residual_powers:
            for cutoff in cutoffs:
                for temperature in temperatures:
                    cal_scores = aggregate_scores(confidence_gate_scores(calibration_knn, calibration_known, residual_power, cutoff, temperature))
                    eval_scores = aggregate_scores(confidence_gate_scores(eval_knn, eval_known, residual_power, cutoff, temperature))
                    threshold = best_f1_threshold(y_cal, cal_scores)["threshold"]
                    cal_metrics = metrics_with_calibrated_threshold(y_cal, cal_scores, threshold, calibration_labels, tail_idx, emerging_idx)
                    eval_metrics = metrics_with_calibrated_threshold(y_eval, eval_scores, threshold, eval_labels, tail_idx, emerging_idx)
                    cal_grid.append({
                        **cal_metrics,
                        "method": "elta_confidence",
                        "residual_power": residual_power,
                        "confidence_cutoff": cutoff,
                        "confidence_temperature": temperature,
                    })
                    eval_grid.append({
                        **eval_metrics,
                        "residual_power": residual_power,
                        "confidence_cutoff": cutoff,
                        "confidence_temperature": temperature,
                    })
        selected = select_setting(cal_grid, cal_baseline, ap_tolerance, f1_tolerance)
        if selected["method"] == "clip_knn":
            held = dict(eval_baseline)
        else:
            held = next(
                row for row in eval_grid
                if row["residual_power"] == selected["residual_power"]
                and row["confidence_cutoff"] == selected["confidence_cutoff"]
                and row["confidence_temperature"] == selected["confidence_temperature"]
            )
        rows.append({
            **held,
            "variant": variant,
            "tail_fraction": tail_fraction,
            "emerging_count": emerging_count,
            "method": "heldout_gate",
            "split": split["name"],
            "tail_count": split["tail_count"],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/openimages_10k_heldout_ultrastrict.yaml")
    parser.add_argument("--output-dir", default="outputs/tecr_robustness_openimages")
    parser.add_argument("--seed-override", type=int)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_seed = int(args.seed_override) if args.seed_override is not None else int(cfg["seed"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    features, labels, text_features, class_names = load_cached_arrays(cfg, Path(cfg["output_dir"]))

    rows = []
    for tail_fraction, emerging_count in [
        (0.10, 15),
        (0.20, 15),
        (0.30, 15),
        (0.20, 10),
        (0.20, 20),
    ]:
        rows.extend(run_variant(cfg, features, labels, text_features, class_names, run_seed, tail_fraction, emerging_count))
    summary = summarize(rows)
    row_fields = ["variant", "tail_fraction", "emerging_count", "method", "split", "tail_count", "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr"]
    summary_fields = [
        "variant", "tail_fraction", "emerging_count", "method", "num_splits", "tail_count_mean",
        "average_precision_mean", "average_precision_std", "best_f1_mean", "best_f1_std",
        "tecr_mean", "tecr_std", "tecr_reduction_pct",
    ]
    write_csv(output_dir / "tecr_robustness_rows.csv", rows, row_fields)
    write_csv(output_dir / "tecr_robustness_summary.csv", summary, summary_fields)
    lines = [
        "# TECR Robustness",
        "",
        "| Variant | Method | AP | F1 | TECR | TECR reduction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['variant']} | {row['method']} | {row['average_precision_mean']:.4f} | "
            f"{row['best_f1_mean']:.4f} | {row['tecr_mean']:.4f} | {row['tecr_reduction_pct']:.1f}% |"
        )
    (output_dir / "tecr_robustness_report.md").write_text("\n".join(lines), encoding="utf-8")
    result = {"status": "tecr_robustness_complete", "output_dir": str(output_dir), "summary": summary}
    (output_dir / "tecr_robustness_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
