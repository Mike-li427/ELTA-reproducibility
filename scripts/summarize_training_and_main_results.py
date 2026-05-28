from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

import numpy as np


METRIC_COLUMNS = [
    "average_precision_mean",
    "auroc_mean",
    "best_f1_mean",
    "tecr_mean",
]


DISPLAY_NAMES = {
    "linear_bce_global_threshold": "Linear BCE, global threshold",
    "linear_bce_class_thresholds": "Linear BCE, class thresholds",
    "class_balanced_bce_global_threshold": "Class-balanced BCE, global threshold",
    "class_balanced_bce_class_thresholds": "Class-balanced BCE, class thresholds",
    "asl_global_threshold": "ASL (ICCV 2021), global threshold",
    "asl_class_thresholds": "ASL (ICCV 2021), class thresholds",
    "db_loss_global_threshold": "DBLoss (ECCV 2020), global threshold",
    "db_loss_class_thresholds": "DBLoss (ECCV 2020), class thresholds",
    "logit_adjusted_bce_global_threshold": "Logit-adjusted BCE, global threshold",
    "logit_adjusted_bce_class_thresholds": "Logit-adjusted BCE, class thresholds",
    "balance_mix_feature_global_threshold": "BalanceMix-style feature mixup, global threshold",
    "balance_mix_feature_class_thresholds": "BalanceMix-style feature mixup, class thresholds",
    "text_init_bce_global_threshold": "Text-initialized BCE, global threshold",
    "text_init_bce_class_thresholds": "Text-initialized BCE, class thresholds",
    "text_init_class_balanced_bce_global_threshold": "Text-initialized class-balanced BCE, global threshold",
    "text_init_class_balanced_bce_class_thresholds": "Text-initialized class-balanced BCE, class thresholds",
    "text_init_asl_global_threshold": "Text-initialized ASL (ICCV 2021), global threshold",
    "text_init_asl_class_thresholds": "Text-initialized ASL (ICCV 2021), class thresholds",
    "text_init_db_loss_global_threshold": "Text-initialized DBLoss (ECCV 2020), global threshold",
    "text_init_db_loss_class_thresholds": "Text-initialized DBLoss (ECCV 2020), class thresholds",
    "text_init_balance_mix_feature_global_threshold": "Text-initialized BalanceMix-style feature mixup, global threshold",
    "text_init_balance_mix_feature_class_thresholds": "Text-initialized BalanceMix-style feature mixup, class thresholds",
    "text_init_logit_adjusted_bce_global_threshold": "Text-initialized logit-adjusted BCE, global threshold",
    "text_init_logit_adjusted_bce_class_thresholds": "Text-initialized logit-adjusted BCE, class thresholds",
    "clip_knn_global_threshold": "CLIP+kNN, global threshold",
    "clip_knn_class_thresholds": "CLIP+kNN, class thresholds",
    "clip_knn_isotonic_global_threshold": "CLIP+kNN isotonic, global threshold",
    "clip_knn_isotonic_class_thresholds": "CLIP+kNN isotonic, class thresholds",
    "clip_knn_platt_global_threshold": "CLIP+kNN Platt, global threshold",
    "clip_knn_platt_class_thresholds": "CLIP+kNN Platt, class thresholds",
    "clip_zeroshot_global_threshold": "CLIP zero-shot, global threshold",
    "clip_zeroshot_class_thresholds": "CLIP zero-shot, class thresholds",
    "heldout_gate_global_threshold": "Held-out gate, global threshold",
    "heldout_gate_class_thresholds": "Held-out gate, class thresholds",
}


def read_csv_dicts(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def config_name(path: Path) -> str:
    return path.parent.name


def canonical_run_key(config: str) -> str:
    key = config
    replacements = [
        ("openimages_10k_published_loss", ""),
        ("openimages_10k_training_baselines", ""),
        ("openimages_10k_heldout_ultrastrict", ""),
        ("coco_published_loss", ""),
        ("coco_training_baselines", ""),
        ("coco_heldout_ultrastrict", ""),
    ]
    for prefix, replacement in replacements:
        if key.startswith(prefix):
            key = replacement + key[len(prefix):]
            break
    key = key.strip("_")
    return key or "base"


def aggregate_summary_files(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    per_config_rows = []
    for path in paths:
        config = config_name(path)
        for row in read_csv_dicts(path):
            item = {"config": config, "run_key": canonical_run_key(config), "method": row["method"]}
            for col in METRIC_COLUMNS:
                item[col] = parse_float(row.get(col))
            per_config_rows.append(item)

    methods = sorted({row["method"] for row in per_config_rows})
    summary_rows = []
    for method in methods:
        group = [row for row in per_config_rows if row["method"] == method]
        out = {"method": method, "display_name": DISPLAY_NAMES.get(method, method), "num_configs": len(group)}
        for col in METRIC_COLUMNS:
            vals = [row[col] for row in group if row[col] is not None]
            out[col] = float(np.mean(vals)) if vals else None
            out[col.replace("_mean", "_std")] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(out)
    return per_config_rows, sorted(summary_rows, key=lambda row: row["method"])


def add_deltas(rows: list[dict], baseline_method: str) -> None:
    baseline = next(row for row in rows if row["method"] == baseline_method)
    for row in rows:
        row["ap_delta_pct"] = 100.0 * (row["average_precision_mean"] - baseline["average_precision_mean"]) / baseline["average_precision_mean"]
        row["f1_delta_pct"] = 100.0 * (row["best_f1_mean"] - baseline["best_f1_mean"]) / baseline["best_f1_mean"]
        row["tecr_reduction_pct"] = 100.0 * (baseline["tecr_mean"] - row["tecr_mean"]) / baseline["tecr_mean"]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def fmt_pct(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}%"


def paired_delta_stats(per_config_rows: list[dict], baseline_method: str, method: str, metric_col: str) -> dict:
    by_config = {}
    for row in per_config_rows:
        by_config.setdefault(row["run_key"], {})[row["method"]] = row
    deltas = []
    for methods in by_config.values():
        if baseline_method in methods and method in methods:
            base = methods[baseline_method][metric_col]
            value = methods[method][metric_col]
            if base is not None and value is not None:
                deltas.append(value - base)
    if not deltas:
        return {"num_pairs": 0, "delta_mean": None, "delta_std": None, "win_rate": None}
    arr = np.asarray(deltas, dtype=np.float64)
    if metric_col == "tecr_mean":
        wins = arr < 0.0
    else:
        wins = arr > 0.0
    return {
        "num_pairs": int(arr.size),
        "delta_mean": float(arr.mean()),
        "delta_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "win_rate": float(wins.mean()),
    }


def write_report(
    path: Path,
    dataset_name: str,
    training_summary: list[dict],
    main_summary: list[dict],
    training_per_config: list[dict],
    main_per_config: list[dict],
) -> None:
    best_training_ap = max(row["average_precision_mean"] for row in training_summary)
    eligible_training = [
        row for row in training_summary
        if row["average_precision_mean"] >= best_training_ap * 0.99
        and row["method"].endswith("class_thresholds")
    ]
    best_training = min(eligible_training or training_summary, key=lambda row: row["tecr_mean"])
    clip_knn = next(row for row in main_summary if row["method"] == "clip_knn_global_threshold")
    isotonic = next(row for row in main_summary if row["method"] == "clip_knn_isotonic_global_threshold")
    heldout_global = next(row for row in main_summary if row["method"] == "heldout_gate_global_threshold")
    heldout_class = next(row for row in main_summary if row["method"] == "heldout_gate_class_thresholds")

    comparison_rows = [
        best_training,
        clip_knn,
        isotonic,
        heldout_global,
        heldout_class,
    ]

    lines = [
        f"# {dataset_name} Training and Main-Method Comparison",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "Training baselines are frozen-CLIP feature heads trained on the retrieval split and selected/evaluated under the same calibration/evaluation protocol.",
        "",
        "## Training Baselines",
        "",
        "| Method | n | AP | F1 | TECR | TECR reduction vs Linear BCE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in training_summary:
        lines.append(
            f"| {row['display_name']} | {row['num_configs']} | {fmt(row['average_precision_mean'])} | "
            f"{fmt(row['best_f1_mean'])} | {fmt(row['tecr_mean'])} | {fmt_pct(row.get('tecr_reduction_pct'))} |"
        )

    lines.extend([
        "",
        "## Paper-Facing Comparison",
        "",
        "| Method | AP | F1 | TECR | TECR reduction vs CLIP+kNN |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in comparison_rows:
        tecr_reduction_vs_clip = 100.0 * (clip_knn["tecr_mean"] - row["tecr_mean"]) / clip_knn["tecr_mean"]
        lines.append(
            f"| {row['display_name']} | {fmt(row['average_precision_mean'])} | "
            f"{fmt(row['best_f1_mean'])} | {fmt(row['tecr_mean'])} | {fmt_pct(tecr_reduction_vs_clip)} |"
        )

    heldout_tecr = paired_delta_stats(main_per_config, "clip_knn_global_threshold", "heldout_gate_global_threshold", "tecr_mean")
    heldout_ap = paired_delta_stats(main_per_config, "clip_knn_global_threshold", "heldout_gate_global_threshold", "average_precision_mean")
    heldout_f1 = paired_delta_stats(main_per_config, "clip_knn_global_threshold", "heldout_gate_global_threshold", "best_f1_mean")
    train_tecr = paired_delta_stats(training_per_config + main_per_config, best_training["method"], "heldout_gate_global_threshold", "tecr_mean")
    lines.extend([
        "",
        "## Paired Diagnostics",
        "",
        "| Comparison | Metric | n | Mean delta | Win rate |",
        "|---|---|---:|---:|---:|",
        f"| Held-out gate vs CLIP+kNN | TECR | {heldout_tecr['num_pairs']} | {fmt(heldout_tecr['delta_mean'])} | {fmt_pct(100.0 * heldout_tecr['win_rate'] if heldout_tecr['win_rate'] is not None else None)} |",
        f"| Held-out gate vs CLIP+kNN | AP | {heldout_ap['num_pairs']} | {fmt(heldout_ap['delta_mean'])} | {fmt_pct(100.0 * heldout_ap['win_rate'] if heldout_ap['win_rate'] is not None else None)} |",
        f"| Held-out gate vs CLIP+kNN | F1 | {heldout_f1['num_pairs']} | {fmt(heldout_f1['delta_mean'])} | {fmt_pct(100.0 * heldout_f1['win_rate'] if heldout_f1['win_rate'] is not None else None)} |",
        f"| Held-out gate vs best training baseline | TECR | {train_tecr['num_pairs']} | {fmt(train_tecr['delta_mean'])} | {fmt_pct(100.0 * train_tecr['win_rate'] if train_tecr['win_rate'] is not None else None)} |",
        "",
        "Notes:",
        "",
        "- The BalanceMix row is a feature-space BalanceMix-style baseline, not an official image-space BalanceMix reproduction.",
        "- Use the training-baseline table as auxiliary horizontal evidence, not as a claim that all representation-learning MLC methods have been beaten.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--training-glob", required=True)
    parser.add_argument("--main-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_paths = sorted(Path().glob(args.training_glob))
    main_paths = sorted(Path().glob(args.main_glob))
    if not training_paths:
        raise FileNotFoundError(f"No training summary files matched {args.training_glob}")
    if not main_paths:
        raise FileNotFoundError(f"No main summary files matched {args.main_glob}")

    training_per_config, training_summary = aggregate_summary_files(training_paths)
    main_per_config, main_summary = aggregate_summary_files(main_paths)
    add_deltas(training_summary, "linear_bce_global_threshold")
    add_deltas(main_summary, "clip_knn_global_threshold")

    fields = [
        "method", "display_name", "num_configs",
        "average_precision_mean", "average_precision_std",
        "auroc_mean", "auroc_std",
        "best_f1_mean", "best_f1_std",
        "tecr_mean", "tecr_std",
        "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct",
    ]
    write_csv(output_dir / "training_baseline_12config_summary.csv", training_summary, fields)
    write_csv(output_dir / "main_method_12config_summary.csv", main_summary, fields)
    write_report(
        output_dir / "training_and_main_comparison.md",
        args.dataset_name,
        training_summary,
        main_summary,
        training_per_config,
        main_per_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
