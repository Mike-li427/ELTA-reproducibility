from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


METRIC_COLUMNS = ("average_precision_mean", "auroc_mean", "best_f1_mean", "tecr_mean")
GROUP_ORDER = ("head", "mid", "tail")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def mean(values: list[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return float(np.mean(kept)) if kept else None


def stdev(values: list[float | None]) -> float:
    kept = [float(value) for value in values if value is not None]
    return float(np.std(kept, ddof=1)) if len(kept) > 1 else 0.0


def add_deltas(rows: list[dict], baseline_method: str) -> None:
    baseline = next(row for row in rows if row["method"] == baseline_method)
    for row in rows:
        row["ap_delta_pct"] = 100.0 * (row["average_precision_mean"] - baseline["average_precision_mean"]) / baseline["average_precision_mean"]
        row["f1_delta_pct"] = 100.0 * (row["best_f1_mean"] - baseline["best_f1_mean"]) / baseline["best_f1_mean"]
        row["tecr_reduction_pct"] = 100.0 * (baseline["tecr_mean"] - row["tecr_mean"]) / baseline["tecr_mean"]


def collect_method_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        config = path.parent.name
        for row in read_rows(path):
            item = {"config": config, "method": row["method"]}
            for column in METRIC_COLUMNS:
                item[column] = parse_float(row.get(column))
            rows.append(item)
    return rows


def summarize_methods(per_config_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for method in sorted({row["method"] for row in per_config_rows}):
        group = [row for row in per_config_rows if row["method"] == method]
        item = {"method": method, "num_configs": len(group)}
        for column in METRIC_COLUMNS:
            values = [row[column] for row in group]
            item[column] = mean(values)
            item[column.replace("_mean", "_std")] = stdev(values)
        out.append(item)
    return out


def main_summary_from_calibrated(calibrated_rows: list[dict]) -> list[dict]:
    method_order = [
        "clip_knn_global_threshold",
        "clip_knn_class_thresholds",
        "clip_knn_isotonic_global_threshold",
        "heldout_gate_global_threshold",
        "heldout_gate_class_thresholds",
    ]
    out: list[dict] = []
    for row in calibrated_rows:
        if row["method"] in method_order:
            out.append(
                {
                    "method": row["method"],
                    "num_configs": row["num_configs"],
                    "ap_mean": row["average_precision_mean"],
                    "ap_std": row["average_precision_std"],
                    "f1_mean": row["best_f1_mean"],
                    "f1_std": row["best_f1_std"],
                    "tecr_mean": row["tecr_mean"],
                    "tecr_std": row["tecr_std"],
                    "tecr_reduction_pct": row["tecr_reduction_pct"],
                }
            )
    return sorted(out, key=lambda row: method_order.index(row["method"]))


def collect_frequency_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        config = path.parent.name
        for row in read_rows(path):
            item = {"config": config, "frequency_group": row["frequency_group"]}
            for column in [
                "num_splits",
                "num_emerging_labels_mean",
                "clip_knn_ap_mean",
                "heldout_ap_mean",
                "ap_delta",
                "clip_knn_f1_mean",
                "heldout_f1_mean",
                "f1_delta",
                "clip_knn_tecr_mean",
                "heldout_tecr_mean",
                "tecr_reduction_pct",
            ]:
                item[column] = parse_float(row.get(column))
            rows.append(item)
    return rows


def summarize_frequency(per_config_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for group_name in GROUP_ORDER:
        group = [row for row in per_config_rows if row["frequency_group"] == group_name]
        clip_tecr = mean([row["clip_knn_tecr_mean"] for row in group])
        held_tecr = mean([row["heldout_tecr_mean"] for row in group])
        clip_ap = mean([row["clip_knn_ap_mean"] for row in group])
        held_ap = mean([row["heldout_ap_mean"] for row in group])
        clip_f1 = mean([row["clip_knn_f1_mean"] for row in group])
        held_f1 = mean([row["heldout_f1_mean"] for row in group])
        tecr_delta = None if clip_tecr is None or held_tecr is None else held_tecr - clip_tecr
        out.append(
            {
                "frequency_group": group_name,
                "num_configs": len(group),
                "num_emerging_labels_mean": mean([row["num_emerging_labels_mean"] for row in group]),
                "clip_knn_ap_mean": clip_ap,
                "heldout_ap_mean": held_ap,
                "ap_delta": None if clip_ap is None or held_ap is None else held_ap - clip_ap,
                "clip_knn_f1_mean": clip_f1,
                "heldout_f1_mean": held_f1,
                "f1_delta": None if clip_f1 is None or held_f1 is None else held_f1 - clip_f1,
                "clip_knn_tecr_mean": clip_tecr,
                "heldout_tecr_mean": held_tecr,
                "tecr_delta": tecr_delta,
                "tecr_reduction_pct": None if clip_tecr in (None, 0.0) or held_tecr is None else 100.0 * (clip_tecr - held_tecr) / clip_tecr,
                "tecr_reduction_pct_mean_across_configs": mean([row["tecr_reduction_pct"] for row in group]),
                "tecr_delta_std": stdev(
                    [
                        None
                        if row["clip_knn_tecr_mean"] is None or row["heldout_tecr_mean"] is None
                        else row["heldout_tecr_mean"] - row["clip_knn_tecr_mean"]
                        for row in group
                    ]
                ),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="outputs/nuswide_heldout_ultrastrict*/")
    parser.add_argument("--output-dir", default="outputs/nuswide_10config_summary")
    args = parser.parse_args()

    run_dirs = sorted(path for path in Path().glob(args.glob) if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No NUS-WIDE run directories matched {args.glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibrated = summarize_methods(collect_method_rows([path / "calibrated_baseline_summary.csv" for path in run_dirs]))
    add_deltas(calibrated, "clip_knn_global_threshold")
    main_summary = main_summary_from_calibrated(calibrated)

    training = summarize_methods(collect_method_rows([path / "training_baseline_summary.csv" for path in run_dirs]))
    add_deltas(training, "asl_global_threshold")

    asl_gate = summarize_methods(collect_method_rows([path / "asl_gate_summary.csv" for path in run_dirs]))
    add_deltas(asl_gate, "asl_global_threshold")

    gate_ablation = summarize_methods(collect_method_rows([path / "gate_ablation_summary.csv" for path in run_dirs]))
    add_deltas(gate_ablation, "clip_knn")

    frequency_rows = collect_frequency_rows([path / "heldout_frequency_group_summary.csv" for path in run_dirs])
    frequency_summary = summarize_frequency(frequency_rows)

    summary_fields = [
        "method",
        "num_configs",
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
    write_rows(
        output_dir / "main_summary_10config.csv",
        main_summary,
        ["method", "num_configs", "ap_mean", "ap_std", "f1_mean", "f1_std", "tecr_mean", "tecr_std", "tecr_reduction_pct"],
    )
    write_rows(output_dir / "calibrated_baseline_10config_summary.csv", calibrated, summary_fields)
    write_rows(output_dir / "training_baseline_10config_summary.csv", training, summary_fields)
    write_rows(output_dir / "asl_gate_10config_summary.csv", asl_gate, summary_fields)
    write_rows(output_dir / "gate_ablation_10config_summary.csv", gate_ablation, summary_fields)
    write_rows(
        output_dir / "frequency_group_10config_summary.csv",
        frequency_summary,
        [
            "frequency_group",
            "num_configs",
            "num_emerging_labels_mean",
            "clip_knn_ap_mean",
            "heldout_ap_mean",
            "ap_delta",
            "clip_knn_f1_mean",
            "heldout_f1_mean",
            "f1_delta",
            "clip_knn_tecr_mean",
            "heldout_tecr_mean",
            "tecr_delta",
            "tecr_reduction_pct",
            "tecr_reduction_pct_mean_across_configs",
            "tecr_delta_std",
        ],
    )

    print(f"Wrote NUS-WIDE 10-config summaries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
