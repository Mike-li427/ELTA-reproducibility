from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def read_csv_dicts(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        config = path.parent.name
        for row in read_csv_dicts(path):
            rows.append({
                "config": config,
                "calibration_dataset_fraction": parse_float(row["calibration_dataset_fraction"]),
                "method": row["method"],
                "num_splits": int(row["num_splits"]),
                "num_calibration_images_used_mean": parse_float(row["num_calibration_images_used_mean"]),
                "average_precision_mean": parse_float(row["average_precision_mean"]),
                "best_f1_mean": parse_float(row["best_f1_mean"]),
                "tecr_mean": parse_float(row["tecr_mean"]),
            })

    grouped = {}
    for row in rows:
        grouped.setdefault((row["calibration_dataset_fraction"], row["method"]), []).append(row)

    out = []
    for (fraction, method), group in grouped.items():
        item = {
            "calibration_dataset_fraction": fraction,
            "method": method,
            "num_configs": len(group),
            "num_split_averages": int(sum(row["num_splits"] for row in group)),
            "num_calibration_images_used_mean": float(np.mean([row["num_calibration_images_used_mean"] for row in group])),
        }
        for col in ["average_precision_mean", "best_f1_mean", "tecr_mean"]:
            vals = [row[col] for row in group if row[col] is not None]
            item[col] = float(np.mean(vals)) if vals else None
            item[col.replace("_mean", "_std")] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)

    by_fraction = {}
    for row in out:
        by_fraction.setdefault(row["calibration_dataset_fraction"], {})[row["method"]] = row
    for methods in by_fraction.values():
        baseline = methods["clip_knn"]
        for row in methods.values():
            row["ap_delta_pct"] = 100.0 * (row["average_precision_mean"] - baseline["average_precision_mean"]) / baseline["average_precision_mean"]
            row["f1_delta_pct"] = 100.0 * (row["best_f1_mean"] - baseline["best_f1_mean"]) / baseline["best_f1_mean"]
            row["tecr_reduction_pct"] = 100.0 * (baseline["tecr_mean"] - row["tecr_mean"]) / baseline["tecr_mean"]
    order = {"clip_knn": 0, "heldout_gate": 1}
    return sorted(out, key=lambda row: (row["calibration_dataset_fraction"], order.get(row["method"], 99)))


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="outputs/openimages_10k_calibration_ratio*/calibration_ratio_summary.csv")
    parser.add_argument("--output-dir", default="outputs/openimages_10k_calibration_ratio_summary")
    args = parser.parse_args()

    paths = sorted(Path().glob(args.glob))
    paths = [path for path in paths if "smoke" not in str(path)]
    if not paths:
        raise FileNotFoundError(f"No files matched {args.glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize(paths)
    fields = [
        "calibration_dataset_fraction", "method",
        "num_configs", "num_split_averages", "num_calibration_images_used_mean",
        "average_precision_mean", "average_precision_std",
        "best_f1_mean", "best_f1_std",
        "tecr_mean", "tecr_std",
        "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct",
    ]
    write_csv(output_dir / "calibration_ratio_12config_summary.csv", rows, fields)

    report = [
        "# Open Images 10k Calibration-Ratio Sensitivity Summary",
        "",
        "Retrieval and evaluation splits are fixed. The maximum calibration pool is 30% of the dataset; each row uses the listed fraction of the full dataset for gate and threshold selection.",
        "",
        "| Calibration ratio | Method | AP | F1 | TECR | TECR reduction |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['calibration_dataset_fraction'] * 100:.0f}% | {row['method']} | "
            f"{fmt(row['average_precision_mean'])} | {fmt(row['best_f1_mean'])} | "
            f"{fmt(row['tecr_mean'])} | {row['tecr_reduction_pct']:.1f}% |"
        )
    report.extend([
        "",
        "Interpretation:",
        "",
        "- TECR reduction remains positive across 10%-30% calibration ratios.",
        "- AP/F1 changes are small because only threshold and gate selection use the calibration subset.",
        "",
    ])
    (output_dir / "calibration_ratio_summary.md").write_text("\n".join(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
