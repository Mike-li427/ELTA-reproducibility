from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_csv_dicts(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


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
                "calibration_fraction_used": parse_float(row["calibration_fraction_used"]),
                "method": row["method"],
                "num_splits": int(row["num_splits"]),
                "average_precision_mean": parse_float(row["average_precision_mean"]),
                "best_f1_mean": parse_float(row["best_f1_mean"]),
                "tecr_mean": parse_float(row["tecr_mean"]),
            })

    grouped = {}
    for row in rows:
        grouped.setdefault((row["calibration_fraction_used"], row["method"]), []).append(row)

    out = []
    for (fraction, method), group in grouped.items():
        item = {
            "calibration_fraction_used": fraction,
            "effective_dataset_fraction": 0.2 * fraction,
            "method": method,
            "num_configs": len(group),
            "num_split_averages": int(sum(row["num_splits"] for row in group)),
        }
        for col in ["average_precision_mean", "best_f1_mean", "tecr_mean"]:
            vals = [row[col] for row in group if row[col] is not None]
            item[col] = float(np.mean(vals)) if vals else None
            item[col.replace("_mean", "_std")] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)

    by_fraction = {}
    for row in out:
        by_fraction.setdefault(row["calibration_fraction_used"], {})[row["method"]] = row
    for methods in by_fraction.values():
        baseline = methods["clip_knn"]
        for row in methods.values():
            row["ap_delta_pct"] = 100.0 * (row["average_precision_mean"] - baseline["average_precision_mean"]) / baseline["average_precision_mean"]
            row["f1_delta_pct"] = 100.0 * (row["best_f1_mean"] - baseline["best_f1_mean"]) / baseline["best_f1_mean"]
            row["tecr_reduction_pct"] = 100.0 * (baseline["tecr_mean"] - row["tecr_mean"]) / baseline["tecr_mean"]
    order = {"clip_knn": 0, "heldout_gate": 1}
    return sorted(out, key=lambda row: (row["calibration_fraction_used"], order.get(row["method"], 99)))


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="outputs/openimages_10k_calibration_size*/calibration_size_summary.csv")
    parser.add_argument("--output-dir", default="outputs/openimages_10k_calibration_size_summary")
    args = parser.parse_args()

    paths = sorted(Path().glob(args.glob))
    paths = [path for path in paths if "smoke" not in str(path)]
    if not paths:
        raise FileNotFoundError(f"No files matched {args.glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize(paths)
    fields = [
        "calibration_fraction_used", "effective_dataset_fraction", "method",
        "num_configs", "num_split_averages",
        "average_precision_mean", "average_precision_std",
        "best_f1_mean", "best_f1_std",
        "tecr_mean", "tecr_std",
        "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct",
    ]
    write_csv(output_dir / "calibration_size_12config_summary.csv", rows, fields)

    report = [
        "# Open Images 10k Calibration-Size Sensitivity Summary",
        "",
        "Date: 2026-05-23",
        "",
        "The 20% calibration pool from the main protocol is subsampled while keeping retrieval and evaluation splits fixed. Thus 12.5%, 25%, 50%, and 100% of the calibration pool correspond to 2.5%, 5%, 10%, and 20% of the dataset.",
        "",
        "| Calibration pool used | Dataset used for calibration | Method | AP | F1 | TECR | TECR reduction |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['calibration_fraction_used'] * 100:.1f}% | {row['effective_dataset_fraction'] * 100:.1f}% | "
            f"{row['method']} | {fmt(row['average_precision_mean'])} | {fmt(row['best_f1_mean'])} | "
            f"{fmt(row['tecr_mean'])} | {row['tecr_reduction_pct']:.1f}% |"
        )
    report.extend([
        "",
        "Interpretation:",
        "",
        "- TECR reduction is stable even when using only a small calibration subset.",
        "- The full 20% calibration setting remains the main protocol because it is the cleanest and most stable selection setting.",
        "",
    ])
    (output_dir / "calibration_size_summary.md").write_text("\n".join(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
