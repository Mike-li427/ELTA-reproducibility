from __future__ import annotations

import argparse
import csv
import json
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
    "asl_global_threshold": "ASL (ICCV 2021), global threshold",
    "asl_class_thresholds": "ASL (ICCV 2021), class thresholds",
    "asl_gate_global_threshold": "ASL + gate, global threshold",
    "asl_gate_class_thresholds": "ASL + gate, class thresholds",
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


def summarize_files(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    per_config_rows = []
    for path in paths:
        config = config_name(path)
        for row in read_csv_dicts(path):
            item = {"config": config, "method": row["method"]}
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


def paired_delta_stats(per_config_rows: list[dict], baseline_method: str, method: str, metric_col: str) -> dict:
    by_config = {}
    for row in per_config_rows:
        by_config.setdefault(row["config"], {})[row["method"]] = row
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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def fmt_pct(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--glob", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path().glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No ASL gate summary files matched {args.glob}")

    per_config_rows, summary_rows = summarize_files(paths)
    baseline = next(row for row in summary_rows if row["method"] == "asl_global_threshold")
    add_deltas(summary_rows, baseline["method"])

    write_csv(
        output_dir / "asl_gate_per_config.csv",
        per_config_rows,
        ["config", "method", "average_precision_mean", "auroc_mean", "best_f1_mean", "tecr_mean"],
    )
    write_csv(
        output_dir / "asl_gate_summary.csv",
        summary_rows,
        [
            "method",
            "display_name",
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
        ],
    )

    gate_vs_asl_tecr = paired_delta_stats(per_config_rows, "asl_global_threshold", "asl_gate_global_threshold", "tecr_mean")
    gate_vs_asl_ap = paired_delta_stats(per_config_rows, "asl_global_threshold", "asl_gate_global_threshold", "average_precision_mean")
    gate_vs_asl_f1 = paired_delta_stats(per_config_rows, "asl_global_threshold", "asl_gate_global_threshold", "best_f1_mean")

    lines = [
        f"# {args.dataset_name} ASL + Gate Comparison",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "ASL is trained on the retrieval split only, and gate parameters are selected on calibration rows only.",
        "",
        "| Method | AP | F1 | TECR | AP delta | F1 delta | TECR reduction vs ASL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['display_name']} | {fmt(row['average_precision_mean'])} | "
            f"{fmt(row['best_f1_mean'])} | {fmt(row['tecr_mean'])} | "
            f"{fmt_pct(row.get('ap_delta_pct'))} | {fmt_pct(row.get('f1_delta_pct'))} | "
            f"{fmt_pct(row.get('tecr_reduction_pct'))} |"
        )

    lines.extend([
        "",
        "## Paired Diagnostics",
        "",
        "| Comparison | Metric | n | Mean delta | Win rate |",
        "|---|---|---:|---:|---:|",
        f"| ASL + gate vs ASL | TECR | {gate_vs_asl_tecr['num_pairs']} | {fmt(gate_vs_asl_tecr['delta_mean'])} | {fmt_pct(100.0 * gate_vs_asl_tecr['win_rate'] if gate_vs_asl_tecr['win_rate'] is not None else None)} |",
        f"| ASL + gate vs ASL | AP | {gate_vs_asl_ap['num_pairs']} | {fmt(gate_vs_asl_ap['delta_mean'])} | {fmt_pct(100.0 * gate_vs_asl_ap['win_rate'] if gate_vs_asl_ap['win_rate'] is not None else None)} |",
        f"| ASL + gate vs ASL | F1 | {gate_vs_asl_f1['num_pairs']} | {fmt(gate_vs_asl_f1['delta_mean'])} | {fmt_pct(100.0 * gate_vs_asl_f1['win_rate'] if gate_vs_asl_f1['win_rate'] is not None else None)} |",
        "",
        "Notes:",
        "",
        "- The gate is post-hoc: it changes only the inference scores, not the ASL training procedure.",
        "- This table is meant to test whether the gate is scorer-agnostic rather than to replace the main CLIP+kNN comparison.",
        "",
    ])
    (output_dir / "asl_gate_report.md").write_text("\n".join(lines), encoding="utf-8")

    result = {
        "dataset_name": args.dataset_name,
        "num_configs": len({row["config"] for row in per_config_rows}),
        "output_dir": str(output_dir),
        "summary": summary_rows,
    }
    (output_dir / "asl_gate_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n".join(lines[:8]))
    return 0


def add_deltas(rows: list[dict], baseline_method: str) -> None:
    baseline = next(row for row in rows if row["method"] == baseline_method)
    for row in rows:
        row["ap_delta_pct"] = 100.0 * (row["average_precision_mean"] - baseline["average_precision_mean"]) / baseline["average_precision_mean"]
        row["f1_delta_pct"] = 100.0 * (row["best_f1_mean"] - baseline["best_f1_mean"]) / baseline["best_f1_mean"]
        row["tecr_reduction_pct"] = 100.0 * (baseline["tecr_mean"] - row["tecr_mean"]) / baseline["tecr_mean"]


if __name__ == "__main__":
    raise SystemExit(main())
