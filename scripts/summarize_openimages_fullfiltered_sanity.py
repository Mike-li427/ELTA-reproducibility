from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


DISPLAY_NAMES = {
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

METRIC_COLUMNS = (
    "average_precision_mean",
    "auroc_mean",
    "best_f1_mean",
    "tecr_mean",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_run_identity(source: str) -> tuple[str, int]:
    split_set = "classB" if "_classB" in source else "classA"
    seed_match = re.search(r"_s(\d{8})$", source)
    seed = int(seed_match.group(1)) if seed_match else 20260522
    return split_set, seed


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def fmt_pct(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}%"


def build_parser(
    *,
    default_glob: str,
    default_output_dir: str,
    default_dataset_name: str,
    default_dataset_id: str,
    default_per_config_name: str,
    default_summary_name: str,
    default_report_name: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default=default_glob)
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--dataset-name", default=default_dataset_name)
    parser.add_argument("--dataset-id", default=default_dataset_id)
    parser.add_argument("--per-config-name", default=default_per_config_name)
    parser.add_argument("--summary-name", default=default_summary_name)
    parser.add_argument("--report-name", default=default_report_name)
    parser.add_argument("--report-intro", default=None)
    parser.add_argument("--report-scope-note", default=None)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    default_glob: str = "outputs/openimages_fullfiltered_validation_heldout_ultrastrict*/calibrated_baseline_summary.csv",
    default_output_dir: str = "results/supplementary",
    default_dataset_name: str = "Larger filtered Open Images validation-subset check",
    default_dataset_id: str = "openimages_full_filtered_validation",
    default_per_config_name: str = "openimages_full_filtered_validation_sanity_per_config.csv",
    default_summary_name: str = "openimages_full_filtered_validation_sanity_summary.csv",
    default_report_name: str = "openimages_full_filtered_validation_sanity_report.md",
) -> int:
    parser = build_parser(
        default_glob=default_glob,
        default_output_dir=default_output_dir,
        default_dataset_name=default_dataset_name,
        default_dataset_id=default_dataset_id,
        default_per_config_name=default_per_config_name,
        default_summary_name=default_summary_name,
        default_report_name=default_report_name,
    )
    args = parser.parse_args(argv)

    summary_paths = sorted(Path().glob(args.glob))
    if not summary_paths:
        raise FileNotFoundError(f"No summary files matched {args.glob}")

    per_config_rows: list[dict] = []
    for path in summary_paths:
        source = path.parent.name
        split_set, seed = parse_run_identity(source)
        for row in read_rows(path):
            item = {
                "dataset": args.dataset_id,
                "split_set": split_set,
                "seed": seed,
                "source": source,
                "method": row["method"],
                "display_name": DISPLAY_NAMES.get(row["method"], row["method"]),
                "num_splits": int(float(row["num_splits"])),
            }
            for col in (
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
            ):
                item[col] = parse_float(row.get(col))
            per_config_rows.append(item)

    methods = sorted({row["method"] for row in per_config_rows})
    summary_rows: list[dict] = []
    for method in methods:
        group = [row for row in per_config_rows if row["method"] == method]
        out = {
            "method": method,
            "display_name": DISPLAY_NAMES.get(method, method),
            "num_configs": len(group),
        }
        for col in METRIC_COLUMNS:
            vals = [row[col] for row in group if row[col] is not None]
            out[col] = float(np.mean(vals)) if vals else None
            out[col.replace("_mean", "_std")] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(out)

    baseline = next(row for row in summary_rows if row["method"] == "clip_knn_global_threshold")
    for row in summary_rows:
        row["ap_delta_pct"] = 100.0 * (row["average_precision_mean"] - baseline["average_precision_mean"]) / baseline["average_precision_mean"]
        row["f1_delta_pct"] = 100.0 * (row["best_f1_mean"] - baseline["best_f1_mean"]) / baseline["best_f1_mean"]
        row["tecr_reduction_pct"] = 100.0 * (baseline["tecr_mean"] - row["tecr_mean"]) / baseline["tecr_mean"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_config_path = output_dir / args.per_config_name
    summary_path = output_dir / args.summary_name
    report_path = output_dir / args.report_name

    write_csv(
        per_config_path,
        per_config_rows,
        [
            "dataset",
            "split_set",
            "seed",
            "source",
            "method",
            "display_name",
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
        ],
    )
    write_csv(
        summary_path,
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

    if args.report_intro:
        intro = args.report_intro
    elif args.dataset_id == "openimages_complete_validation":
        intro = (
            "This summary aggregates the 12 configuration-level calibrated-baseline outputs "
            "from the complete 41,620-image Open Images validation pool under the same held-out "
            "40/20/40 protocol family used in the main paper, while freezing the reported 120-label slice."
        )
    else:
        intro = (
            "This summary aggregates the 12 configuration-level calibrated-baseline outputs "
            "from a larger filtered Open Images validation subset under the same held-out 40/20/40 "
            "protocol family used in the main paper."
        )

    if args.report_scope_note:
        scope_note = args.report_scope_note
    elif args.dataset_id == "openimages_complete_validation":
        scope_note = (
            "This reviewer-facing check is supplementary evidence only. It does not replace the "
            "repeated Open Images 10k benchmark reported in the manuscript."
        )
    else:
        scope_note = (
            "This reviewer-facing check is supplementary scope evidence only. It is not the raw "
            "Open Images validation dump and does not replace the repeated Open Images 10k benchmark "
            "reported in the manuscript."
        )

    lines = [
        f"# {args.dataset_name}",
        "",
        intro,
        "",
        scope_note,
        "",
        "| Method | n | AP | F1 | TECR | AP delta | F1 delta | TECR reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['display_name']} | {row['num_configs']} | {fmt(row['average_precision_mean'])} | "
            f"{fmt(row['best_f1_mean'])} | {fmt(row['tecr_mean'])} | {fmt_pct(row['ap_delta_pct'])} | "
            f"{fmt_pct(row['f1_delta_pct'])} | {fmt_pct(row['tecr_reduction_pct'])} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
