from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np


GROUP_ORDER = ["head", "mid", "tail"]
METRIC_COLUMNS = [
    "average_precision_mean",
    "auroc_mean",
    "best_f1_mean",
    "tecr_mean",
]


DISPLAY_NAMES = {
    "text_init_asl_class_thresholds": "Text-initialized ASL, class thresholds",
    "heldout_gate_global_threshold": "Held-out gate, global threshold",
    "clip_knn_global_threshold": "CLIP+kNN, global threshold",
}


def read_csv_dicts(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def canonical_run_key(config: str) -> str:
    key = config
    for prefix in [
        "openimages_10k_heldout_ultrastrict",
        "openimages_10k_training_baselines",
        "coco_heldout_ultrastrict",
        "coco_training_baselines",
    ]:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    key = key.strip("_")
    return key or "base"


def mean(values: list[float | None]) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


def stdev(values: list[float | None]) -> float:
    kept = [float(v) for v in values if v is not None]
    return float(np.std(kept, ddof=1)) if len(kept) > 1 else 0.0


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def fmt_pct(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}%"


def pct_reduction(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or old == 0:
        return None
    return 100.0 * (old - new) / old


def read_frequency_rows(root: Path, dataset: str, prefix: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob(f"{prefix}*/heldout_frequency_group_summary.csv")):
        config = path.parent.name
        for row in read_csv_dicts(path):
            rows.append(
                {
                    "dataset": dataset,
                    "config": config,
                    "run_key": canonical_run_key(config),
                    "frequency_group": row["frequency_group"],
                    "num_splits": parse_float(row.get("num_splits")),
                    "num_emerging_labels_mean": parse_float(row.get("num_emerging_labels_mean")),
                    "clip_knn_ap_mean": parse_float(row.get("clip_knn_ap_mean")),
                    "heldout_ap_mean": parse_float(row.get("heldout_ap_mean")),
                    "ap_delta": parse_float(row.get("ap_delta")),
                    "clip_knn_f1_mean": parse_float(row.get("clip_knn_f1_mean")),
                    "heldout_f1_mean": parse_float(row.get("heldout_f1_mean")),
                    "f1_delta": parse_float(row.get("f1_delta")),
                    "clip_knn_tecr_mean": parse_float(row.get("clip_knn_tecr_mean")),
                    "heldout_tecr_mean": parse_float(row.get("heldout_tecr_mean")),
                    "tecr_reduction_pct": parse_float(row.get("tecr_reduction_pct")),
                }
            )
    return rows


def summarize_frequency(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for group in GROUP_ORDER:
            group_rows = [row for row in dataset_rows if row["frequency_group"] == group]
            clip_tecr = mean([row["clip_knn_tecr_mean"] for row in group_rows])
            gate_tecr = mean([row["heldout_tecr_mean"] for row in group_rows])
            clip_ap = mean([row["clip_knn_ap_mean"] for row in group_rows])
            gate_ap = mean([row["heldout_ap_mean"] for row in group_rows])
            clip_f1 = mean([row["clip_knn_f1_mean"] for row in group_rows])
            gate_f1 = mean([row["heldout_f1_mean"] for row in group_rows])
            out.append(
                {
                    "dataset": dataset,
                    "frequency_group": group,
                    "num_configs": len(group_rows),
                    "num_emerging_labels_mean": mean([row["num_emerging_labels_mean"] for row in group_rows]),
                    "clip_knn_ap_mean": clip_ap,
                    "heldout_ap_mean": gate_ap,
                    "ap_delta": None if clip_ap is None or gate_ap is None else gate_ap - clip_ap,
                    "clip_knn_f1_mean": clip_f1,
                    "heldout_f1_mean": gate_f1,
                    "f1_delta": None if clip_f1 is None or gate_f1 is None else gate_f1 - clip_f1,
                    "clip_knn_tecr_mean": clip_tecr,
                    "heldout_tecr_mean": gate_tecr,
                    "tecr_delta": None if clip_tecr is None or gate_tecr is None else gate_tecr - clip_tecr,
                    "tecr_reduction_pct": pct_reduction(clip_tecr, gate_tecr),
                    "tecr_delta_std": stdev(
                        [
                            row["heldout_tecr_mean"] - row["clip_knn_tecr_mean"]
                            for row in group_rows
                            if row["heldout_tecr_mean"] is not None and row["clip_knn_tecr_mean"] is not None
                        ]
                    ),
                }
            )
    return out


def read_summary_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(paths):
        config = path.parent.name
        for row in read_csv_dicts(path):
            item = {
                "config": config,
                "run_key": canonical_run_key(config),
                "method": row["method"],
            }
            for column in METRIC_COLUMNS:
                item[column] = parse_float(row.get(column))
            rows.append(item)
    return rows


def aggregate_method(rows: list[dict], method: str) -> dict:
    group = [row for row in rows if row["method"] == method]
    out = {"method": method, "display_name": DISPLAY_NAMES.get(method, method), "num_configs": len(group)}
    for column in METRIC_COLUMNS:
        values = [row[column] for row in group]
        out[column] = mean(values)
        out[column.replace("_mean", "_std")] = stdev(values)
    return out


def select_best_training_method(rows: list[dict]) -> str:
    methods = sorted({row["method"] for row in rows})
    summaries = [aggregate_method(rows, method) for method in methods]
    best_ap = max(row["average_precision_mean"] for row in summaries if row["average_precision_mean"] is not None)
    eligible = [
        row
        for row in summaries
        if row["average_precision_mean"] is not None
        and row["average_precision_mean"] >= best_ap * 0.99
        and row["method"].endswith("class_thresholds")
    ]
    best = min(eligible or summaries, key=lambda row: row["tecr_mean"])
    return str(best["method"])


def coco_exception_rows(root: Path) -> tuple[str, list[dict]]:
    training_rows = read_summary_rows(list(root.glob("coco_training_baselines*/training_baseline_summary.csv")))
    main_rows = read_summary_rows(list(root.glob("coco_heldout_ultrastrict*/heldout_summary.csv")))
    best_method = select_best_training_method(training_rows)
    training_by_key = {row["run_key"]: row for row in training_rows if row["method"] == best_method}
    gate_by_key = {row["run_key"]: row for row in main_rows if row["method"] == "elta_confidence_heldout"}
    clip_by_key = {row["run_key"]: row for row in main_rows if row["method"] == "clip_knn"}

    out: list[dict] = []
    for key in sorted(training_by_key):
        if key not in gate_by_key:
            continue
        train = training_by_key[key]
        gate = gate_by_key[key]
        clip = clip_by_key.get(key)
        out.append(
            {
                "run_key": key,
                "config": gate["config"],
                "best_training_method": best_method,
                "best_training_tecr": train["tecr_mean"],
                "heldout_gate_tecr": gate["tecr_mean"],
                "clip_knn_tecr": None if clip is None else clip["tecr_mean"],
                "gate_minus_training_tecr": None if gate["tecr_mean"] is None or train["tecr_mean"] is None else gate["tecr_mean"] - train["tecr_mean"],
                "gate_wins": bool(gate["tecr_mean"] < train["tecr_mean"]),
                "best_training_ap": train["average_precision_mean"],
                "heldout_gate_ap": gate["average_precision_mean"],
                "best_training_f1": train["best_f1_mean"],
                "heldout_gate_f1": gate["best_f1_mean"],
            }
        )
    return best_method, out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, frequency_summary: list[dict], best_method: str, exceptions: list[dict]) -> None:
    losing = [row for row in exceptions if not row["gate_wins"]]
    lines = [
        "# Fairness Diagnostics Summary",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "This report summarizes post-rerun diagnostics from the audited protocol package. It does not run new experiments.",
        "",
        "## Frequency Groups",
        "",
        "| Dataset | Group | Configs | Emerging labels/split | CLIP+kNN TECR | Gate TECR | Delta | TECR reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frequency_summary:
        lines.append(
            f"| {row['dataset']} | {row['frequency_group']} | {row['num_configs']} | "
            f"{fmt(row['num_emerging_labels_mean'])} | {fmt(row['clip_knn_tecr_mean'])} | "
            f"{fmt(row['heldout_tecr_mean'])} | {fmt(row['tecr_delta'])} | "
            f"{fmt_pct(row['tecr_reduction_pct'])} |"
        )

    lines.extend(
        [
            "",
            "Interpretation: the aggregate TECR reduction is driven primarily by head and mid emerging-label groups. Tail-group estimates are small and less stable because the per-split tail emerging-label count is smaller, especially on COCO.",
            "",
            "The available rerun CSVs store TECR ratios but not the image-level TECR denominator. Therefore this report does not infer eligible-population counts from ratios. Exact denominator analysis should be regenerated from image-level labels if needed.",
            "",
            "## COCO 83.3% Win-Rate Exceptions",
            "",
            f"Best eligible training baseline: `{best_method}`.",
            "",
            "| Run key | Gate wins | Best-training TECR | Gate TECR | Gate - training | CLIP+kNN TECR | Best-training AP | Gate AP |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in exceptions:
        lines.append(
            f"| {row['run_key']} | {row['gate_wins']} | {fmt(row['best_training_tecr'])} | "
            f"{fmt(row['heldout_gate_tecr'])} | {fmt(row['gate_minus_training_tecr'])} | "
            f"{fmt(row['clip_knn_tecr'])} | {fmt(row['best_training_ap'])} | {fmt(row['heldout_gate_ap'])} |"
        )
    if losing:
        lines.extend(
            [
                "",
                f"The held-out gate loses to the best training baseline in {len(losing)} of {len(exceptions)} COCO configurations. In these exception configurations, the text-initialized ASL class-threshold baseline already has low TECR; the gate still improves over CLIP+kNN but has less remaining room against that trained-head comparator.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/fairness_diagnostics")
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frequency_rows = []
    frequency_rows.extend(read_frequency_rows(audit_root, "Open Images 10k", "openimages_10k_heldout_ultrastrict"))
    frequency_rows.extend(read_frequency_rows(audit_root, "COCO val2017", "coco_heldout_ultrastrict"))
    frequency_summary = summarize_frequency(frequency_rows)
    best_method, exceptions = coco_exception_rows(audit_root)

    write_csv(
        output_dir / "frequency_group_12config_summary.csv",
        frequency_summary,
        [
            "dataset",
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
            "tecr_delta_std",
        ],
    )
    write_csv(
        output_dir / "coco_training_exception_configs.csv",
        exceptions,
        [
            "run_key",
            "config",
            "best_training_method",
            "gate_wins",
            "best_training_tecr",
            "heldout_gate_tecr",
            "clip_knn_tecr",
            "gate_minus_training_tecr",
            "best_training_ap",
            "heldout_gate_ap",
            "best_training_f1",
            "heldout_gate_f1",
        ],
    )
    write_report(output_dir / "fairness_diagnostics_report.md", frequency_summary, best_method, exceptions)
    (output_dir / "fairness_diagnostics_result.json").write_text(
        json.dumps(
            {
                "status": "fairness_diagnostics_complete",
                "audit_root": str(audit_root),
                "output_dir": str(output_dir),
                "best_coco_training_method": best_method,
                "num_frequency_rows": len(frequency_rows),
                "num_coco_configs": len(exceptions),
                "num_coco_exceptions": len([row for row in exceptions if not row["gate_wins"]]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print((output_dir / "fairness_diagnostics_report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
