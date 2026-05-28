from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: str | float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pct(value: str | float) -> str:
    return f"{float(value):.1f}%"


def print_table(title: str, rows: list[dict], method_names: dict[str, str]) -> None:
    print(f"\n## {title}\n")
    print("| Method | AP | F1 | TECR | TECR reduction |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        method = row["method"]
        name = method_names.get(method, method)
        if "average_precision_mean" in row:
            ap = row["average_precision_mean"]
            f1 = row["best_f1_mean"]
            tecr = row["tecr_mean"]
        else:
            ap = row["ap_mean"]
            f1 = row["f1_mean"]
            tecr = row["tecr_mean"]
        reduction = row.get("tecr_reduction_pct", "0")
        print(f"| {name} | {fmt(ap)} | {fmt(f1)} | {fmt(tecr)} | {fmt_pct(reduction)} |")


def print_posthoc_table() -> None:
    method_names = {
        "clip_knn": "CLIP+kNN",
        "temperature_scaling": "Temperature scaling",
        "split_conformal": "Split conformal prediction",
        "maxlogit_known_reject": "MaxLogit/MCM-style rejection",
        "entropy_reject": "Entropy selective rejection",
        "elta_confidence_heldout": "ELTA held-out confidence gate",
    }
    rows = read_rows(ROOT / "results" / "supplementary" / "posthoc_combined_summary.csv")
    rows = [row for method in method_names for row in rows if row["method"] == method]

    print("\n## Table 7: Open Images post-hoc baseline strategies\n")
    print("| Method | AP | F1 | TECR | Avg. labels | Coverage | TECR reduction |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        method = row["method"]
        print(
            f"| {method_names[method]} | "
            f"{fmt(row['average_precision_mean'])} | "
            f"{fmt(row['best_f1_mean'])} | "
            f"{fmt(row['tecr_mean'])} | "
            f"{fmt(row['avg_predicted_labels_mean'])} | "
            f"{fmt(row['coverage_mean'])} | "
            f"{fmt_pct(row['tecr_reduction_pct'])} |"
        )


def print_mkt_table() -> None:
    rows = read_rows(ROOT / "results" / "supplementary" / "mkt_combined_summary.csv")

    print("\n## Supplementary: Adapted public MKT checkpoint sanity check\n")
    print("| Method | AP | F1 | TECR |")
    print("|---|---:|---:|---:|")
    for row in rows:
        name = "Adapted public MKT checkpoint" if row["method"] == "mkt_open_vocab" else row["method"]
        print(
            f"| {name} | "
            f"{fmt(row['average_precision_mean'])} | "
            f"{fmt(row['best_f1_mean'])} | "
            f"{fmt(row['tecr_mean'])} |"
        )


def main() -> None:
    method_names = {
        "clip_knn_global_threshold": "CLIP+kNN global threshold",
        "clip_knn_class_thresholds": "CLIP+kNN class thresholds",
        "clip_knn_isotonic_global_threshold": "CLIP+kNN isotonic global threshold",
        "heldout_gate_global_threshold": "Held-out gate global threshold",
        "heldout_gate_class_thresholds": "Held-out gate class thresholds",
    }
    order = list(method_names)

    openimages_rows = read_rows(ROOT / "results" / "openimages_10k" / "main_method_12config_summary.csv")
    openimages_rows = [row for method in order for row in openimages_rows if row["method"] == method]
    print_table("Table 1: Open Images 10k", openimages_rows, method_names)

    coco_rows = read_rows(ROOT / "results" / "coco_val2017" / "main_method_12config_summary.csv")
    coco_rows = [row for method in order for row in coco_rows if row["method"] == method]
    print_table("Table 2: COCO val2017", coco_rows, method_names)

    nuswide_rows = read_rows(ROOT / "results" / "nuswide" / "main_summary_10config.csv")
    nuswide_rows = [row for method in order for row in nuswide_rows if row["method"] == method]
    print_table("Table 3: NUS-WIDE stress test", nuswide_rows, method_names)

    print_posthoc_table()
    print_mkt_table()


if __name__ == "__main__":
    main()
