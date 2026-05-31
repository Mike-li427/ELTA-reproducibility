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


def print_odin_table() -> None:
    rows = read_rows(ROOT / "results" / "supplementary" / "odin_combined_summary.csv")
    method_names = {
        "clip_knn": "CLIP+kNN",
        "odin_low_reject": "ODIN low-confidence rejection",
    }

    print("\n## Supplementary: ODIN low-confidence rejection\n")
    print("| Method | AP | F1 | TECR | Avg. labels | Coverage | TECR reduction |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {method_names.get(row['method'], row['method'])} | "
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


def print_openimages_ablation_tables() -> None:
    method_names = {
        "clip_knn": "CLIP+kNN",
        "pure_residual_heldout": "Pure residual gate",
        "full_gate_tol_0.00pct": "Full gate, 0.00% tolerance",
        "full_gate_tol_0.25pct": "Full gate, 0.25% tolerance",
        "full_gate_tol_0.50pct": "Full gate, 0.50% tolerance",
        "full_gate_tol_1.00pct": "Full gate, 1.00% tolerance",
    }
    rows = read_rows(ROOT / "results" / "openimages_10k" / "gate_ablation_12config_summary.csv")
    rows = [row for method in method_names for row in rows if row["method"] == method]
    print_table("Supplementary: Open Images gate ablation", rows, method_names)

    rows = read_rows(ROOT / "results" / "openimages_10k" / "calibration_size_12config_summary.csv")
    rows = [row for row in rows if row["method"] == "heldout_gate"]
    print("\n## Supplementary: Open Images calibration-size sensitivity\n")
    print("| Calibration pool used | Dataset used for calibration | AP | F1 | TECR | TECR reduction vs main baseline |")
    print("|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        pool_fraction = float(row["calibration_fraction_used"])
        dataset_fraction = pool_fraction * 0.2
        reduction = row.get("tecr_reduction_pct_vs_main_clip_knn", row.get("tecr_reduction_pct", "0"))
        print(
            f"| {pool_fraction * 100:.1f}% | {dataset_fraction * 100:.1f}% | "
            f"{fmt(row['average_precision_mean'])} | "
            f"{fmt(row['best_f1_mean'])} | "
            f"{fmt(row['tecr_mean'])} | "
            f"{fmt_pct(reduction)} |"
        )

    rows = read_rows(ROOT / "results" / "supplementary" / "calibration_ratio_summary.csv")
    rows = [row for row in rows if row["method"] == "heldout_gate"]
    print("\n## Supplementary: Open Images calibration-ratio sensitivity\n")
    print("| Calibration fraction | AP | F1 | TECR | TECR reduction |")
    print("|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {float(row['calibration_dataset_fraction']) * 100:.0f}% | "
            f"{fmt(row['average_precision_mean'])} | "
            f"{fmt(row['best_f1_mean'])} | "
            f"{fmt(row['tecr_mean'])} | "
            f"{fmt_pct(row['tecr_reduction_pct'])} |"
        )


def print_nuswide_supplementary_tables() -> None:
    method_names = {
        "asl_class_thresholds": "ASL class thresholds",
        "db_loss_class_thresholds": "DBLoss class thresholds",
        "asl_global_threshold": "ASL global threshold",
        "asl_gate_global_threshold": "ASL + gate global threshold",
    }
    rows = read_rows(ROOT / "results" / "nuswide" / "training_baseline_10config_summary.csv")
    gate_rows = read_rows(ROOT / "results" / "nuswide" / "asl_gate_10config_summary.csv")
    selected = [row for row in rows if row["method"] in {"asl_class_thresholds", "db_loss_class_thresholds"}]
    selected.extend(row for row in gate_rows if row["method"] in {"asl_global_threshold", "asl_gate_global_threshold"})
    selected = [row for method in method_names for row in selected if row["method"] == method]
    print_table("Supplementary: NUS-WIDE trained-head and ASL+gate", selected, method_names)

    rows = read_rows(ROOT / "results" / "nuswide" / "frequency_group_10config_summary.csv")
    print("\n## Supplementary: NUS-WIDE frequency groups\n")
    print("| Group | Labels/split | CLIP+kNN TECR | Gate TECR | Delta | TECR reduction |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        reduction = row.get("tecr_reduction_pct_mean_across_configs", row["tecr_reduction_pct"])
        print(
            f"| {row['frequency_group']} | "
            f"{fmt(row['num_emerging_labels_mean'], 1)} | "
            f"{fmt(row['clip_knn_tecr_mean'])} | "
            f"{fmt(row['heldout_tecr_mean'])} | "
            f"{fmt(row['tecr_delta'])} | "
            f"{fmt_pct(reduction)} |"
        )


def print_robustness_and_diagnostics() -> None:
    rows = read_rows(ROOT / "results" / "supplementary" / "tecr_robustness_summary.csv")
    rows = [row for row in rows if row["method"] == "heldout_gate"]
    print("\n## Supplementary: TECR definition robustness\n")
    print("| Variant | Gate TECR | TECR reduction |")
    print("|---|---:|---:|")
    for row in rows:
        print(f"| {row['variant']} | {fmt(row['tecr_mean'])} | {fmt_pct(row['tecr_reduction_pct'])} |")

    rows = read_rows(ROOT / "results" / "supplementary" / "gate_parameter_stability_summary.csv")
    print("\n## Supplementary: Gate parameter stability\n")
    print("| Dataset | n | p mean | b mean | tau mean |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['dataset']} | {row['num_selected']} | "
            f"{fmt(row['residual_power_mean'], 3)} | "
            f"{fmt(row['confidence_cutoff_mean'], 3)} | "
            f"{fmt(row['confidence_temperature_mean'], 3)} |"
        )

    rows = read_rows(ROOT / "results" / "supplementary" / "openimages_vitb16_heldout_summary.csv")
    method_names = {
        "clip_knn": "CLIP+kNN ViT-B/16",
        "elta_confidence_heldout": "Held-out gate ViT-B/16",
    }
    rows = [row for method in method_names for row in rows if row["method"] == method]
    print_table("Supplementary: Open Images ViT-B/16 check", rows, method_names)


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
    print_odin_table()
    print_mkt_table()
    print_openimages_ablation_tables()
    print_nuswide_supplementary_tables()
    print_robustness_and_diagnostics()


if __name__ == "__main__":
    main()
