from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from run_openimages_heldout_calibration import (
    aggregate_scores,
    best_f1_threshold,
    build_splits,
    confidence_gate_scores,
    metrics_with_calibrated_threshold,
    split_retrieval_calibration_eval,
)
from run_openimages_margin_gate import load_cached_arrays
from run_openimages_pilot import knn_score_matrix


def frequency_groups(reference_counts: np.ndarray, class_names: list[str]) -> dict[str, str]:
    order = sorted(range(len(class_names)), key=lambda i: (-reference_counts[i], class_names[i].lower()))
    groups = {}
    n = len(order)
    for rank, idx in enumerate(order):
        if rank < n / 3:
            group = "head"
        elif rank < 2 * n / 3:
            group = "mid"
        else:
            group = "tail"
        groups[class_names[idx]] = group
    return groups


def read_selected_settings(path: Path) -> dict[str, dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        out[row["split"]] = {
            "selection_status": row["selection_status"],
            "residual_power": float(row["residual_power"]),
            "confidence_cutoff": float(row["confidence_cutoff"]),
            "confidence_temperature": float(row["confidence_temperature"]),
        }
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float | None]) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def pct_reduction(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or old == 0:
        return None
    return 100.0 * (old - new) / old


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for group in ["head", "mid", "tail"]:
        base_rows = [row for row in rows if row["frequency_group"] == group and row["method"] == "clip_knn"]
        held_rows = [row for row in rows if row["frequency_group"] == group and row["method"] == "elta_confidence_heldout"]
        by_split_base = {row["split"]: row for row in base_rows}
        by_split_held = {row["split"]: row for row in held_rows}
        common = sorted(set(by_split_base) & set(by_split_held))
        base_ap = mean([by_split_base[s]["average_precision"] for s in common])
        held_ap = mean([by_split_held[s]["average_precision"] for s in common])
        base_f1 = mean([by_split_base[s]["best_f1"] for s in common])
        held_f1 = mean([by_split_held[s]["best_f1"] for s in common])
        base_tecr = mean([by_split_base[s]["tecr"] for s in common])
        held_tecr = mean([by_split_held[s]["tecr"] for s in common])
        out.append({
            "frequency_group": group,
            "num_splits": len(common),
            "num_emerging_labels_mean": mean([by_split_base[s]["num_emerging_labels"] for s in common]),
            "clip_knn_ap_mean": base_ap,
            "heldout_ap_mean": held_ap,
            "ap_delta": None if base_ap is None or held_ap is None else held_ap - base_ap,
            "clip_knn_f1_mean": base_f1,
            "heldout_f1_mean": held_f1,
            "f1_delta": None if base_f1 is None or held_f1 is None else held_f1 - base_f1,
            "clip_knn_tecr_mean": base_tecr,
            "heldout_tecr_mean": held_tecr,
            "tecr_reduction_pct": pct_reduction(base_tecr, held_tecr),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/openimages_10k_heldout_ultrastrict.yaml")
    parser.add_argument("--heldout-dir", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    heldout_dir = Path(args.heldout_dir)
    output_dir = Path(args.output_dir) if args.output_dir else heldout_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = heldout_dir / "heldout_calibration_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_seed = int(result.get("seed", cfg["seed"]))
    selected = read_selected_settings(heldout_dir / "heldout_selected_settings.csv")

    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))

    features, labels, text_features, class_names = load_cached_arrays(cfg, Path(cfg["output_dir"]))
    (
        retrieval_features,
        retrieval_labels,
        calibration_features,
        calibration_labels,
        eval_features,
        eval_labels,
    ) = split_retrieval_calibration_eval(
        features,
        labels,
        run_seed,
        retrieval_fraction,
        calibration_fraction,
    )
    splits = build_splits(class_names, retrieval_labels, calibration_labels, cfg["protocol"], run_seed)
    reference_counts = retrieval_labels.sum(axis=0) + calibration_labels.sum(axis=0)
    label_groups = frequency_groups(reference_counts, class_names)

    retrieval_logits_raw = retrieval_features @ text_features.T
    logit_mean = retrieval_logits_raw.mean(axis=0, keepdims=True)
    logit_std = retrieval_logits_raw.std(axis=0, keepdims=True) + 1e-8
    calibration_logits = (calibration_features @ text_features.T - logit_mean) / logit_std
    eval_logits = (eval_features @ text_features.T - logit_mean) / logit_std

    rows = []
    for split in splits:
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
        known_idx = [i for i, name in enumerate(class_names) if name not in split["emerging_labels"]]

        calibration_known = calibration_logits[:, known_idx].max(axis=1)
        calibration_known = 1.0 / (1.0 + np.exp(-calibration_known))
        eval_known = eval_logits[:, known_idx].max(axis=1)
        eval_known = 1.0 / (1.0 + np.exp(-eval_known))

        calibration_knn = knn_score_matrix(retrieval_features, retrieval_labels, calibration_features, emerging_idx, cfg["knn"])
        eval_knn = knn_score_matrix(retrieval_features, retrieval_labels, eval_features, emerging_idx, cfg["knn"])

        setting = selected[split["name"]]
        calibration_heldout = confidence_gate_scores(
            calibration_knn,
            calibration_known,
            setting["residual_power"],
            setting["confidence_cutoff"],
            setting["confidence_temperature"],
        )
        eval_heldout = confidence_gate_scores(
            eval_knn,
            eval_known,
            setting["residual_power"],
            setting["confidence_cutoff"],
            setting["confidence_temperature"],
        )

        for group in ["head", "mid", "tail"]:
            local_indices = [
                i for i, class_idx in enumerate(emerging_idx)
                if label_groups[class_names[class_idx]] == group
            ]
            if not local_indices:
                continue
            group_idx = [emerging_idx[i] for i in local_indices]
            y_cal = calibration_labels[:, group_idx].max(axis=1).astype(int)
            y_eval = eval_labels[:, group_idx].max(axis=1).astype(int)

            cal_base_scores = aggregate_scores(calibration_knn[:, local_indices])
            eval_base_scores = aggregate_scores(eval_knn[:, local_indices])
            base_threshold = best_f1_threshold(y_cal, cal_base_scores)["threshold"]
            base_metrics = metrics_with_calibrated_threshold(
                y_eval,
                eval_base_scores,
                base_threshold,
                eval_labels,
                tail_idx,
                group_idx,
            )
            rows.append({
                **base_metrics,
                "split": split["name"],
                "seed": run_seed,
                "frequency_group": group,
                "method": "clip_knn",
                "num_emerging_labels": len(local_indices),
                "residual_power": None,
                "confidence_cutoff": None,
                "confidence_temperature": None,
            })

            cal_held_scores = aggregate_scores(calibration_heldout[:, local_indices])
            eval_held_scores = aggregate_scores(eval_heldout[:, local_indices])
            held_threshold = best_f1_threshold(y_cal, cal_held_scores)["threshold"]
            held_metrics = metrics_with_calibrated_threshold(
                y_eval,
                eval_held_scores,
                held_threshold,
                eval_labels,
                tail_idx,
                group_idx,
            )
            rows.append({
                **held_metrics,
                "split": split["name"],
                "seed": run_seed,
                "frequency_group": group,
                "method": "elta_confidence_heldout",
                "num_emerging_labels": len(local_indices),
                "residual_power": setting["residual_power"],
                "confidence_cutoff": setting["confidence_cutoff"],
                "confidence_temperature": setting["confidence_temperature"],
            })

    summary = summarize(rows)
    row_fields = [
        "split", "seed", "frequency_group", "method", "num_emerging_labels",
        "residual_power", "confidence_cutoff", "confidence_temperature",
        "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    summary_fields = [
        "frequency_group", "num_splits", "num_emerging_labels_mean",
        "clip_knn_ap_mean", "heldout_ap_mean", "ap_delta",
        "clip_knn_f1_mean", "heldout_f1_mean", "f1_delta",
        "clip_knn_tecr_mean", "heldout_tecr_mean", "tecr_reduction_pct",
    ]
    write_csv(output_dir / "heldout_frequency_group_rows.csv", rows, row_fields)
    write_csv(output_dir / "heldout_frequency_group_summary.csv", summary, summary_fields)

    report = [
        "# Held-Out Calibration Frequency Diagnostics",
        "",
        "Date: 2026-05-22",
        "",
        f"Seed: `{run_seed}`.",
        "",
        "| Group | Splits | Labels/split | CLIP+kNN AP | Held-out AP | CLIP+kNN F1 | Held-out F1 | CLIP+kNN TECR | Held-out TECR | TECR reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        reduction = row["tecr_reduction_pct"]
        reduction_text = "" if reduction is None else f"{reduction:.1f}%"
        report.append(
            f"| {row['frequency_group']} | {row['num_splits']} | "
            f"{fmt(row['num_emerging_labels_mean'])} | {fmt(row['clip_knn_ap_mean'])} | "
            f"{fmt(row['heldout_ap_mean'])} | {fmt(row['clip_knn_f1_mean'])} | "
            f"{fmt(row['heldout_f1_mean'])} | {fmt(row['clip_knn_tecr_mean'])} | "
            f"{fmt(row['heldout_tecr_mean'])} | {reduction_text} |"
        )
    report.extend([
        "",
        "Files:",
        "",
        "- `heldout_frequency_group_rows.csv`",
        "- `heldout_frequency_group_summary.csv`",
        "",
    ])
    (output_dir / "heldout_frequency_group_report.md").write_text("\n".join(report), encoding="utf-8")
    result_out = {
        "status": "heldout_frequency_groups_complete",
        "seed": run_seed,
        "heldout_dir": str(heldout_dir),
        "summary": summary,
    }
    (output_dir / "heldout_frequency_group_result.json").write_text(
        json.dumps(result_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result_out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
