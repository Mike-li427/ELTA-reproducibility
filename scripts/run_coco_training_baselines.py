from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from run_coco_heldout_calibration import build_splits
from run_coco_pilot import COCO_CLASSES
from run_openimages_heldout_calibration import split_retrieval_calibration_eval
from run_openimages_training_baselines import (
    add_global_and_class_rows,
    apply_logit_adjustment,
    best_f1_threshold,
    metrics_with_threshold,
    predict_scores,
    summarize,
    train_head,
)


def load_cached_coco_arrays(cfg: dict, output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data_cfg = cfg["data"]
    cache_dir = output_dir / "cache"
    max_images = data_cfg.get("max_images")
    cache_tag = f"coco2017_{data_cfg['image_set']}_{max_images or 'full'}_{cfg['clip']['model'].replace('/', '-')}"
    paths = {
        "features": cache_dir / f"{cache_tag}_image_features.npy",
        "labels": cache_dir / f"{cache_tag}_labels.npy",
        "text": cache_dir / f"{cache_tag}_text_features.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing cached COCO arrays: " + ", ".join(missing))
    return (
        np.load(paths["features"]),
        np.load(paths["labels"]),
        np.load(paths["text"]),
        list(COCO_CLASSES),
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/coco_heldout_ultrastrict.yaml")
    parser.add_argument("--output-dir", default="outputs/coco_training_baselines")
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    args = parser.parse_args()

    start = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_seed = int(args.seed_override) if args.seed_override is not None else int(cfg["seed"])
    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))

    features, labels, text_features, class_names = load_cached_coco_arrays(cfg, Path(cfg["output_dir"]))
    (
        retrieval_features,
        retrieval_labels,
        calibration_features,
        calibration_labels,
        eval_features,
        eval_labels,
    ) = split_retrieval_calibration_eval(features, labels, run_seed, retrieval_fraction, calibration_fraction)
    splits = build_splits(class_names, retrieval_labels, calibration_labels, cfg["protocol"], run_seed)

    rows = []
    selected_rows = []
    methods = ["linear_bce", "class_balanced_bce", "balance_mix_feature", "asl", "db_loss"]
    text_init_methods = [
        "text_init_bce",
        "text_init_class_balanced_bce",
        "text_init_balance_mix_feature",
        "text_init_asl",
        "text_init_db_loss",
    ]
    for split_idx, split in enumerate(splits):
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
        y_cal = calibration_labels[:, emerging_idx].max(axis=1).astype(int)
        y_eval = eval_labels[:, emerging_idx].max(axis=1).astype(int)
        train_y = retrieval_labels[:, emerging_idx]
        priors = np.clip(train_y.mean(axis=0), 1e-5, 1.0 - 1e-5)

        trained_scores = {}
        for method in methods + text_init_methods:
            train_method = method.replace("text_init_", "")
            init_weights = text_features[emerging_idx] if method.startswith("text_init_") else None
            model = train_head(
                retrieval_features,
                train_y,
                train_method,
                device,
                run_seed + split_idx * 100 + len(method),
                args.epochs,
                args.lr,
                args.weight_decay,
                args.mixup_alpha,
                init_weights=init_weights,
            )
            cal_scores = predict_scores(model, calibration_features, device)
            eval_scores = predict_scores(model, eval_features, device)
            trained_scores[method] = (cal_scores, eval_scores)
            add_global_and_class_rows(
                rows,
                method,
                calibration_labels[:, emerging_idx],
                cal_scores,
                eval_scores,
                y_cal,
                y_eval,
                eval_labels,
                tail_idx,
                emerging_idx,
                split["name"],
                run_seed,
            )

        for base_method, output_method in [
            ("linear_bce", "logit_adjusted_bce"),
            ("text_init_bce", "text_init_logit_adjusted_bce"),
        ]:
            base_cal, base_eval = trained_scores[base_method]
            candidates = []
            for strength in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
                cal_adj = apply_logit_adjustment(base_cal, priors, strength)
                eval_adj = apply_logit_adjustment(base_eval, priors, strength)
                cal_agg = cal_adj.max(axis=1)
                threshold = best_f1_threshold(y_cal, cal_agg)["threshold"]
                cal_metrics = metrics_with_threshold(y_cal, cal_agg, threshold, calibration_labels, tail_idx, emerging_idx)
                candidates.append((strength, cal_metrics, cal_adj, eval_adj))
            base_cal_agg = base_cal.max(axis=1)
            base_threshold = best_f1_threshold(y_cal, base_cal_agg)["threshold"]
            base_cal_metrics = metrics_with_threshold(y_cal, base_cal_agg, base_threshold, calibration_labels, tail_idx, emerging_idx)
            valid = [
                item for item in candidates
                if item[1]["average_precision"] is not None
                and item[1]["best_f1"] is not None
                and item[1]["tecr"] is not None
                and item[1]["average_precision"] >= base_cal_metrics["average_precision"] * (1.0 - 0.0025)
                and item[1]["best_f1"] >= base_cal_metrics["best_f1"] * (1.0 - 0.0025)
            ]
            selected = min(valid or candidates, key=lambda item: (item[1]["tecr"] if item[1]["tecr"] is not None else 1e9, -item[1]["best_f1"]))
            strength, _cal_metrics, cal_adj, eval_adj = selected
            selected_rows.append({"split": split["name"], "seed": run_seed, "method": output_method, "strength": strength})
            add_global_and_class_rows(
                rows,
                output_method,
                calibration_labels[:, emerging_idx],
                cal_adj,
                eval_adj,
                y_cal,
                y_eval,
                eval_labels,
                tail_idx,
                emerging_idx,
                split["name"],
                run_seed,
            )

    summary = summarize(rows)
    row_fields = [
        "method", "split", "seed", "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    summary_fields = [
        "method", "num_splits", "average_precision_mean", "average_precision_std",
        "auroc_mean", "auroc_std", "best_f1_mean", "best_f1_std", "tecr_mean", "tecr_std",
        "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct",
    ]
    write_csv(output_dir / "training_baseline_rows.csv", rows, row_fields)
    write_csv(output_dir / "training_baseline_summary.csv", summary, summary_fields)
    write_csv(output_dir / "training_baseline_selected_settings.csv", selected_rows, ["split", "seed", "method", "strength"])

    report = [
        "# COCO Training Baselines",
        "",
        "Date: 2026-05-23",
        "",
        f"Seed: `{run_seed}`.",
        f"Device: `{device}`.",
        f"Epochs: `{args.epochs}`.",
        "",
        "| Method | AP | F1 | TECR | AP delta | F1 delta | TECR reduction vs linear BCE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['method']} | {row['average_precision_mean']:.4f} | "
            f"{row['best_f1_mean']:.4f} | {row['tecr_mean']:.4f} | "
            f"{row.get('ap_delta_pct', 0.0):.2f}% | {row.get('f1_delta_pct', 0.0):.2f}% | "
            f"{row.get('tecr_reduction_pct', 0.0):.1f}% |"
        )
    report.extend([
        "",
        "Notes:",
        "",
        "- These are frozen-CLIP feature training baselines under the same retrieval/calibration/evaluation protocol.",
        "- `balance_mix_feature` is a feature-space BalanceMix-style baseline, not the original image-space BalanceMix implementation.",
        "- `asl` is Asymmetric Loss for Multi-Label Classification (ICCV 2021), adapted as a frozen-feature linear-head baseline.",
        "- `db_loss` is Distribution-Balanced Loss for long-tailed multi-label classification (ECCV 2020), adapted as a frozen-feature linear-head baseline.",
        "- `text_init_*` baselines initialize the linear classifier with CLIP text embeddings before training.",
        "- `*_logit_adjusted_bce` selects the prior-adjustment strength on the calibration split.",
        "",
    ])
    (output_dir / "training_baseline_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "status": "coco_training_baselines_complete",
        "seed": run_seed,
        "time_seconds": round(time.time() - start, 3),
        "output_dir": str(output_dir),
        "summary": summary,
    }
    (output_dir / "training_baseline_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TIME_ESTIMATE:", max(1, int(time.time() - start)))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
