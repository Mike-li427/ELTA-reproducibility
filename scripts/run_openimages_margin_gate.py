from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from run_openimages_pilot import (
    build_splits,
    evaluate_scores,
    knn_score_matrix,
    split_retrieval_eval,
)


METRICS = ["average_precision", "auroc", "best_f1", "tecr"]


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def pct_change(old: float, new: float) -> float:
    return 100.0 * (new - old) / old


def pct_reduction(old: float, new: float) -> float:
    return 100.0 * (old - new) / old


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def load_cached_arrays(cfg: dict, output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data_cfg = cfg["data"]
    cache_dir = Path(cfg.get("feature_cache_dir", output_dir / "cache"))
    cache_tag = (
        f"openimages_{data_cfg['split']}_{data_cfg['max_images']}_"
        f"{data_cfg['num_classes']}_{cfg['clip']['model'].replace('/', '-')}"
    )
    paths = {
        "features": cache_dir / f"{cache_tag}_image_features.npy",
        "labels": cache_dir / f"{cache_tag}_labels.npy",
        "text": cache_dir / f"{cache_tag}_text_features.npy",
        "classes": cache_dir / f"{cache_tag}_classes.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing cached arrays: " + ", ".join(missing))
    return (
        np.load(paths["features"]),
        np.load(paths["labels"]),
        np.load(paths["text"]),
        json.loads(paths["classes"].read_text(encoding="utf-8")),
    )


def read_run_splits(cfg: dict, output_dir: Path, class_names: list[str], retrieval_labels: np.ndarray) -> list[dict]:
    metrics_path = output_dir / "openimages_pareto_metrics.json"
    if metrics_path.exists():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        if result.get("splits"):
            return result["splits"]
    return build_splits(class_names, retrieval_labels, cfg["protocol"], int(cfg["seed"]))


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (
            row["method"],
            row.get("margin_strength"),
            row.get("margin_temperature"),
            row.get("tail_weight"),
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for (method, strength, temperature, tail_weight), group in grouped.items():
        out = {
            "method": method,
            "margin_strength": strength,
            "margin_temperature": temperature,
            "tail_weight": tail_weight,
        }
        for metric in METRICS:
            vals = [row[metric] for row in group if row[metric] is not None]
            out[f"{metric}_mean"] = float(np.mean(vals)) if vals else None
            out[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(out)
    summary.sort(key=lambda row: (
        row["method"],
        -1.0 if row["margin_strength"] is None else float(row["margin_strength"]),
        -1.0 if row["margin_temperature"] is None else float(row["margin_temperature"]),
        -1.0 if row["tail_weight"] is None else float(row["tail_weight"]),
    ))
    return summary


def non_dominated(rows: list[dict], quality_metric: str) -> list[dict]:
    candidates = [row for row in rows if row["method"] == "elta_margin"]
    front = []
    for row in candidates:
        q = row[f"{quality_metric}_mean"]
        t = row["tecr_mean"]
        if q is None or t is None:
            continue
        dominated = False
        for other in candidates:
            if other is row:
                continue
            oq = other[f"{quality_metric}_mean"]
            ot = other["tecr_mean"]
            if oq is None or ot is None:
                continue
            if oq >= q and ot <= t and (oq > q or ot < t):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda row: (row["tecr_mean"], -row[f"{quality_metric}_mean"]))


def best_under_constraint(rows: list[dict], metric: str, baseline: float, tolerance: float) -> dict | None:
    threshold = baseline * (1.0 - tolerance)
    candidates = [
        row
        for row in rows
        if row["method"] == "elta_margin"
        and row[f"{metric}_mean"] is not None
        and row["tecr_mean"] is not None
        and row[f"{metric}_mean"] >= threshold
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["tecr_mean"], -row[f"{metric}_mean"]))


def setting(row: dict) -> str:
    if row["method"] == "clip_knn":
        return "CLIP+kNN"
    return (
        f"ELTA-M s={row['margin_strength']:.2f}, "
        f"tau={row['margin_temperature']:.2f}, w={row['tail_weight']:.2f}"
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, baseline: dict, best: dict, matched: list[dict]) -> None:
    lines = [
        "# Open Images Margin Gate Analysis",
        "",
        "Date: 2026-05-22",
        "",
        "Baseline:",
        "",
        "| Method | AP | AUROC | F1 | TECR |",
        "|---|---:|---:|---:|---:|",
        (
            f"| CLIP+kNN | {baseline['average_precision_mean']:.4f} | "
            f"{baseline['auroc_mean']:.4f} | {baseline['best_f1_mean']:.4f} | "
            f"{baseline['tecr_mean']:.4f} |"
        ),
        "",
        "Best margin-gate Pareto point:",
        "",
        "| Setting | AP | AUROC | F1 | TECR | TECR reduction |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| {setting(best)} | {best['average_precision_mean']:.4f} | "
            f"{best['auroc_mean']:.4f} | {best['best_f1_mean']:.4f} | "
            f"{best['tecr_mean']:.4f} | "
            f"{pct_reduction(baseline['tecr_mean'], best['tecr_mean']):.1f}% |"
        ),
        "",
        "Matched-quality operating points:",
        "",
        "| Constraint | Setting | AP | F1 | TECR | AP delta | F1 delta | TECR reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in matched:
        lines.append(
            f"| {row['constraint']} <= {row['tolerance'] * 100:.1f}% loss | "
            f"{row['setting']} | {row['average_precision_mean']:.4f} | "
            f"{row['best_f1_mean']:.4f} | {row['tecr_mean']:.4f} | "
            f"{row['ap_delta_pct']:.2f}% | {row['f1_delta_pct']:.2f}% | "
            f"{row['tecr_reduction_pct']:.1f}% |"
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- ELTA-M gates emerging scores only when known-label evidence exceeds emerging retrieval evidence.",
        "- Treat this as an exploratory variant until it is validated on larger samples and additional baselines.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/openimages_10k_heldout_ultrastrict.yaml")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.get("margin_output_dir", "outputs/openimages_10k_margin"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    features, labels, text_features, class_names = load_cached_arrays(cfg, output_dir)
    retrieval_features, retrieval_labels, eval_features, eval_labels = split_retrieval_eval(
        features,
        labels,
        int(cfg["seed"]),
        float(cfg["data"].get("retrieval_fraction", 0.5)),
    )
    splits = read_run_splits(cfg, output_dir, class_names, retrieval_labels)

    retrieval_logits_raw = retrieval_features @ text_features.T
    logit_mean = retrieval_logits_raw.mean(axis=0, keepdims=True)
    logit_std = retrieval_logits_raw.std(axis=0, keepdims=True) + 1e-8
    logits = (eval_features @ text_features.T - logit_mean) / logit_std

    margin_cfg = cfg.get("margin_gate", {})
    strengths = margin_cfg.get("strengths", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    temperatures = margin_cfg.get("temperatures", [0.05, 0.1, 0.2, 0.35])
    tail_weights = margin_cfg.get("tail_weights", [0.0, 0.15])

    rows = []
    raw = []
    for split in splits:
        emerging_idx = [class_names.index(x) for x in split["emerging_labels"]]
        tail_idx = [class_names.index(x) for x in split["tail_known_labels"]]
        known_idx = [i for i, name in enumerate(class_names) if name not in split["emerging_labels"]]

        known_explanation = logits[:, known_idx].max(axis=1)
        known_explanation = 1.0 / (1.0 + np.exp(-known_explanation))
        tail_text = text_features[tail_idx]
        emerging_text = text_features[emerging_idx]
        tail_conf = np.maximum(0.0, emerging_text @ tail_text.T).max(axis=1)

        knn_scores = knn_score_matrix(retrieval_features, retrieval_labels, eval_features, emerging_idx, cfg["knn"])
        base = evaluate_scores(knn_scores, eval_labels, emerging_idx, tail_idx)
        base_row = {
            "method": "clip_knn",
            "split": split["name"],
            "margin_strength": None,
            "margin_temperature": None,
            "tail_weight": None,
            "average_precision": base["average_precision"],
            "auroc": base["auroc"],
            "best_f1": base["best_f1"]["f1"],
            "tecr": base["tecr"],
        }
        rows.append(base_row)
        raw.append({**base_row, "threshold": base["best_f1"]["threshold"]})

        best_emerging_local = knn_scores.argmax(axis=1)
        sample_tail_conf = tail_conf[best_emerging_local]
        for strength in strengths:
            strength = float(strength)
            for temperature in temperatures:
                temperature = float(temperature)
                margin = (known_explanation[:, None] - knn_scores) / max(temperature, 1e-6)
                margin_gate = 1.0 - strength * sigmoid(margin)
                for tail_weight in tail_weights:
                    tail_weight = float(tail_weight)
                    semantic_gate = np.maximum(0.0, 1.0 - tail_weight * sample_tail_conf)
                    scores = knn_scores * margin_gate * semantic_gate[:, None]
                    metrics = evaluate_scores(scores, eval_labels, emerging_idx, tail_idx)
                    row = {
                        "method": "elta_margin",
                        "split": split["name"],
                        "margin_strength": strength,
                        "margin_temperature": temperature,
                        "tail_weight": tail_weight,
                        "average_precision": metrics["average_precision"],
                        "auroc": metrics["auroc"],
                        "best_f1": metrics["best_f1"]["f1"],
                        "tecr": metrics["tecr"],
                    }
                    rows.append(row)
                    raw.append({**row, "threshold": metrics["best_f1"]["threshold"]})

    summary = summarize_rows(rows)
    baseline = next(row for row in summary if row["method"] == "clip_knn")
    front = non_dominated(summary, "best_f1")
    best = min(front, key=lambda row: (row["tecr_mean"], -row["best_f1_mean"]))

    matched = []
    for metric, label in [("best_f1", "F1"), ("average_precision", "AP")]:
        for tolerance in [0.005, 0.01, 0.02]:
            row = best_under_constraint(summary, metric, baseline[f"{metric}_mean"], tolerance)
            if row is None:
                continue
            matched.append({
                "constraint": label,
                "tolerance": tolerance,
                "setting": setting(row),
                "average_precision_mean": row["average_precision_mean"],
                "best_f1_mean": row["best_f1_mean"],
                "tecr_mean": row["tecr_mean"],
                "ap_delta_pct": pct_change(baseline["average_precision_mean"], row["average_precision_mean"]),
                "f1_delta_pct": pct_change(baseline["best_f1_mean"], row["best_f1_mean"]),
                "tecr_reduction_pct": pct_reduction(baseline["tecr_mean"], row["tecr_mean"]),
            })

    summary_fields = [
        "method", "margin_strength", "margin_temperature", "tail_weight",
        "average_precision_mean", "average_precision_std",
        "auroc_mean", "auroc_std",
        "best_f1_mean", "best_f1_std",
        "tecr_mean", "tecr_std",
    ]
    raw_fields = [
        "method", "split", "margin_strength", "margin_temperature", "tail_weight",
        "average_precision", "auroc", "best_f1", "tecr", "threshold",
    ]
    matched_fields = [
        "constraint", "tolerance", "setting", "average_precision_mean",
        "best_f1_mean", "tecr_mean", "ap_delta_pct", "f1_delta_pct",
        "tecr_reduction_pct",
    ]
    write_csv(output_dir / "openimages_margin_summary.csv", summary, summary_fields)
    write_csv(output_dir / "openimages_margin_raw.csv", raw, raw_fields)
    write_csv(output_dir / "openimages_margin_pareto_front.csv", front, summary_fields)
    write_csv(output_dir / "matched_comparisons.csv", matched, matched_fields)
    write_report(output_dir / "margin_gate_analysis.md", baseline, best, matched)

    result = {
        "status": "openimages_margin_gate_complete",
        "output_dir": str(output_dir),
        "num_images": int(labels.shape[0]),
        "num_eval_images": int(eval_labels.shape[0]),
        "num_splits": len(splits),
        "baseline": baseline,
        "best": best,
        "matched": matched,
    }
    (output_dir / "openimages_margin_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
