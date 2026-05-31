from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import clip
import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from run_openimages_heldout_calibration import (
    aggregate_scores,
    best_f1_threshold,
    build_splits,
    metrics_with_calibrated_threshold,
    pct_change,
    pct_reduction,
)
from run_openimages_pilot import knn_score_matrix


ImageFile.LOAD_TRUNCATED_IMAGES = True


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class ImageIdDataset(Dataset):
    def __init__(self, image_dir: Path, image_ids: list[str], preprocess):
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_dir / f"{self.image_ids[idx]}.jpg").convert("RGB")
        return self.preprocess(image)


def cache_paths(cfg: dict, output_dir: Path) -> dict[str, Path]:
    data_cfg = cfg["data"]
    cache_dir = Path(cfg.get("feature_cache_dir", output_dir / "cache"))
    cache_tag = (
        f"openimages_{data_cfg['split']}_{data_cfg['max_images']}_"
        f"{data_cfg['num_classes']}_{cfg['clip']['model'].replace('/', '-')}"
    )
    return {
        "features": cache_dir / f"{cache_tag}_image_features.npy",
        "labels": cache_dir / f"{cache_tag}_labels.npy",
        "text": cache_dir / f"{cache_tag}_text_features.npy",
        "classes": cache_dir / f"{cache_tag}_classes.json",
        "image_ids": cache_dir / f"{cache_tag}_image_ids.json",
        "tag": Path(cache_tag),
    }


def load_cached_arrays_with_ids(cfg: dict, output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    paths = cache_paths(cfg, output_dir)
    required = [paths["features"], paths["labels"], paths["text"], paths["classes"], paths["image_ids"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing cached arrays: " + ", ".join(missing))
    return (
        np.load(paths["features"]),
        np.load(paths["labels"]),
        np.load(paths["text"]),
        json.loads(paths["classes"].read_text(encoding="utf-8")),
        json.loads(paths["image_ids"].read_text(encoding="utf-8")),
    )


def split_indices(n: int, seed: int, retrieval_fraction: float, calibration_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_retrieval = int(round(n * retrieval_fraction))
    n_calibration = int(round(n * calibration_fraction))
    retrieval_idx = np.sort(indices[:n_retrieval])
    calibration_idx = np.sort(indices[n_retrieval:n_retrieval + n_calibration])
    eval_idx = np.sort(indices[n_retrieval + n_calibration:])
    return retrieval_idx, calibration_idx, eval_idx


@torch.no_grad()
def encode_text_from_names(model, class_names: list[str], device: str) -> torch.Tensor:
    tokens = clip.tokenize([f"a photo of {name}" for name in class_names]).to(device)
    text = model.encode_text(tokens).float()
    return F.normalize(text, dim=-1)


def compute_odin_scores(
    cfg: dict,
    class_names: list[str],
    image_ids: list[str],
    output_dir: Path,
    odin_temperature: float,
    odin_epsilon: float,
    batch_size: int,
    device: str,
) -> np.ndarray:
    paths = cache_paths(cfg, Path(cfg["output_dir"]))
    odin_cache = output_dir / "odin_cache" / (
        f"{paths['tag'].name}_T{odin_temperature:g}_eps{odin_epsilon:g}_scores.npy"
    )
    if odin_cache.exists():
        return np.load(odin_cache)

    model, preprocess = clip.load(cfg["clip"]["model"], device=device)
    model.eval()
    text_features = encode_text_from_names(model, class_names, device)
    image_dir = Path(cfg["data"]["root"]) / str(cfg["data"]["split"])
    dataset = ImageIdDataset(image_dir, image_ids, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["clip"].get("num_workers", 2)),
        pin_memory=(device == "cuda"),
    )
    lower = ((0.0 - CLIP_MEAN) / CLIP_STD).to(device)
    upper = ((1.0 - CLIP_MEAN) / CLIP_STD).to(device)
    scores = []
    scale = model.logit_scale.exp().float().detach()
    for batch in loader:
        images = batch.to(device, non_blocking=True)
        images.requires_grad_(True)
        image_features = model.encode_image(images).float()
        image_features = F.normalize(image_features, dim=-1)
        logits = scale * image_features @ text_features.T
        logits_t = logits / max(odin_temperature, 1e-6)
        pred = logits_t.argmax(dim=1)
        loss = F.cross_entropy(logits_t, pred)
        model.zero_grad(set_to_none=True)
        loss.backward()
        perturbed = images - float(odin_epsilon) * torch.sign(images.grad)
        perturbed = torch.max(torch.min(perturbed, upper), lower).detach()
        with torch.no_grad():
            perturbed_features = model.encode_image(perturbed).float()
            perturbed_features = F.normalize(perturbed_features, dim=-1)
            perturbed_logits = scale * perturbed_features @ text_features.T
            odin_score = F.softmax(perturbed_logits / max(odin_temperature, 1e-6), dim=1).max(dim=1).values
        scores.append(odin_score.cpu().numpy())
    out = np.concatenate(scores, axis=0)
    odin_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(odin_cache, out)
    return out


def reject_low_odin(scores: np.ndarray, odin_scores: np.ndarray, cutoff: float) -> np.ndarray:
    out = np.array(scores, copy=True)
    out[odin_scores <= cutoff] = 0.0
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    out = []
    for method, group in grouped.items():
        item = {"method": method, "num_splits": len(group)}
        for key in ["average_precision", "auroc", "precision", "recall", "best_f1", "tecr", "avg_predicted_labels", "coverage"]:
            vals = [row[key] for row in group if row.get(key) is not None]
            item[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    return sorted(out, key=lambda row: row["method"])


def add_prediction_load(metrics: dict, score_matrix: np.ndarray, threshold: float, labels: np.ndarray, emerging_idx: list[int]) -> dict:
    pred_matrix = score_matrix >= threshold
    metrics["avg_predicted_labels"] = float(pred_matrix.sum(axis=1).mean())
    y_true = labels[:, emerging_idx].max(axis=1).astype(bool)
    metrics["coverage"] = float(pred_matrix.max(axis=1)[y_true].mean()) if y_true.any() else None
    return metrics


def select_cutoff(rows: list[dict], baseline: dict, ap_tolerance: float, f1_tolerance: float) -> dict | None:
    candidates = [
        row
        for row in rows
        if row.get("average_precision") is not None
        and row.get("best_f1") is not None
        and row.get("tecr") is not None
        and row["average_precision"] >= baseline["average_precision"] * (1.0 - ap_tolerance)
        and row["best_f1"] >= baseline["best_f1"] * (1.0 - f1_tolerance)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["tecr"], -row["best_f1"], -row["average_precision"]))


def run_one(config_path: Path, output_dir: Path, seed_override: int | None, odin_temperature: float, odin_epsilon: float) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_seed = int(seed_override) if seed_override is not None else int(cfg["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    features, labels, text_features, class_names, image_ids = load_cached_arrays_with_ids(cfg, Path(cfg["output_dir"]))
    odin_scores = compute_odin_scores(
        cfg,
        class_names,
        image_ids,
        output_dir.parent,
        odin_temperature,
        odin_epsilon,
        int(cfg["clip"].get("batch_size", 96)),
        device,
    )

    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))
    ap_tolerance = float(calibration_cfg.get("ap_tolerance", 0.0025))
    f1_tolerance = float(calibration_cfg.get("f1_tolerance", 0.0025))
    retrieval_idx, calibration_idx, eval_idx = split_indices(labels.shape[0], run_seed, retrieval_fraction, calibration_fraction)
    retrieval_features = features[retrieval_idx]
    retrieval_labels = labels[retrieval_idx]
    calibration_features = features[calibration_idx]
    calibration_labels = labels[calibration_idx]
    eval_features = features[eval_idx]
    eval_labels = labels[eval_idx]
    cal_odin = odin_scores[calibration_idx]
    eval_odin = odin_scores[eval_idx]

    splits = build_splits(class_names, retrieval_labels, calibration_labels, cfg["protocol"], run_seed)
    cutoff_grid = sorted(set(np.quantile(cal_odin, np.linspace(0.0, 1.0, 51)).tolist() + [-1.0, 2.0]))

    eval_rows: list[dict] = []
    grid_rows: list[dict] = []
    selection_rows: list[dict] = []
    for split in splits:
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
        cal_knn = knn_score_matrix(retrieval_features, retrieval_labels, calibration_features, emerging_idx, cfg["knn"])
        eval_knn = knn_score_matrix(retrieval_features, retrieval_labels, eval_features, emerging_idx, cfg["knn"])
        y_cal = calibration_labels[:, emerging_idx].max(axis=1).astype(int)
        y_eval = eval_labels[:, emerging_idx].max(axis=1).astype(int)
        cal_base_scores = aggregate_scores(cal_knn)
        eval_base_scores = aggregate_scores(eval_knn)
        cal_threshold = best_f1_threshold(y_cal, cal_base_scores)
        cal_baseline = metrics_with_calibrated_threshold(
            y_cal, cal_base_scores, cal_threshold["threshold"], calibration_labels, tail_idx, emerging_idx
        )
        eval_baseline = metrics_with_calibrated_threshold(
            y_eval, eval_base_scores, cal_threshold["threshold"], eval_labels, tail_idx, emerging_idx
        )
        add_prediction_load(eval_baseline, eval_knn, cal_threshold["threshold"], eval_labels, emerging_idx)
        eval_baseline.update({"method": "clip_knn", "split": split["name"], "selection_status": "baseline", "odin_cutoff": None})
        eval_rows.append(eval_baseline)

        eval_by_cutoff = {}
        cal_candidates = []
        for cutoff in cutoff_grid:
            cutoff = float(cutoff)
            cal_scores = reject_low_odin(cal_base_scores, cal_odin, cutoff)
            threshold = best_f1_threshold(y_cal, cal_scores)
            cal_metrics = metrics_with_calibrated_threshold(
                y_cal, cal_scores, threshold["threshold"], calibration_labels, tail_idx, emerging_idx
            )
            cal_metrics.update({"method": "odin_low_reject", "split": split["name"], "odin_cutoff": cutoff})
            cal_candidates.append(cal_metrics)
            grid_rows.append(cal_metrics)

            eval_matrix = reject_low_odin(eval_knn, eval_odin, cutoff)
            eval_scores = aggregate_scores(eval_matrix)
            eval_metrics = metrics_with_calibrated_threshold(
                y_eval, eval_scores, threshold["threshold"], eval_labels, tail_idx, emerging_idx
            )
            add_prediction_load(eval_metrics, eval_matrix, threshold["threshold"], eval_labels, emerging_idx)
            eval_by_cutoff[cutoff] = eval_metrics

        selected = select_cutoff(cal_candidates, cal_baseline, ap_tolerance, f1_tolerance)
        if selected is None:
            cutoff = -1.0
            status = "fallback_baseline"
        else:
            cutoff = float(selected["odin_cutoff"])
            status = "selected_under_constraints"
        eval_selected = dict(eval_by_cutoff.get(cutoff, eval_baseline))
        eval_selected.update({"method": "odin_low_reject", "split": split["name"], "selection_status": status, "odin_cutoff": cutoff})
        eval_rows.append(eval_selected)
        selection_rows.append({
            "split": split["name"],
            "selection_status": status,
            "odin_cutoff": cutoff,
            "baseline_calibration_ap": cal_baseline["average_precision"],
            "baseline_calibration_f1": cal_baseline["best_f1"],
            "baseline_calibration_tecr": cal_baseline["tecr"],
            "selected_calibration_ap": None if selected is None else selected["average_precision"],
            "selected_calibration_f1": None if selected is None else selected["best_f1"],
            "selected_calibration_tecr": None if selected is None else selected["tecr"],
        })

    summary = summarize(eval_rows)
    baseline_summary = next(row for row in summary if row["method"] == "clip_knn")
    for row in summary:
        if row["method"] == "clip_knn":
            row["ap_delta_pct"] = 0.0
            row["f1_delta_pct"] = 0.0
            row["tecr_reduction_pct"] = 0.0
        else:
            row["ap_delta_pct"] = pct_change(baseline_summary["average_precision_mean"], row["average_precision_mean"])
            row["f1_delta_pct"] = pct_change(baseline_summary["best_f1_mean"], row["best_f1_mean"])
            row["tecr_reduction_pct"] = pct_reduction(baseline_summary["tecr_mean"], row["tecr_mean"])

    eval_fields = [
        "method", "split", "selection_status", "odin_cutoff",
        "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
        "avg_predicted_labels", "coverage",
    ]
    summary_fields = [
        "method", "num_splits", "average_precision_mean", "average_precision_std",
        "auroc_mean", "auroc_std", "precision_mean", "precision_std", "recall_mean", "recall_std",
        "best_f1_mean", "best_f1_std", "tecr_mean", "tecr_std",
        "avg_predicted_labels_mean", "avg_predicted_labels_std", "coverage_mean", "coverage_std",
        "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct",
    ]
    write_csv(output_dir / "odin_eval_rows.csv", eval_rows, eval_fields)
    write_csv(output_dir / "odin_summary.csv", summary, summary_fields)
    write_csv(output_dir / "odin_selection_rows.csv", selection_rows, [
        "split", "selection_status", "odin_cutoff", "baseline_calibration_ap", "baseline_calibration_f1",
        "baseline_calibration_tecr", "selected_calibration_ap", "selected_calibration_f1", "selected_calibration_tecr",
    ])
    write_csv(output_dir / "odin_calibration_grid.csv", grid_rows, [
        "method", "split", "odin_cutoff", "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ])

    report = [
        "# Open Images ODIN Baseline",
        "",
        f"Config: `{config_path}`",
        f"Seed: `{run_seed}`",
        f"ODIN temperature: `{odin_temperature}`",
        f"ODIN epsilon: `{odin_epsilon}`",
        "",
        "ODIN is used as a standard image-level OOD score: images with low ODIN confidence are rejected by zeroing emerging-label scores. The cutoff is selected on calibration under the same AP/F1 preservation constraints.",
        "",
        "| Method | AP | F1 | TECR | Avg. labels | Coverage | AP delta | F1 delta | TECR reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['method']} | {row['average_precision_mean']:.4f} | {row['best_f1_mean']:.4f} | "
            f"{row['tecr_mean']:.4f} | {row['avg_predicted_labels_mean']:.4f} | "
            f"{row['coverage_mean']:.4f} | {row['ap_delta_pct']:.2f}% | "
            f"{row['f1_delta_pct']:.2f}% | {row['tecr_reduction_pct']:.1f}% |"
        )
    (output_dir / "odin_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "status": "odin_baseline_complete",
        "config": str(config_path),
        "seed": run_seed,
        "output_dir": str(output_dir),
        "odin_temperature": odin_temperature,
        "odin_epsilon": odin_epsilon,
        "summary": summary,
    }
    (output_dir / "odin_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def combine_results(root_dir: Path) -> None:
    rows = []
    for path in sorted(root_dir.glob("*/odin_summary.csv")):
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["run"] = path.parent.name
                rows.append(row)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    summary = []
    for method, group in grouped.items():
        out = {"method": method, "n": len(group)}
        for key in [
            "average_precision_mean",
            "best_f1_mean",
            "tecr_mean",
            "avg_predicted_labels_mean",
            "coverage_mean",
            "ap_delta_pct",
            "f1_delta_pct",
            "tecr_reduction_pct",
        ]:
            vals = [float(row[key]) for row in group if row.get(key) not in (None, "")]
            out[key] = float(np.mean(vals)) if vals else None
            out[key.replace("_mean", "") + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(out)
    summary.sort(key=lambda row: 0 if row["method"] == "clip_knn" else 1)
    fields = [
        "method", "n", "average_precision_mean", "average_precision_std", "best_f1_mean", "best_f1_std",
        "tecr_mean", "tecr_std", "avg_predicted_labels_mean", "avg_predicted_labels_std",
        "coverage_mean", "coverage_std", "ap_delta_pct", "ap_delta_pct_std",
        "f1_delta_pct", "f1_delta_pct_std", "tecr_reduction_pct", "tecr_reduction_pct_std",
    ]
    write_csv(root_dir / "odin_combined_summary.csv", summary, fields)
    lines = [
        "# ODIN Baseline Combined Summary",
        "",
        f"Output directory: `{root_dir}`",
        "",
        "| Method | n | AP | F1 | TECR | Avg. labels | Coverage | AP delta | F1 delta | TECR reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def fmt(key: str) -> str:
            if row[key] is None:
                return ""
            std_key = key.replace("_mean", "") + "_std"
            return f"{row[key]:.4f} +/- {row.get(std_key, 0.0):.4f}"

        lines.append(
            f"| {row['method']} | {row['n']} | {fmt('average_precision_mean')} | "
            f"{fmt('best_f1_mean')} | {fmt('tecr_mean')} | {fmt('avg_predicted_labels_mean')} | "
            f"{fmt('coverage_mean')} | {row['ap_delta_pct']:.2f}% | "
            f"{row['f1_delta_pct']:.2f}% | {row['tecr_reduction_pct']:.1f}% |"
        )
    (root_dir / "odin_combined_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--odin-temperature", type=float, default=1000.0)
    parser.add_argument("--odin-epsilon", type=float, default=0.0014)
    parser.add_argument("--combine-root")
    args = parser.parse_args()
    if args.combine_root:
        combine_results(Path(args.combine_root))
        return 0
    if not args.config or not args.output_dir:
        parser.error("--config and --output-dir are required unless --combine-root is used")
    run_one(Path(args.config), Path(args.output_dir), args.seed_override, args.odin_temperature, args.odin_epsilon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
