from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from run_openimages_heldout_calibration import (
    best_f1_threshold,
    build_splits,
    metrics_with_calibrated_threshold,
    pct_change,
    pct_reduction,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageIdDataset(Dataset):
    def __init__(self, image_dir: Path, image_ids: list[str], transform):
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_dir / f"{self.image_ids[idx]}.jpg").convert("RGB")
        return self.transform(image)


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
        "classes": cache_dir / f"{cache_tag}_classes.json",
        "image_ids": cache_dir / f"{cache_tag}_image_ids.json",
    }


def load_labels_classes_ids(cfg: dict) -> tuple[np.ndarray, list[str], list[str]]:
    paths = cache_paths(cfg, Path(cfg["output_dir"]))
    required = [paths["labels"], paths["classes"], paths["image_ids"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing cached arrays: " + ", ".join(missing))
    return (
        np.load(paths["labels"]),
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


def import_mkt_modules(mkt_root: Path):
    sys.path.insert(0, str(mkt_root))
    import clip  # type: ignore
    from models.clip_vit import CLIPVIT  # type: ignore
    from models.prompt_model import PromptLearner  # type: ignore
    from utils.misc import convert_models_to_fp32  # type: ignore
    from utils.transforms import build_transform  # type: ignore

    return clip, CLIPVIT, PromptLearner, convert_models_to_fp32, build_transform


def mkt_args(clip_model: str) -> SimpleNamespace:
    return SimpleNamespace(
        device="cuda" if torch.cuda.is_available() else "cpu",
        clip_path=clip_model,
        input_size=224,
        bert_embed_dim=512,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        topk=18,
    )


@torch.no_grad()
def mkt_text_features(text_encoder, class_names: list[str]) -> torch.Tensor:
    labels = [name + "\n" for name in class_names]
    text_encoder.load_label_emb(labels)
    return text_encoder("all").float()


def load_state(path: Path, device: str):
    return torch.load(path, map_location=device)


@torch.no_grad()
def compute_mkt_scores(
    cfg: dict,
    class_names: list[str],
    image_ids: list[str],
    output_root: Path,
    mkt_root: Path,
    first_stage_ckpt: Path,
    second_stage_ckpt: Path,
    batch_size: int,
    device: str,
) -> np.ndarray:
    cache_name = (
        f"openimages_{cfg['data']['split']}_{cfg['data']['max_images']}_{cfg['data']['num_classes']}_"
        f"{cfg['clip']['model'].replace('/', '-')}_mkt_scores.npy"
    )
    cache_path = output_root / "mkt_cache" / cache_name
    if cache_path.exists():
        return np.load(cache_path)

    clip, CLIPVIT, PromptLearner, convert_models_to_fp32, build_transform = import_mkt_modules(mkt_root)
    args = mkt_args("ViT-B/16")
    args.device = device

    clip_model, _ = clip.load(args.clip_path, device=device, jit=False)
    image_encoder = CLIPVIT(args, clip_model)
    convert_models_to_fp32(image_encoder)
    image_encoder.load_state_dict(load_state(first_stage_ckpt, device), strict=True)
    image_encoder = image_encoder.eval().to(device)

    text_encoder = PromptLearner(args).to(device)
    txt_ckpt = load_state(second_stage_ckpt, device)
    if next(iter(txt_ckpt.items()))[0].startswith("module"):
        txt_ckpt = {k[len("module."):]: v for k, v in txt_ckpt.items()}
    text_encoder.load_state_dict(txt_ckpt, strict=True)
    text_encoder = text_encoder.eval()
    txt_feat = mkt_text_features(text_encoder, class_names)

    transform = build_transform(False, args)
    image_dir = Path(cfg["data"]["root"]) / str(cfg["data"]["split"])
    dataset = ImageIdDataset(image_dir, image_ids, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["clip"].get("num_workers", 2)),
        pin_memory=(device == "cuda"),
    )

    scores = []
    for images in loader:
        images = images.to(device, non_blocking=True)
        pred_feat, dist_feat = image_encoder.encode_img(images)
        score1 = torch.topk(pred_feat @ txt_feat.t(), k=image_encoder.topk, dim=1)[0].mean(dim=1)
        score2 = dist_feat @ txt_feat.t()
        score1 = score1 / score1.norm(dim=-1, keepdim=True)
        score2 = score2 / score2.norm(dim=-1, keepdim=True)
        logits = (score1 + score2) / 2
        scores.append(logits.float().cpu().numpy())
    out = np.concatenate(scores, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, out)
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
        for key in ["average_precision", "auroc", "precision", "recall", "best_f1", "tecr"]:
            vals = [row[key] for row in group if row.get(key) is not None]
            item[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    return sorted(out, key=lambda row: row["method"])


def run_one(
    config_path: Path,
    output_dir: Path,
    seed_override: int | None,
    mkt_root: Path,
    first_stage_ckpt: Path,
    second_stage_ckpt: Path,
) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_seed = int(seed_override) if seed_override is not None else int(cfg["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    labels, class_names, image_ids = load_labels_classes_ids(cfg)
    scores = compute_mkt_scores(
        cfg,
        class_names,
        image_ids,
        output_dir.parent,
        mkt_root,
        first_stage_ckpt,
        second_stage_ckpt,
        int(cfg["clip"].get("batch_size", 96)),
        device,
    )

    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))
    retrieval_idx, calibration_idx, eval_idx = split_indices(labels.shape[0], run_seed, retrieval_fraction, calibration_fraction)
    retrieval_labels = labels[retrieval_idx]
    calibration_labels = labels[calibration_idx]
    eval_labels = labels[eval_idx]
    calibration_scores = scores[calibration_idx]
    eval_scores = scores[eval_idx]
    splits = build_splits(class_names, retrieval_labels, calibration_labels, cfg["protocol"], run_seed)

    rows = []
    for split in splits:
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
        cal_matrix = calibration_scores[:, emerging_idx]
        eval_matrix = eval_scores[:, emerging_idx]
        cal_image_scores = cal_matrix.max(axis=1)
        eval_image_scores = eval_matrix.max(axis=1)
        y_cal = calibration_labels[:, emerging_idx].max(axis=1).astype(int)
        y_eval = eval_labels[:, emerging_idx].max(axis=1).astype(int)
        threshold = best_f1_threshold(y_cal, cal_image_scores)
        metrics = metrics_with_calibrated_threshold(
            y_eval,
            eval_image_scores,
            threshold["threshold"],
            eval_labels,
            tail_idx,
            emerging_idx,
        )
        metrics.update({"method": "mkt_open_vocab", "split": split["name"]})
        rows.append(metrics)

    summary = summarize(rows)
    write_csv(output_dir / "mkt_eval_rows.csv", rows, [
        "method", "split", "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ])
    write_csv(output_dir / "mkt_summary.csv", summary, [
        "method", "num_splits", "average_precision_mean", "average_precision_std", "auroc_mean", "auroc_std",
        "precision_mean", "precision_std", "recall_mean", "recall_std", "best_f1_mean", "best_f1_std",
        "tecr_mean", "tecr_std",
    ])
    report = [
        "# MKT Open-Vocabulary Baseline",
        "",
        f"Config: `{config_path}`",
        f"Seed: `{run_seed}`",
        "",
        "| Method | AP | F1 | TECR |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['method']} | {row['average_precision_mean']:.4f} | "
            f"{row['best_f1_mean']:.4f} | {row['tecr_mean']:.4f} |"
        )
    (output_dir / "mkt_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "status": "mkt_baseline_complete",
        "config": str(config_path),
        "seed": run_seed,
        "output_dir": str(output_dir),
        "summary": summary,
    }
    (output_dir / "mkt_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def combine_results(root_dir: Path) -> None:
    rows = []
    for path in sorted(root_dir.glob("*/mkt_summary.csv")):
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
        for key in ["average_precision_mean", "best_f1_mean", "tecr_mean"]:
            vals = [float(row[key]) for row in group if row.get(key) not in (None, "")]
            out[key] = float(np.mean(vals)) if vals else None
            out[key.replace("_mean", "") + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(out)
    write_csv(root_dir / "mkt_combined_summary.csv", summary, [
        "method", "n", "average_precision_mean", "average_precision_std",
        "best_f1_mean", "best_f1_std", "tecr_mean", "tecr_std",
    ])
    lines = [
        "# MKT Baseline Combined Summary",
        "",
        f"Output directory: `{root_dir}`",
        "",
        "| Method | n | AP | F1 | TECR |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        def fmt(key: str) -> str:
            std_key = key.replace("_mean", "") + "_std"
            return f"{row[key]:.4f} +/- {row.get(std_key, 0.0):.4f}"

        lines.append(
            f"| {row['method']} | {row['n']} | {fmt('average_precision_mean')} | "
            f"{fmt('best_f1_mean')} | {fmt('tecr_mean')} |"
        )
    (root_dir / "mkt_combined_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--mkt-root")
    parser.add_argument("--first-stage-ckpt")
    parser.add_argument("--second-stage-ckpt")
    parser.add_argument("--combine-root")
    args = parser.parse_args()
    if args.combine_root:
        combine_results(Path(args.combine_root))
        return 0
    if not args.config or not args.output_dir:
        parser.error("--config and --output-dir are required unless --combine-root is used")
    if not args.mkt_root or not args.first_stage_ckpt or not args.second_stage_ckpt:
        parser.error("--mkt-root, --first-stage-ckpt, and --second-stage-ckpt are required for MKT runs")
    run_one(
        Path(args.config),
        Path(args.output_dir),
        args.seed_override,
        Path(args.mkt_root),
        Path(args.first_stage_ckpt),
        Path(args.second_stage_ckpt),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
