from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch
import torch.nn.functional as F
import yaml

from run_openimages_heldout_calibration import build_splits, split_retrieval_calibration_eval
from run_openimages_margin_gate import load_cached_arrays


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    idx = int(np.nanargmax(f1))
    threshold = float(thresholds[idx]) if idx < len(thresholds) else float(scores.max() + 1e-6)
    return {
        "threshold": threshold,
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(f1[idx]),
    }


def tecr_from_predictions(pred_any: np.ndarray, labels: np.ndarray, tail_idx: list[int], emerging_idx: list[int]) -> float | None:
    eligible = (labels[:, tail_idx].max(axis=1) > 0) & (labels[:, emerging_idx].max(axis=1) == 0)
    if eligible.sum() == 0:
        return None
    return float((pred_any.astype(bool) & eligible).sum() / eligible.sum())


def metrics_with_threshold(
    y_true: np.ndarray,
    aggregate_scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
) -> dict:
    if len(np.unique(y_true)) < 2:
        ap = None
        auroc = None
    else:
        ap = float(average_precision_score(y_true, aggregate_scores))
        auroc = float(roc_auc_score(y_true, aggregate_scores))
    pred = aggregate_scores >= threshold
    tp = float(((pred == 1) & (y_true == 1)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "average_precision": ap,
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "best_f1": f1,
        "threshold": threshold,
        "tecr": tecr_from_predictions(pred, labels, tail_idx, emerging_idx),
    }


def class_thresholds(calibration_labels: np.ndarray, calibration_scores: np.ndarray) -> np.ndarray:
    thresholds = []
    for j in range(calibration_scores.shape[1]):
        y = calibration_labels[:, j].astype(int)
        scores = calibration_scores[:, j]
        if len(np.unique(y)) < 2:
            thresholds.append(float(scores.max() + 1e-6))
        else:
            thresholds.append(best_f1_threshold(y, scores)["threshold"])
    return np.asarray(thresholds, dtype=np.float32)


def aggregate_metrics_from_predictions(
    y_true: np.ndarray,
    aggregate_scores: np.ndarray,
    pred_any: np.ndarray,
    labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
) -> dict:
    if len(np.unique(y_true)) < 2:
        ap = None
        auroc = None
    else:
        ap = float(average_precision_score(y_true, aggregate_scores))
        auroc = float(roc_auc_score(y_true, aggregate_scores))
    tp = float(((pred_any == 1) & (y_true == 1)).sum())
    fp = float(((pred_any == 1) & (y_true == 0)).sum())
    fn = float(((pred_any == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "average_precision": ap,
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "best_f1": f1,
        "threshold": None,
        "tecr": tecr_from_predictions(pred_any, labels, tail_idx, emerging_idx),
    }


class LinearHead(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def sample_weight_from_labels(labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum(dim=0).clamp_min(1.0)
    inv = 1.0 / positives
    weights = (labels * inv[None, :]).sum(dim=1)
    weights = weights + weights.mean().clamp_min(1e-6)
    return weights / weights.sum()


def asymmetric_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """ASL, adapted from the official ICCV 2021 multi-label implementation."""
    probs = torch.sigmoid(logits)
    pos = probs
    neg = 1.0 - probs
    if clip > 0:
        neg = (neg + clip).clamp(max=1.0)
    loss = targets * torch.log(pos.clamp(min=eps)) + (1.0 - targets) * torch.log(neg.clamp(min=eps))
    if gamma_neg > 0 or gamma_pos > 0:
        with torch.no_grad():
            pt = pos * targets + neg * (1.0 - targets)
            gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
            weight = torch.pow(1.0 - pt, gamma)
        loss = loss * weight
    return -loss.mean()


def distribution_balanced_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_freq: torch.Tensor,
    train_size: int,
    beta: float = 0.9999,
    gamma: float = 2.0,
    map_alpha: float = 0.1,
    map_beta: float = 10.0,
    map_gamma: float = 0.2,
    neg_scale: float = 2.0,
    init_bias: float = 0.05,
) -> torch.Tensor:
    """DBLoss-style rebalanced focal BCE for long-tailed multi-label learning.

    This follows the two key operations from Wu et al., ECCV 2020: distribution
    re-balancing for co-occurring labels and negative-tolerant regularization.
    """
    class_freq = class_freq.clamp_min(1.0)
    freq_inv = 1.0 / class_freq
    repeat_rate = (targets * class_freq[None, :]).sum(dim=1, keepdim=True).clamp_min(1.0)
    rebalance = targets * freq_inv[None, :] / repeat_rate
    rebalance = torch.sigmoid(map_beta * (rebalance - map_gamma)) + map_alpha

    effective_num = 1.0 - torch.pow(torch.full_like(class_freq, beta), class_freq)
    cb_weight = (1.0 - beta) / effective_num.clamp_min(1e-12)
    cb_weight = cb_weight / cb_weight.mean().clamp_min(1e-12)
    weight = rebalance * cb_weight[None, :]

    logits = logits.clone()
    logits = logits + (1.0 - targets) * np.log(init_bias / (1.0 - init_bias)) / neg_scale
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal = torch.pow(1.0 - pt, gamma)
    loss = loss * focal * weight
    loss = loss * (targets + (1.0 - targets) * neg_scale)
    return loss.mean()


def train_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    method: str,
    device: str,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    mixup_alpha: float,
    init_weights: np.ndarray | None = None,
) -> LinearHead:
    set_seed(seed)
    x = torch.from_numpy(train_features).float().to(device)
    y = torch.from_numpy(train_labels).float().to(device)
    model = LinearHead(x.shape[1], y.shape[1]).to(device)
    if init_weights is not None:
        with torch.no_grad():
            weights = torch.from_numpy(init_weights).float().to(device)
            model.linear.weight.copy_(weights)
            model.linear.bias.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos = y.sum(dim=0)
    neg = y.shape[0] - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 50.0)
    sample_prob = sample_weight_from_labels(y)

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        if method == "balance_mix_feature":
            lam = float(np.random.beta(mixup_alpha, mixup_alpha)) if mixup_alpha > 0 else 1.0
            idx = torch.multinomial(sample_prob, num_samples=x.shape[0], replacement=True)
            perm = torch.randperm(x.shape[0], device=device)
            xb = lam * x[idx] + (1.0 - lam) * x[perm]
            yb = lam * y[idx] + (1.0 - lam) * y[perm]
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
        else:
            logits = model(x)
            if method == "class_balanced_bce":
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            elif method == "asl":
                loss = asymmetric_loss_with_logits(logits, y)
            elif method == "db_loss":
                loss = distribution_balanced_loss_with_logits(logits, y, pos, y.shape[0])
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def predict_scores(model: LinearHead, features: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(features).float().to(device)
    logits = model(x)
    return torch.sigmoid(logits).cpu().numpy()


def apply_logit_adjustment(scores: np.ndarray, priors: np.ndarray, strength: float) -> np.ndarray:
    logits = np.log(np.clip(scores, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - scores, 1e-6, 1.0))
    prior_logits = np.log(np.clip(priors, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - priors, 1e-6, 1.0))
    adjusted = logits - strength * prior_logits[None, :]
    return 1.0 / (1.0 + np.exp(-np.clip(adjusted, -50.0, 50.0)))


def add_global_and_class_rows(
    rows: list[dict],
    method: str,
    calibration_labels: np.ndarray,
    calibration_scores: np.ndarray,
    eval_scores: np.ndarray,
    y_cal: np.ndarray,
    y_eval: np.ndarray,
    eval_labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
    split_name: str,
    seed: int,
) -> None:
    cal_agg = calibration_scores.max(axis=1)
    eval_agg = eval_scores.max(axis=1)
    threshold = best_f1_threshold(y_cal, cal_agg)["threshold"]
    rows.append({
        **metrics_with_threshold(y_eval, eval_agg, threshold, eval_labels, tail_idx, emerging_idx),
        "method": f"{method}_global_threshold",
        "split": split_name,
        "seed": seed,
    })
    thresholds = class_thresholds(calibration_labels, calibration_scores)
    pred_any = (eval_scores >= thresholds[None, :]).max(axis=1)
    rows.append({
        **aggregate_metrics_from_predictions(y_eval, eval_agg, pred_any, eval_labels, tail_idx, emerging_idx),
        "method": f"{method}_class_thresholds",
        "split": split_name,
        "seed": seed,
    })


def summarize(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    out = []
    for method, group in grouped.items():
        item = {"method": method, "num_splits": len(group)}
        for key in ["average_precision", "auroc", "best_f1", "tecr"]:
            vals = [row[key] for row in group if row[key] is not None]
            item[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    baseline = next((row for row in out if row["method"] == "linear_bce_global_threshold"), None)
    if baseline:
        for row in out:
            row["ap_delta_pct"] = 0.0 if row is baseline else 100.0 * (row["average_precision_mean"] - baseline["average_precision_mean"]) / baseline["average_precision_mean"]
            row["f1_delta_pct"] = 0.0 if row is baseline else 100.0 * (row["best_f1_mean"] - baseline["best_f1_mean"]) / baseline["best_f1_mean"]
            row["tecr_reduction_pct"] = 0.0 if row is baseline else 100.0 * (baseline["tecr_mean"] - row["tecr_mean"]) / baseline["tecr_mean"]
    order = {
        "linear_bce_global_threshold": 0,
        "linear_bce_class_thresholds": 1,
        "class_balanced_bce_global_threshold": 2,
        "class_balanced_bce_class_thresholds": 3,
        "logit_adjusted_bce_global_threshold": 4,
        "logit_adjusted_bce_class_thresholds": 5,
        "asl_global_threshold": 6,
        "asl_class_thresholds": 7,
        "db_loss_global_threshold": 8,
        "db_loss_class_thresholds": 9,
        "balance_mix_feature_global_threshold": 10,
        "balance_mix_feature_class_thresholds": 11,
    }
    return sorted(out, key=lambda row: (order.get(row["method"], 99), row["method"]))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/openimages_10k_heldout_ultrastrict.yaml")
    parser.add_argument("--output-dir", default="outputs/openimages_10k_training_baselines")
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

    features, labels, text_features, class_names = load_cached_arrays(cfg, Path(cfg["output_dir"]))
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
        "# Open Images Training Baselines",
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
        "- `asl` is Asymmetric Loss for Multi-Label Classification (ICCV 2021), adapted as a frozen-feature linear-head baseline.",
        "- `db_loss` is Distribution-Balanced Loss for long-tailed multi-label classification (ECCV 2020), adapted as a frozen-feature linear-head baseline.",
        "- `balance_mix_feature` is a feature-space BalanceMix-style baseline, not the original image-space BalanceMix implementation.",
        "- `text_init_*` baselines initialize the linear classifier with CLIP text embeddings before training.",
        "- `*_logit_adjusted_bce` selects the prior-adjustment strength on the calibration split.",
        "",
    ])
    (output_dir / "training_baseline_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "status": "training_baselines_complete",
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
