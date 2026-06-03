from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
import yaml


METHODS = [
    "clip_knn",
    "score_only_matched_logistic",
    "permuted_known_confidence_logistic",
    "known_aware_two_feature_logistic",
]

METHOD_ORDER = {method: i for i, method in enumerate(METHODS)}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def infer_dataset(config_path: Path, explicit: str | None = None) -> str:
    if explicit and explicit != "auto":
        return explicit
    name = config_path.name.lower()
    if "coco" in name:
        return "coco"
    if "nuswide" in name:
        return "nuswide"
    return "openimages"


def model_tag(cfg: dict) -> str:
    return str(cfg["clip"]["model"]).replace("/", "-")


def split_retrieval_calibration_eval(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    retrieval_fraction: float,
    calibration_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    n_retrieval = int(round(len(indices) * retrieval_fraction))
    n_calibration = int(round(len(indices) * calibration_fraction))
    retrieval_idx = np.sort(indices[:n_retrieval])
    calibration_idx = np.sort(indices[n_retrieval:n_retrieval + n_calibration])
    eval_idx = np.sort(indices[n_retrieval + n_calibration:])
    return (
        features[retrieval_idx],
        labels[retrieval_idx],
        features[calibration_idx],
        labels[calibration_idx],
        features[eval_idx],
        labels[eval_idx],
    )


def build_splits(
    class_names: list[str],
    retrieval_labels: np.ndarray,
    calibration_labels: np.ndarray,
    protocol_cfg: dict,
    seed: int,
) -> list[dict]:
    retrieval_counts = retrieval_labels.sum(axis=0)
    calibration_counts = calibration_labels.sum(axis=0)
    eligible = [
        class_names[i]
        for i in range(len(class_names))
        if retrieval_counts[i] >= int(protocol_cfg.get("min_retrieval_positives", 5))
        and calibration_counts[i] >= int(protocol_cfg.get("min_calibration_positives", 3))
    ]
    split_seeds = protocol_cfg.get("split_seeds") or [seed + i for i in range(5)]
    emerging_count = min(int(protocol_cfg.get("emerging_count", 15)), len(eligible))
    tail_count = int(protocol_cfg.get("tail_count", 10))
    tail_pool = sorted(eligible, key=lambda name: (retrieval_counts[class_names.index(name)], name.lower()))
    splits = []
    for i, split_seed in enumerate(split_seeds):
        rng = random.Random(int(split_seed))
        shuffled = list(eligible)
        rng.shuffle(shuffled)
        emerging = sorted(shuffled[:emerging_count])
        tail_candidates = [name for name in tail_pool if name not in emerging]
        tail = sorted(tail_candidates[:tail_count])
        splits.append({
            "name": f"split_{i}",
            "seed": int(split_seed),
            "emerging_labels": emerging,
            "tail_known_labels": tail,
        })
    return splits


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if len(np.unique(y_true)) < 2:
        threshold = float(scores.max() + 1e-6) if int(y_true.sum()) == 0 else float(scores.min() - 1e-6)
        precision = 0.0 if int(y_true.sum()) == 0 else 1.0
        recall = 0.0 if int(y_true.sum()) == 0 else 1.0
        return {"threshold": threshold, "precision": precision, "recall": recall, "f1": precision}
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


def tecr_at_threshold(
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
) -> float | None:
    if not tail_idx or not emerging_idx:
        return None
    eligible = (labels[:, tail_idx].max(axis=1) > 0) & (labels[:, emerging_idx].max(axis=1) == 0)
    if eligible.sum() == 0:
        return None
    return float(((scores >= threshold) & eligible).sum() / eligible.sum())


def metrics_with_calibrated_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
) -> dict:
    if len(np.unique(y_true)) < 2:
        ap = None
        auroc = None
    else:
        ap = float(average_precision_score(y_true, scores))
        auroc = float(roc_auc_score(y_true, scores))
    pred = scores >= threshold
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
        "tecr": tecr_at_threshold(scores, threshold, labels, tail_idx, emerging_idx),
    }


def aggregate_scores(score_matrix: np.ndarray) -> np.ndarray:
    if score_matrix.ndim == 1:
        return score_matrix
    return score_matrix.max(axis=1)


def load_openimages_cached_arrays(cfg: dict, output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data_cfg = cfg["data"]
    cache_dir = Path(cfg.get("feature_cache_dir", output_dir / "cache"))
    cache_tag = (
        f"openimages_{data_cfg['split']}_{data_cfg['max_images']}_"
        f"{data_cfg['num_classes']}_{model_tag(cfg)}"
    )
    paths = {
        "features": cache_dir / f"{cache_tag}_image_features.npy",
        "labels": cache_dir / f"{cache_tag}_labels.npy",
        "text": cache_dir / f"{cache_tag}_text_features.npy",
        "classes": cache_dir / f"{cache_tag}_classes.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Open Images cached arrays: " + ", ".join(missing))
    return (
        np.load(paths["features"]),
        np.load(paths["labels"]),
        np.load(paths["text"]),
        json.loads(paths["classes"].read_text(encoding="utf-8")),
    )


def load_coco_classes() -> list[str]:
    try:
        from run_coco_pilot import COCO_CLASSES  # type: ignore

        return list(COCO_CLASSES)
    except ModuleNotFoundError:
        source_path = Path(__file__).with_name("run_coco_pilot.py")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        categories = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "COCO_CLASSES":
                        try:
                            return list(ast.literal_eval(node.value))
                        except (ValueError, SyntaxError):
                            pass
                    if isinstance(target, ast.Name) and target.id == "COCO_CATEGORIES":
                        categories = ast.literal_eval(node.value)
        if categories is not None:
            return [name for _, name in categories]
        raise RuntimeError(f"Could not recover COCO_CLASSES from {source_path}")


def load_coco_cached_arrays(cfg: dict, output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data_cfg = cfg["data"]
    cache_dir = Path(cfg.get("feature_cache_dir", output_dir / "cache"))
    max_images = data_cfg.get("max_images")
    cache_tag = f"coco2017_{data_cfg['image_set']}_{max_images or 'full'}_{model_tag(cfg)}"
    paths = {
        "features": cache_dir / f"{cache_tag}_image_features.npy",
        "labels": cache_dir / f"{cache_tag}_labels.npy",
        "text": cache_dir / f"{cache_tag}_text_features.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing COCO cached arrays: " + ", ".join(missing))
    return (
        np.load(paths["features"]),
        np.load(paths["labels"]),
        np.load(paths["text"]),
        load_coco_classes(),
    )


def load_nuswide_cached_arrays(cfg: dict, output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data_cfg = cfg["data"]
    cache_dir = Path(cfg.get("feature_cache_dir", output_dir / "cache"))
    max_images = data_cfg.get("max_images", "full")
    num_classes = data_cfg.get("num_classes", 81)
    image_pool_seed = int(data_cfg.get("image_pool_seed", cfg["seed"]))
    tags = [
        f"nuswide_{max_images}_{num_classes}_s{image_pool_seed}_{model_tag(cfg)}",
        f"nuswide_{max_images}_{num_classes}_{model_tag(cfg)}",
    ]
    for cache_tag in tags:
        paths = {
            "features": cache_dir / f"{cache_tag}_image_features.npy",
            "labels": cache_dir / f"{cache_tag}_labels.npy",
            "text": cache_dir / f"{cache_tag}_text_features.npy",
            "classes": cache_dir / f"{cache_tag}_classes.json",
        }
        if all(path.exists() for path in paths.values()):
            return (
                np.load(paths["features"]),
                np.load(paths["labels"]),
                np.load(paths["text"]),
                json.loads(paths["classes"].read_text(encoding="utf-8")),
            )
    expected = ", ".join(str(cache_dir / f"{tag}_image_features.npy") for tag in tags)
    raise FileNotFoundError("Missing NUS-WIDE cached arrays; tried " + expected)


def load_cached_dataset(
    dataset: str,
    cfg: dict,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if dataset == "openimages":
        return load_openimages_cached_arrays(cfg, output_dir)
    if dataset == "coco":
        return load_coco_cached_arrays(cfg, output_dir)
    if dataset == "nuswide":
        return load_nuswide_cached_arrays(cfg, output_dir)
    raise ValueError(f"Unknown dataset: {dataset}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def logit_feature(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def knn_topk_weights(
    train_features: np.ndarray,
    test_features: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(int(cfg.get("k", 20)), train_features.shape[0])
    if k <= 0:
        raise ValueError("kNN retrieval split is empty.")
    temperature = float(cfg.get("temperature", 0.07))
    sims = test_features @ train_features.T
    top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    top_sims = np.take_along_axis(sims, top_idx, axis=1)
    scaled = top_sims / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    weights = np.exp(scaled)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return top_idx, weights


def knn_scores_from_topk(
    train_labels: np.ndarray,
    top_idx: np.ndarray,
    weights: np.ndarray,
    emerging_idx: list[int],
) -> np.ndarray:
    neighbor_labels = train_labels[top_idx][:, :, emerging_idx]
    return (weights[:, :, None] * neighbor_labels).sum(axis=1)


def make_pair_features(score_matrix: np.ndarray, known_scores: np.ndarray | None) -> np.ndarray:
    columns = [logit_feature(score_matrix.reshape(-1))]
    if known_scores is not None:
        repeated_known = np.repeat(known_scores, score_matrix.shape[1])
        columns.append(logit_feature(repeated_known))
    return np.stack(columns, axis=1)


def fit_logistic_matrix(
    cal_labels: np.ndarray,
    cal_scores: np.ndarray,
    eval_scores: np.ndarray,
    cal_known: np.ndarray | None,
    eval_known: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    y_pair = cal_labels.reshape(-1).astype(int)
    if len(np.unique(y_pair)) < 2:
        fill = float(np.mean(y_pair))
        return (
            np.full_like(cal_scores, fill, dtype=np.float64),
            np.full_like(eval_scores, fill, dtype=np.float64),
            "constant_one_class_calibration",
        )
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced")
    x_cal = make_pair_features(cal_scores, cal_known)
    x_eval = make_pair_features(eval_scores, eval_known)
    model.fit(x_cal, y_pair)
    cal_out = model.predict_proba(x_cal)[:, 1].reshape(cal_scores.shape)
    eval_out = model.predict_proba(x_eval)[:, 1].reshape(eval_scores.shape)
    return cal_out, eval_out, "pairwise_fixed_c1_balanced"


def summarize(rows: list[dict], group_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    out = []
    for key, group in grouped.items():
        item = {name: value for name, value in zip(group_keys, key)}
        item["num_split_level_rows"] = len(group)
        for metric in ["average_precision", "auroc", "precision", "recall", "best_f1", "tecr"]:
            vals = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            item[f"{metric}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    def sort_key(row: dict) -> tuple:
        keys = []
        for key_name in group_keys:
            if key_name == "method":
                keys.append(METHOD_ORDER.get(row[key_name], 99))
            else:
                keys.append(str(row[key_name]))
        return tuple(keys)

    return sorted(out, key=sort_key)


def summarize_by_configuration(rows: list[dict], group_keys: list[str]) -> list[dict]:
    config_grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        config_key = tuple(row[key] for key in group_keys) + (row["config_name"], row["seed"])
        config_grouped.setdefault(config_key, []).append(row)

    config_rows: list[dict] = []
    for key, group in config_grouped.items():
        item = {name: value for name, value in zip(group_keys, key[:len(group_keys)])}
        item["config_name"] = key[-2]
        item["seed"] = key[-1]
        item["num_split_level_rows"] = len(group)
        for metric in ["average_precision", "auroc", "precision", "recall", "best_f1", "tecr"]:
            vals = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            item[metric] = float(np.mean(vals)) if vals else None
        config_rows.append(item)

    grouped: dict[tuple, list[dict]] = {}
    for row in config_rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    out = []
    for key, group in grouped.items():
        item = {name: value for name, value in zip(group_keys, key)}
        item["num_configs"] = len(group)
        item["num_split_level_rows"] = int(sum(int(row["num_split_level_rows"]) for row in group))
        for metric in ["average_precision", "auroc", "precision", "recall", "best_f1", "tecr"]:
            vals = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            item[f"{metric}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)

    def sort_key(row: dict) -> tuple:
        keys = []
        for key_name in group_keys:
            if key_name == "method":
                keys.append(METHOD_ORDER.get(row[key_name], 99))
            else:
                keys.append(str(row[key_name]))
        return tuple(keys)

    return sorted(out, key=sort_key)


def safe_pct_change(old: float | None, new: float | None) -> float | None:
    if old in (None, 0) or new is None:
        return None
    return 100.0 * (float(new) - float(old)) / float(old)


def safe_pct_reduction(old: float | None, new: float | None) -> float | None:
    if old in (None, 0) or new is None:
        return None
    return 100.0 * (float(old) - float(new)) / float(old)


def add_deltas(summary_rows: list[dict], group_key: str = "dataset") -> None:
    baselines = {
        row[group_key]: row for row in summary_rows
        if row.get("method") == "clip_knn"
    }
    for row in summary_rows:
        baseline = baselines.get(row[group_key])
        if row.get("method") == "clip_knn" or baseline is None:
            row["ap_delta_pct"] = 0.0 if row.get("method") == "clip_knn" else None
            row["f1_delta_pct"] = 0.0 if row.get("method") == "clip_knn" else None
            row["tecr_reduction_pct"] = 0.0 if row.get("method") == "clip_knn" else None
            continue
        row["ap_delta_pct"] = safe_pct_change(
            baseline["average_precision_mean"],
            row["average_precision_mean"],
        )
        row["f1_delta_pct"] = safe_pct_change(
            baseline["best_f1_mean"],
            row["best_f1_mean"],
        )
        row["tecr_reduction_pct"] = safe_pct_reduction(
            baseline["tecr_mean"],
            row["tecr_mean"],
        )


def calibrator_row(
    method: str,
    split_name: str,
    dataset: str,
    config_name: str,
    run_seed: int,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
    selection_status: str,
) -> dict:
    metrics = metrics_with_calibrated_threshold(
        y_true,
        scores,
        threshold,
        labels,
        tail_idx,
        emerging_idx,
    )
    return {
        **metrics,
        "dataset": dataset,
        "config_name": config_name,
        "seed": run_seed,
        "split": split_name,
        "method": method,
        "selection_status": selection_status,
    }


def run_one(config_path: Path, output_dir: Path, dataset: str, seed_override: int | None) -> dict:
    start = time.time()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_seed = int(seed_override) if seed_override is not None else int(cfg["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))

    features, labels, text_features, class_names = load_cached_dataset(dataset, cfg, Path(cfg["output_dir"]))
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

    rough_seconds = max(1, int((calibration_features.shape[0] + eval_features.shape[0]) * retrieval_features.shape[0] / 2.5e7))
    print(f"TIME_ESTIMATE: {rough_seconds} seconds", flush=True)

    cal_top_idx, cal_weights = knn_topk_weights(retrieval_features, calibration_features, cfg["knn"])
    eval_top_idx, eval_weights = knn_topk_weights(retrieval_features, eval_features, cfg["knn"])

    retrieval_logits = retrieval_features @ text_features.T
    logit_mean = retrieval_logits.mean(axis=0, keepdims=True)
    logit_std = retrieval_logits.std(axis=0, keepdims=True) + 1e-8
    calibration_logits = (calibration_features @ text_features.T - logit_mean) / logit_std
    eval_logits = (eval_features @ text_features.T - logit_mean) / logit_std

    eval_rows: list[dict] = []
    calibration_rows: list[dict] = []
    selected_rows: list[dict] = []

    for split in splits:
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
        known_idx = [i for i, name in enumerate(class_names) if name not in split["emerging_labels"]]

        cal_known = sigmoid(calibration_logits[:, known_idx].max(axis=1)) if known_idx else np.zeros(calibration_logits.shape[0])
        eval_known = sigmoid(eval_logits[:, known_idx].max(axis=1)) if known_idx else np.zeros(eval_logits.shape[0])
        perm_rng = np.random.default_rng(run_seed + int(split["seed"]) + 104729)
        cal_known_perm = perm_rng.permutation(cal_known)
        eval_known_perm = np.random.default_rng(run_seed + int(split["seed"]) + 130363).permutation(eval_known)

        cal_knn = knn_scores_from_topk(retrieval_labels, cal_top_idx, cal_weights, emerging_idx)
        eval_knn = knn_scores_from_topk(retrieval_labels, eval_top_idx, eval_weights, emerging_idx)
        y_cal = calibration_labels[:, emerging_idx].max(axis=1).astype(int)
        y_eval = eval_labels[:, emerging_idx].max(axis=1).astype(int)
        cal_base = aggregate_scores(cal_knn)
        eval_base = aggregate_scores(eval_knn)

        base_threshold = best_f1_threshold(y_cal, cal_base)["threshold"]
        cal_baseline = calibrator_row(
            "clip_knn",
            split["name"],
            dataset,
            config_path.stem,
            run_seed,
            y_cal,
            cal_base,
            base_threshold,
            calibration_labels,
            tail_idx,
            emerging_idx,
            "baseline_threshold_from_calibration",
        )
        eval_baseline = calibrator_row(
            "clip_knn",
            split["name"],
            dataset,
            config_path.stem,
            run_seed,
            y_eval,
            eval_base,
            base_threshold,
            eval_labels,
            tail_idx,
            emerging_idx,
            "baseline_threshold_from_calibration",
        )
        calibration_rows.append(cal_baseline)
        eval_rows.append(eval_baseline)

        variants = [
            ("score_only_matched_logistic", None, None),
            ("permuted_known_confidence_logistic", cal_known_perm, eval_known_perm),
            ("known_aware_two_feature_logistic", cal_known, eval_known),
        ]
        for method, train_known, test_known in variants:
            cal_matrix, eval_matrix, fit_status = fit_logistic_matrix(
                calibration_labels[:, emerging_idx],
                cal_knn,
                eval_knn,
                train_known,
                test_known,
            )
            cal_scores = aggregate_scores(cal_matrix)
            eval_scores = aggregate_scores(eval_matrix)
            threshold = best_f1_threshold(y_cal, cal_scores)["threshold"]
            cal_row = calibrator_row(
                method,
                split["name"],
                dataset,
                config_path.stem,
                run_seed,
                y_cal,
                cal_scores,
                threshold,
                calibration_labels,
                tail_idx,
                emerging_idx,
                fit_status,
            )
            eval_row = calibrator_row(
                method,
                split["name"],
                dataset,
                config_path.stem,
                run_seed,
                y_eval,
                eval_scores,
                threshold,
                eval_labels,
                tail_idx,
                emerging_idx,
                fit_status,
            )
            calibration_rows.append(cal_row)
            eval_rows.append(eval_row)
            selected_rows.append({
                "dataset": dataset,
                "config_name": config_path.stem,
                "seed": run_seed,
                "split": split["name"],
                "method": method,
                "fit_status": fit_status,
                "threshold": threshold,
                "calibration_ap": cal_row["average_precision"],
                "calibration_f1": cal_row["best_f1"],
                "calibration_tecr": cal_row["tecr"],
                "baseline_calibration_ap": cal_baseline["average_precision"],
                "baseline_calibration_f1": cal_baseline["best_f1"],
                "baseline_calibration_tecr": cal_baseline["tecr"],
                "num_calibration_positive": int(calibration_labels[:, emerging_idx].sum()),
                "num_calibration_negative": int(calibration_labels[:, emerging_idx].size - calibration_labels[:, emerging_idx].sum()),
            })

    summary = summarize(eval_rows, ["dataset", "method"])
    add_deltas(summary)

    eval_fields = [
        "dataset", "config_name", "seed", "split", "method", "selection_status",
        "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    selection_fields = [
        "dataset", "config_name", "seed", "split", "method", "fit_status", "threshold",
        "calibration_ap", "calibration_f1", "calibration_tecr",
        "baseline_calibration_ap", "baseline_calibration_f1", "baseline_calibration_tecr",
        "num_calibration_positive", "num_calibration_negative",
    ]
    summary_fields = [
        "dataset", "method", "num_split_level_rows",
        "average_precision_mean", "average_precision_std",
        "auroc_mean", "auroc_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "best_f1_mean", "best_f1_std",
        "tecr_mean", "tecr_std",
        "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct",
    ]
    outputs = {
        "eval_rows": output_dir / "known_aware_posthoc_eval_rows.csv",
        "calibration_rows": output_dir / "known_aware_posthoc_calibration_rows.csv",
        "selected_settings": output_dir / "known_aware_posthoc_selected_settings.csv",
        "summary": output_dir / "known_aware_posthoc_summary.csv",
        "report": output_dir / "known_aware_posthoc_report.md",
        "result": output_dir / "known_aware_posthoc_result.json",
    }
    write_csv(outputs["eval_rows"], eval_rows, eval_fields)
    write_csv(outputs["calibration_rows"], calibration_rows, eval_fields)
    write_csv(outputs["selected_settings"], selected_rows, selection_fields)
    write_csv(outputs["summary"], summary, summary_fields)

    # Backward-compatible aliases for the first local draft of this script.
    write_csv(output_dir / "known_aware_eval_rows.csv", eval_rows, eval_fields)
    write_csv(output_dir / "known_aware_calibration_rows.csv", calibration_rows, eval_fields)
    write_csv(output_dir / "known_aware_selected_settings.csv", selected_rows, selection_fields)
    write_csv(output_dir / "known_aware_summary.csv", summary, summary_fields)

    report = [
        "# Known-Aware Post-Hoc Baselines",
        "",
        f"Dataset: `{dataset}`",
        f"Config: `{config_path}`",
        f"Seed: `{run_seed}`",
        "",
        "All calibrators are fit on the calibration split and evaluated on the held-out evaluation split. "
        "`known_aware_two_feature_logistic` uses each emerging-label kNN score plus image-level known-label confidence; "
        "`score_only_matched_logistic` uses the same logistic calibration protocol without known-label confidence; "
        "`permuted_known_confidence_logistic` keeps a two-feature model while permuting known confidence to break image alignment.",
        "",
        "| Method | AP | F1 | TECR | AP delta | F1 delta | TECR reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['method']} | {row['average_precision_mean']:.4f} | "
            f"{row['best_f1_mean']:.4f} | {row['tecr_mean']:.4f} | "
            f"{row['ap_delta_pct']:.2f}% | {row['f1_delta_pct']:.2f}% | "
            f"{row['tecr_reduction_pct']:.1f}% |"
        )
    outputs["report"].write_text("\n".join(report), encoding="utf-8")
    (output_dir / "known_aware_report.md").write_text("\n".join(report), encoding="utf-8")

    result = {
        "status": "known_aware_posthoc_complete",
        "time_seconds": round(time.time() - start, 3),
        "dataset": dataset,
        "config": str(config_path),
        "seed": run_seed,
        "output_dir": str(output_dir),
        "num_images": int(labels.shape[0]),
        "num_retrieval": int(retrieval_features.shape[0]),
        "num_calibration": int(calibration_features.shape[0]),
        "num_eval": int(eval_features.shape[0]),
        "summary": summary,
        "selected_settings": selected_rows,
        "files": {key: str(path) for key, path in outputs.items()},
    }
    outputs["result"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "known_aware_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def parse_float(value: str) -> float | None:
    if value in ("", "None", "nan", "NaN"):
        return None
    return float(value)


def combine_outputs(root: Path, output_dir: Path) -> dict:
    rows: list[dict] = []
    for path in sorted(root.rglob("known_aware_eval_rows.csv")):
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for key in ["average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr"]:
                    row[key] = parse_float(row.get(key, ""))
                row["seed"] = int(row["seed"])
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No known_aware_eval_rows.csv files found under {root}")
    summary = summarize_by_configuration(rows, ["dataset", "method"])
    add_deltas(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_fields = [
        "dataset", "config_name", "seed", "split", "method", "selection_status",
        "average_precision", "auroc", "precision", "recall", "best_f1", "threshold", "tecr",
    ]
    summary_fields = [
        "dataset", "method", "num_configs", "num_split_level_rows",
        "average_precision_mean", "average_precision_std",
        "auroc_mean", "auroc_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "best_f1_mean", "best_f1_std",
        "tecr_mean", "tecr_std",
        "ap_delta_pct", "f1_delta_pct", "tecr_reduction_pct",
    ]
    write_csv(output_dir / "known_aware_combined_eval_rows.csv", rows, eval_fields)
    write_csv(output_dir / "known_aware_combined_summary.csv", summary, summary_fields)

    report = [
        "# Combined Known-Aware Post-Hoc Baselines",
        "",
        f"Combined rows: {len(rows)} from `{root}`.",
        "",
        "| Dataset | Method | Configs | Split-level rows | AP | F1 | TECR | TECR reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['dataset']} | {row['method']} | {row['num_configs']} | {row['num_split_level_rows']} | "
            f"{row['average_precision_mean']:.4f} | {row['best_f1_mean']:.4f} | "
            f"{row['tecr_mean']:.4f} | {row['tecr_reduction_pct']:.1f}% |"
        )
    (output_dir / "known_aware_combined_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "status": "known_aware_combined_complete",
        "root": str(root),
        "output_dir": str(output_dir),
        "num_rows": len(rows),
        "summary": summary,
    }
    (output_dir / "known_aware_combined_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=["auto", "openimages", "coco", "nuswide"], default="auto")
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--combine-root")
    args = parser.parse_args()

    if args.combine_root:
        combine_outputs(Path(args.combine_root), Path(args.output_dir))
        return 0
    if not args.config:
        parser.error("--config is required unless --combine-root is used")
    config_path = Path(args.config)
    dataset = infer_dataset(config_path, args.dataset)
    run_one(config_path, Path(args.output_dir), dataset, args.seed_override)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
