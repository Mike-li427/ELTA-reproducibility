from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import clip
import numpy as np
from PIL import Image, ImageFile
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml


ImageFile.LOAD_TRUNCATED_IMAGES = True

META_URLS = {
    "class_descriptions": "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv",
    "human_labels": "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv",
    "image_metadata": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_url(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {path}", flush=True)
    urllib.request.urlretrieve(url, path)


def ensure_metadata(root: Path) -> dict[str, Path]:
    meta_dir = root / "meta"
    paths = {
        "class_descriptions": meta_dir / "oidv6-class-descriptions.csv",
        "human_labels": meta_dir / "validation-annotations-human-imagelabels.csv",
        "image_metadata": meta_dir / "validation-images-with-rotation.csv",
    }
    for key, url in META_URLS.items():
        download_url(url, paths[key])
    return paths


def read_class_names(path: Path) -> dict[str, str]:
    out = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["LabelName"]] = row["DisplayName"]
    return out


def count_positive_labels(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Confidence"] == "1":
                counts[row["LabelName"]] = counts.get(row["LabelName"], 0) + 1
    return counts


def choose_classes(class_names: dict[str, str], counts: dict[str, int], num_classes: int) -> list[str]:
    candidates = [label for label, count in counts.items() if label in class_names and count > 0]
    candidates.sort(key=lambda label: (-counts[label], class_names[label].lower()))
    return candidates[:num_classes]


def build_image_label_index(labels_path: Path, selected_labels: list[str]) -> dict[str, set[str]]:
    selected = set(selected_labels)
    image_to_labels: dict[str, set[str]] = {}
    with labels_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["LabelName"]
            if row["Confidence"] == "1" and label in selected:
                image_to_labels.setdefault(row["ImageID"], set()).add(label)
    return image_to_labels


def select_images(image_to_labels: dict[str, set[str]], max_images: int, seed: int) -> list[str]:
    image_ids = [image_id for image_id, labels in image_to_labels.items() if labels]
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    return sorted(image_ids[:max_images])


def image_url(image_id: str, split: str) -> str:
    return f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"


def download_one_image(image_id: str, split: str, image_dir: Path) -> tuple[str, bool, str | None]:
    path = image_dir / f"{image_id}.jpg"
    if path.exists() and path.stat().st_size > 0:
        return image_id, True, None
    url = image_url(image_id, split)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=25) as response:
                data = response.read()
            if data:
                path.write_bytes(data)
                return image_id, True, None
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                if path.exists() and path.stat().st_size == 0:
                    path.unlink()
                return image_id, False, repr(exc)
    return image_id, False, "unknown download failure"


def download_images(image_ids: list[str], split: str, image_dir: Path, workers: int) -> list[str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    ok = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(download_one_image, image_id, split, image_dir) for image_id in image_ids]
        for i, future in enumerate(as_completed(futures), start=1):
            image_id, success, err = future.result()
            if success:
                ok.append(image_id)
            else:
                failures.append({"image_id": image_id, "error": err})
            if i % 250 == 0:
                print(f"Downloaded/checkpointed {i}/{len(image_ids)} images; ok={len(ok)} failed={len(failures)}", flush=True)
    if failures:
        (image_dir.parent / "download_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return sorted(ok)


def labels_matrix(image_ids: list[str], selected_labels: list[str], image_to_labels: dict[str, set[str]]) -> np.ndarray:
    label_to_idx = {label: i for i, label in enumerate(selected_labels)}
    labels = np.zeros((len(image_ids), len(selected_labels)), dtype=np.float32)
    for row_idx, image_id in enumerate(image_ids):
        for label in image_to_labels.get(image_id, set()):
            idx = label_to_idx.get(label)
            if idx is not None:
                labels[row_idx, idx] = 1.0
    return labels


class OpenImagesDataset(Dataset):
    def __init__(self, image_dir: Path, image_ids: list[str], labels: np.ndarray, preprocess):
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_dir / f"{self.image_ids[idx]}.jpg").convert("RGB")
        return self.preprocess(image), torch.from_numpy(self.labels[idx])


def collate(batch):
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(labels, dim=0)


@torch.no_grad()
def encode_images(model, loader, device: str):
    feats = []
    labels = []
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features.float(), dim=-1).cpu()
        feats.append(image_features)
        labels.append(batch_labels.float())
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


@torch.no_grad()
def encode_text(model, class_names: list[str], device: str):
    prompts = [f"a photo of {name}" for name in class_names]
    tokens = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(tokens)
    text_features = F.normalize(text_features.float(), dim=-1)
    return text_features.cpu().numpy()


def load_or_compute_features(cfg: dict, model, preprocess, device: str, output_dir: Path):
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    cache_dir = Path(cfg.get("feature_cache_dir", output_dir / "cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = f"openimages_{data_cfg['split']}_{data_cfg['max_images']}_{data_cfg['num_classes']}_{cfg['clip']['model'].replace('/', '-')}"
    feature_cache = cache_dir / f"{cache_tag}_image_features.npy"
    label_cache = cache_dir / f"{cache_tag}_labels.npy"
    text_cache = cache_dir / f"{cache_tag}_text_features.npy"
    classes_cache = cache_dir / f"{cache_tag}_classes.json"
    image_ids_cache = cache_dir / f"{cache_tag}_image_ids.json"
    if all(path.exists() for path in [feature_cache, label_cache, text_cache, classes_cache, image_ids_cache]):
        return (
            np.load(feature_cache),
            np.load(label_cache),
            np.load(text_cache),
            json.loads(classes_cache.read_text(encoding="utf-8")),
            json.loads(image_ids_cache.read_text(encoding="utf-8")),
            True,
        )

    paths = ensure_metadata(root)
    class_names = read_class_names(paths["class_descriptions"])
    counts = count_positive_labels(paths["human_labels"])
    selected_labels = choose_classes(class_names, counts, int(data_cfg["num_classes"]))
    display_names = [class_names[label] for label in selected_labels]
    image_to_labels = build_image_label_index(paths["human_labels"], selected_labels)
    selected_image_ids = select_images(image_to_labels, int(data_cfg["max_images"]), int(cfg["seed"]))
    image_dir = root / data_cfg["split"]
    ok_image_ids = download_images(
        selected_image_ids,
        str(data_cfg["split"]),
        image_dir,
        int(data_cfg.get("download_workers", 8)),
    )
    labels = labels_matrix(ok_image_ids, selected_labels, image_to_labels)
    dataset = OpenImagesDataset(image_dir, ok_image_ids, labels, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["clip"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["clip"].get("num_workers", 2)),
        pin_memory=(device == "cuda"),
        collate_fn=collate,
    )
    image_features, labels = encode_images(model, loader, device)
    text_features = encode_text(model, display_names, device)
    np.save(feature_cache, image_features)
    np.save(label_cache, labels)
    np.save(text_cache, text_features)
    classes_cache.write_text(json.dumps(display_names, indent=2, ensure_ascii=False), encoding="utf-8")
    image_ids_cache.write_text(json.dumps(ok_image_ids, indent=2), encoding="utf-8")
    return image_features, labels, text_features, display_names, ok_image_ids, False


def split_retrieval_eval(features: np.ndarray, labels: np.ndarray, seed: int, retrieval_fraction: float):
    rng = np.random.default_rng(seed)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    n_retrieval = int(round(len(indices) * retrieval_fraction))
    retrieval_idx = np.sort(indices[:n_retrieval])
    eval_idx = np.sort(indices[n_retrieval:])
    return features[retrieval_idx], labels[retrieval_idx], features[eval_idx], labels[eval_idx]


def build_splits(class_names: list[str], retrieval_labels: np.ndarray, protocol_cfg: dict, seed: int) -> list[dict]:
    retrieval_counts = retrieval_labels.sum(axis=0)
    eligible = [
        class_names[i]
        for i in range(len(class_names))
        if retrieval_counts[i] >= int(protocol_cfg.get("min_retrieval_positives", 5))
    ]
    split_seeds = protocol_cfg.get("split_seeds") or [seed + i for i in range(3)]
    emerging_count = min(int(protocol_cfg.get("emerging_count", 20)), len(eligible))
    tail_count = int(protocol_cfg.get("tail_count", 15))
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


def best_f1_from_scores(y_true: np.ndarray, scores: np.ndarray):
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


def binary_metrics(y_true: np.ndarray, scores: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return {
            "average_precision": None,
            "auroc": None,
            "best_f1": {"threshold": None, "precision": None, "recall": None, "f1": None},
        }
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "best_f1": best_f1_from_scores(y_true, scores),
    }


def knn_score_matrix(train_features: np.ndarray, train_labels: np.ndarray, test_features: np.ndarray, emerging_idx: list[int], cfg: dict):
    k = min(int(cfg.get("k", 20)), train_features.shape[0])
    temperature = float(cfg.get("temperature", 0.07))
    sims = test_features @ train_features.T
    top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    top_sims = np.take_along_axis(sims, top_idx, axis=1)
    weights = np.exp(top_sims / max(temperature, 1e-6))
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    neighbor_labels = train_labels[top_idx][:, :, emerging_idx]
    return (weights[:, :, None] * neighbor_labels).sum(axis=1)


def tecr(scores: np.ndarray, threshold: float | None, labels: np.ndarray, tail_idx: list[int], emerging_idx: list[int]):
    if threshold is None:
        return None
    is_known_tail_positive = labels[:, tail_idx].max(axis=1) > 0
    is_emerging_negative = labels[:, emerging_idx].max(axis=1) == 0
    eligible = is_known_tail_positive & is_emerging_negative
    if eligible.sum() == 0:
        return None
    predicted_emerging = scores >= threshold
    return float((predicted_emerging & eligible).sum() / eligible.sum())


def evaluate_scores(scores: np.ndarray, labels: np.ndarray, emerging_idx: list[int], tail_idx: list[int]):
    y_emerging = labels[:, emerging_idx].max(axis=1).astype(int)
    max_scores = scores.max(axis=1) if scores.ndim == 2 else scores
    out = binary_metrics(y_emerging, max_scores)
    out["tecr"] = tecr(max_scores, out["best_f1"]["threshold"], labels, tail_idx, emerging_idx)
    return out


def summarize_rows(rows: list[dict]):
    grouped = {}
    for row in rows:
        key = (row["method"], row.get("residual_power"), row.get("tail_weight"))
        grouped.setdefault(key, []).append(row)
    summary = []
    for (method, residual_power, tail_weight), group in grouped.items():
        out = {"method": method, "residual_power": residual_power, "tail_weight": tail_weight}
        for metric in ["average_precision", "auroc", "best_f1", "tecr"]:
            vals = [g[metric] for g in group if g[metric] is not None]
            out[f"{metric}_mean"] = float(np.mean(vals)) if vals else None
            out[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(out)
    summary.sort(key=lambda x: (
        x["method"],
        -1 if x["residual_power"] is None else float(x["residual_power"]),
        -1 if x["tail_weight"] is None else float(x["tail_weight"]),
    ))
    return summary


def pareto_front(summary: list[dict]):
    candidates = [r for r in summary if r["method"].startswith("elta_r")]
    front = []
    for row in candidates:
        f1 = row["best_f1_mean"]
        tecr_value = row["tecr_mean"]
        if f1 is None or tecr_value is None:
            continue
        dominated = False
        for other in candidates:
            if other is row:
                continue
            other_f1 = other["best_f1_mean"]
            other_tecr = other["tecr_mean"]
            if other_f1 is None or other_tecr is None:
                continue
            if other_f1 >= f1 and other_tecr <= tecr_value and (other_f1 > f1 or other_tecr < tecr_value):
                dominated = True
                break
        if not dominated:
            front.append(row)
    front.sort(key=lambda x: (x["tecr_mean"], -x["best_f1_mean"]))
    return front


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "method", "residual_power", "tail_weight",
        "average_precision_mean", "average_precision_std",
        "auroc_mean", "auroc_std",
        "best_f1_mean", "best_f1_std",
        "tecr_mean", "tecr_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))
    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    start = time.time()
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    model, preprocess = clip.load(cfg["clip"]["model"], device=device)
    model.eval()
    features, labels, text_features, class_names, image_ids, cache_hit = load_or_compute_features(
        cfg, model, preprocess, device, output_dir
    )
    if args.cache_only:
        result = {
            "status": "openimages_cache_ready",
            "time_seconds": round(time.time() - start, 3),
            "device": device,
            "cache_hit": cache_hit,
            "num_images_total": int(labels.shape[0]),
            "num_classes": len(class_names),
            "num_image_ids": len(image_ids),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    retrieval_features, retrieval_labels, eval_features, eval_labels = split_retrieval_eval(
        features,
        labels,
        int(cfg["seed"]),
        float(cfg["data"].get("retrieval_fraction", 0.5)),
    )
    retrieval_logits_raw = retrieval_features @ text_features.T
    logit_mean = retrieval_logits_raw.mean(axis=0, keepdims=True)
    logit_std = retrieval_logits_raw.std(axis=0, keepdims=True) + 1e-8
    logits = (eval_features @ text_features.T - logit_mean) / logit_std
    splits = build_splits(class_names, retrieval_labels, cfg["protocol"], int(cfg["seed"]))

    rows = []
    raw = []
    for split in splits:
        emerging_idx = [class_names.index(x) for x in split["emerging_labels"]]
        tail_idx = [class_names.index(x) for x in split["tail_known_labels"]]
        known_idx = [i for i, x in enumerate(class_names) if x not in split["emerging_labels"]]
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
            "residual_power": None,
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
        for residual_power in cfg["pareto"]["residual_powers"]:
            residual_gate = np.maximum(0.0, 1.0 - known_explanation) ** float(residual_power)
            for tail_weight in cfg["pareto"]["tail_weights"]:
                semantic_gate = np.maximum(0.0, 1.0 - float(tail_weight) * sample_tail_conf)
                scores = knn_scores * residual_gate[:, None] * semantic_gate[:, None]
                metrics = evaluate_scores(scores, eval_labels, emerging_idx, tail_idx)
                row = {
                    "method": "elta_r_grid",
                    "split": split["name"],
                    "residual_power": float(residual_power),
                    "tail_weight": float(tail_weight),
                    "average_precision": metrics["average_precision"],
                    "auroc": metrics["auroc"],
                    "best_f1": metrics["best_f1"]["f1"],
                    "tecr": metrics["tecr"],
                }
                rows.append(row)
                raw.append({**row, "threshold": metrics["best_f1"]["threshold"]})

    summary = summarize_rows(rows)
    front = pareto_front(summary)
    write_summary_csv(output_dir / "openimages_pareto_summary.csv", summary)
    write_summary_csv(output_dir / "openimages_pareto_front.csv", front)
    with (output_dir / "openimages_pareto_raw.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "split", "residual_power", "tail_weight", "average_precision", "auroc", "best_f1", "tecr", "threshold"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(raw)

    result = {
        "status": "openimages_pilot_complete",
        "warning": "Open Images validation subset pilot; not a full paper result.",
        "time_seconds": round(time.time() - start, 3),
        "device": device,
        "cache_hit": cache_hit,
        "num_images_total": int(labels.shape[0]),
        "num_retrieval_images": int(retrieval_labels.shape[0]),
        "num_eval_images": int(eval_labels.shape[0]),
        "num_classes": len(class_names),
        "classes": class_names,
        "splits": splits,
        "summary": summary,
        "pareto_front": front,
    }
    (output_dir / "openimages_pareto_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TIME_ESTIMATE:", max(1, int(time.time() - start)))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
