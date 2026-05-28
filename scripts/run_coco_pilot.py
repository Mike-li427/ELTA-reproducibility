from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import clip
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml


COCO_CATEGORIES = [
    (1, "person"), (2, "bicycle"), (3, "car"), (4, "motorcycle"), (5, "airplane"),
    (6, "bus"), (7, "train"), (8, "truck"), (9, "boat"), (10, "traffic light"),
    (11, "fire hydrant"), (13, "stop sign"), (14, "parking meter"), (15, "bench"),
    (16, "bird"), (17, "cat"), (18, "dog"), (19, "horse"), (20, "sheep"),
    (21, "cow"), (22, "elephant"), (23, "bear"), (24, "zebra"), (25, "giraffe"),
    (27, "backpack"), (28, "umbrella"), (31, "handbag"), (32, "tie"), (33, "suitcase"),
    (34, "frisbee"), (35, "skis"), (36, "snowboard"), (37, "sports ball"), (38, "kite"),
    (39, "baseball bat"), (40, "baseball glove"), (41, "skateboard"), (42, "surfboard"),
    (43, "tennis racket"), (44, "bottle"), (46, "wine glass"), (47, "cup"),
    (48, "fork"), (49, "knife"), (50, "spoon"), (51, "bowl"), (52, "banana"),
    (53, "apple"), (54, "sandwich"), (55, "orange"), (56, "broccoli"), (57, "carrot"),
    (58, "hot dog"), (59, "pizza"), (60, "donut"), (61, "cake"), (62, "chair"),
    (63, "couch"), (64, "potted plant"), (65, "bed"), (67, "dining table"),
    (70, "toilet"), (72, "tv"), (73, "laptop"), (74, "mouse"), (75, "remote"),
    (76, "keyboard"), (77, "cell phone"), (78, "microwave"), (79, "oven"),
    (80, "toaster"), (81, "sink"), (82, "refrigerator"), (84, "book"),
    (85, "clock"), (86, "vase"), (87, "scissors"), (88, "teddy bear"),
    (89, "hair drier"), (90, "toothbrush"),
]

COCO_IDS = [cat_id for cat_id, _ in COCO_CATEGORIES]
COCO_CLASSES = [name for _, name in COCO_CATEGORIES]
ID_TO_INDEX = {cat_id: i for i, cat_id in enumerate(COCO_IDS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_file(url: str, path: Path, download_source: str = "official", hf_repo_id: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if download_source == "huggingface_mirror":
        from huggingface_hub import hf_hub_download

        repo_id = hf_repo_id or "pcuenq/coco-2017-mirror"
        print(f"Downloading from Hugging Face mirror {repo_id}: {path.name}", flush=True)
        hf_path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=path.name))
        if hf_path != path:
            if not path.exists() or path.stat().st_size != hf_path.stat().st_size:
                shutil.copyfile(hf_path, path)
        return hf_path

    if path.exists():
        if path.stat().st_size > 1024 * 1024 * 10:
            return path
        path.unlink()
    print(f"Downloading {url} -> {path}", flush=True)
    urllib.request.urlretrieve(url, path)
    return path


def extract_zip(path: Path, root: Path, marker: Path) -> None:
    if marker.exists():
        return
    print(f"Extracting {path}", flush=True)
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"Invalid or incomplete zip file: {path}")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(root)
    marker.touch()


def ensure_coco_val(root: Path, download_source: str = "official", hf_repo_id: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    val_zip = root / "val2017.zip"
    ann_zip = root / "annotations_trainval2017.zip"
    val_zip = download_file(
        "http://images.cocodataset.org/zips/val2017.zip",
        val_zip,
        download_source=download_source,
        hf_repo_id=hf_repo_id,
    )
    ann_zip = download_file(
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        ann_zip,
        download_source=download_source,
        hf_repo_id=hf_repo_id,
    )
    extract_zip(val_zip, root, root / ".val2017_extracted")
    extract_zip(ann_zip, root, root / ".annotations_extracted")


class COCOMultiLabelDataset(Dataset):
    def __init__(self, root: Path, annotation_file: str, image_set: str, preprocess, max_images: int | None = None):
        ann_path = root / annotation_file
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        image_by_id = {item["id"]: item for item in data["images"]}
        labels_by_id = {image_id: np.zeros(len(COCO_CLASSES), dtype=np.float32) for image_id in image_by_id}
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            cat_id = ann["category_id"]
            if image_id in labels_by_id and cat_id in ID_TO_INDEX:
                labels_by_id[image_id][ID_TO_INDEX[cat_id]] = 1.0
        images = sorted(image_by_id.values(), key=lambda item: item["id"])
        if max_images is not None:
            images = images[:max_images]
        self.root = root
        self.image_set = image_set
        self.preprocess = preprocess
        self.items = [(item["file_name"], labels_by_id[item["id"]]) for item in images]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        file_name, labels = self.items[idx]
        image = Image.open(self.root / self.image_set / file_name).convert("RGB")
        return self.preprocess(image), torch.from_numpy(labels)


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
    prompts = [f"a photo of a {name}" for name in class_names]
    tokens = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(tokens)
    text_features = F.normalize(text_features.float(), dim=-1)
    return text_features.cpu().numpy()


def load_or_compute_features(cfg: dict, model, preprocess, device: str, output_dir: Path):
    data_cfg = cfg["data"]
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_images = data_cfg.get("max_images")
    cache_tag = f"coco2017_{data_cfg['image_set']}_{max_images or 'full'}_{cfg['clip']['model'].replace('/', '-')}"
    image_cache = cache_dir / f"{cache_tag}_image_features.npy"
    label_cache = cache_dir / f"{cache_tag}_labels.npy"
    text_cache = cache_dir / f"{cache_tag}_text_features.npy"
    if image_cache.exists() and label_cache.exists() and text_cache.exists():
        return np.load(image_cache), np.load(label_cache), np.load(text_cache), True

    root = Path(data_cfg["root"])
    if data_cfg.get("download", True):
        ensure_coco_val(
            root,
            download_source=str(data_cfg.get("download_source", "official")),
            hf_repo_id=data_cfg.get("hf_repo_id"),
        )
    dataset = COCOMultiLabelDataset(
        root=root,
        annotation_file=data_cfg["annotation_file"],
        image_set=data_cfg["image_set"],
        preprocess=preprocess,
        max_images=max_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["clip"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["clip"].get("num_workers", 2)),
        pin_memory=(device == "cuda"),
        collate_fn=collate,
    )
    image_features, labels = encode_images(model, loader, device)
    text_features = encode_text(model, COCO_CLASSES, device)
    np.save(image_cache, image_features)
    np.save(label_cache, labels)
    np.save(text_cache, text_features)
    return image_features, labels, text_features, False


def split_retrieval_eval(features: np.ndarray, labels: np.ndarray, seed: int, retrieval_fraction: float):
    rng = np.random.default_rng(seed)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    n_retrieval = int(round(len(indices) * retrieval_fraction))
    retrieval_idx = np.sort(indices[:n_retrieval])
    eval_idx = np.sort(indices[n_retrieval:])
    return features[retrieval_idx], labels[retrieval_idx], features[eval_idx], labels[eval_idx]


def build_splits(labels: np.ndarray, protocol_cfg: dict, seed: int) -> list[dict]:
    class_counts = labels.sum(axis=0)
    eligible = [
        COCO_CLASSES[i]
        for i, count in enumerate(class_counts)
        if count >= int(protocol_cfg.get("min_retrieval_positives", 5))
    ]
    split_seeds = protocol_cfg.get("split_seeds") or [seed + i for i in range(3)]
    emerging_count = int(protocol_cfg.get("emerging_count", 20))
    tail_count = int(protocol_cfg.get("tail_count", 15))
    count_order = sorted(range(len(COCO_CLASSES)), key=lambda i: (class_counts[i], COCO_CLASSES[i].lower()))
    tail_pool = [COCO_CLASSES[i] for i in count_order if COCO_CLASSES[i] in eligible]
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


def write_csv(path: Path, rows: list[dict]) -> None:
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
    features, labels, text_features, cache_hit = load_or_compute_features(cfg, model, preprocess, device, output_dir)
    if args.cache_only:
        result = {
            "status": "coco_cache_ready",
            "time_seconds": round(time.time() - start, 3),
            "device": device,
            "cache_hit": cache_hit,
            "num_images_total": int(labels.shape[0]),
            "num_classes": len(COCO_CLASSES),
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
    splits = build_splits(retrieval_labels, cfg["protocol"], int(cfg["seed"]))

    rows = []
    raw = []
    for split in splits:
        emerging_idx = [COCO_CLASSES.index(x) for x in split["emerging_labels"]]
        tail_idx = [COCO_CLASSES.index(x) for x in split["tail_known_labels"]]
        known_idx = [i for i, x in enumerate(COCO_CLASSES) if x not in split["emerging_labels"]]
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
    write_csv(output_dir / "coco_pareto_summary.csv", summary)
    write_csv(output_dir / "coco_pareto_front.csv", front)
    with (output_dir / "coco_pareto_raw.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "split", "residual_power", "tail_weight", "average_precision", "auroc", "best_f1", "tecr", "threshold"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(raw)

    result = {
        "status": "coco_pilot_complete",
        "warning": "COCO val-only retrieval/eval pilot; not a full paper result.",
        "time_seconds": round(time.time() - start, 3),
        "device": device,
        "cache_hit": cache_hit,
        "num_images_total": int(labels.shape[0]),
        "num_retrieval_images": int(retrieval_labels.shape[0]),
        "num_eval_images": int(eval_labels.shape[0]),
        "splits": splits,
        "summary": summary,
        "pareto_front": front,
    }
    (output_dir / "coco_pareto_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TIME_ESTIMATE:", max(1, int(time.time() - start)))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
