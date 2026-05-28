from __future__ import annotations

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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True


SOURCE_FILES = [
    "cats",
    "train.json",
    "test.json",
    "NUS-WIDE-urls.txt",
]

SMALL_SOURCE_FILES = [
    "cats",
    "small_train.json",
    "small_test.json",
    "NUS-WIDE-urls.txt",
]


def download_url(url: str, path: Path, timeout: int = 120, retries: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response, tmp_path.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp_path.replace(path)
            print(f"NUS-WIDE metadata downloaded: {path.name} bytes={path.stat().st_size}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            tmp_path.unlink(missing_ok=True)
            print(f"NUS-WIDE metadata download retry {attempt}/{retries}: {path.name} error={exc!r}", flush=True)
            if attempt < retries:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"failed to download NUS-WIDE metadata {path.name}") from last_error


def ensure_nuswide_metadata(root: Path, base_url: str, small_only: bool = False) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    files = SMALL_SOURCE_FILES if small_only else SOURCE_FILES
    out = {}
    for name in files:
        path = root / name
        print(f"NUS-WIDE metadata ready/checking: {name}", flush=True)
        download_url(f"{base_url.rstrip('/')}/{name}", path)
        out[name] = path
    return out


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def read_samples(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        marker = '"samples"'
        marker_pos = text.find(marker)
        if marker_pos < 0:
            raise
        array_start = text.find("[", marker_pos)
        if array_start < 0:
            raise
        samples, _ = json.JSONDecoder().raw_decode(text[array_start:])
        data = {"samples": samples}
    samples = []
    for item in data.get("samples", []):
        labels = [normalize_label(x) for x in item.get("image_labels", [])]
        if labels:
            samples.append({"image_name": item["image_name"], "labels": labels})
    return samples


def read_concepts(path: Path) -> list[str]:
    return [normalize_label(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def windows_path_to_image_name(path_text: str) -> str:
    return path_text.replace("\\", "/").split("/")[-1]


def image_name_to_photo_id(image_name: str) -> str | None:
    stem = Path(image_name).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[1]
    return None


def normalize_flickr_url(url: str) -> list[str]:
    variants = [url]
    if "static.flickr.com" in url:
        staticflickr = url.replace("http://", "https://").replace("static.flickr.com", "staticflickr.com")
        live = staticflickr.replace("farm1.staticflickr.com", "live.staticflickr.com")
        live = live.replace("farm2.staticflickr.com", "live.staticflickr.com")
        live = live.replace("farm3.staticflickr.com", "live.staticflickr.com")
        live = live.replace("farm4.staticflickr.com", "live.staticflickr.com")
        live = live.replace("farm5.staticflickr.com", "live.staticflickr.com")
        live = live.replace("farm6.staticflickr.com", "live.staticflickr.com")
        live = live.replace("farm7.staticflickr.com", "live.staticflickr.com")
        live = live.replace("farm8.staticflickr.com", "live.staticflickr.com")
        live = live.replace("farm9.staticflickr.com", "live.staticflickr.com")
        variants = [live, staticflickr, url]
    return list(dict.fromkeys(variants))


def read_url_map(path: Path) -> dict[str, list[str]]:
    url_map: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            image_name = windows_path_to_image_name(parts[0])
            photo_id = parts[1]
            urls = []
            for u in parts[2:]:
                if u.lower() == "null":
                    continue
                urls.extend(normalize_flickr_url(u))
            if urls:
                url_map[image_name] = urls
                url_map[photo_id] = urls
    return url_map


def select_balanced_samples(
    samples: list[dict],
    concepts: list[str],
    max_images: int,
    min_label_positives: int,
    seed: int,
) -> tuple[list[dict], list[str]]:
    concept_set = set(concepts)
    filtered = []
    counts = {label: 0 for label in concepts}
    for item in samples:
        labels = [label for label in item["labels"] if label in concept_set]
        if labels:
            filtered.append({"image_name": item["image_name"], "labels": labels})
            for label in labels:
                counts[label] += 1
    selected_concepts = [label for label in concepts if counts[label] >= min_label_positives]
    selected_set = set(selected_concepts)
    filtered = [
        {"image_name": item["image_name"], "labels": [label for label in item["labels"] if label in selected_set]}
        for item in filtered
    ]
    filtered = [item for item in filtered if item["labels"]]
    rng = random.Random(seed)
    rng.shuffle(filtered)
    return sorted(filtered[:max_images], key=lambda x: x["image_name"]), selected_concepts


def download_one(sample: dict, url_map: dict[str, list[str]], image_dir: Path) -> tuple[str, bool, str | None]:
    image_name = sample["image_name"]
    path = image_dir / image_name
    if path.exists() and path.stat().st_size > 0:
        try:
            with Image.open(path) as image:
                image.verify()
            return image_name, True, None
        except Exception:  # noqa: BLE001
            path.unlink(missing_ok=True)
    urls = url_map.get(image_name, [])
    if not urls:
        photo_id = image_name_to_photo_id(image_name)
        urls = url_map.get(photo_id, []) if photo_id else []
    last_error = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as response:
                data = response.read()
            if data:
                path.write_bytes(data)
                with Image.open(path) as image:
                    image.verify()
                return image_name, True, None
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            path.unlink(missing_ok=True)
    return image_name, False, last_error if urls else "no_url"


def download_images(samples: list[dict], url_map: dict[str, list[str]], image_dir: Path, workers: int) -> list[dict]:
    image_dir.mkdir(parents=True, exist_ok=True)
    by_name = {sample["image_name"]: sample for sample in samples}
    ok_names = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(download_one, sample, url_map, image_dir) for sample in samples]
        for i, future in enumerate(as_completed(futures), start=1):
            image_name, ok, error = future.result()
            if ok:
                ok_names.append(image_name)
            else:
                failures.append({"image_name": image_name, "error": error})
            if i % 500 == 0:
                print(f"NUS-WIDE images checked {i}/{len(samples)} ok={len(ok_names)} failed={len(failures)}", flush=True)
    (image_dir.parent / "download_failures.json").write_text(json.dumps(failures[:5000], indent=2), encoding="utf-8")
    return [by_name[name] for name in sorted(ok_names)]


def labels_matrix(samples: list[dict], class_names: list[str]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(class_names)}
    labels = np.zeros((len(samples), len(class_names)), dtype=np.float32)
    for i, sample in enumerate(samples):
        for label in sample["labels"]:
            j = idx.get(label)
            if j is not None:
                labels[i, j] = 1.0
    return labels


class NUSWideDataset(Dataset):
    def __init__(self, image_dir: Path, samples: list[dict], labels: np.ndarray, preprocess):
        self.image_dir = image_dir
        self.samples = samples
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_dir / self.samples[idx]["image_name"]).convert("RGB")
        return self.preprocess(image), torch.from_numpy(self.labels[idx])


def collate(batch):
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(labels, dim=0)


@torch.no_grad()
def encode_images(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
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
def encode_text(model, class_names: list[str], device: str) -> np.ndarray:
    prompts = [f"a photo of {name.replace('_', ' ')}" for name in class_names]
    tokens = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(tokens)
    text_features = F.normalize(text_features.float(), dim=-1)
    return text_features.cpu().numpy()


def load_or_compute_nuswide_features(cfg: dict, model, preprocess, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], bool]:
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    cache_dir = Path(cfg.get("feature_cache_dir", Path(cfg["output_dir"]) / "cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_pool_seed = int(data_cfg.get("image_pool_seed", cfg["seed"]))
    cache_tag = (
        f"nuswide_{data_cfg.get('max_images', 'full')}_{data_cfg.get('num_classes', 81)}_s{image_pool_seed}_"
        f"{cfg['clip']['model'].replace('/', '-')}"
    )
    paths = {
        "features": cache_dir / f"{cache_tag}_image_features.npy",
        "labels": cache_dir / f"{cache_tag}_labels.npy",
        "text": cache_dir / f"{cache_tag}_text_features.npy",
        "classes": cache_dir / f"{cache_tag}_classes.json",
        "images": cache_dir / f"{cache_tag}_image_names.json",
    }
    legacy_cache_tag = (
        f"nuswide_{data_cfg.get('max_images', 'full')}_{data_cfg.get('num_classes', 81)}_"
        f"{cfg['clip']['model'].replace('/', '-')}"
    )
    legacy_paths = {
        "features": cache_dir / f"{legacy_cache_tag}_image_features.npy",
        "labels": cache_dir / f"{legacy_cache_tag}_labels.npy",
        "text": cache_dir / f"{legacy_cache_tag}_text_features.npy",
        "classes": cache_dir / f"{legacy_cache_tag}_classes.json",
        "images": cache_dir / f"{legacy_cache_tag}_image_names.json",
    }
    if all(path.exists() for path in paths.values()):
        return (
            np.load(paths["features"]),
            np.load(paths["labels"]),
            np.load(paths["text"]),
            json.loads(paths["classes"].read_text(encoding="utf-8")),
            json.loads(paths["images"].read_text(encoding="utf-8")),
            True,
        )
    legacy_seed_matches = image_pool_seed == int(cfg["seed"])
    if legacy_seed_matches and all(path.exists() for path in legacy_paths.values()):
        for key, legacy_path in legacy_paths.items():
            if not paths[key].exists():
                if legacy_path.suffix == ".npy":
                    np.save(paths[key], np.load(legacy_path))
                else:
                    paths[key].write_text(legacy_path.read_text(encoding="utf-8"), encoding="utf-8")
        return (
            np.load(paths["features"]),
            np.load(paths["labels"]),
            np.load(paths["text"]),
            json.loads(paths["classes"].read_text(encoding="utf-8")),
            json.loads(paths["images"].read_text(encoding="utf-8")),
            True,
        )

    ensure_nuswide_metadata(root, data_cfg["source_base_url"], small_only=bool(data_cfg.get("small_only", False)))
    train_path = root / data_cfg.get("train_json", "train.json")
    test_path = root / data_cfg.get("test_json", "test.json")
    if bool(data_cfg.get("small_only", False)):
        train_path = root / "small_train.json"
        test_path = root / "small_test.json"
    samples = read_samples(train_path) + read_samples(test_path)
    concepts = read_concepts(root / "cats")[: int(data_cfg.get("num_classes", 81))]
    selected, class_names = select_balanced_samples(
        samples,
        concepts,
        int(data_cfg.get("max_images", 12000)),
        int(data_cfg.get("min_label_positives", 20)),
        image_pool_seed,
    )
    url_map = read_url_map(root / data_cfg.get("urls_file", "NUS-WIDE-urls.txt"))
    image_dir = root / "images"
    downloaded = download_images(selected, url_map, image_dir, int(data_cfg.get("download_workers", 24)))
    if len(downloaded) < int(data_cfg.get("min_downloaded_images", 1000)):
        raise RuntimeError(f"NUS-WIDE image download yielded only {len(downloaded)} usable images")
    labels = labels_matrix(downloaded, class_names)
    positive_counts = labels.sum(axis=0)
    keep = positive_counts >= int(data_cfg.get("min_label_positives", 20))
    class_names = [name for name, flag in zip(class_names, keep) if flag]
    labels = labels[:, keep]
    dataset = NUSWideDataset(image_dir, downloaded, labels, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["clip"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["clip"].get("num_workers", 2)),
        pin_memory=(device == "cuda"),
        collate_fn=collate,
    )
    features, labels = encode_images(model, loader, device)
    text_features = encode_text(model, class_names, device)
    image_names = [sample["image_name"] for sample in downloaded]
    np.save(paths["features"], features)
    np.save(paths["labels"], labels)
    np.save(paths["text"], text_features)
    paths["classes"].write_text(json.dumps(class_names, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["images"].write_text(json.dumps(image_names, indent=2), encoding="utf-8")
    return features, labels, text_features, class_names, image_names, False
