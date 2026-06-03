from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from run_known_aware_posthoc_baselines import (
    build_splits,
    infer_dataset,
    load_cached_dataset,
    model_tag,
    split_retrieval_calibration_eval,
    write_csv,
)


DEFAULT_CONFIGS = [
    "configs/openimages_10k_heldout_ultrastrict.yaml",
    "configs/openimages_10k_heldout_ultrastrict_classB.yaml",
    "configs/coco_heldout_ultrastrict.yaml",
    "configs/coco_heldout_ultrastrict_classB.yaml",
    "configs/nuswide_heldout_ultrastrict.yaml",
    "configs/nuswide_heldout_ultrastrict_classB.yaml",
]


ROW_FIELDS = [
    "dataset",
    "config",
    "config_name",
    "seed",
    "run_seed",
    "split",
    "split_seed",
    "num_images_total",
    "eval_images",
    "num_eval_images",
    "retrieval_images",
    "num_retrieval_images",
    "calibration_images",
    "num_calibration_images",
    "num_classes",
    "num_eligible_classes",
    "num_tail_labels",
    "num_tail_known_labels",
    "num_emerging_labels",
    "tail_positive_eval_images",
    "eval_tail_known_positive_images",
    "emerging_positive_eval_images",
    "eval_emerging_positive_images",
    "risk_set_denominator",
    "tecr_denominator",
    "risk_set_fraction",
    "denominator_fraction_of_eval",
    "denominator_fraction_of_total",
    "retrieval_tail_known_positive_images",
    "calibration_tail_known_positive_images",
    "tail_known_labels",
    "emerging_labels",
    "label_cache",
    "classes_cache",
]


DATASET_SUMMARY_FIELDS = [
    "dataset",
    "num_rows",
    "num_configs",
    "num_run_seeds",
    "num_label_splits",
    "configs",
    "run_seeds",
    "tecr_denominator_mean",
    "tecr_denominator_std",
    "tecr_denominator_min",
    "tecr_denominator_max",
    "denominator_fraction_of_eval_mean",
    "denominator_fraction_of_eval_std",
    "denominator_fraction_of_eval_min",
    "denominator_fraction_of_eval_max",
    "eval_tail_known_positive_images_mean",
    "eval_tail_known_positive_images_std",
    "eval_tail_known_positive_images_min",
    "eval_tail_known_positive_images_max",
    "eval_emerging_positive_images_mean",
    "eval_emerging_positive_images_std",
    "eval_emerging_positive_images_min",
    "eval_emerging_positive_images_max",
    "num_eval_images_mean",
    "num_eval_images_std",
    "num_eval_images_min",
    "num_eval_images_max",
    "zero_denominator_rows",
]


LEGACY_SUMMARY_FIELDS = [
    "dataset",
    "num_splits",
    "risk_set_denominator_mean",
    "risk_set_denominator_std",
    "risk_set_denominator_min",
    "risk_set_denominator_max",
    "risk_set_fraction_mean",
    "risk_set_fraction_std",
    "eval_images_mean",
    "eval_images_std",
]


def cache_paths(dataset: str, cfg: dict, output_dir: Path) -> dict[str, str]:
    data_cfg = cfg["data"]
    cache_dir = Path(cfg.get("feature_cache_dir", output_dir / "cache"))
    if dataset == "openimages":
        tag = (
            f"openimages_{data_cfg['split']}_{data_cfg['max_images']}_"
            f"{data_cfg['num_classes']}_{model_tag(cfg)}"
        )
        return {
            "label_cache": str(cache_dir / f"{tag}_labels.npy"),
            "classes_cache": str(cache_dir / f"{tag}_classes.json"),
        }
    if dataset == "coco":
        max_images = data_cfg.get("max_images")
        tag = f"coco2017_{data_cfg['image_set']}_{max_images or 'full'}_{model_tag(cfg)}"
        return {
            "label_cache": str(cache_dir / f"{tag}_labels.npy"),
            "classes_cache": "",
        }
    if dataset == "nuswide":
        max_images = data_cfg.get("max_images", "full")
        num_classes = data_cfg.get("num_classes", 81)
        image_pool_seed = int(data_cfg.get("image_pool_seed", cfg["seed"]))
        tag = f"nuswide_{max_images}_{num_classes}_s{image_pool_seed}_{model_tag(cfg)}"
        label_cache = cache_dir / f"{tag}_labels.npy"
        classes_cache = cache_dir / f"{tag}_classes.json"
        if not label_cache.exists() or not classes_cache.exists():
            legacy_tag = f"nuswide_{max_images}_{num_classes}_{model_tag(cfg)}"
            legacy_label = cache_dir / f"{legacy_tag}_labels.npy"
            legacy_classes = cache_dir / f"{legacy_tag}_classes.json"
            if legacy_label.exists() and legacy_classes.exists():
                label_cache = legacy_label
                classes_cache = legacy_classes
        return {
            "label_cache": str(label_cache),
            "classes_cache": str(classes_cache),
        }
    return {"label_cache": "", "classes_cache": ""}


def parse_run_seeds(args: argparse.Namespace, cfg_seed: int) -> list[int]:
    seeds: list[int] = []
    if args.seed_override is not None:
        seeds.append(int(args.seed_override))
    for seed in args.seed or []:
        seeds.append(int(seed))
    for chunk in args.seeds or []:
        seeds.extend(int(part.strip()) for part in chunk.split(",") if part.strip())
    if not seeds:
        seeds.append(int(cfg_seed))
    return list(dict.fromkeys(seeds))


def any_positive(labels: np.ndarray, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.zeros(labels.shape[0], dtype=bool)
    return labels[:, indices].max(axis=1) > 0


def count_eligible_classes(
    class_names: list[str],
    retrieval_labels: np.ndarray,
    calibration_labels: np.ndarray,
    protocol_cfg: dict,
) -> int:
    retrieval_counts = retrieval_labels.sum(axis=0)
    calibration_counts = calibration_labels.sum(axis=0)
    min_retrieval = int(protocol_cfg.get("min_retrieval_positives", 5))
    min_calibration = int(protocol_cfg.get("min_calibration_positives", 3))
    return int(sum(
        1
        for i in range(len(class_names))
        if retrieval_counts[i] >= min_retrieval and calibration_counts[i] >= min_calibration
    ))


def compute_rows_for_config(config_path: Path, dataset: str, run_seeds: list[int]) -> list[dict]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))

    features, labels, _text_features, class_names = load_cached_dataset(dataset, cfg, Path(cfg["output_dir"]))
    cache_info = cache_paths(dataset, cfg, Path(cfg["output_dir"]))
    rows: list[dict] = []

    for run_seed in run_seeds:
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
            int(run_seed),
            retrieval_fraction,
            calibration_fraction,
        )
        splits = build_splits(class_names, retrieval_labels, calibration_labels, cfg["protocol"], int(run_seed))
        eligible_count = count_eligible_classes(class_names, retrieval_labels, calibration_labels, cfg["protocol"])

        for split in splits:
            emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
            tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
            tail_positive = any_positive(eval_labels, tail_idx)
            emerging_positive = any_positive(eval_labels, emerging_idx)
            risk_set = tail_positive & ~emerging_positive
            retrieval_tail_positive = any_positive(retrieval_labels, tail_idx)
            calibration_tail_positive = any_positive(calibration_labels, tail_idx)
            denominator = int(risk_set.sum())
            eval_count = int(eval_labels.shape[0])
            row = {
                "dataset": dataset,
                "config": str(config_path),
                "config_name": config_path.stem,
                "seed": int(run_seed),
                "run_seed": int(run_seed),
                "split": split["name"],
                "split_seed": int(split["seed"]),
                "num_images_total": int(labels.shape[0]),
                "eval_images": eval_count,
                "num_eval_images": eval_count,
                "retrieval_images": int(retrieval_features.shape[0]),
                "num_retrieval_images": int(retrieval_features.shape[0]),
                "calibration_images": int(calibration_features.shape[0]),
                "num_calibration_images": int(calibration_features.shape[0]),
                "num_classes": int(labels.shape[1]),
                "num_eligible_classes": eligible_count,
                "num_tail_labels": int(len(tail_idx)),
                "num_tail_known_labels": int(len(tail_idx)),
                "num_emerging_labels": int(len(emerging_idx)),
                "tail_positive_eval_images": int(tail_positive.sum()),
                "eval_tail_known_positive_images": int(tail_positive.sum()),
                "emerging_positive_eval_images": int(emerging_positive.sum()),
                "eval_emerging_positive_images": int(emerging_positive.sum()),
                "risk_set_denominator": denominator,
                "tecr_denominator": denominator,
                "risk_set_fraction": float(denominator / max(eval_count, 1)),
                "denominator_fraction_of_eval": float(denominator / max(eval_count, 1)),
                "denominator_fraction_of_total": float(denominator / max(int(labels.shape[0]), 1)),
                "retrieval_tail_known_positive_images": int(retrieval_tail_positive.sum()),
                "calibration_tail_known_positive_images": int(calibration_tail_positive.sum()),
                "tail_known_labels": ";".join(split["tail_known_labels"]),
                "emerging_labels": ";".join(split["emerging_labels"]),
                **cache_info,
            }
            rows.append(row)
    return rows


def denominator_rows(
    config_path: Path,
    output_dir: Path,
    dataset: str,
    seed_override: int | None,
    seeds: list[int] | None = None,
) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_seeds = seeds if seeds is not None else [
        int(seed_override) if seed_override is not None else int(cfg["seed"])
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    rows = compute_rows_for_config(config_path, dataset, run_seeds)
    dataset_rows = dataset_summary(rows)
    legacy_rows = summarize(rows, ["dataset"])
    write_outputs(output_dir, rows, dataset_rows, legacy_rows, combined=False)

    result = {
        "status": "tecr_denominators_complete",
        "dataset": dataset,
        "config": str(config_path),
        "seeds": [int(seed) for seed in run_seeds],
        "output_dir": str(output_dir),
        "num_rows": len(rows),
        "summary": dataset_rows,
    }
    (output_dir / "tecr_denominator_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def normalize_row(row: dict) -> dict:
    out = dict(row)
    out.setdefault("config", "")
    out.setdefault("config_name", Path(str(out.get("config", ""))).stem)
    out.setdefault("run_seed", out.get("seed", 0))
    out.setdefault("seed", out.get("run_seed", 0))
    out.setdefault("split_seed", out.get("split", ""))
    out.setdefault("num_eval_images", out.get("eval_images", 0))
    out.setdefault("eval_images", out.get("num_eval_images", 0))
    out.setdefault("num_retrieval_images", out.get("retrieval_images", 0))
    out.setdefault("retrieval_images", out.get("num_retrieval_images", 0))
    out.setdefault("num_calibration_images", out.get("calibration_images", 0))
    out.setdefault("calibration_images", out.get("num_calibration_images", 0))
    out.setdefault("tecr_denominator", out.get("risk_set_denominator", 0))
    out.setdefault("risk_set_denominator", out.get("tecr_denominator", 0))
    out.setdefault("denominator_fraction_of_eval", out.get("risk_set_fraction", 0.0))
    out.setdefault("risk_set_fraction", out.get("denominator_fraction_of_eval", 0.0))
    out.setdefault("denominator_fraction_of_total", 0.0)
    out.setdefault("eval_tail_known_positive_images", out.get("tail_positive_eval_images", 0))
    out.setdefault("tail_positive_eval_images", out.get("eval_tail_known_positive_images", 0))
    out.setdefault("eval_emerging_positive_images", out.get("emerging_positive_eval_images", 0))
    out.setdefault("emerging_positive_eval_images", out.get("eval_emerging_positive_images", 0))
    out.setdefault("num_images_total", 0)
    out.setdefault("num_classes", 0)
    out.setdefault("num_eligible_classes", 0)
    out.setdefault("num_tail_labels", out.get("num_tail_known_labels", 0))
    out.setdefault("num_tail_known_labels", out.get("num_tail_labels", 0))
    out.setdefault("num_emerging_labels", 0)
    out.setdefault("retrieval_tail_known_positive_images", 0)
    out.setdefault("calibration_tail_known_positive_images", 0)
    out.setdefault("tail_known_labels", "")
    out.setdefault("emerging_labels", "")
    out.setdefault("label_cache", "")
    out.setdefault("classes_cache", "")

    int_keys = [
        "seed", "run_seed", "num_images_total", "eval_images", "num_eval_images",
        "retrieval_images", "num_retrieval_images", "calibration_images", "num_calibration_images",
        "num_classes", "num_eligible_classes", "num_tail_labels", "num_tail_known_labels",
        "num_emerging_labels", "tail_positive_eval_images", "eval_tail_known_positive_images",
        "emerging_positive_eval_images", "eval_emerging_positive_images", "risk_set_denominator",
        "tecr_denominator", "retrieval_tail_known_positive_images", "calibration_tail_known_positive_images",
    ]
    for key in int_keys:
        if out.get(key) not in (None, ""):
            out[key] = int(float(out[key]))
    for key in ["risk_set_fraction", "denominator_fraction_of_eval", "denominator_fraction_of_total"]:
        if out.get(key) not in (None, ""):
            out[key] = float(out[key])
    return out


def summarize(rows: list[dict], group_keys: list[str]) -> list[dict]:
    normalized = [normalize_row(row) for row in rows]
    grouped: dict[tuple, list[dict]] = {}
    for row in normalized:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    out = []
    for key, group in grouped.items():
        item = {name: value for name, value in zip(group_keys, key)}
        item["num_splits"] = len(group)
        for metric in ["risk_set_denominator", "risk_set_fraction", "eval_images"]:
            vals = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = float(np.mean(vals))
            item[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            item[f"{metric}_min"] = float(np.min(vals))
            item[f"{metric}_max"] = float(np.max(vals))
        return_keys = tuple(str(item[key]) for key in group_keys)
        item["_sort_key"] = return_keys
        out.append(item)
    out.sort(key=lambda row: row.pop("_sort_key"))
    return out


def metric_stats(rows: list[dict], key: str) -> dict[str, float | None]:
    vals = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    if not vals:
        return {
            f"{key}_mean": None,
            f"{key}_std": None,
            f"{key}_min": None,
            f"{key}_max": None,
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        f"{key}_mean": float(arr.mean()),
        f"{key}_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        f"{key}_min": float(arr.min()),
        f"{key}_max": float(arr.max()),
    }


def dataset_summary(rows: list[dict]) -> list[dict]:
    normalized = [normalize_row(row) for row in rows]
    out = []
    for dataset in sorted({row["dataset"] for row in normalized}):
        group = [row for row in normalized if row["dataset"] == dataset]
        item = {
            "dataset": dataset,
            "num_rows": len(group),
            "num_configs": len({row["config_name"] for row in group}),
            "num_run_seeds": len({row["run_seed"] for row in group}),
            "num_label_splits": len({row.get("split_seed", row["split"]) for row in group}),
            "configs": ";".join(sorted({row["config_name"] for row in group if row["config_name"]})),
            "run_seeds": ";".join(str(seed) for seed in sorted({int(row["run_seed"]) for row in group})),
            "zero_denominator_rows": sum(1 for row in group if int(row["tecr_denominator"]) == 0),
        }
        for key in [
            "tecr_denominator",
            "denominator_fraction_of_eval",
            "eval_tail_known_positive_images",
            "eval_emerging_positive_images",
            "num_eval_images",
        ]:
            item.update(metric_stats(group, key))
        out.append(item)
    return out


def parse_row(row: dict) -> dict:
    return normalize_row(row)


def fmt(value: object, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def write_report(path: Path, rows: list[dict], summary: list[dict], combined: bool) -> None:
    normalized = [normalize_row(row) for row in rows]
    title = "Combined TECR Risk-Set Denominators" if combined else "TECR Risk-Set Denominators"
    lines = [
        f"# {title}",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "Definition: `|C|` is the number of held-out evaluation images with at least one tail-known positive label and no emerging positive label for the same dataset/config/run-seed/label-split.",
        "The script reuses the existing cached-array loader, `split_retrieval_calibration_eval`, and `build_splits` protocol from `run_known_aware_posthoc_baselines.py`.",
        "No scorer, threshold, kNN prediction, or gate selection is used for this denominator audit.",
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Rows | Configs | Run seeds | Label splits | C mean | C std | C min | C max | Eval fraction mean | Zero rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['num_rows']} | {row['num_configs']} | "
            f"{row['num_run_seeds']} | {row['num_label_splits']} | "
            f"{fmt(row['tecr_denominator_mean'])} | {fmt(row['tecr_denominator_std'])} | "
            f"{fmt(row['tecr_denominator_min'])} | {fmt(row['tecr_denominator_max'])} | "
            f"{fmt(row['denominator_fraction_of_eval_mean'], 4)} | {row['zero_denominator_rows']} |"
        )
    lines.extend([
        "",
        "## Inputs",
        "",
        "| Dataset | Config | Run seeds | Label cache | Classes cache |",
        "|---|---|---|---|---|",
    ])
    seen = set()
    for row in normalized:
        key = (row["dataset"], row["config_name"], row.get("label_cache", ""))
        if key in seen:
            continue
        seen.add(key)
        seeds = sorted({
            int(item["run_seed"])
            for item in normalized
            if item["dataset"] == row["dataset"] and item["config_name"] == row["config_name"]
        })
        lines.append(
            f"| {row['dataset']} | {row['config_name']} | "
            f"{';'.join(str(seed) for seed in seeds)} | `{row.get('label_cache', '')}` | "
            f"`{row.get('classes_cache', '')}` |"
        )
    lines.extend([
        "",
        "## Audit Notes",
        "",
        "- `run_seed`/`seed` is the held-out image split seed; `split_seed` is the class split seed from `protocol.split_seeds`.",
        "- The denominator is counted only on the evaluation partition. Retrieval and calibration labels are used to define eligible labels and tail-known ordering.",
        "- `eval_tail_known_positive_images` can exceed `|C|` because images that also contain an emerging positive are excluded from the TECR risk set.",
        "- A zero-denominator row means TECR is undefined for that dataset/config/seed/split and should not be averaged as zero.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, rows: list[dict], dataset_rows: list[dict], legacy_rows: list[dict], combined: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = [normalize_row(row) for row in rows]
    write_csv(output_dir / "tecr_denominator_rows.csv", normalized, ROW_FIELDS)
    write_csv(output_dir / "tecr_denominator_dataset_summary.csv", dataset_rows, DATASET_SUMMARY_FIELDS)
    write_csv(output_dir / "tecr_denominator_summary.csv", legacy_rows, LEGACY_SUMMARY_FIELDS)
    write_report(output_dir / "tecr_denominator_report.md", normalized, dataset_rows, combined=combined)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [normalize_row(row) for row in csv.DictReader(f)]


def combine_outputs(root: Path, output_dir: Path) -> dict:
    rows = []
    output_resolved = output_dir.resolve()
    for path in sorted(root.rglob("tecr_denominator_rows.csv")):
        try:
            if path.resolve().is_relative_to(output_resolved):
                continue
        except AttributeError:
            pass
        with path.open("r", newline="", encoding="utf-8") as f:
            rows.extend(parse_row(row) for row in csv.DictReader(f))
    if not rows:
        raise FileNotFoundError(f"No tecr_denominator_rows.csv files found under {root}")
    dataset_rows = dataset_summary(rows)
    legacy_rows = summarize(rows, ["dataset"])
    write_outputs(output_dir, rows, dataset_rows, legacy_rows, combined=True)
    result = {
        "status": "tecr_denominators_combined_complete",
        "root": str(root),
        "output_dir": str(output_dir),
        "num_rows": len(rows),
        "summary": dataset_rows,
    }
    (output_dir / "tecr_denominator_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def run_many(config_paths: list[Path], output_dir: Path, dataset_arg: str, args: argparse.Namespace) -> dict:
    all_rows: list[dict] = []
    for config_path in config_paths:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dataset = infer_dataset(config_path, dataset_arg)
        run_seeds = parse_run_seeds(args, int(cfg["seed"]))
        subdir = output_dir / config_path.stem if len(config_paths) > 1 else output_dir
        denominator_rows(config_path, subdir, dataset, args.seed_override, seeds=run_seeds)
        all_rows.extend(read_rows(subdir / "tecr_denominator_rows.csv"))
    dataset_rows = dataset_summary(all_rows)
    legacy_rows = summarize(all_rows, ["dataset"])
    if len(config_paths) > 1:
        write_outputs(output_dir, all_rows, dataset_rows, legacy_rows, combined=False)
    result = {
        "status": "tecr_denominators_complete",
        "output_dir": str(output_dir),
        "num_rows": len(all_rows),
        "datasets": sorted({row["dataset"] for row in all_rows}),
        "summary": dataset_rows,
    }
    (output_dir / "tecr_denominator_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        help="Config YAML path. Repeat for multiple configs. Defaults to the six main held-out configs.",
    )
    parser.add_argument("--output-dir", default="outputs/tecr_denominators")
    parser.add_argument("--dataset", choices=["auto", "openimages", "coco", "nuswide"], default="auto")
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--seed", action="append", type=int, help="Held-out image split seed. Repeatable.")
    parser.add_argument("--seeds", action="append", help="Comma-separated held-out image split seeds.")
    parser.add_argument("--combine-root")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.combine_root:
        combine_outputs(Path(args.combine_root), output_dir)
        return 0
    config_paths = [Path(path) for path in (args.config or DEFAULT_CONFIGS)]
    if len(config_paths) == 1:
        config_path = config_paths[0]
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dataset = infer_dataset(config_path, args.dataset)
        run_seeds = parse_run_seeds(args, int(cfg["seed"]))
        denominator_rows(config_path, output_dir, dataset, args.seed_override, seeds=run_seeds)
        return 0
    run_many(config_paths, output_dir, args.dataset, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
