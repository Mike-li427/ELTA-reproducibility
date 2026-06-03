from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy import __version__ as scipy_version
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_inputs(input_path: str | None, input_glob: str | None) -> tuple[list[dict[str, str]], str]:
    if bool(input_path) == bool(input_glob):
        raise ValueError("Specify exactly one of --input or --input-glob.")
    if input_path:
        path = (ROOT / input_path).resolve() if not Path(input_path).is_absolute() else Path(input_path)
        return read_rows(path), f"`{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}`"

    paths = sorted(ROOT.glob(input_glob or ""))
    if not paths:
        raise FileNotFoundError(f"No input files matched --input-glob {input_glob!r}")
    rows: list[dict[str, str]] = []
    for path in paths:
        source = path.parent.name
        for row in read_rows(path):
            item = dict(row)
            item.setdefault("source", source)
            rows.append(item)
    return rows, f"`{input_glob}` ({len(paths)} files)"


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def value_column(rows: list[dict[str, str]], metric: str) -> str:
    aliases = {
        "ap": ["ap", "average_precision", "average_precision_mean"],
        "f1": ["f1", "best_f1", "best_f1_mean"],
        "tecr": ["tecr", "tecr_mean"],
    }
    candidates = aliases.get(metric, [metric])
    fields = set(rows[0].keys()) if rows else set()
    for candidate in candidates:
        if candidate in fields:
            return candidate
    raise ValueError(f"Could not find a value column for metric '{metric}'. Tried: {', '.join(candidates)}")


def unit_key(row: dict[str, str], unit_cols: list[str]) -> tuple[str, ...]:
    missing = [col for col in unit_cols if col not in row]
    if missing:
        raise ValueError(f"Input is missing unit column(s): {', '.join(missing)}")
    return tuple(row[col] for col in unit_cols)


def keep_dataset(row: dict[str, str], dataset: str | None) -> bool:
    return dataset is None or row.get("dataset") == dataset


def build_method_values(
    rows: list[dict[str, str]],
    metric_col: str,
    unit_cols: list[str],
    methods: set[str],
    dataset: str | None,
) -> dict[tuple[str, ...], dict[str, float]]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not keep_dataset(row, dataset):
            continue
        method = row.get("method")
        if method not in methods:
            continue
        value = row.get(metric_col)
        if value in (None, ""):
            continue
        grouped[unit_key(row, unit_cols)][method].append(float(value))

    out: dict[tuple[str, ...], dict[str, float]] = {}
    for key, by_method in grouped.items():
        out[key] = {method: mean(values) for method, values in by_method.items()}
    return out


def paired_arrays(
    values: dict[tuple[str, ...], dict[str, float]],
    baseline_method: str,
    method: str,
) -> tuple[list[tuple[str, ...]], np.ndarray, np.ndarray]:
    keys: list[tuple[str, ...]] = []
    baseline: list[float] = []
    treatment: list[float] = []
    for key in sorted(values):
        methods = values[key]
        if baseline_method in methods and method in methods:
            keys.append(key)
            baseline.append(methods[baseline_method])
            treatment.append(methods[method])
    if not keys:
        raise ValueError("No paired configuration-level units contain both requested methods.")
    return keys, np.asarray(baseline, dtype=np.float64), np.asarray(treatment, dtype=np.float64)


def signed_rank_pvalue(deltas: np.ndarray, alternative: str, zero_method: str) -> tuple[float, float]:
    if np.all(deltas == 0.0):
        return 0.0, 1.0
    result = wilcoxon(deltas, alternative=alternative, zero_method=zero_method, correction=False, method="auto")
    return float(result.statistic), float(result.pvalue)


def fmt(value: float) -> str:
    return f"{value:.6g}"


def run(args: argparse.Namespace) -> int:
    rows, input_label = read_inputs(args.input, args.input_glob)
    if not rows:
        raise ValueError("No rows found in input.")

    unit_cols = split_csv(args.unit_cols)
    metric_col = value_column(rows, args.metric)
    values = build_method_values(
        rows,
        metric_col,
        unit_cols,
        {args.baseline_method, args.method},
        args.dataset,
    )
    keys, baseline, treatment = paired_arrays(values, args.baseline_method, args.method)

    if args.delta == "baseline_minus_method":
        deltas = baseline - treatment
        delta_label = f"{args.baseline_method} - {args.method}"
    else:
        deltas = treatment - baseline
        delta_label = f"{args.method} - {args.baseline_method}"

    statistic, pvalue = signed_rank_pvalue(deltas, args.alternative, args.zero_method)
    nonzero = int(np.count_nonzero(deltas))
    ties = int(deltas.size - nonzero)
    wins = int(np.sum(deltas > 0.0))
    losses = int(np.sum(deltas < 0.0))

    print("# Descriptive Wilcoxon Signed-Rank Check")
    print()
    print(f"Input: {input_label}")
    if args.dataset:
        print(f"Dataset filter: `{args.dataset}`")
    print(f"Configuration-level unit columns: `{','.join(unit_cols)}`")
    print(f"Metric column: `{metric_col}`")
    print(f"Delta: `{delta_label}`")
    print(f"SciPy: `{scipy_version}`")
    print(f"Test: `scipy.stats.wilcoxon(deltas, alternative='{args.alternative}', zero_method='{args.zero_method}', correction=False, method='auto')`")
    print()
    print("| n pairs | nonzero | ties | positive deltas | negative deltas | mean delta | median delta | statistic | one-sided p |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {deltas.size} | {nonzero} | {ties} | {wins} | {losses} | "
        f"{fmt(float(np.mean(deltas)))} | {fmt(float(np.median(deltas)))} | "
        f"{fmt(statistic)} | {fmt(pvalue)} |"
    )
    print()
    print("Configuration-level paired deltas:")
    for key, delta in zip(keys, deltas):
        print(f"- `{','.join(key)}`: {fmt(float(delta))}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute descriptive Wilcoxon signed-rank p values from configuration-level "
            "result rows. Split-level rows may be provided only when --unit-cols groups "
            "them into configuration-level paired units first."
        )
    )
    parser.add_argument("--input")
    parser.add_argument(
        "--input-glob",
        help=(
            "Glob of per-configuration summary CSVs. The parent directory name is added "
            "as a synthetic configuration-level `source` column."
        ),
    )
    parser.add_argument("--dataset", help="Optional dataset column filter, for combined CSVs.")
    parser.add_argument("--unit-cols", default="split_set,seed")
    parser.add_argument("--metric", default="tecr", help="Metric name or column alias: tecr, ap, f1.")
    parser.add_argument("--baseline-method", default="clip_knn_global_threshold")
    parser.add_argument("--method", default="heldout_gate_global_threshold")
    parser.add_argument(
        "--delta",
        choices=("baseline_minus_method", "method_minus_baseline"),
        default="baseline_minus_method",
        help="Use baseline_minus_method for TECR reductions where positive deltas favor the new method.",
    )
    parser.add_argument(
        "--alternative",
        choices=("greater", "less", "two-sided"),
        default="greater",
        help="One-sided alternative applied to the signed deltas; default tests positive deltas.",
    )
    parser.add_argument(
        "--zero-method",
        choices=("wilcox", "pratt", "zsplit"),
        default="wilcox",
        help="SciPy zero/tie convention for zero deltas. Default drops zero differences.",
    )
    args = parser.parse_args()
    if args.input is None and args.input_glob is None:
        args.input = "results/nuswide/main_per_config_rows.csv"
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
