from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from run_openimages_heldout_calibration import (
    aggregate_scores,
    best_f1_threshold,
    build_splits,
    confidence_gate_scores,
    metrics_with_calibrated_threshold,
    pct_change,
    pct_reduction,
    split_retrieval_calibration_eval,
)
from run_openimages_margin_gate import load_cached_arrays
from run_openimages_pilot import knn_score_matrix


def normalized_entropy(scores: np.ndarray) -> np.ndarray:
    probs = np.maximum(scores, 0.0)
    denom = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, np.maximum(denom, 1e-12))
    ent = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    return ent / max(np.log(scores.shape[1]), 1e-12)


def selective_reject_scores(scores: np.ndarray, entropy: np.ndarray, cutoff: float) -> np.ndarray:
    out = np.array(scores, copy=True)
    out[entropy >= cutoff] = 0.0
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
        for key in [
            "average_precision",
            "auroc",
            "precision",
            "recall",
            "best_f1",
            "tecr",
            "avg_predicted_labels",
            "set_size",
            "coverage",
        ]:
            vals = [row[key] for row in group if row.get(key) is not None]
            item[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    return sorted(out, key=lambda row: row["method"])


def add_prediction_load(
    metrics: dict,
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    emerging_idx: list[int],
) -> dict:
    pred_matrix = scores >= threshold
    if pred_matrix.ndim == 1:
        metrics["avg_predicted_labels"] = float(pred_matrix.mean())
    else:
        metrics["avg_predicted_labels"] = float(pred_matrix.sum(axis=1).mean())
    y_true = labels[:, emerging_idx].max(axis=1).astype(bool)
    pred_any = pred_matrix if pred_matrix.ndim == 1 else pred_matrix.max(axis=1)
    metrics["coverage"] = float(pred_any[y_true].mean()) if y_true.any() else None
    metrics["set_size"] = None
    return metrics


def binary_nll_temperature(y_true: np.ndarray, scores: np.ndarray, temperatures: list[float]) -> tuple[float, list[dict]]:
    eps = 1e-6
    logits = np.log(np.clip(scores, eps, 1.0 - eps) / np.clip(1.0 - scores, eps, 1.0 - eps))
    rows = []
    best_temp = float(temperatures[0])
    best_nll = float("inf")
    y = y_true.astype(float)
    for temp in temperatures:
        temp = float(temp)
        scaled = 1.0 / (1.0 + np.exp(-np.clip(logits / max(temp, 1e-6), -50.0, 50.0)))
        nll = -np.mean(y * np.log(np.clip(scaled, eps, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - scaled, eps, 1.0)))
        row = {"temperature": temp, "nll": float(nll)}
        rows.append(row)
        if nll < best_nll:
            best_nll = float(nll)
            best_temp = temp
    return best_temp, rows


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    eps = 1e-6
    logits = np.log(np.clip(scores, eps, 1.0 - eps) / np.clip(1.0 - scores, eps, 1.0 - eps))
    return 1.0 / (1.0 + np.exp(-np.clip(logits / max(float(temperature), 1e-6), -50.0, 50.0)))


def split_conformal_metrics(
    y_eval: np.ndarray,
    eval_knn: np.ndarray,
    qhat: float,
    eval_labels: np.ndarray,
    tail_idx: list[int],
    emerging_idx: list[int],
) -> dict:
    threshold = float(1.0 - qhat)
    base = metrics_with_calibrated_threshold(
        y_eval,
        aggregate_scores(eval_knn),
        threshold,
        eval_labels,
        tail_idx,
        emerging_idx,
    )
    pred_matrix = eval_knn >= (1.0 - qhat)
    pred_any = pred_matrix.max(axis=1)
    coverage = float(pred_any[y_eval == 1].mean()) if (y_eval == 1).any() else None
    base.update({
        "avg_predicted_labels": float(pred_matrix.sum(axis=1).mean()),
        "set_size": float(pred_matrix.sum(axis=1).mean()),
        "coverage": coverage,
    })
    return base


def conformal_quantile(cal_knn: np.ndarray, cal_labels: np.ndarray, emerging_idx: list[int], alpha: float) -> float:
    class_labels = cal_labels[:, emerging_idx].astype(int)
    conformity = np.where(class_labels == 1, cal_knn, 1.0 - cal_knn)
    scores = 1.0 - conformity.reshape(-1)
    n = scores.size
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(np.sort(scores)[rank - 1])


def select_by_constraints(
    rows: list[dict],
    baseline: dict,
    ap_tolerance: float,
    f1_tolerance: float,
    method: str,
) -> dict | None:
    candidates = [
        row
        for row in rows
        if row["method"] == method
        and row.get("average_precision") is not None
        and row.get("best_f1") is not None
        and row.get("tecr") is not None
        and row["average_precision"] >= baseline["average_precision"] * (1.0 - ap_tolerance)
        and row["best_f1"] >= baseline["best_f1"] * (1.0 - f1_tolerance)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["tecr"], -row["best_f1"], -row["average_precision"]))


def known_reject_scores(scores: np.ndarray, known_scores: np.ndarray, cutoff: float) -> np.ndarray:
    out = np.array(scores, copy=True)
    out[known_scores >= cutoff] = 0.0
    return out


def run_one(config_path: Path, output_dir: Path, seed_override: int | None = None) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_seed = int(seed_override) if seed_override is not None else int(cfg["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    calibration_cfg = cfg.get("heldout_calibration", {})
    retrieval_fraction = float(calibration_cfg.get("retrieval_fraction", 0.4))
    calibration_fraction = float(calibration_cfg.get("calibration_fraction", 0.2))
    ap_tolerance = float(calibration_cfg.get("ap_tolerance", 0.0025))
    f1_tolerance = float(calibration_cfg.get("f1_tolerance", 0.0025))
    residual_powers = calibration_cfg.get("residual_powers", [0.0, 0.1, 0.2, 0.3, 0.35, 0.4])
    confidence_cutoffs = calibration_cfg.get("confidence_cutoffs", [0.2, 0.3, 0.4, 0.5, 0.6])
    confidence_temperatures = calibration_cfg.get("confidence_temperatures", [0.01, 0.02, 0.05, 0.1])
    entropy_cutoffs = [float(x) for x in np.linspace(0.0, 1.0, 51).tolist()] + [1.01]
    known_cutoffs = [float(x) for x in np.linspace(0.0, 1.0, 51).tolist()] + [1.01]
    ts_temperatures = [0.25, 0.33, 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    conformal_alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]

    features, labels, text_features, class_names = load_cached_arrays(cfg, Path(cfg["output_dir"]))
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

    retrieval_logits_raw = retrieval_features @ text_features.T
    logit_mean = retrieval_logits_raw.mean(axis=0, keepdims=True)
    logit_std = retrieval_logits_raw.std(axis=0, keepdims=True) + 1e-8
    calibration_logits = (calibration_features @ text_features.T - logit_mean) / logit_std
    eval_logits = (eval_features @ text_features.T - logit_mean) / logit_std

    eval_rows: list[dict] = []
    selection_rows: list[dict] = []
    grid_rows: list[dict] = []

    for split in splits:
        emerging_idx = [class_names.index(name) for name in split["emerging_labels"]]
        tail_idx = [class_names.index(name) for name in split["tail_known_labels"]]
        known_idx = [i for i, name in enumerate(class_names) if name not in split["emerging_labels"]]

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
        eval_baseline.update({
            "method": "clip_knn",
            "split": split["name"],
            "selection_status": "baseline",
            "selected_param": "",
        })
        eval_rows.append(eval_baseline)

        best_temp, temp_grid = binary_nll_temperature(y_cal, cal_base_scores, ts_temperatures)
        for temp_row in temp_grid:
            temp_row.update({"split": split["name"], "method": "temperature_scaling"})
            grid_rows.append(temp_row)
        cal_ts_scores = apply_temperature(cal_base_scores, best_temp)
        eval_ts_scores = apply_temperature(eval_base_scores, best_temp)
        ts_threshold = best_f1_threshold(y_cal, cal_ts_scores)
        eval_ts = metrics_with_calibrated_threshold(
            y_eval, eval_ts_scores, ts_threshold["threshold"], eval_labels, tail_idx, emerging_idx
        )
        add_prediction_load(eval_ts, apply_temperature(eval_knn, best_temp), ts_threshold["threshold"], eval_labels, emerging_idx)
        eval_ts.update({
            "method": "temperature_scaling",
            "split": split["name"],
            "selection_status": "nll_selected_on_calibration",
            "selected_param": f"T={best_temp:g}",
        })
        eval_rows.append(eval_ts)
        selection_rows.append({
            "split": split["name"],
            "method": "temperature_scaling",
            "selection_status": "nll_selected_on_calibration",
            "selected_param": f"T={best_temp:g}",
            "baseline_calibration_ap": cal_baseline["average_precision"],
            "baseline_calibration_f1": cal_baseline["best_f1"],
            "baseline_calibration_tecr": cal_baseline["tecr"],
        })

        con_rows = []
        for alpha in conformal_alphas:
            qhat = conformal_quantile(cal_knn, calibration_labels, emerging_idx, alpha)
            cal_conformal = split_conformal_metrics(y_cal, cal_knn, qhat, calibration_labels, tail_idx, emerging_idx)
            cal_conformal.update({"method": "split_conformal", "split": split["name"], "selected_param": f"alpha={alpha:g}"})
            con_rows.append(cal_conformal)
            grid_rows.append(cal_conformal)
        selected_conformal = select_by_constraints(con_rows, cal_baseline, ap_tolerance, f1_tolerance, "split_conformal")
        if selected_conformal is None:
            selected_alpha = 0.1
            selection_status = "fallback_alpha_0.1"
        else:
            selected_alpha = float(selected_conformal["selected_param"].split("=")[1])
            selection_status = "selected_under_constraints"
        qhat = conformal_quantile(cal_knn, calibration_labels, emerging_idx, selected_alpha)
        eval_conf = split_conformal_metrics(y_eval, eval_knn, qhat, eval_labels, tail_idx, emerging_idx)
        eval_conf.update({
            "method": "split_conformal",
            "split": split["name"],
            "selection_status": selection_status,
            "selected_param": f"alpha={selected_alpha:g};q={qhat:.6g}",
        })
        eval_rows.append(eval_conf)
        selection_rows.append({
            "split": split["name"],
            "method": "split_conformal",
            "selection_status": selection_status,
            "selected_param": f"alpha={selected_alpha:g};q={qhat:.6g}",
            "baseline_calibration_ap": cal_baseline["average_precision"],
            "baseline_calibration_f1": cal_baseline["best_f1"],
            "baseline_calibration_tecr": cal_baseline["tecr"],
        })

        cal_entropy = normalized_entropy(cal_knn)
        eval_entropy = normalized_entropy(eval_knn)
        entropy_rows = []
        entropy_eval_by_cutoff = {}
        for cutoff in entropy_cutoffs:
            cal_scores = selective_reject_scores(cal_base_scores, cal_entropy, cutoff)
            threshold = best_f1_threshold(y_cal, cal_scores)
            cal_metrics = metrics_with_calibrated_threshold(
                y_cal, cal_scores, threshold["threshold"], calibration_labels, tail_idx, emerging_idx
            )
            cal_metrics.update({"method": "entropy_reject", "split": split["name"], "selected_param": f"cutoff={cutoff:g}"})
            entropy_rows.append(cal_metrics)
            grid_rows.append(cal_metrics)

            eval_scores = selective_reject_scores(eval_base_scores, eval_entropy, cutoff)
            eval_metrics = metrics_with_calibrated_threshold(
                y_eval, eval_scores, threshold["threshold"], eval_labels, tail_idx, emerging_idx
            )
            eval_matrix = selective_reject_scores(eval_knn, eval_entropy, cutoff)
            add_prediction_load(eval_metrics, eval_matrix, threshold["threshold"], eval_labels, emerging_idx)
            entropy_eval_by_cutoff[cutoff] = eval_metrics
        selected_entropy = select_by_constraints(entropy_rows, cal_baseline, ap_tolerance, f1_tolerance, "entropy_reject")
        if selected_entropy is None:
            entropy_cutoff = 1.01
            selection_status = "fallback_baseline"
        else:
            entropy_cutoff = float(selected_entropy["selected_param"].split("=")[1])
            selection_status = "selected_under_constraints"
        eval_entropy_row = dict(entropy_eval_by_cutoff.get(entropy_cutoff, eval_baseline))
        eval_entropy_row.update({
            "method": "entropy_reject",
            "split": split["name"],
            "selection_status": selection_status,
            "selected_param": f"cutoff={entropy_cutoff:g}",
        })
        eval_rows.append(eval_entropy_row)

        cal_known = 1.0 / (1.0 + np.exp(-calibration_logits[:, known_idx].max(axis=1)))
        eval_known = 1.0 / (1.0 + np.exp(-eval_logits[:, known_idx].max(axis=1)))
        known_rows = []
        known_eval_by_cutoff = {}
        for cutoff in known_cutoffs:
            cal_scores = known_reject_scores(cal_base_scores, cal_known, cutoff)
            threshold = best_f1_threshold(y_cal, cal_scores)
            cal_metrics = metrics_with_calibrated_threshold(
                y_cal, cal_scores, threshold["threshold"], calibration_labels, tail_idx, emerging_idx
            )
            cal_metrics.update({"method": "maxlogit_known_reject", "split": split["name"], "selected_param": f"cutoff={cutoff:g}"})
            known_rows.append(cal_metrics)
            grid_rows.append(cal_metrics)

            eval_scores = known_reject_scores(eval_base_scores, eval_known, cutoff)
            eval_metrics = metrics_with_calibrated_threshold(
                y_eval, eval_scores, threshold["threshold"], eval_labels, tail_idx, emerging_idx
            )
            eval_matrix = known_reject_scores(eval_knn, eval_known, cutoff)
            add_prediction_load(eval_metrics, eval_matrix, threshold["threshold"], eval_labels, emerging_idx)
            known_eval_by_cutoff[cutoff] = eval_metrics
        selected_known = select_by_constraints(known_rows, cal_baseline, ap_tolerance, f1_tolerance, "maxlogit_known_reject")
        if selected_known is None:
            known_cutoff = 1.01
            selection_status = "fallback_baseline"
        else:
            known_cutoff = float(selected_known["selected_param"].split("=")[1])
            selection_status = "selected_under_constraints"
        eval_known_row = dict(known_eval_by_cutoff.get(known_cutoff, eval_baseline))
        eval_known_row.update({
            "method": "maxlogit_known_reject",
            "split": split["name"],
            "selection_status": selection_status,
            "selected_param": f"cutoff={known_cutoff:g}",
        })
        eval_rows.append(eval_known_row)

        gate_rows = []
        gate_eval_rows = []
        for residual_power in residual_powers:
            for cutoff in confidence_cutoffs:
                for temperature in confidence_temperatures:
                    residual_power = float(residual_power)
                    cutoff = float(cutoff)
                    temperature = float(temperature)
                    cal_scores = aggregate_scores(confidence_gate_scores(cal_knn, cal_known, residual_power, cutoff, temperature))
                    threshold = best_f1_threshold(y_cal, cal_scores)
                    cal_metrics = metrics_with_calibrated_threshold(
                        y_cal, cal_scores, threshold["threshold"], calibration_labels, tail_idx, emerging_idx
                    )
                    param = f"p={residual_power:g};c={cutoff:g};tau={temperature:g}"
                    cal_metrics.update({"method": "elta_confidence", "split": split["name"], "selected_param": param})
                    gate_rows.append(cal_metrics)
                    grid_rows.append(cal_metrics)

                    eval_scores = confidence_gate_scores(eval_knn, eval_known, residual_power, cutoff, temperature)
                    eval_metrics = metrics_with_calibrated_threshold(
                        y_eval, aggregate_scores(eval_scores), threshold["threshold"], eval_labels, tail_idx, emerging_idx
                    )
                    add_prediction_load(eval_metrics, eval_scores, threshold["threshold"], eval_labels, emerging_idx)
                    eval_metrics.update({"selected_param": param})
                    gate_eval_rows.append(eval_metrics)
        selected_gate = select_by_constraints(gate_rows, cal_baseline, ap_tolerance, f1_tolerance, "elta_confidence")
        if selected_gate is None:
            eval_gate = dict(eval_baseline)
            eval_gate.update({"selection_status": "fallback_baseline", "selected_param": ""})
        else:
            eval_gate = next(row for row in gate_eval_rows if row["selected_param"] == selected_gate["selected_param"])
            eval_gate = dict(eval_gate)
            eval_gate.update({"selection_status": "selected_under_constraints"})
        eval_gate.update({"method": "elta_confidence_heldout", "split": split["name"]})
        eval_rows.append(eval_gate)

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
        "method",
        "split",
        "selection_status",
        "selected_param",
        "average_precision",
        "auroc",
        "precision",
        "recall",
        "best_f1",
        "threshold",
        "tecr",
        "avg_predicted_labels",
        "set_size",
        "coverage",
    ]
    summary_fields = [
        "method",
        "num_splits",
        "average_precision_mean",
        "average_precision_std",
        "auroc_mean",
        "auroc_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "best_f1_mean",
        "best_f1_std",
        "tecr_mean",
        "tecr_std",
        "avg_predicted_labels_mean",
        "avg_predicted_labels_std",
        "set_size_mean",
        "set_size_std",
        "coverage_mean",
        "coverage_std",
        "ap_delta_pct",
        "f1_delta_pct",
        "tecr_reduction_pct",
    ]
    grid_fields = [
        "method",
        "split",
        "selected_param",
        "temperature",
        "nll",
        "average_precision",
        "auroc",
        "precision",
        "recall",
        "best_f1",
        "threshold",
        "tecr",
        "avg_predicted_labels",
        "set_size",
        "coverage",
    ]
    write_csv(output_dir / "posthoc_eval_rows.csv", eval_rows, eval_fields)
    write_csv(output_dir / "posthoc_summary.csv", summary, summary_fields)
    write_csv(output_dir / "posthoc_selection_rows.csv", selection_rows, [
        "split",
        "method",
        "selection_status",
        "selected_param",
        "baseline_calibration_ap",
        "baseline_calibration_f1",
        "baseline_calibration_tecr",
    ])
    write_csv(output_dir / "posthoc_calibration_grid.csv", grid_rows, grid_fields)

    report = [
        "# Open Images Post-hoc Baseline Comparison",
        "",
        f"Config: `{config_path}`",
        f"Seed: `{run_seed}`",
        "",
        "All alternatives are selected on the calibration split and evaluated once on the held-out eval split. AP/F1-constrained methods use the same preservation tolerances as the ELTA held-out gate.",
        "",
        "| Method | AP | F1 | TECR | Avg. labels | Coverage | AP delta | F1 delta | TECR reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        coverage = "" if row["coverage_mean"] is None else f"{row['coverage_mean']:.4f}"
        avg_labels = "" if row["avg_predicted_labels_mean"] is None else f"{row['avg_predicted_labels_mean']:.4f}"
        report.append(
            f"| {row['method']} | {row['average_precision_mean']:.4f} | "
            f"{row['best_f1_mean']:.4f} | {row['tecr_mean']:.4f} | "
            f"{avg_labels} | {coverage} | {row['ap_delta_pct']:.2f}% | "
            f"{row['f1_delta_pct']:.2f}% | {row['tecr_reduction_pct']:.1f}% |"
        )
    (output_dir / "posthoc_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "status": "posthoc_baselines_complete",
        "config": str(config_path),
        "seed": run_seed,
        "output_dir": str(output_dir),
        "summary": summary,
    }
    (output_dir / "posthoc_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def combine_results(root_dir: Path) -> None:
    rows = []
    for path in sorted(root_dir.glob("*/posthoc_summary.csv")):
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
    method_order = {
        "clip_knn": 0,
        "temperature_scaling": 1,
        "split_conformal": 2,
        "maxlogit_known_reject": 3,
        "entropy_reject": 4,
        "elta_confidence_heldout": 5,
    }
    summary.sort(key=lambda row: method_order.get(row["method"], 99))
    fields = [
        "method",
        "n",
        "average_precision_mean",
        "average_precision_std",
        "best_f1_mean",
        "best_f1_std",
        "tecr_mean",
        "tecr_std",
        "avg_predicted_labels_mean",
        "avg_predicted_labels_std",
        "coverage_mean",
        "coverage_std",
        "ap_delta_pct",
        "ap_delta_pct_std",
        "f1_delta_pct",
        "f1_delta_pct_std",
        "tecr_reduction_pct",
        "tecr_reduction_pct_std",
    ]
    write_csv(root_dir / "posthoc_combined_summary.csv", summary, fields)
    lines = [
        "# Post-hoc Baseline Combined Summary",
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
    lines.extend([
        "",
        "Temperature scaling is fit by calibration NLL. Split conformal reports prediction-set activation against TECR and coverage. MaxLogit/MCM-style rejection uses the strongest known-label CLIP logit as an image-level in-distribution score, illustrating the granularity mismatch between OOD rejection and route-level TECR.",
        "",
    ])
    (root_dir / "posthoc_combined_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--combine-root")
    args = parser.parse_args()

    if args.combine_root:
        combine_results(Path(args.combine_root))
        return 0
    if not args.config or not args.output_dir:
        parser.error("--config and --output-dir are required unless --combine-root is used")
    run_one(Path(args.config), Path(args.output_dir), args.seed_override)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
