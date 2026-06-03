from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path, dataset: str) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            if row.get("selection_status") != "selected_under_constraints":
                continue
            rows.append({
                "dataset": dataset,
                "source": str(path),
                "config": path.parent.name,
                "split": row["split"],
                "residual_power": float(row["residual_power"]),
                "confidence_cutoff": float(row["confidence_cutoff"]),
                "confidence_temperature": float(row["confidence_temperature"]),
                "calibration_tecr": float(row["calibration_tecr"]) if row.get("calibration_tecr") else None,
            })
        return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", default=".")
    parser.add_argument("--output-dir", default="outputs/gate_parameter_stability")
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, glob in [
        ("Open Images 10k", "outputs/openimages_10k_heldout_ultrastrict*/heldout_selected_settings.csv"),
        ("COCO val2017", "outputs/coco_heldout_ultrastrict*/heldout_selected_settings.csv"),
    ]:
        for path in sorted(audit_root.glob(glob)):
            rows.extend(read_rows(path, dataset))
    summary = []
    for dataset in sorted({row["dataset"] for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset]
        item = {"dataset": dataset, "num_selected": len(group)}
        for key in ["residual_power", "confidence_cutoff", "confidence_temperature"]:
            vals = np.asarray([row[key] for row in group], dtype=np.float64)
            item[f"{key}_mean"] = float(vals.mean())
            item[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            item[f"{key}_min"] = float(vals.min())
            item[f"{key}_max"] = float(vals.max())
        summary.append(item)
    write_csv(output_dir / "gate_parameter_selected_rows.csv", rows, ["dataset", "config", "split", "residual_power", "confidence_cutoff", "confidence_temperature", "calibration_tecr", "source"])
    write_csv(output_dir / "gate_parameter_stability_summary.csv", summary, [
        "dataset", "num_selected",
        "residual_power_mean", "residual_power_std", "residual_power_min", "residual_power_max",
        "confidence_cutoff_mean", "confidence_cutoff_std", "confidence_cutoff_min", "confidence_cutoff_max",
        "confidence_temperature_mean", "confidence_temperature_std", "confidence_temperature_min", "confidence_temperature_max",
    ])
    lines = ["# Gate Parameter Stability", "", "| Dataset | n | p mean±std | b mean±std | tau mean±std | Range p/b/tau |", "|---|---:|---:|---:|---:|---|"]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['num_selected']} | {row['residual_power_mean']:.3f}±{row['residual_power_std']:.3f} | "
            f"{row['confidence_cutoff_mean']:.3f}±{row['confidence_cutoff_std']:.3f} | "
            f"{row['confidence_temperature_mean']:.3f}±{row['confidence_temperature_std']:.3f} | "
            f"{row['residual_power_min']:.2f}-{row['residual_power_max']:.2f} / "
            f"{row['confidence_cutoff_min']:.2f}-{row['confidence_cutoff_max']:.2f} / "
            f"{row['confidence_temperature_min']:.2f}-{row['confidence_temperature_max']:.2f} |"
        )
    (output_dir / "gate_parameter_stability_report.md").write_text("\n".join(lines), encoding="utf-8")
    result = {"status": "gate_parameter_stability_complete", "output_dir": str(output_dir), "summary": summary}
    (output_dir / "gate_parameter_stability_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
