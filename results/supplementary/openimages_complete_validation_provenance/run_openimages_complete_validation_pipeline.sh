#!/usr/bin/env bash
set -euo pipefail
WORK_DIR="${WORK_DIR:-/path/to/ELTA_repro_fullval_20260607}"
PY="${PYTHON_BIN:-python3}"
cd "$WORK_DIR"
SEEDS=(20260522 20260523 20260524 20260525 20260526 20260527)
: > fullval_pipeline.log
exec > >(tee -a fullval_pipeline.log) 2>&1

echo "[START] $(date -Is) complete Open Images validation pipeline"
$PY scripts/run_openimages_pilot.py --config configs/openimages_complete_validation_heldout_ultrastrict.yaml --cache-only
for SPLIT in classA classB; do
  if [ "$SPLIT" = "classA" ]; then
    CFG=configs/openimages_complete_validation_heldout_ultrastrict.yaml
  else
    CFG=configs/openimages_complete_validation_heldout_ultrastrict_classB.yaml
  fi
  for SEED in "${SEEDS[@]}"; do
    OUT="outputs/openimages_complete_validation_heldout_ultrastrict_${SPLIT}_s${SEED}"
    echo "[RUN] $(date -Is) $SPLIT seed=$SEED"
    $PY scripts/run_openimages_heldout_calibration.py --config "$CFG" --seed-override "$SEED" --output-dir "$OUT"
    $PY scripts/run_openimages_calibrated_baselines.py --config "$CFG" --heldout-dir "$OUT" --output-dir "$OUT"
  done
done
mkdir -p results/supplementary
$PY scripts/summarize_openimages_fullfiltered_sanity.py   --glob "outputs/openimages_complete_validation_heldout_ultrastrict_*/calibrated_baseline_summary.csv"   --output-dir results/supplementary   --dataset-name "Complete Open Images validation check"   --dataset-id "openimages_complete_validation"   --per-config-name "openimages_complete_validation_per_config.csv"   --summary-name "openimages_complete_validation_summary.csv"   --report-name "openimages_complete_validation_report.md"

echo "[DONE] $(date -Is) complete Open Images validation pipeline"
