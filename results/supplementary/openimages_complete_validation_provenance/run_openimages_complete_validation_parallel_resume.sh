#!/usr/bin/env bash
set -euo pipefail
WORK_DIR="${WORK_DIR:-/path/to/ELTA_repro_fullval_20260607}"
PY="${PYTHON_BIN:-python3}"
cd "$WORK_DIR"
MAX_JOBS=4
LOG=parallel_resume.log
exec > >(tee -a "$LOG") 2>&1

echo "[RESUME-START] $(date -Is) complete Open Images validation parallel resume"

tasks=(
  "outputs/openimages_complete_validation_heldout_ultrastrict_classA_s20260522|configs/openimages_complete_validation_heldout_ultrastrict.yaml|20260522"
  "outputs/openimages_complete_validation_heldout_ultrastrict_classA_s20260523|configs/openimages_complete_validation_heldout_ultrastrict.yaml|20260523"
  "outputs/openimages_complete_validation_heldout_ultrastrict_classA_s20260524|configs/openimages_complete_validation_heldout_ultrastrict.yaml|20260524"
  "outputs/openimages_complete_validation_heldout_ultrastrict_classA_s20260525|configs/openimages_complete_validation_heldout_ultrastrict.yaml|20260525"
  "outputs/openimages_complete_validation_heldout_ultrastrict_classA_s20260526|configs/openimages_complete_validation_heldout_ultrastrict.yaml|20260526"
  "outputs/openimages_complete_validation_heldout_ultrastrict_classA_s20260527|configs/openimages_complete_validation_heldout_ultrastrict.yaml|20260527"
  "outputs/openimages_complete_validation_heldout_ultrastrict_classB_s20260522|configs/openimages_complete_validation_heldout_ultrastrict_classB.yaml|20260522"
  "outputs/openimages_complete_validation_heldout_ultrastrict_classB_s20260523|configs/openimages_complete_validation_heldout_ultrastrict_classB.yaml|20260523"
  "outputs/openimages_complete_validation_heldout_ultrastrict_classB_s20260524|configs/openimages_complete_validation_heldout_ultrastrict_classB.yaml|20260524"
  "outputs/openimages_complete_validation_heldout_ultrastrict_classB_s20260525|configs/openimages_complete_validation_heldout_ultrastrict_classB.yaml|20260525"
  "outputs/openimages_complete_validation_heldout_ultrastrict_classB_s20260526|configs/openimages_complete_validation_heldout_ultrastrict_classB.yaml|20260526"
  "outputs/openimages_complete_validation_heldout_ultrastrict_classB_s20260527|configs/openimages_complete_validation_heldout_ultrastrict_classB.yaml|20260527"
)

run_one() {
  local outdir="$1"
  local cfg="$2"
  local seed="$3"
  mkdir -p "$outdir"
  if [[ -f "$outdir/calibrated_baseline_summary.csv" ]]; then
    echo "[SKIP] $(date -Is) $outdir already complete"
    return 0
  fi
  echo "[START] $(date -Is) $outdir seed=$seed cfg=$cfg"
  "$PY" scripts/run_openimages_heldout_calibration.py --config "$cfg" --seed-override "$seed" --output-dir "$outdir" > "$outdir/heldout.stdout.log" 2>&1
  "$PY" scripts/run_openimages_calibrated_baselines.py --config "$cfg" --heldout-dir "$outdir" --output-dir "$outdir" > "$outdir/calibrated.stdout.log" 2>&1
  echo "[DONE] $(date -Is) $outdir"
}

running_jobs() {
  jobs -pr | wc -l
}

for task in "${tasks[@]}"; do
  IFS='|' read -r outdir cfg seed <<< "$task"
  while [[ $(running_jobs) -ge $MAX_JOBS ]]; do
    sleep 5
  done
  run_one "$outdir" "$cfg" "$seed" &
done
wait

mkdir -p results/supplementary
"$PY" scripts/summarize_openimages_fullfiltered_sanity.py \
  --glob "outputs/openimages_complete_validation_heldout_ultrastrict_*/calibrated_baseline_summary.csv" \
  --output-dir results/supplementary \
  --dataset-name "Complete Open Images validation check" \
  --dataset-id "openimages_complete_validation" \
  --per-config-name "openimages_complete_validation_per_config.csv" \
  --summary-name "openimages_complete_validation_summary.csv" \
  --report-name "openimages_complete_validation_report.md"

echo "[RESUME-DONE] $(date -Is) complete Open Images validation parallel resume"
