# Reproduction Guide

## Processed-result check

Run:

```bash
python scripts/print_main_tables.py
```

This prints manuscript-ready summaries from `results/`:

- Table 1: Open Images 10k main methods
- Table 2: COCO val2017 main methods
- Table 3: NUS-WIDE recoverable-subset check
- Table 7: Open Images post-hoc baseline strategies
- Main and supplementary ablation/sensitivity summaries
- NUS-WIDE trained-head, ASL+gate, and frequency-group summaries
- ODIN, MKT, ViT-B/16, TECR-robustness, and gate-parameter-stability summaries
- Known-context logistic controls and TECR risk-set denominator summaries
- Supplementary MKT adapted-checkpoint sanity check
- Supplementary 37,591-image filtered Open Images scope-check summary
- Supplementary 41,620-image complete Open Images validation reviewer-facing summary

For a quick reviewer pass, run `python scripts/print_main_tables.py` and `python scripts/audit_reproducibility.py` from the repository root. These two checks read committed processed results and manifests only; Python 3.10+ is enough. Use `requirements-lock.txt` for cache-dependent reruns and SciPy-backed Wilcoxon diagnostics.

Run the completeness audit before packaging or releasing the artifact:

```bash
python scripts/audit_reproducibility.py
```

The audit does not rerun raw-data experiments. It checks that every paper-facing experiment family has a script, config, and processed-result CSV with the expected method rows and key manuscript/supplementary values. For the 37,591-image filtered supplementary slot and the 41,620-image complete-validation reviewer release, it also checks the report/provenance files and verifies that the held-out gate stays directionally better in the released per-configuration rows. Neither slot is treated as a replacement for the main Open Images 10k benchmark.

## Config path policy

Configs in this artifact use portable `data/...` roots. The protocol fields are synchronized with the reported Open Images, COCO, NUS-WIDE, and ViT-B/16 settings.

## Full experiment outline

1. Prepare public datasets under `data/`.
2. Install `requirements-lock.txt` for cache-dependent reruns, or `requirements.txt` for a looser development install.
3. Run the held-out calibration scripts for both class-split sets and the listed seeds.
4. For Open Images, run `scripts/run_openimages_calibrated_baselines.py` and `scripts/analyze_openimages_heldout_groups.py` against each held-out output directory to generate class-threshold/isotonic rows and frequency-group diagnostics.
5. Run `scripts/run_openimages_posthoc_baselines.py` and `scripts/run_openimages_odin_baseline.py` for the Open Images post-hoc baseline comparison.
6. Run `scripts/run_known_aware_posthoc_baselines.py` for the known-context logistic control and `scripts/summarize_tecr_denominators.py` for the TECR risk-set denominator audit.
7. Run the ablation/sensitivity scripts listed below for gate ablation, calibration size, calibration ratio, ASL+gate, TECR-definition robustness, and parameter stability.
8. Optionally run `scripts/run_openimages_mkt_baseline.py` with a local MKT checkout and public NUS-WIDE checkpoints for the adapted-checkpoint sanity check.
9. Run the summarization scripts to aggregate per-configuration outputs. For NUS-WIDE, use `scripts/summarize_nuswide_full_suite.py` to aggregate the 10 per-configuration full-suite outputs.
10. Compare generated summary CSVs with `results/`.

The `--seed-override` option in the matrix commands is the configuration-level image/run seed. The `protocol.split_seeds` values in each config are the internal retrieval/calibration/evaluation split seeds; classB configs intentionally use a separate `202606xx` split-seed series.

In this guide, "fairness" or "fairness diagnostics" refers to label-frequency slices and documented exception cases used to check whether reported TECR behavior is concentrated in only one part of the label distribution. It does not refer to demographic fairness evaluation.

The protocol uses:

- retrieval/calibration/evaluation split: 40% / 20% / 40%
- kNN: `k=20`, temperature `0.07`
- gate grid: residual powers `[0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4]`, confidence cutoffs `[0.2, 0.3, 0.4, 0.5, 0.6]`, temperatures `[0.01, 0.02, 0.05, 0.1]`
- AP/F1 preservation tolerance: `0.25%`

## Paper-claim to artifact map

| Paper-facing / reviewer-facing result | Rerun entry point | Processed result |
|---|---|---|
| Open Images main gate and calibrated baselines | `run_openimages_heldout_calibration.py`, `run_openimages_calibrated_baselines.py` | `results/openimages_10k/main_method_12config_summary.csv`, `results/openimages_10k/main_per_config_rows.csv` |
| COCO main gate and calibrated baselines | `run_coco_heldout_calibration.py` | `results/coco_val2017/main_method_12config_summary.csv`, `results/coco_val2017/main_per_config_rows.csv` |
| Frozen-feature ASL/DBLoss baselines | `run_openimages_training_baselines.py`, `run_coco_training_baselines.py`, `run_nuswide_full_suite.py` | `results/*/training_baseline_*summary.csv` |
| Gate ablation | `run_openimages_gate_ablation.py` | `results/openimages_10k/gate_ablation_12config_summary.csv` |
| Calibration-size sensitivity | `run_openimages_calibration_size_sensitivity.py`, `summarize_calibration_size_sensitivity.py` | `results/openimages_10k/calibration_size_12config_summary.csv` |
| Calibration-ratio sensitivity | `run_openimages_calibration_ratio_sensitivity.py`, `summarize_calibration_ratio_sensitivity.py` | `results/supplementary/calibration_ratio_summary.csv` |
| ASL+gate add-on | `run_asl_gate_baselines.py`, `summarize_asl_gate_results.py`, `run_nuswide_full_suite.py` | `results/openimages_10k/asl_gate_summary.csv`, `results/coco_val2017/asl_gate_summary.csv`, `results/nuswide/asl_gate_10config_summary.csv` |
| NUS-WIDE recoverable-subset suite | `run_nuswide_full_suite.py`, `summarize_nuswide_full_suite.py` | `results/nuswide/main_summary_10config.csv`, `results/nuswide/*_10config_summary.csv` |
| TECR definition robustness | `run_tecr_robustness.py` | `results/supplementary/tecr_robustness_summary.csv` |
| Selective and generic post-hoc baselines | `run_openimages_posthoc_baselines.py`, `run_openimages_selective_baseline.py`, `run_openimages_odin_baseline.py` | `results/supplementary/posthoc_combined_summary.csv`, `results/supplementary/odin_combined_summary.csv` |
| Known-context logistic controls | `run_known_aware_posthoc_baselines.py` | `results/supplementary/known_aware_combined_summary.csv` |
| TECR risk-set denominator audit | `summarize_tecr_denominators.py` | `results/supplementary/tecr_denominator_combined_summary.csv` |
| Adapted MKT checkpoint | `run_openimages_mkt_baseline.py` | `results/supplementary/mkt_combined_summary.csv` |
| Gate parameter stability | `summarize_gate_parameter_stability.py` | `results/supplementary/gate_parameter_stability_summary.csv` |
| ViT-B/16 check | `run_openimages_pilot.py`, `run_openimages_heldout_calibration.py` | `results/supplementary/openimages_vitb16_heldout_summary.csv` |
| 37,591-image filtered Open Images scope check | `run_openimages_pilot.py`, `run_openimages_heldout_calibration.py`, `run_openimages_calibrated_baselines.py`, `summarize_openimages_fullval_sanity.py` | `results/supplementary/openimages_full_filtered_validation_sanity_summary.csv`, `results/supplementary/openimages_full_filtered_validation_sanity_report.md`, `results/supplementary/openimages_full_filtered_validation_sanity_per_config.csv` |
| 41,620-image complete Open Images validation reviewer-facing evidence | `run_openimages_pilot.py`, `run_openimages_heldout_calibration.py`, `run_openimages_calibrated_baselines.py`, `summarize_openimages_fullfiltered_sanity.py` | `results/supplementary/openimages_complete_validation_summary.csv`, `results/supplementary/openimages_complete_validation_report.md`, `results/supplementary/openimages_complete_validation_per_config.csv` |
| Frequency-group diagnostics and COCO exceptions | `analyze_openimages_heldout_groups.py`, `run_coco_heldout_calibration.py`, `summarize_fairness_diagnostics.py` | `results/supplementary/frequency_group_12config_summary.csv`, `results/supplementary/coco_training_exception_configs.csv` |

## Per-Configuration Combine Entry Points

After regenerating one directory per config/seed, combine the supplementary families with:

```bash
python scripts/run_openimages_posthoc_baselines.py --combine-root outputs/openimages_posthoc_12config
python scripts/run_openimages_odin_baseline.py --combine-root outputs/openimages_odin_12config
python scripts/run_known_aware_posthoc_baselines.py --output-dir outputs/known_aware_combined --combine-root outputs/known_aware_all_datasets
python scripts/run_openimages_mkt_baseline.py --combine-root outputs/openimages_mkt_12config
```

The known-aware combine root should contain the Open Images, COCO, and NUS-WIDE per-configuration `known_aware_eval_rows.csv` files. The MKT combine root should contain per-configuration `mkt_summary.csv` files; checkpoint paths are needed only for the per-configuration MKT runs.

## Post-hoc baseline summary

The manuscript Table 7 evidence is in `results/supplementary/posthoc_combined_summary.csv`. Under the 12-configuration Open Images protocol, entropy selective rejection reports TECR `0.2562` and a `1.1%` reduction, while the held-out reliability gate reports TECR `0.2126` and an `18.0%` reduction.

## MKT adapted-checkpoint summary

The supplementary MKT sanity-check evidence is in `results/supplementary/mkt_combined_summary.csv`. It uses public NUS-WIDE MKT checkpoints as an adapted Open Images scorer by replacing the label list, so it is not a same-protocol MKT reproduction. The 12-configuration result is AP `0.6263`, F1 `0.6626`, and TECR `0.6399`.

To rerun it, provide `--mkt-root`, `--first-stage-ckpt`, and `--second-stage-ckpt` explicitly. The checkpoint files are intentionally not committed.

## Open Images supplementary scope check and complete-validation reviewer evidence

The supplementary 37,591-image filtered Open Images scope-check evidence is in:

- `results/supplementary/openimages_full_filtered_validation_sanity_summary.csv`
- `results/supplementary/openimages_full_filtered_validation_sanity_report.md`
- `results/supplementary/openimages_full_filtered_validation_sanity_per_config.csv`
- `results/supplementary/openimages_full_filtered_validation_provenance/`

This slot uses a larger filtered `37,591`-image Open Images validation subset under the same `120`-label held-out protocol family as the main Open Images `10k` benchmark. It is bundled as reviewer-facing scope evidence rather than as a replacement benchmark.

The bundled reviewer-facing evidence is intentionally split into a small provenance chain: retained image-id list, cache-manifest/status metadata, aggregate summary, per-configuration rows, reviewer report, and a lightweight provenance directory for supporting logs or copied run receipts. This makes the filtered supplementary slot easier to audit fairly without requiring redistribution of dataset-derived caches.

The retained-pool metadata for this supplementary slot is recorded in:

- `data_manifest/openimages_full_filtered_validation_image_ids.json`
- `data_manifest/openimages_full_filtered_validation_cache_manifest.json`
- `data_manifest/openimages_full_filtered_validation_cache_status.json`

To regenerate the slot from public data, first build the filtered cache and then rerun the 12 held-out/calibrated-baseline jobs with the `openimages_fullval_filtered_*` configs:

```bash
python scripts/run_openimages_pilot.py --config configs/openimages_fullval_filtered_heldout_ultrastrict.yaml --cache-only
python scripts/summarize_openimages_fullval_sanity.py
```

The paired held-out/calibrated-baseline per-configuration outputs are summarized into the reviewer-facing fixed filenames above. See `docs/openimages_filtered_validation_sanity_check.md` for naming conventions and interpretation guardrails.

The reviewer-facing complete Open Images validation release is in:

- `results/supplementary/openimages_complete_validation_summary.csv`
- `results/supplementary/openimages_complete_validation_report.md`
- `results/supplementary/openimages_complete_validation_per_config.csv`
- `results/supplementary/openimages_complete_validation_provenance/`
- `data_manifest/openimages_complete_validation_cache_manifest.json`
- `data_manifest/openimages_selected_label_ids_top120.json`

This release fixes the same `120` labels, expands the image pool to the complete `41,620`-image Open Images validation set, and keeps the same held-out 40/20/40 protocol family. It remains reviewer-facing supplementary evidence rather than a replacement for the repeated Open Images `10k` benchmark.

To regenerate the complete-validation release from public data, first build the full validation cache and then rerun the 12 held-out/calibrated-baseline jobs with the `openimages_complete_validation_*` configs:

```bash
python scripts/run_openimages_pilot.py --config configs/openimages_complete_validation_heldout_ultrastrict.yaml --cache-only
python scripts/summarize_openimages_fullfiltered_sanity.py --glob "outputs/openimages_complete_validation_heldout_ultrastrict_*/calibrated_baseline_summary.csv" --output-dir results/supplementary --dataset-name "Complete Open Images validation check" --dataset-id "openimages_complete_validation" --per-config-name "openimages_complete_validation_per_config.csv" --summary-name "openimages_complete_validation_summary.csv" --report-name "openimages_complete_validation_report.md"
```

The 37,591-image filtered slot remains a supplementary scope check, and the 41,620-image complete-validation bundle is released as reviewer-facing evidence. Both remain supplementary to the main Open Images `10k` benchmark.

## Known-context controls and TECR denominator audit

The known-context control evidence is in `results/supplementary/known_aware_combined_summary.csv`. It compares score-only logistic, permuted-known logistic, real known-aware logistic, and the held-out gate under the same calibration/evaluation split discipline. The real known-aware feature reduces TECR on Open Images (`4.2%`), COCO (`12.0%`), and NUS-WIDE (`8.1%`), while the permuted-known feature is weak or adverse.

The denominator audit is in `results/supplementary/tecr_denominator_combined_summary.csv`. It reports `|C|`, the number of tail-known, emerging-negative evaluation images used as the TECR denominator, with means and min/max ranges over split configurations. This audit checks the TECR risk-set denominator only; it does not use scorer predictions, thresholds, or gate selection.

## Descriptive Wilcoxon check

The Wilcoxon script is `scripts/wilcoxon_descriptive_pvalues.py`. It is a descriptive processed-result check over configuration-level paired units, not a split-level significance test. For the released configuration-level rows, run:

```bash
python scripts/wilcoxon_descriptive_pvalues.py --input results/openimages_10k/main_per_config_rows.csv --unit-cols split_set,seed --metric tecr --baseline-method clip_knn_global_threshold --method heldout_gate_global_threshold --delta baseline_minus_method --alternative greater --zero-method wilcox
python scripts/wilcoxon_descriptive_pvalues.py --input results/coco_val2017/main_per_config_rows.csv --unit-cols split_set,seed --metric tecr --baseline-method clip_knn_global_threshold --method heldout_gate_global_threshold --delta baseline_minus_method --alternative greater --zero-method wilcox
python scripts/wilcoxon_descriptive_pvalues.py --input results/nuswide/main_per_config_rows.csv --unit-cols split_set,seed --metric tecr --baseline-method clip_knn_global_threshold --method heldout_gate_global_threshold --delta baseline_minus_method --alternative greater --zero-method wilcox
```

The Open Images and COCO commands use the released `main_per_config_rows.csv` files, so they do not require raw images, feature caches, or regenerated output directories. If reviewers regenerate the 12 per-configuration `calibrated_baseline_summary.csv` files, they can also use `--input-glob` with `--unit-cols source` as shown in the README. The script prints the installed SciPy version and the exact `scipy.stats.wilcoxon(...)` call; SciPy is required for this script and is pinned in `requirements-lock.txt`.

## Important boundary

The artifact is designed for reproducibility of the post-hoc frozen-feature protocol. It does not redistribute raw datasets, large CLIP feature caches, MKT checkpoints, or official fully trained ASL/DBLoss model checkpoints.
