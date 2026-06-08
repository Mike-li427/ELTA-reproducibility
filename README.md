# Reproducibility Artifact for Visual Knowledge-Base Maintenance

This journal-facing reproducibility artifact supports the paper **Reducing False Routes in Visual Knowledge-Base Maintenance: A Decision-Layer Reliability Gate**.

The code evaluates a post-hoc held-out confidence gate for reducing Tail-Emerging Confusion Rate (TECR) in open-vocabulary multi-label recognition. The repository contains the core TECR/gate implementation, experiment scripts, protocol configs, NUS-WIDE exact image/class manifests, and processed summary CSVs used to check the main paper tables.

For Open Images, the paper-facing benchmark remains the repeated `10k` validation-subset protocol reported in the manuscript. The repository also bundles two supplementary reviewer-facing checks under the same held-out protocol family: a larger filtered `37,591`-image validation-subset scope check and a complete-validation `41,620`-image release built over the frozen `120`-label slice. Neither check replaces the manuscript's main benchmark tables. The filtered slot remains directional scope evidence, while the complete-validation slot adds a stronger release-time direction check with run receipts and configuration-level outputs. See `docs/openimages_filtered_validation_sanity_check.md`.

## Quick Review

- Purpose: a compact reproducibility artifact for checking the reported frozen-feature, post-hoc reliability-gating results.
- Fastest checks: run `python scripts/print_main_tables.py` and `python scripts/audit_reproducibility.py` from the repository root; both read only `results/` and the committed manifests.
- Minimal table-audit environment: Python 3.10+ is sufficient for the two fastest checks. Install SciPy only for the descriptive Wilcoxon scripts. Install `requirements-lock.txt` only for cache-dependent or full rerun workflows.
- Config-level TECR checks: the released rows under `results/*/main_per_config_rows.csv` support the descriptive Wilcoxon commands below without raw images or CLIP caches.
- Main processed-result finding: the held-out gate reduces TECR by `18.0%` on Open Images 10k, `27.4%` on COCO val2017, and `25.3%` on the NUS-WIDE recoverable-subset stress check.
- Supplementary scope-check finding: on the larger filtered `37,591`-image Open Images validation subset, the held-out gate reduces TECR from `0.2399` to `0.2111` (`12.0%`) while AP/F1 remain near the CLIP+kNN baseline.
- Supplementary direction check: in the released larger filtered Open Images per-configuration rows, the held-out gate lowers TECR versus `clip_knn_global_threshold` in all `12/12` class-split-by-seed pairs.
- Supplementary complete-validation finding: on the complete `41,620`-image Open Images validation release, the held-out gate reduces TECR from `0.2241` to `0.1914` (`14.6%`) while AP is unchanged to three decimals and F1 changes by `-0.1%` relative to the CLIP+kNN global-threshold baseline.
- Supplementary complete-validation direction check: in the released complete-validation per-configuration rows, the held-out gate lowers TECR versus `clip_knn_global_threshold` in all `12/12` class-split-by-seed pairs.
- Supplementary provenance: the retained image-id list, cache-definition/status manifests, per-configuration rows, reviewer report, and reserved `results/supplementary/openimages_full_filtered_validation_provenance/` directory together document the origin of the larger-slot evidence without redistributing dataset-derived caches.
- Supplementary complete-validation provenance: the cache manifest, selected-label manifest, per-configuration rows, reviewer report, and `results/supplementary/openimages_complete_validation_provenance/` run receipts document the full-validation release without redistributing downloaded images or feature arrays.
- Not included: raw datasets, downloaded images, large CLIP feature caches, model checkpoints, and MKT checkpoints.
- Release boundary: this is a journal-facing author artifact; create a separate anonymized copy if a venue requires double-anonymous review.

## Open Images Naming Map

| Paper-facing scope | Config prefix | Main processed outputs |
|---|---|---|
| Open Images 10k controlled benchmark | `openimages_10k_...` | `results/openimages_10k/*` |
| Filtered 37,591-image scale check | `openimages_fullval_filtered_...` | `results/supplementary/openimages_full_filtered_validation_sanity_*` |
| Complete 41,620-image validation check | `openimages_complete_validation_...` | `results/supplementary/openimages_complete_validation_*` |

The older `openimages_fullfiltered_validation_...` names are retained only as compatibility aliases for scripts that predate the final release naming.

## Repository Layout

```text
configs/        Protocol configs, seeds, class-split settings, and hyperparameters.
src/elta/       Minimal core gate, TECR, threshold, and selection utilities.
scripts/        Full experiment and summary scripts.
data_manifest/  Exact NUS-WIDE manifests plus Open Images filtered-subset pool-definition files.
results/        Processed summary CSVs for main and supplementary tables.
docs/           Reproduction, data-availability, and supplementary scope-check notes.
REPRODUCIBILITY_PACKAGE_MANIFEST.md
                Release manifest for included/excluded files and audit coverage.
```

See `REPRODUCIBILITY_PACKAGE_MANIFEST.md` for the package boundary, directory-by-directory file-purpose map, excluded raw/cache/checkpoint artifacts, processed-result audit coverage, and the 12/10 configuration matrices.

## Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

For a looser development environment, use:

```bash
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

`requirements-lock.txt` pins the OpenAI CLIP dependency to a fixed commit. `requirements.txt` intentionally leaves it as a floating repository dependency for local development.

The original experiments used frozen CLIP ViT-B/32 unless otherwise noted. GPU is recommended for feature extraction, but the post-hoc gate itself is lightweight.

The optional MKT sanity-check script additionally expects a local checkout of the official MKT repository, its PyTorch/vision dependencies, and the public NUS-WIDE checkpoints. The checkpoints are not redistributed in this artifact.


## Config Path Policy

This reproducibility artifact uses portable relative data roots under `data/...`. Apart from that data-root normalization, the Open Images, COCO, NUS-WIDE, and ViT-B/16 configs are intended to match the reported protocol.

## Data

Raw Open Images, COCO, and NUS-WIDE images are not redistributed. Download or place them under:

```text
data/openimages_v6/
data/coco2017/
data/nuswide/
```

The NUS-WIDE exact usable subset is recorded in:

```text
data_manifest/nuswide_image_names.json
data_manifest/nuswide_classes.json
```

The larger filtered Open Images validation-subset check records its retained image pool and cache-definition metadata in:

```text
data_manifest/openimages_full_filtered_validation_image_ids.json
data_manifest/openimages_full_filtered_validation_cache_manifest.json
data_manifest/openimages_full_filtered_validation_cache_status.json
```

Together with `results/supplementary/openimages_full_filtered_validation_sanity_per_config.csv`, `results/supplementary/openimages_full_filtered_validation_sanity_report.md`, and the reserved `results/supplementary/openimages_full_filtered_validation_provenance/` directory, these files provide the reviewer-facing provenance chain for the larger filtered Open Images supplementary slot: retained image pool, cache-definition metadata, released per-configuration aggregates, and a stable place to attach future rerun receipts if a full validation slot is later requested.

The complete Open Images validation release records its frozen full-validation cache definition in:

```text
data_manifest/openimages_complete_validation_cache_manifest.json
data_manifest/openimages_selected_label_ids_top120.json
```

Together with `results/supplementary/openimages_complete_validation_summary.csv`, `results/supplementary/openimages_complete_validation_per_config.csv`, `results/supplementary/openimages_complete_validation_report.md`, and `results/supplementary/openimages_complete_validation_provenance/`, these files provide the reviewer-facing provenance chain for the complete-validation supplementary slot: the `41,620`-image validation scope, the fixed `120`-label slice, released per-configuration aggregates, and the run receipts used to produce the local release.

CLIP feature caches are intentionally not committed because they may be large and dataset-derived redistribution can be license-dependent. They can be regenerated by the scripts from the public datasets and recorded manifests.

## Cache and Audit Preconditions

There are three distinct reproduction modes:

- **Processed-result audit, no raw data/cache required:** `python scripts/print_main_tables.py` and `python scripts/audit_reproducibility.py` read only files already under `results/`. The configuration-level Wilcoxon commands below also read only `results/`, but they require the SciPy dependency declared in `requirements.txt` and pinned in `requirements-lock.txt`.
- **Scripts that require existing CLIP feature caches:** the Open Images held-out/gate/ablation/post-hoc/training scripts use `load_cached_arrays(...)` and expect cache files under the config's `output_dir` or `feature_cache_dir`. Generate the Open Images cache first with:

```bash
python scripts/run_openimages_pilot.py --config configs/openimages_10k_heldout_ultrastrict.yaml --cache-only
```

- **Scripts that can regenerate caches from public data:** COCO held-out runs call `load_or_compute_features(...)`; NUS-WIDE full-suite runs call `load_or_compute_nuswide_features(...)`. These can populate caches when public data/manifests are present:

```bash
python scripts/run_coco_pilot.py --config configs/coco_heldout_ultrastrict.yaml --cache-only
python scripts/run_nuswide_full_suite.py --config configs/nuswide_heldout_ultrastrict.yaml --seed-override 20260522 --output-dir outputs/nuswide_heldout_ultrastrict_s20260522
```

Open Images and COCO cache locations are read from the selected config file and are shared across classA/classB configs because the image pool, class list, and CLIP model are the same; classA/classB changes the config-internal `protocol.split_seeds`. NUS-WIDE caches are keyed by `image_pool_seed`, max images, class count, and CLIP model. In the matrix commands below, `--seed-override` is the configuration-level image/run seed; `protocol.split_seeds` are the split seeds used inside each configuration. The classB configs intentionally use the `202606xx` internal split-seed series.

In this artifact, reviewer-facing "fairness diagnostics" refers to label-frequency slices and documented exception configurations, not demographic fairness claims.

## Quick Check: Print Main Tables

To print the processed summary tables used by the manuscript and supplementary material:

```bash
python scripts/print_main_tables.py
```

This reads `results/` only and does not require raw images. It prints the three main dataset tables plus the key ablation, sensitivity, post-hoc, known-context control, TECR denominator, NUS-WIDE, MKT, ODIN, robustness, parameter-stability, larger filtered Open Images supplementary summaries, and the complete Open Images validation supplementary summary.

To check that every paper-facing experiment claim has a runnable script, protocol config, processed-result file, and key manuscript/supplementary numeric value:

```bash
python scripts/audit_reproducibility.py
```

This is a processed-result completeness audit. It verifies that the released result CSVs, protocol configs, scripts, and reported numeric values are mutually present and aligned; it is not a full raw-data rerun audit from images or regenerated CLIP features.

## Full Reproduction Entry Points

The main full-run scripts are:

```bash
python scripts/run_openimages_heldout_calibration.py --config configs/openimages_10k_heldout_ultrastrict.yaml --output-dir outputs/openimages_10k_heldout
python scripts/run_openimages_calibrated_baselines.py --config configs/openimages_10k_heldout_ultrastrict.yaml --heldout-dir outputs/openimages_10k_heldout --output-dir outputs/openimages_10k_heldout
python scripts/analyze_openimages_heldout_groups.py --config configs/openimages_10k_heldout_ultrastrict.yaml --heldout-dir outputs/openimages_10k_heldout --output-dir outputs/openimages_10k_heldout
python scripts/run_coco_heldout_calibration.py --config configs/coco_heldout_ultrastrict.yaml --output-dir outputs/coco_val2017_heldout
python scripts/run_nuswide_full_suite.py --config configs/nuswide_heldout_ultrastrict.yaml --output-dir outputs/nuswide_heldout
python scripts/run_openimages_posthoc_baselines.py --config configs/openimages_10k_heldout_ultrastrict.yaml --output-dir outputs/openimages_posthoc_baselines
python scripts/run_known_aware_posthoc_baselines.py --config configs/openimages_10k_heldout_ultrastrict.yaml --output-dir outputs/openimages_known_aware
python scripts/summarize_tecr_denominators.py --output-dir outputs/tecr_denominators
python scripts/run_openimages_mkt_baseline.py --config configs/openimages_10k_vitb16_heldout.yaml --output-dir outputs/openimages_mkt_baseline --mkt-root /path/to/MKT --first-stage-ckpt /path/to/mkt_nus_first_stage.pth --second-stage-ckpt /path/to/mkt_nus_second_stage.pth
```

For the MKT command, replace the placeholder paths with a local checkout of the official MKT repository and locally obtained public checkpoint files.

The paper reports averages over two class-split sets and multiple image seeds. The 12 Open Images and COCO configurations are classA/classB by six image/run seeds; the 10 NUS-WIDE configurations are classA/classB by five image/run seeds.

### Open Images 12-Configuration Matrix

Open Images held-out runs require the cache command above first. The classA config is `configs/openimages_10k_heldout_ultrastrict.yaml`; the classB config is `configs/openimages_10k_heldout_ultrastrict_classB.yaml`. The six image seeds are `20260522 20260523 20260524 20260525 20260526 20260527`.

PowerShell template:

```powershell
$runs = @(
  @{Tag="classA"; Config="configs/openimages_10k_heldout_ultrastrict.yaml"},
  @{Tag="classB"; Config="configs/openimages_10k_heldout_ultrastrict_classB.yaml"}
)
$seeds = @(20260522, 20260523, 20260524, 20260525, 20260526, 20260527)
foreach ($run in $runs) {
  foreach ($seed in $seeds) {
    $out = "outputs/openimages_10k_heldout_ultrastrict_$($run.Tag)_s$seed"
    python scripts/run_openimages_heldout_calibration.py --config $run.Config --seed-override $seed --output-dir $out
    python scripts/run_openimages_calibrated_baselines.py --config $run.Config --heldout-dir $out --output-dir $out
    python scripts/analyze_openimages_heldout_groups.py --config $run.Config --heldout-dir $out --output-dir $out
  }
}
python scripts/summarize_training_and_main_results.py --dataset-name "Open Images 10k" --training-glob "outputs/openimages_10k_training_baselines*/training_baseline_summary.csv" --main-glob "outputs/openimages_10k_heldout_ultrastrict_*/calibrated_baseline_summary.csv" --output-dir outputs/openimages_10k_12config_summary
```

If training baselines are being regenerated too, run `scripts/run_openimages_training_baselines.py` for the same 12 `--config`/`--seed-override` pairs into directories matching the `--training-glob`.

### Optional Larger Filtered Open Images Validation-Subset Check

This supplementary reviewer-facing scope check keeps the same `120` labels and held-out `40/20/40` protocol family but expands the Open Images pool to a larger filtered `37,591`-image validation subset. It is not a replacement for the repeated Open Images `10k` benchmark used in the manuscript main tables.

The config aliases for this supplementary check are:

- `configs/openimages_fullval_filtered_heldout_ultrastrict.yaml`
- `configs/openimages_fullval_filtered_heldout_ultrastrict_classB.yaml`
- `configs/openimages_fullfiltered_validation_heldout_ultrastrict.yaml`
- `configs/openimages_fullfiltered_validation_heldout_ultrastrict_classB.yaml`

Minimal rerun outline:

```bash
python scripts/run_openimages_pilot.py --config configs/openimages_fullval_filtered_heldout_ultrastrict.yaml --cache-only
```

Then rerun the same 12 held-out/calibrated-baseline jobs as the main Open Images matrix, but with the `openimages_fullval_filtered_...` configs and output prefixes. After the per-configuration jobs complete, aggregate them with:

```bash
python scripts/summarize_openimages_fullval_sanity.py
```

The released processed summaries for this supplementary slot are:

- `results/supplementary/openimages_full_filtered_validation_sanity_summary.csv`
- `results/supplementary/openimages_full_filtered_validation_sanity_report.md`
- `results/supplementary/openimages_full_filtered_validation_sanity_per_config.csv`
- `results/supplementary/openimages_full_filtered_validation_provenance/`

The reviewer-facing interpretation and naming conventions for this slot are documented in `docs/openimages_filtered_validation_sanity_check.md`.

If a future journal workflow explicitly asks for this supplementary slot to become a full validation slot, keep the same release-facing filenames, rerun the full `12`-configuration matrix against the recorded retained pool, populate the reserved provenance directory with run receipts or copied config snapshots, and only then freeze paper-facing numeric targets for that promoted slot. Here, "full validation slot" means that future rerun-plus-receipts release, not the current bundled supplementary evidence.

### Supplementary Complete Open Images Validation Release

This supplementary reviewer-facing release keeps the same `120` labels and held-out `40/20/40` protocol family, but uses the complete `41,620`-image Open Images validation pool defined by `validation-images-with-rotation.csv` while freezing the selected `120` labels through `data_manifest/openimages_selected_label_ids_top120.json`. It remains supplementary reviewer evidence rather than a replacement for the manuscript's repeated Open Images `10k` benchmark.

The configs for this release are:

- `configs/openimages_complete_validation_heldout_ultrastrict.yaml`
- `configs/openimages_complete_validation_heldout_ultrastrict_classB.yaml`

The released processed summaries for this supplementary slot are:

- `results/supplementary/openimages_complete_validation_summary.csv`
- `results/supplementary/openimages_complete_validation_report.md`
- `results/supplementary/openimages_complete_validation_per_config.csv`
- `results/supplementary/openimages_complete_validation_provenance/`
- `data_manifest/openimages_complete_validation_cache_manifest.json`
- `data_manifest/openimages_selected_label_ids_top120.json`

The main supplementary comparison in the released summary is:

- `CLIP+kNN global threshold`: AP `0.8589`, F1 `0.7774`, TECR `0.2241`
- `Held-out gate global threshold`: AP `0.8588`, F1 `0.7763`, TECR `0.1914`
- TECR reduction: `14.6%`

The released per-config rows also keep the same direction at the configuration level: `12/12` class-split-by-seed pairs show lower TECR for `heldout_gate_global_threshold` than for `clip_knn_global_threshold`, with mean paired TECR delta `0.0327`.

### COCO 12-Configuration Matrix

The classA config is `configs/coco_heldout_ultrastrict.yaml`; the classB config is `configs/coco_heldout_ultrastrict_classB.yaml`. The six image seeds are `20260522 20260523 20260524 20260525 20260526 20260527`.

```powershell
$runs = @(
  @{Tag="classA"; Config="configs/coco_heldout_ultrastrict.yaml"},
  @{Tag="classB"; Config="configs/coco_heldout_ultrastrict_classB.yaml"}
)
$seeds = @(20260522, 20260523, 20260524, 20260525, 20260526, 20260527)
foreach ($run in $runs) {
  foreach ($seed in $seeds) {
    $out = "outputs/coco_heldout_ultrastrict_$($run.Tag)_s$seed"
    python scripts/run_coco_heldout_calibration.py --config $run.Config --seed-override $seed --output-dir $out
  }
}
python scripts/summarize_training_and_main_results.py --dataset-name "COCO" --training-glob "outputs/coco_training_baselines*/training_baseline_summary.csv" --main-glob "outputs/coco_heldout_ultrastrict_*/calibrated_baseline_summary.csv" --output-dir outputs/coco_12config_summary
```

If training baselines are being regenerated too, run `scripts/run_coco_training_baselines.py` for the same 12 `--config`/`--seed-override` pairs into directories matching the `--training-glob`.

### NUS-WIDE 10-Configuration Matrix

The classA config is `configs/nuswide_heldout_ultrastrict.yaml`; the classB config is `configs/nuswide_heldout_ultrastrict_classB.yaml`. The five image seeds are `20260522 20260523 20260524 20260525 20260526`.

```powershell
$runs = @(
  @{Tag="classA"; Config="configs/nuswide_heldout_ultrastrict.yaml"},
  @{Tag="classB"; Config="configs/nuswide_heldout_ultrastrict_classB.yaml"}
)
$seeds = @(20260522, 20260523, 20260524, 20260525, 20260526)
foreach ($run in $runs) {
  foreach ($seed in $seeds) {
    $out = "outputs/nuswide_heldout_ultrastrict_$($run.Tag)_s$seed"
    python scripts/run_nuswide_full_suite.py --config $run.Config --seed-override $seed --output-dir $out
  }
}
python scripts/summarize_nuswide_full_suite.py --glob "outputs/nuswide_heldout_ultrastrict_*/" --output-dir outputs/nuswide_10config_summary
```

The released config-level rows corresponding to these matrices are `results/openimages_10k/main_per_config_rows.csv`, `results/coco_val2017/main_per_config_rows.csv`, and `results/nuswide/main_per_config_rows.csv`.

Row-level CSVs and JSON metadata under `results/supplementary/` are included for traceability of the known-context controls and TECR denominator audit.

## Supplementary Ablation Entry Points

These scripts reproduce the supplementary ablations and diagnostics from public datasets after regenerating any needed features:

```bash
python scripts/run_openimages_gate_ablation.py --config configs/openimages_10k_heldout_ultrastrict.yaml --output-dir outputs/openimages_gate_ablation
python scripts/run_openimages_calibration_size_sensitivity.py --config configs/openimages_10k_heldout_ultrastrict.yaml --output-dir outputs/openimages_calibration_size
python scripts/run_openimages_calibration_ratio_sensitivity.py --config configs/openimages_10k_heldout_ultrastrict.yaml --output-dir outputs/openimages_calibration_ratio
python scripts/run_asl_gate_baselines.py --dataset openimages --config configs/openimages_10k_heldout_ultrastrict.yaml --output-dir outputs/asl_gate_openimages
python scripts/run_tecr_robustness.py --config configs/openimages_10k_heldout_ultrastrict.yaml --output-dir outputs/tecr_robustness_openimages
python scripts/run_openimages_odin_baseline.py --config configs/openimages_10k_heldout_ultrastrict.yaml --output-dir outputs/openimages_odin
python scripts/summarize_gate_parameter_stability.py --audit-root . --output-dir outputs/gate_parameter_stability
```

After running per-configuration jobs, use the corresponding `summarize_*` scripts to aggregate to 12-configuration or 10-configuration CSVs. For gate-parameter stability, `--audit-root` should point to the parent directory containing `outputs/openimages_10k_heldout_ultrastrict*/` and `outputs/coco_heldout_ultrastrict*/`. The processed CSVs shipped under `results/` are the values used for manuscript and supplementary table checks.

For the NUS-WIDE 10-configuration suite, aggregate the per-configuration `run_nuswide_full_suite.py` outputs with:

```bash
python scripts/summarize_nuswide_full_suite.py --glob "outputs/nuswide_heldout_ultrastrict*/" --output-dir outputs/nuswide_10config_summary
```

For per-configuration supplementary baselines that write one output directory per config/seed, combine the generated directories with the script-level `--combine-root` entry points:

```bash
python scripts/run_openimages_posthoc_baselines.py --combine-root outputs/openimages_posthoc_12config
python scripts/run_openimages_odin_baseline.py --combine-root outputs/openimages_odin_12config
python scripts/run_known_aware_posthoc_baselines.py --output-dir outputs/known_aware_combined --combine-root outputs/known_aware_all_datasets
python scripts/run_openimages_mkt_baseline.py --combine-root outputs/openimages_mkt_12config
```

For the known-context controls, `--combine-root` should point to a parent directory containing the Open Images, COCO, and NUS-WIDE per-configuration `known_aware_eval_rows.csv` files. For MKT, the combine step reads per-configuration `mkt_summary.csv` files and does not require checkpoint files again.

## Descriptive Wilcoxon P-Value Check

Wilcoxon signed-rank p values are descriptive reproduction checks, not split-level significance claims. The paired unit is the **configuration-level** result: class split set by image seed. Do not treat the five protocol splits inside a configuration as independent observations.

The released Open Images, COCO, and NUS-WIDE config-level rows test whether TECR is lower for the held-out gate than CLIP+kNN. The script forms deltas as `clip_knn_global_threshold - heldout_gate_global_threshold`, so the one-sided alternative is `greater`. Zero deltas are handled with SciPy's `zero_method='wilcox'`, which drops zero differences; use `--zero-method pratt` or `--zero-method zsplit` only as a sensitivity check.

```bash
python scripts/wilcoxon_descriptive_pvalues.py --input results/openimages_10k/main_per_config_rows.csv --unit-cols split_set,seed --metric tecr --baseline-method clip_knn_global_threshold --method heldout_gate_global_threshold --delta baseline_minus_method --alternative greater --zero-method wilcox
python scripts/wilcoxon_descriptive_pvalues.py --input results/coco_val2017/main_per_config_rows.csv --unit-cols split_set,seed --metric tecr --baseline-method clip_knn_global_threshold --method heldout_gate_global_threshold --delta baseline_minus_method --alternative greater --zero-method wilcox
python scripts/wilcoxon_descriptive_pvalues.py --input results/nuswide/main_per_config_rows.csv --unit-cols split_set,seed --metric tecr --baseline-method clip_knn_global_threshold --method heldout_gate_global_threshold --delta baseline_minus_method --alternative greater --zero-method wilcox
```

The command reports the installed SciPy version and the exact `scipy.stats.wilcoxon(...)` call. SciPy is required for this script and is pinned in `requirements-lock.txt`.

If reviewers regenerate the 12 Open Images or COCO configuration directories, they can also run the same script over the per-run summary glob, using the parent directory name as the configuration-level unit:

```bash
python scripts/wilcoxon_descriptive_pvalues.py --input-glob "outputs/openimages_10k_heldout_ultrastrict_*/calibrated_baseline_summary.csv" --unit-cols source --metric tecr --baseline-method clip_knn_global_threshold --method heldout_gate_global_threshold --delta baseline_minus_method --alternative greater
python scripts/wilcoxon_descriptive_pvalues.py --input-glob "outputs/coco_heldout_ultrastrict_*/calibrated_baseline_summary.csv" --unit-cols source --metric tecr --baseline-method clip_knn_global_threshold --method heldout_gate_global_threshold --delta baseline_minus_method --alternative greater
```

Do not run Wilcoxon on `heldout_eval_rows.csv` or other split-level rows unless `--unit-cols` groups them back to configuration-level units such as `config_name,seed` first.

## Main Reported TECR Results

| Dataset | CLIP+kNN TECR | Gate TECR | Relative reduction |
|---|---:|---:|---:|
| Open Images 10k | 0.2592 | 0.2126 | 18.0% |
| COCO val2017 | 0.2545 | 0.1848 | 27.4% |
| NUS-WIDE recoverable-subset check | 0.2571 | 0.1920 | 25.3% |

## Post-Hoc Baseline Check

The Open Images post-hoc baseline comparison is stored in:

```text
results/supplementary/posthoc_combined_summary.csv
results/supplementary/posthoc_combined_report.md
```

The entropy selective rejection row follows the manuscript Table 7 value: TECR `0.2562`, a `1.1%` reduction from CLIP+kNN under the same 12-configuration protocol.

## Known-Context Logistic Controls

The 2026-06-02 reviewer add-on evidence is stored in:

```text
results/supplementary/known_aware_combined_summary.csv
results/supplementary/known_aware_combined_report.md
```

It trains score-only, permuted-known, and real known-aware logistic calibrators on calibration image--emerging-label pairs, then evaluates max-pooled image-level emerging decisions. The known-aware control lowers TECR on all three datasets: Open Images `0.2483` (`4.2%`), COCO `0.2240` (`12.0%`), and NUS-WIDE `0.2362` (`8.1%`). The permuted-known negative control is weak or adverse, supporting the claim that real known-label evidence is informative rather than a generic score-calibration artifact.

## TECR Risk-Set Denominators

The 2026-06-02 denominator audit is stored in:

```text
results/supplementary/tecr_denominator_combined_summary.csv
results/supplementary/tecr_denominator_combined_report.md
```

It reports the size of the conditioned risk population `C` across protocol configurations: Open Images `252.9 +/- 33.7` evaluation images, COCO `142.8 +/- 33.2`, and NUS-WIDE `103.6 +/- 29.2`. These values are reported so TECR is not interpreted without the underlying risk-set size.

## MKT Sanity Check

The adapted public MKT checkpoint check is stored in:

```text
results/supplementary/mkt_combined_summary.csv
results/supplementary/mkt_combined_report.md
```

This is a cross-protocol sanity check using public NUS-WIDE MKT checkpoints with the Open Images label list substituted. It reports AP `0.6263`, F1 `0.6626`, and TECR `0.6399` over 12 Open Images split configurations. It is not a same-protocol reproduction of MKT's benchmark performance.

## Scope Notes

- ASL and DBLoss are adapted as same-protocol frozen-feature trained-head baselines, not official full image-backbone reproductions.
- MKT checkpoints are not redistributed; use the public checkpoint links from the official MKT repository or locally provided checkpoint files.
- The gate is a post-hoc reliability layer and can be applied on top of different scorers.
- NUS-WIDE raw URL availability may drift over time; use the included exact image-name list for comparison with the reported subset.

## Review and Release Metadata

Before journal upload, align repository URLs, DOI fields, and release information with the metadata requirements of the target journal. This package is prepared for journal-facing release metadata; author names are present in `CITATION.cff`, but personal email addresses are intentionally omitted from the public artifact. If an anonymized reproducibility artifact is requested by another venue, create a separate anonymized copy with `.git` metadata/remotes and author-identifying fields removed.
