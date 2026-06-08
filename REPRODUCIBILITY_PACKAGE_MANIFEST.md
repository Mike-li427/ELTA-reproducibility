# Reproducibility Package Manifest

This journal-facing reproducibility package supports the paper **Reducing False Routes in Visual Knowledge-Base Maintenance: A Decision-Layer Reliability Gate**.

The package is designed for two review tasks:

- inspect and rerun the post-hoc frozen-feature reliability protocol when public datasets and caches are available;
- verify the released processed results against the manuscript-facing claims without raw datasets or feature caches.

It is not a raw-dataset archive and does not redistribute model checkpoints or dataset-derived feature files.

For a fast reviewer check, run `python scripts/print_main_tables.py` plus `python scripts/audit_reproducibility.py` from the package root. These two processed-result checks require only Python 3.10+ and the committed files under `results/` and `data_manifest/`; install SciPy for the descriptive Wilcoxon scripts and `requirements-lock.txt` for cache-dependent reruns.

## Included File Groups

| Group | Paths | Purpose |
|---|---|---|
| Code | `src/elta/` | Minimal TECR metric, thresholding, selection, and held-out confidence-gate utilities used by the scripts. |
| Configs | `configs/*.yaml` | Portable protocol configs with relative `data/...` roots, class-split variants, seed settings, CLIP model choices, kNN settings, and held-out gate search grids. |
| Scripts | `scripts/*.py` | Experiment entry points, post-hoc baselines, supplementary diagnostics, result summarizers, table printing, processed-result audit, and descriptive Wilcoxon checks. |
| Results | `results/**` | Processed CSV/JSON/Markdown summaries for main and supplementary tables, including row-level CSVs, reviewer-facing reports, and reserved provenance directories for traceability. These are aggregate or split/config-level processed outputs, not raw images or feature tensors. |
| Manifests | `data_manifest/**`, `REPRODUCIBILITY_PACKAGE_MANIFEST.md` | Exact NUS-WIDE usable image-name/class manifests and this package-level release manifest. |
| Docs | `README.md`, `docs/*.md`, `CITATION.cff`, `LICENSE` | Setup, reproduction, data availability, claim-to-artifact mapping, citation metadata, and license text. |
| Dependencies | `requirements.txt`, `requirements-lock.txt` | Loose and review-time pinned Python dependency lists. |

## Excluded Artifacts

The following files are intentionally not included:

- raw Open Images, COCO, and NUS-WIDE images or annotation dumps beyond the included NUS-WIDE image/class manifests;
- precomputed CLIP feature files, cached arrays, NumPy archives, pickle files, and other dataset-derived cache files;
- external checkpoints, including public MKT checkpoint files and any trained ASL/DBLoss/backbone checkpoints;
- local `outputs/` run directories, logs, virtual environments, archives, and bytecode caches.

In the Git repository, these exclusions are reflected in `.gitignore` through entries such as `data/`, `features/`, `checkpoints/`, `outputs/`, `*.npy`, `*.npz`, `*.pkl`, `*.pt`, and `*.pth`. If this manifest is read inside an anonymized reviewer ZIP that omits `.gitignore`, the same exclusions still define the package boundary.

## Processed-Result Audit

Run the processed-result completeness audit from the package root:

```bash
python scripts/audit_reproducibility.py
```

This audit checks released processed-result CSVs, runnable script paths, protocol configs, required manifests, reviewer-facing report/provenance paths where applicable, and key manuscript/supplementary numeric values. It does not rerun experiments from raw images or regenerated CLIP features.

The audit covers these result groups:

| Claim group | Checked processed results |
|---|---|
| Open Images 10k main held-out protocol | `results/openimages_10k/main_method_12config_summary.csv`, `results/openimages_10k/main_per_config_rows.csv`, `results/openimages_10k/training_baseline_12config_summary.csv` |
| COCO val2017 main held-out protocol | `results/coco_val2017/main_method_12config_summary.csv`, `results/coco_val2017/main_per_config_rows.csv`, `results/coco_val2017/training_baseline_12config_summary.csv` |
| Open Images gate ablation | `results/openimages_10k/gate_ablation_12config_summary.csv` |
| Open Images calibration-size sensitivity | `results/openimages_10k/calibration_size_12config_summary.csv` |
| Open Images calibration-ratio sensitivity | `results/supplementary/calibration_ratio_summary.csv` |
| ASL+gate add-on | `results/openimages_10k/asl_gate_summary.csv`, `results/coco_val2017/asl_gate_summary.csv`, `results/nuswide/asl_gate_10config_summary.csv` |
| NUS-WIDE recoverable-subset check and supplementary suite | `results/nuswide/main_summary_10config.csv`, `results/nuswide/main_per_config_rows.csv`, `results/nuswide/training_baseline_10config_summary.csv`, `results/nuswide/frequency_group_10config_summary.csv`, `results/nuswide/gate_ablation_10config_summary.csv` |
| TECR definition robustness | `results/supplementary/tecr_robustness_summary.csv` |
| Alternative post-hoc and selective baselines | `results/supplementary/posthoc_combined_summary.csv`, `results/supplementary/odin_combined_summary.csv` |
| Known-context logistic controls | `results/supplementary/known_aware_combined_summary.csv` |
| TECR risk-set denominator audit | `results/supplementary/tecr_denominator_combined_summary.csv` |
| Adapted public MKT checkpoint sanity check | `results/supplementary/mkt_combined_summary.csv` |
| Larger filtered Open Images validation-subset sanity check | `results/supplementary/openimages_full_filtered_validation_sanity_summary.csv`, `results/supplementary/openimages_full_filtered_validation_sanity_report.md`, `results/supplementary/openimages_full_filtered_validation_sanity_per_config.csv`, `results/supplementary/openimages_full_filtered_validation_provenance/`, `data_manifest/openimages_full_filtered_validation_image_ids.json`, `data_manifest/openimages_full_filtered_validation_cache_manifest.json`, `data_manifest/openimages_full_filtered_validation_cache_status.json` |
| Complete Open Images validation release | `results/supplementary/openimages_complete_validation_summary.csv`, `results/supplementary/openimages_complete_validation_report.md`, `results/supplementary/openimages_complete_validation_per_config.csv`, `results/supplementary/openimages_complete_validation_provenance/`, `data_manifest/openimages_complete_validation_cache_manifest.json`, `data_manifest/openimages_selected_label_ids_top120.json` |
| Gate parameter stability | `results/supplementary/gate_parameter_stability_summary.csv` |
| ViT-B/16 backbone check | `results/supplementary/openimages_vitb16_heldout_summary.csv` |
| Frequency-group diagnostics and COCO exceptions | `results/supplementary/frequency_group_12config_summary.csv`, `results/supplementary/coco_training_exception_configs.csv` |

The larger filtered Open Images validation-subset slot is treated as scale-check evidence. Its summary/report/per-config/provenance files are schema-audited by `scripts/audit_reproducibility.py`, and the per-configuration rows are direction-checked against the bundled summary and expected 12/12 TECR directionality. The slot broadens the Open Images scope without replacing the controlled 10k benchmark.

The complete Open Images validation release is also treated as supplementary reviewer evidence. Unlike the filtered slot, it uses the complete `41,620`-image validation pool together with a frozen `120`-label slice and bundled run receipts. The audit checks its summary/per-config/provenance files and the `12/12` configuration-level TECR directionality of the main reviewer-facing pair, but it still does not promote the slot into a new manuscript headline benchmark.

For manuscript-style table printing from released results only, run:

```bash
python scripts/print_main_tables.py
```

## Cache Preconditions

There are three reproduction modes:

- Processed-result audit: requires only this repository and the committed files under `results/`.
- Cache-dependent Open Images runs: require existing CLIP feature caches under the config `output_dir` or `feature_cache_dir`; generate the Open Images cache first with `scripts/run_openimages_pilot.py --cache-only`.
- Cache-regenerating COCO/NUS-WIDE runs: can compute caches from public data when the expected raw data/manifests are present, using `scripts/run_coco_pilot.py --cache-only` and `scripts/run_nuswide_full_suite.py`.

Open Images and COCO classA/classB configs share cache locations because they use the same image pool, label list, and CLIP model; only protocol split seeds differ. NUS-WIDE caches are keyed by image-pool seed, maximum images, class count, and CLIP model.

## Configuration Matrices

The released main protocol uses configuration-level units:

| Dataset | Class-split configs | Image seeds | Config count | Primary processed result |
|---|---|---|---:|---|
| Open Images 10k | `configs/openimages_10k_heldout_ultrastrict.yaml`, `configs/openimages_10k_heldout_ultrastrict_classB.yaml` | `20260522` to `20260527` | 12 | `results/openimages_10k/main_method_12config_summary.csv`, `results/openimages_10k/main_per_config_rows.csv` |
| Open Images larger filtered validation-subset check | `configs/openimages_fullval_filtered_heldout_ultrastrict.yaml`, `configs/openimages_fullval_filtered_heldout_ultrastrict_classB.yaml` | `20260522` to `20260527` | 12 | `results/supplementary/openimages_full_filtered_validation_sanity_summary.csv`, `results/supplementary/openimages_full_filtered_validation_sanity_per_config.csv`, `results/supplementary/openimages_full_filtered_validation_sanity_report.md`, `results/supplementary/openimages_full_filtered_validation_provenance/` |
| Open Images complete validation release | `configs/openimages_complete_validation_heldout_ultrastrict.yaml`, `configs/openimages_complete_validation_heldout_ultrastrict_classB.yaml` | `20260522` to `20260527` | 12 | `results/supplementary/openimages_complete_validation_summary.csv`, `results/supplementary/openimages_complete_validation_per_config.csv`, `results/supplementary/openimages_complete_validation_report.md`, `results/supplementary/openimages_complete_validation_provenance/` |
| COCO val2017 | `configs/coco_heldout_ultrastrict.yaml`, `configs/coco_heldout_ultrastrict_classB.yaml` | `20260522` to `20260527` | 12 | `results/coco_val2017/main_method_12config_summary.csv`, `results/coco_val2017/main_per_config_rows.csv` |
| NUS-WIDE recoverable-subset check | `configs/nuswide_heldout_ultrastrict.yaml`, `configs/nuswide_heldout_ultrastrict_classB.yaml` | `20260522` to `20260526` | 10 | `results/nuswide/main_summary_10config.csv`, `results/nuswide/main_per_config_rows.csv` |

The config-internal protocol split seeds are `20260522` to `20260526`. For Open Images and COCO, the 12-configuration matrix is formed by two class-split configs and six image seeds. For NUS-WIDE, the 10-configuration matrix is formed by two class-split configs and five image seeds.

The larger filtered Open Images validation-subset check is intentionally supplementary reviewer evidence. It keeps the same `120` labels and held-out `40/20/40` protocol family as the main Open Images `10k` study, but uses a larger filtered `37,591`-image validation pool defined through the recorded image-id and cache-manifest files. It should be interpreted as a directional scope check, not as a replacement benchmark for the main manuscript tables or as an already-promoted full-validation result.

For reviewer-facing completeness, the bundled materials for this slot form a human-readable provenance chain:

- retained pool definition: `data_manifest/openimages_full_filtered_validation_image_ids.json`;
- cache definition and availability metadata: `data_manifest/openimages_full_filtered_validation_cache_manifest.json`, `data_manifest/openimages_full_filtered_validation_cache_status.json`;
- released aggregate and per-configuration evidence: `results/supplementary/openimages_full_filtered_validation_sanity_summary.csv`, `results/supplementary/openimages_full_filtered_validation_sanity_per_config.csv`;
- reviewer caveats and interpretation: `results/supplementary/openimages_full_filtered_validation_sanity_report.md`;
- reserved future-validation attachment point: `results/supplementary/openimages_full_filtered_validation_provenance/`.

If a future journal workflow explicitly asks for this larger-slot check to become a complete validation slot, the intended path is to keep these release-facing names stable, rerun the same 12 configuration-level jobs against the recorded retained pool, populate the reserved provenance directory with run receipts/config snapshots, and only then freeze paper-facing numeric targets for that promoted slot. Until that happens, the bundled files remain supplementary reviewer evidence only.

The complete Open Images validation release is also supplementary reviewer evidence. It uses the full `41,620`-image validation pool referenced by `validation-images-with-rotation.csv`, but freezes the `120`-label slice through `data_manifest/openimages_selected_label_ids_top120.json` so reruns do not depend on re-ranking labels from the same validation split.

For reviewer-facing completeness, the bundled materials for this slot form a second provenance chain:

- frozen full-validation cache definition: `data_manifest/openimages_complete_validation_cache_manifest.json`;
- frozen label slice: `data_manifest/openimages_selected_label_ids_top120.json`;
- released aggregate and per-configuration evidence: `results/supplementary/openimages_complete_validation_summary.csv`, `results/supplementary/openimages_complete_validation_per_config.csv`;
- reviewer caveats and interpretation: `results/supplementary/openimages_complete_validation_report.md`;
- run receipts and launch scripts: `results/supplementary/openimages_complete_validation_provenance/`.

## Wilcoxon Descriptive Checks

The Wilcoxon script is:

```text
scripts/wilcoxon_descriptive_pvalues.py
```

For the released configuration-level rows, the descriptive checks are:

```bash
python scripts/wilcoxon_descriptive_pvalues.py --input results/openimages_10k/main_per_config_rows.csv --unit-cols split_set,seed --metric tecr --baseline-method clip_knn_global_threshold --method heldout_gate_global_threshold --delta baseline_minus_method --alternative greater --zero-method wilcox
python scripts/wilcoxon_descriptive_pvalues.py --input results/coco_val2017/main_per_config_rows.csv --unit-cols split_set,seed --metric tecr --baseline-method clip_knn_global_threshold --method heldout_gate_global_threshold --delta baseline_minus_method --alternative greater --zero-method wilcox
python scripts/wilcoxon_descriptive_pvalues.py --input results/nuswide/main_per_config_rows.csv --unit-cols split_set,seed --metric tecr --baseline-method clip_knn_global_threshold --method heldout_gate_global_threshold --delta baseline_minus_method --alternative greater --zero-method wilcox
```

The paired unit is the configuration-level result: class-split set by image seed. Do not treat split-level rows as independent observations. If reviewers regenerate the Open Images or COCO per-configuration output directories, the same script can also be run with `--input-glob` over the per-run `calibrated_baseline_summary.csv` files and `--unit-cols source`.

## Review and Release-Metadata Boundary

This package is prepared for journal-facing release metadata. Author names are present in `CITATION.cff`, but personal email addresses are intentionally omitted from the public artifact. Repository URLs, DOI fields, and final release identifiers should be completed according to the journal's submission and research-data workflow. The included processed results and manifests are sufficient for the processed-result audit, while full reruns depend on public datasets and externally obtained caches/checkpoints as described above.
