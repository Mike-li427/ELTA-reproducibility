# Larger Filtered Open Images Validation-Subset Check

This note documents the bundled reviewer-facing sanity check on a larger filtered Open Images validation subset. The mirrored files are included in the current blind package dated `2026-06-07`, but the check remains supplementary reviewer evidence rather than a required manuscript main-table dependency.

The repository also carries a separate complete-validation supplementary release under `results/supplementary/openimages_complete_validation_*`. That release uses the complete `41,620`-image validation pool with a frozen `120`-label slice. This note is only about the filtered `37,591`-image slot, so the two releases should not be merged conceptually.

## Boundary and interpretation

- This check is supplementary reviewer evidence only. It is not a replacement for the manuscript's main Open Images `10k` benchmark tables.
- The intent is to check whether the held-out reliability-gating comparison remains directionally similar when the Open Images study is expanded to a larger filtered subset under the same protocol family.
- The bundled run uses a filtered `37,591`-image validation subset over the same `120` labels, not the raw Open Images validation dump.
- The fairness interpretation for this slot is conservative: it is meant to show that the gate's TECR benefit is not confined to a single small retained pool, while still avoiding any claim that this larger slot supersedes the paper's pre-declared benchmark.
- Because the retained pool is filtered rather than the raw validation dump, this slot should be read as additional scope evidence under the same evaluation discipline, not as a hidden benchmark upgrade.

## Naming note

- The release-facing reviewer filenames use the stable `openimages_full_filtered_validation_sanity_*` form.
- Internal configs/scripts may still carry either `openimages_fullval_filtered_*` or `openimages_fullfiltered_validation_*` aliases for this same supplementary check.
- In this package, those aliases are collapsed into the single release-facing filename set listed below.

## Protocol guardrails

- Keep the same Open Images label space, held-out calibration family, and retrieval/calibration/evaluation split discipline used by the reported `10k` benchmark.
- Keep the held-out split fractions at `40% / 20% / 40%` unless a future report explicitly documents and justifies a deviation.
- Treat `classA` and `classB` as paired class-split families rather than as one base split plus an add-on pass.
- In reader-facing summary tables, `classA` and `classB` should share the same outer experiment-seed labels. Internally, the `*_classB` config family should keep its own distinct class-split seed stream rather than reusing the `classA` partition.
- The bundled summary should be framed as a scope sanity check for directionality, not as a benchmark replacement or a new paper-facing requirement.
- Reviewer-facing writeups should preserve the same comparison unit as the main paper: configuration-level pairs formed by class split set and outer seed. Avoid treating the five internal split averages inside one config as independent evidence.

## Provenance chain

The bundled larger-slot evidence is meant to be inspectable in a human-readable order:

1. Retained image pool: `data_manifest/openimages_full_filtered_validation_image_ids.json`
2. Cache definition and status: `data_manifest/openimages_full_filtered_validation_cache_manifest.json`, `data_manifest/openimages_full_filtered_validation_cache_status.json`
3. Released aggregate comparison: `results/supplementary/openimages_full_filtered_validation_sanity_summary.csv`
4. Released configuration-level rows: `results/supplementary/openimages_full_filtered_validation_sanity_per_config.csv`
5. Reviewer-facing caveats and interpretation: `results/supplementary/openimages_full_filtered_validation_sanity_report.md`
6. Reserved future-validation attachment point: `results/supplementary/openimages_full_filtered_validation_provenance/`

The provenance directory is intentionally part of the completeness story even when it contains only lightweight receipts. It reserves a stable place for future run logs, copied config snapshots, or checksum notes without forcing those bulky or environment-specific files into the default paper-facing bundle.

## Bundled processed-result files

- `results/supplementary/openimages_full_filtered_validation_sanity_summary.csv`
- `results/supplementary/openimages_full_filtered_validation_sanity_report.md`
- `results/supplementary/openimages_full_filtered_validation_sanity_per_config.csv`
- `results/supplementary/openimages_full_filtered_validation_provenance/`
- `data_manifest/openimages_full_filtered_validation_cache_manifest.json`
- `data_manifest/openimages_full_filtered_validation_image_ids.json`
- `data_manifest/openimages_full_filtered_validation_cache_status.json`

The summary CSV is intended for compact table printing and audit checks. The Markdown report is intended for reviewer-facing caveats such as filtering rules, image counts, seed conventions, and any deviations from the main `10k` protocol.

The per-config CSV is the fairness and directionality anchor for this slot. It shows whether the held-out gate remains better or worse than the baseline at the configuration level instead of letting one aggregate mean hide mixed-sign behavior.

## Minimal summary schema

The audit and table-printing scripts expect the summary CSV to contain at least these columns:

- `method`
- `num_configs`
- `average_precision_mean`
- `best_f1_mean`
- `tecr_mean`
- `tecr_reduction_pct`

Additional columns may be included. If `display_name` is present, the reporting script may use it for friendlier method labels.

## Required method rows

The bundled summary includes these main reviewer-facing method identifiers:

- `clip_knn_global_threshold`
- `heldout_gate_global_threshold`

Additional calibrated or post-hoc rows are included in the bundled summary, but the main reviewer-facing comparison remains whether the held-out gate stays directionally similar to the main Open Images `10k` result on the larger filtered subset.

## Bundled directional summary

- Validation subset size: `37,591` retained images
- Label space: `120` labels
- Reader-facing configurations: `12` (`classA/classB` paired families over six outer seeds)
- Main supplementary comparison:
  - `CLIP+kNN global threshold`: AP `0.8690`, F1 `0.7901`, TECR `0.2399`
  - `Held-out gate global threshold`: AP `0.8689`, F1 `0.7894`, TECR `0.2111`
  - TECR reduction: `12.0%`
- Configuration-level direction check:
  - `12/12` released class-split-by-seed pairs show lower TECR for `heldout_gate_global_threshold` than for `clip_knn_global_threshold`
  - Mean paired TECR delta (`clip_knn_global_threshold - heldout_gate_global_threshold`): `0.0288`

The main takeaway is directional consistency: on the larger filtered subset, the held-out gate still lowers TECR while leaving AP and F1 near the CLIP+kNN baseline.

This direction check matters because it is a fairness-style guard against a misleading average: reviewers can see that the aggregate reduction is not coming from only a small subset of favorable configurations.

## What this slot does not claim

- It does not replace the manuscript's repeated Open Images `10k` benchmark.
- It does not certify performance on the unfiltered full Open Images validation dump.
- It does not stand in for the separate complete-validation release bundled elsewhere in `results/supplementary/openimages_complete_validation_*`.
- It does not add a new paper-facing significance claim beyond the released configuration-level descriptive evidence.
- It does not ship dataset-derived feature caches or raw images.

## Future complete-validation readiness

The bundle is prepared so that a future journal request for a fuller validation release can reuse the same outward-facing filenames and interpretation rules.

If that happens, the minimal readiness checklist is:

1. Regenerate the filtered retained pool from public data and confirm it matches the recorded image-id manifest.
2. Rebuild the required cache metadata and confirm the same config aliases are used.
3. Rerun the full 12 configuration-level held-out and calibrated-baseline jobs.
4. Update the summary, per-config, and report files in place.
5. Populate `results/supplementary/openimages_full_filtered_validation_provenance/` with run receipts, copied config snapshots, or checksum notes.
6. Only after those steps, decide whether a journal actually wants paper-facing frozen numeric targets for this promoted slot.

## Audit and printing behavior

- `python scripts/print_main_tables.py` prints a supplementary section automatically from the bundled summary CSV.
- `python scripts/audit_reproducibility.py` checks the bundled report path, provenance directory path, required columns, required method rows, and the configuration-level directionality of the main reviewer-facing pair.
- The audit does not hard-code numeric targets for this slot in advance, because it is treated as descriptive supplementary evidence rather than as a new paper-facing main benchmark.
- Audit and printing behavior still treat the slot as supplementary reviewer evidence rather than as a replacement main table.
