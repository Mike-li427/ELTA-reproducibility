# Provenance Notes for Larger Filtered Open Images Validation Slot

This directory is intentionally lightweight in the released reproducibility bundle.

It exists as the reviewer-facing attachment point for supplementary provenance that
is specific to the larger filtered `37,591`-image Open Images validation-subset
check. In this bundle, "validation-subset" means the filtered reviewer check named
by `openimages_full_filtered_validation_*`; it does not mean a promoted
"complete validation" release. The released package already includes the core
bundle used to interpret this slot:

- `../openimages_full_filtered_validation_sanity_summary.csv`
- `../openimages_full_filtered_validation_sanity_per_config.csv`
- `../openimages_full_filtered_validation_sanity_report.md`
- `../../../data_manifest/openimages_full_filtered_validation_image_ids.json`
- `../../../data_manifest/openimages_full_filtered_validation_cache_manifest.json`
- `../../../data_manifest/openimages_full_filtered_validation_cache_status.json`

Why this directory is still present:

1. It gives the bundle a stable location for optional rerun receipts, copied config
   snapshots, checksum notes, or environment logs if a journal later asks for a
   fuller validation-release trail.
2. It avoids overloading the main paper-facing directories with bulky or
   environment-specific files that are not required to inspect the released
   processed evidence.

Current release status:

- No extra rerun receipts are required for the present submission package beyond
  the released summary/per-config/report/manifests listed above.
- If a future review cycle requests a promoted complete-validation release, this is
  the directory that should receive the additional rerun-specific attachments so
  the filtered subset artifacts stay clearly separated from that fuller release.
