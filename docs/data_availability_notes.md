# Data Availability Notes

The experiments use public datasets:

- Open Images validation images and image-level labels
- COCO val2017 images and annotations
- NUS-WIDE image lists and labels

Raw images are not included in this repository. The exact NUS-WIDE recoverable subset is recorded in `data_manifest/nuswide_image_names.json` and `data_manifest/nuswide_classes.json` to reduce ambiguity caused by changing URL availability.

For Open Images, the paper-facing benchmark remains the repeated `10k` validation-subset protocol reported in the manuscript. The repository also bundles two supplementary reviewer-facing checks under the same held-out protocol family: a larger filtered `37,591`-image validation-subset scope check and a complete-validation `41,620`-image release over the frozen `120`-label slice. These supplementary slots are not replacement benchmarks. The filtered slot records its retained pool and cache-definition metadata in `data_manifest/openimages_full_filtered_validation_image_ids.json`, `data_manifest/openimages_full_filtered_validation_cache_manifest.json`, and `data_manifest/openimages_full_filtered_validation_cache_status.json`. The complete-validation slot records its frozen full-pool cache definition in `data_manifest/openimages_complete_validation_cache_manifest.json` and its frozen label slice in `data_manifest/openimages_selected_label_ids_top120.json`.

Those manifests are paired with `results/supplementary/openimages_full_filtered_validation_sanity_summary.csv`, `results/supplementary/openimages_full_filtered_validation_sanity_per_config.csv`, `results/supplementary/openimages_full_filtered_validation_sanity_report.md`, and the reserved `results/supplementary/openimages_full_filtered_validation_provenance/` directory so reviewers can inspect the filtered-slot evidence as a human-readable provenance chain rather than as a bare aggregate number. The complete-validation release is paired with `results/supplementary/openimages_complete_validation_summary.csv`, `results/supplementary/openimages_complete_validation_per_config.csv`, `results/supplementary/openimages_complete_validation_report.md`, and `results/supplementary/openimages_complete_validation_provenance/` for the same reason.

CLIP feature caches are not committed because they may be large and dataset-derived redistribution can be license-dependent. They can be regenerated from the public data and the recorded manifests.

The processed summaries in `results/supplementary/` include the known-context logistic controls, TECR risk-set denominator audit, the larger filtered Open Images validation-subset sanity-check bundle, and the complete Open Images validation release bundle. These files do not contain raw images; they record aggregate protocol evidence, retained-pool or full-pool metadata, frozen label-slice metadata, configuration-level direction checks, and per-split denominator rows needed to interpret the paper's TECR claims fairly. Here, "fairly" means with the relevant risk-set and label-frequency context visible, not as a claim of demographic-fairness coverage.
