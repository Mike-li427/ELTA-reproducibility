# NUS-WIDE Reproducibility Manifest

This folder stores the exact NUS-WIDE subset identifiers used by the reported supplementary experiments.

- `nuswide_image_names.json`: 10,707 downloaded and verified image names.
- `nuswide_classes.json`: 72 concepts retained after minimum-positive filtering.

The manifest corresponds to `image_pool_seed=20260522`. A/B class-split configurations share this image pool and vary image partition seeds and class split seeds. Raw NUS-WIDE image URL availability can drift over time, so this manifest should be released with the processed results.
