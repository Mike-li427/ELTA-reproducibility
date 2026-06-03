# Data Availability Notes

The experiments use public datasets:

- Open Images validation images and image-level labels
- COCO val2017 images and annotations
- NUS-WIDE image lists and labels

Raw images are not included in this repository. The exact NUS-WIDE subset used in the stress test is recorded in `data_manifest/nuswide_image_names.json` and `data_manifest/nuswide_classes.json` to reduce ambiguity caused by changing URL availability.

CLIP feature caches are not committed because they may be large and dataset-derived redistribution can be license-dependent. They can be regenerated from the public data and the recorded manifests.

The processed summaries in `results/supplementary/` include the known-context logistic controls and TECR risk-set denominator audit. These files do not contain raw images; they record aggregate protocol evidence and per-split denominator rows needed to interpret the paper's TECR claims.
