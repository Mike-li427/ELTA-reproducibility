# Data Availability Notes

The experiments use public datasets:

- Open Images validation images and image-level labels
- COCO val2017 images and annotations
- NUS-WIDE image lists and labels

Raw images are not included in this repository. The exact NUS-WIDE subset used in the stress test is recorded in `data_manifest/` to reduce ambiguity caused by changing URL availability.

Pre-computed CLIP features are not committed because they may be large. They can be regenerated from the public data with the included scripts. Where dataset terms allow, feature caches can be shared with reviewers separately to avoid URL drift during review.
