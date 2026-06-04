# Post-hoc Baseline Combined Summary

Output directory: `outputs/openimages_posthoc_baselines_20260528_2310_full`

| Method | n | AP | F1 | TECR | Avg. labels | Coverage | AP delta | F1 delta | TECR reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clip_knn | 12 | 0.8519 +/- 0.0078 | 0.7713 +/- 0.0079 | 0.2592 +/- 0.0217 | 0.7561 +/- 0.0328 | 0.8329 +/- 0.0185 | 0.00% | 0.00% | 0.0% |
| temperature_scaling | 12 | 0.8519 +/- 0.0078 | 0.7713 +/- 0.0079 | 0.2592 +/- 0.0217 | 0.7561 +/- 0.0328 | 0.8329 +/- 0.0185 | 0.01% | 0.00% | 0.0% |
| split_conformal | 12 | 0.8519 +/- 0.0078 | 0.2274 +/- 0.0309 | 0.0014 +/- 0.0011 | 0.0618 +/- 0.0097 | 0.1312 +/- 0.0198 | 0.00% | -70.55% | 99.5% |
| maxlogit_known_reject | 12 | 0.8519 +/- 0.0078 | 0.7713 +/- 0.0079 | 0.2592 +/- 0.0217 | 0.7561 +/- 0.0328 | 0.8329 +/- 0.0185 | 0.00% | 0.00% | 0.0% |
| entropy_reject | 12 | 0.8513 +/- 0.0077 | 0.7710 +/- 0.0078 | 0.2562 +/- 0.0209 | 0.7493 +/- 0.0319 | 0.8298 +/- 0.0184 | -0.06% | -0.05% | 1.1% |
| elta_confidence_heldout | 12 | 0.8520 +/- 0.0075 | 0.7704 +/- 0.0068 | 0.2126 +/- 0.0218 | 0.6970 +/- 0.0339 | 0.8021 +/- 0.0166 | 0.02% | -0.12% | 18.0% |

Temperature scaling is fit by calibration NLL. Split conformal reports prediction-set activation against TECR and coverage. MaxLogit/MCM-style rejection uses the strongest known-label CLIP logit as an image-level in-distribution score, illustrating the granularity mismatch between OOD rejection and route-level TECR.
