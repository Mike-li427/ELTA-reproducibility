# Larger Filtered Open Images Validation-Subset Check

This summary aggregates the 12 configuration-level calibrated-baseline outputs from a larger filtered `37,591`-image Open Images validation subset under the same held-out `40/20/40` protocol family used in the main paper.

This is supplementary reviewer-facing scope evidence only. It is not the raw Open Images validation dump and does not replace the repeated Open Images `10k` benchmark reported in the manuscript.

| Method | n | AP | F1 | TECR | AP delta | F1 delta | TECR reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP+kNN, class thresholds | 12 | 0.8690 | 0.7875 | 0.2302 | 0.0% | -0.3% | 4.0% |
| CLIP+kNN, global threshold | 12 | 0.8690 | 0.7901 | 0.2399 | 0.0% | 0.0% | 0.0% |
| CLIP+kNN isotonic, class thresholds | 12 | 0.8652 | 0.7874 | 0.2303 | -0.4% | -0.3% | 4.0% |
| CLIP+kNN isotonic, global threshold | 12 | 0.8652 | 0.7919 | 0.2330 | -0.4% | 0.2% | 2.9% |
| CLIP+kNN Platt, class thresholds | 12 | 0.8672 | 0.7875 | 0.2302 | -0.2% | -0.3% | 4.0% |
| CLIP+kNN Platt, global threshold | 12 | 0.8672 | 0.7909 | 0.2383 | -0.2% | 0.1% | 0.7% |
| CLIP zero-shot, class thresholds | 12 | 0.5676 | 0.5983 | 0.7720 | -34.7% | -24.3% | -221.8% |
| CLIP zero-shot, global threshold | 12 | 0.5676 | 0.6214 | 0.9214 | -34.7% | -21.4% | -284.0% |
| Held-out gate, class thresholds | 12 | 0.8689 | 0.7852 | 0.2266 | -0.0% | -0.6% | 5.6% |
| Held-out gate, global threshold | 12 | 0.8689 | 0.7894 | 0.2111 | -0.0% | -0.1% | 12.0% |

Main reviewer-facing comparison:

- `CLIP+kNN global threshold`: AP `0.8690`, F1 `0.7901`, TECR `0.2399`
- `Held-out gate global threshold`: AP `0.8689`, F1 `0.7894`, TECR `0.2111`
- TECR reduction: `12.0%`

The directional pattern therefore remains consistent on this larger filtered subset: the held-out gate reduces TECR while leaving AP and F1 near the CLIP+kNN baseline.
