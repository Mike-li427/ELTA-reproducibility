# Open Images 10k Reproduction Training and Main-Method Comparison

Date: 2026-05-26

Training baselines are frozen-CLIP feature heads trained on the retrieval split and selected/evaluated under the same calibration/evaluation protocol.

## Training Baselines

| Method | n | AP | F1 | TECR | TECR reduction vs Linear BCE |
|---|---:|---:|---:|---:|---:|
| ASL (ICCV 2021), class thresholds | 12 | 0.8421 | 0.7734 | 0.2490 | 43.8% |
| ASL (ICCV 2021), global threshold | 12 | 0.8421 | 0.7729 | 0.2629 | 40.6% |
| BalanceMix-style feature mixup, class thresholds | 12 | 0.6885 | 0.7451 | 0.2929 | 33.8% |
| BalanceMix-style feature mixup, global threshold | 12 | 0.6885 | 0.6816 | 0.5790 | -30.8% |
| Class-balanced BCE, class thresholds | 12 | 0.7410 | 0.7485 | 0.2902 | 34.4% |
| Class-balanced BCE, global threshold | 12 | 0.7410 | 0.7219 | 0.4471 | -1.0% |
| DBLoss (ECCV 2020), class thresholds | 12 | 0.8363 | 0.7674 | 0.2573 | 41.9% |
| DBLoss (ECCV 2020), global threshold | 12 | 0.8363 | 0.7679 | 0.2682 | 39.4% |
| Linear BCE, class thresholds | 12 | 0.7183 | 0.6483 | 0.5337 | -20.6% |
| Linear BCE, global threshold | 12 | 0.7183 | 0.6767 | 0.4427 | 0.0% |
| Logit-adjusted BCE, class thresholds | 12 | 0.7183 | 0.6483 | 0.5337 | -20.6% |
| Logit-adjusted BCE, global threshold | 12 | 0.7183 | 0.6770 | 0.4427 | 0.0% |
| Text-initialized ASL (ICCV 2021), class thresholds | 12 | 0.8383 | 0.7688 | 0.2543 | 42.6% |
| Text-initialized ASL (ICCV 2021), global threshold | 12 | 0.8383 | 0.7680 | 0.2623 | 40.7% |
| Text-initialized BalanceMix-style feature mixup, class thresholds | 12 | 0.6916 | 0.7468 | 0.2931 | 33.8% |
| Text-initialized BalanceMix-style feature mixup, global threshold | 12 | 0.6916 | 0.6836 | 0.5748 | -29.8% |
| Text-initialized BCE, class thresholds | 12 | 0.7092 | 0.6476 | 0.5074 | -14.6% |
| Text-initialized BCE, global threshold | 12 | 0.7092 | 0.6759 | 0.4732 | -6.9% |
| Text-initialized class-balanced BCE, class thresholds | 12 | 0.7365 | 0.7487 | 0.2872 | 35.1% |
| Text-initialized class-balanced BCE, global threshold | 12 | 0.7365 | 0.7187 | 0.4574 | -3.3% |
| Text-initialized DBLoss (ECCV 2020), class thresholds | 12 | 0.8299 | 0.7634 | 0.2709 | 38.8% |
| Text-initialized DBLoss (ECCV 2020), global threshold | 12 | 0.8299 | 0.7627 | 0.2807 | 36.6% |
| Text-initialized logit-adjusted BCE, class thresholds | 12 | 0.7092 | 0.6476 | 0.5074 | -14.6% |
| Text-initialized logit-adjusted BCE, global threshold | 12 | 0.7092 | 0.6759 | 0.4732 | -6.9% |

## Paper-Facing Comparison

| Method | AP | F1 | TECR | TECR reduction vs CLIP+kNN |
|---|---:|---:|---:|---:|
| ASL (ICCV 2021), class thresholds | 0.8421 | 0.7734 | 0.2490 | 4.0% |
| CLIP+kNN, global threshold | 0.8519 | 0.7713 | 0.2592 | 0.0% |
| CLIP+kNN isotonic, global threshold | 0.8416 | 0.7746 | 0.2352 | 9.3% |
| Held-out gate, global threshold | 0.8520 | 0.7704 | 0.2126 | 18.0% |
| Held-out gate, class thresholds | 0.8520 | 0.7683 | 0.2206 | 14.9% |

## Paired Diagnostics

| Comparison | Metric | n | Mean delta | Win rate |
|---|---|---:|---:|---:|
| Held-out gate vs CLIP+kNN | TECR | 12 | -0.0467 | 100.0% |
| Held-out gate vs CLIP+kNN | AP | 12 | 0.0001 | 50.0% |
| Held-out gate vs CLIP+kNN | F1 | 12 | -0.0010 | 25.0% |
| Held-out gate vs best training baseline | TECR | 12 | -0.0364 | 100.0% |

Notes:

- The BalanceMix row is a feature-space BalanceMix-style baseline, not an official image-space BalanceMix reproduction.
- Use the training-baseline table as auxiliary horizontal evidence, not as a claim that all representation-learning MLC methods have been beaten.
