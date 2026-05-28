# COCO Reproduction Training and Main-Method Comparison

Date: 2026-05-26

Training baselines are frozen-CLIP feature heads trained on the retrieval split and selected/evaluated under the same calibration/evaluation protocol.

## Training Baselines

| Method | n | AP | F1 | TECR | TECR reduction vs Linear BCE |
|---|---:|---:|---:|---:|---:|
| ASL (ICCV 2021), class thresholds | 12 | 0.8846 | 0.8002 | 0.2220 | 58.2% |
| ASL (ICCV 2021), global threshold | 12 | 0.8846 | 0.7988 | 0.2508 | 52.8% |
| BalanceMix-style feature mixup, class thresholds | 12 | 0.7935 | 0.7689 | 0.3050 | 42.6% |
| BalanceMix-style feature mixup, global threshold | 12 | 0.7935 | 0.7456 | 0.5420 | -2.1% |
| Class-balanced BCE, class thresholds | 12 | 0.8306 | 0.7749 | 0.2875 | 45.8% |
| Class-balanced BCE, global threshold | 12 | 0.8306 | 0.7685 | 0.3938 | 25.8% |
| DBLoss (ECCV 2020), class thresholds | 12 | 0.8788 | 0.7952 | 0.2247 | 57.7% |
| DBLoss (ECCV 2020), global threshold | 12 | 0.8788 | 0.7958 | 0.2572 | 51.6% |
| Linear BCE, class thresholds | 12 | 0.7151 | 0.6551 | 0.4898 | 7.8% |
| Linear BCE, global threshold | 12 | 0.7151 | 0.6926 | 0.5310 | 0.0% |
| Logit-adjusted BCE, class thresholds | 12 | 0.7157 | 0.6551 | 0.4898 | 7.8% |
| Logit-adjusted BCE, global threshold | 12 | 0.7157 | 0.6948 | 0.4901 | 7.7% |
| Text-initialized ASL (ICCV 2021), class thresholds | 12 | 0.8828 | 0.7987 | 0.2213 | 58.3% |
| Text-initialized ASL (ICCV 2021), global threshold | 12 | 0.8828 | 0.7981 | 0.2452 | 53.8% |
| Text-initialized BalanceMix-style feature mixup, class thresholds | 12 | 0.8001 | 0.7719 | 0.2988 | 43.7% |
| Text-initialized BalanceMix-style feature mixup, global threshold | 12 | 0.8001 | 0.7473 | 0.5182 | 2.4% |
| Text-initialized BCE, class thresholds | 12 | 0.7177 | 0.6503 | 0.4224 | 20.4% |
| Text-initialized BCE, global threshold | 12 | 0.7177 | 0.6954 | 0.5244 | 1.2% |
| Text-initialized class-balanced BCE, class thresholds | 12 | 0.8333 | 0.7779 | 0.2829 | 46.7% |
| Text-initialized class-balanced BCE, global threshold | 12 | 0.8333 | 0.7686 | 0.3946 | 25.7% |
| Text-initialized DBLoss (ECCV 2020), class thresholds | 12 | 0.8745 | 0.7941 | 0.2137 | 59.8% |
| Text-initialized DBLoss (ECCV 2020), global threshold | 12 | 0.8745 | 0.7933 | 0.2583 | 51.4% |
| Text-initialized logit-adjusted BCE, class thresholds | 12 | 0.7183 | 0.6503 | 0.4224 | 20.4% |
| Text-initialized logit-adjusted BCE, global threshold | 12 | 0.7183 | 0.6970 | 0.4811 | 9.4% |

## Paper-Facing Comparison

| Method | AP | F1 | TECR | TECR reduction vs CLIP+kNN |
|---|---:|---:|---:|---:|
| Text-initialized ASL (ICCV 2021), class thresholds | 0.8828 | 0.7987 | 0.2213 | 13.1% |
| CLIP+kNN, global threshold | 0.8756 | 0.7887 | 0.2545 | 0.0% |
| CLIP+kNN isotonic, global threshold | 0.8683 | 0.7890 | 0.2258 | 11.3% |
| Held-out gate, global threshold | 0.8805 | 0.7934 | 0.1848 | 27.4% |
| Held-out gate, class thresholds | 0.8805 | 0.7889 | 0.1840 | 27.7% |

## Paired Diagnostics

| Comparison | Metric | n | Mean delta | Win rate |
|---|---|---:|---:|---:|
| Held-out gate vs CLIP+kNN | TECR | 12 | -0.0698 | 100.0% |
| Held-out gate vs CLIP+kNN | AP | 12 | 0.0049 | 100.0% |
| Held-out gate vs CLIP+kNN | F1 | 12 | 0.0046 | 83.3% |
| Held-out gate vs best training baseline | TECR | 12 | -0.0365 | 83.3% |

Notes:

- The BalanceMix row is a feature-space BalanceMix-style baseline, not an official image-space BalanceMix reproduction.
- Use the training-baseline table as auxiliary horizontal evidence, not as a claim that all representation-learning MLC methods have been beaten.
