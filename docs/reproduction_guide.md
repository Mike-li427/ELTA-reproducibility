# Reproduction Guide

## Processed-result check

Run:

```bash
python scripts/print_main_tables.py
```

This prints manuscript-ready summaries from `results/`:

- Table 1: Open Images 10k main methods
- Table 2: COCO val2017 main methods
- Table 3: NUS-WIDE stress test

## Full experiment outline

1. Prepare public datasets under `data/`.
2. Install `requirements.txt`.
3. Run the held-out calibration scripts for both class-split sets and the listed seeds.
4. Run the summarization scripts to aggregate per-configuration outputs.
5. Compare generated summary CSVs with `results/`.

The protocol uses:

- retrieval/calibration/evaluation split: 40% / 20% / 40%
- kNN: `k=20`, temperature `0.07`
- gate grid: residual powers `[0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4]`, confidence cutoffs `[0.2, 0.3, 0.4, 0.5, 0.6]`, temperatures `[0.01, 0.02, 0.05, 0.1]`
- AP/F1 preservation tolerance: `0.25%`

## Important boundary

The artifact is designed for reproducibility of the post-hoc frozen-feature protocol. It does not redistribute raw datasets, large CLIP feature caches, or official fully trained ASL/DBLoss model checkpoints.
