# Combined Known-Aware Post-Hoc Baselines

Combined rows: 680 from `results/supplementary/known_aware_combined_eval_rows.csv`.
Configuration-level summaries average the five class splits inside each image/class configuration before reporting means over 12 Open Images/COCO configurations or 10 NUS-WIDE configurations.

| Dataset | Method | Configs | Split-level rows | AP | F1 | TECR | TECR reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| coco | clip_knn | 12 | 60 | 0.8756 | 0.7887 | 0.2545 | 0.0% |
| coco | known_aware_logistic | 12 | 60 | 0.8754 | 0.7898 | 0.2240 | 12.0% |
| coco | permuted_known_logistic | 12 | 60 | 0.8729 | 0.7862 | 0.2615 | -2.7% |
| coco | score_only_logistic | 12 | 60 | 0.8755 | 0.7887 | 0.2545 | 0.0% |
| nuswide | clip_knn | 10 | 50 | 0.7863 | 0.7339 | 0.2571 | 0.0% |
| nuswide | known_aware_logistic | 10 | 50 | 0.7851 | 0.7315 | 0.2362 | 8.1% |
| nuswide | permuted_known_logistic | 10 | 50 | 0.7839 | 0.7322 | 0.2555 | 0.6% |
| nuswide | score_only_logistic | 10 | 50 | 0.7863 | 0.7339 | 0.2571 | 0.0% |
| openimages | clip_knn | 12 | 60 | 0.8519 | 0.7713 | 0.2592 | 0.0% |
| openimages | known_aware_logistic | 12 | 60 | 0.8508 | 0.7708 | 0.2483 | 4.2% |
| openimages | permuted_known_logistic | 12 | 60 | 0.8510 | 0.7711 | 0.2616 | -0.9% |
| openimages | score_only_logistic | 12 | 60 | 0.8519 | 0.7713 | 0.2592 | 0.0% |
