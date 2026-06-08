# Complete Open Images validation check

This summary aggregates the 12 configuration-level calibrated-baseline outputs from the complete 41,620-image Open Images validation pool under the same held-out 40/20/40 protocol family used in the main paper, while freezing the reported 120-label slice.

This reviewer-facing check is supplementary evidence only. It does not replace the repeated Open Images 10k benchmark reported in the manuscript.

| Method | n | AP | F1 | TECR | AP delta | F1 delta | TECR reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP+kNN, class thresholds | 12 | 0.8589 | 0.7741 | 0.2212 | 0.0% | -0.4% | 1.3% |
| CLIP+kNN, global threshold | 12 | 0.8589 | 0.7774 | 0.2241 | 0.0% | 0.0% | 0.0% |
| CLIP+kNN isotonic, class thresholds | 12 | 0.8549 | 0.7740 | 0.2214 | -0.5% | -0.4% | 1.2% |
| CLIP+kNN isotonic, global threshold | 12 | 0.8549 | 0.7795 | 0.2144 | -0.5% | 0.3% | 4.3% |
| CLIP+kNN Platt, class thresholds | 12 | 0.8566 | 0.7741 | 0.2212 | -0.3% | -0.4% | 1.3% |
| CLIP+kNN Platt, global threshold | 12 | 0.8566 | 0.7783 | 0.2210 | -0.3% | 0.1% | 1.4% |
| CLIP zero-shot, class thresholds | 12 | 0.5266 | 0.5601 | 0.7797 | -38.7% | -28.0% | -247.9% |
| CLIP zero-shot, global threshold | 12 | 0.5266 | 0.5810 | 0.9001 | -38.7% | -25.3% | -301.7% |
| Held-out gate, class thresholds | 12 | 0.8588 | 0.7724 | 0.2162 | -0.0% | -0.6% | 3.5% |
| Held-out gate, global threshold | 12 | 0.8588 | 0.7763 | 0.1914 | -0.0% | -0.1% | 14.6% |