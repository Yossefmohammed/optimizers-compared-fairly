| Optimizer | Best LR | Test acc (mean ± std) | Test loss | Epochs to 95% val-acc | LRs within 1.5x of best |
|---|---|---|---|---|---|
| SGD | 0.03 | 0.9660 ± 0.0075 | 0.0838 | 28 (5/5 seeds) | 2/8 |
| SGD+Momentum | 0.003 | 0.9723 ± 0.0053 | 0.0764 | 29 (5/5 seeds) | 5/8 |
| RMSprop | 0.0003 | 0.9720 ± 0.0029 | 0.0738 | 10 (5/5 seeds) | 6/8 |
| Adam | 0.001 | 0.9703 ± 0.0031 | 0.0763 | 5 (5/5 seeds) | 7/8 |
| AdamW | 0.001 | 0.9700 ± 0.0032 | 0.0761 | 5 (5/5 seeds) | 7/8 |