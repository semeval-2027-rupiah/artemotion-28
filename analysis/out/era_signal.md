# Metadata signal for ST3 (train split, grouped CV)

16,019 train pairs, 5 folds grouped by painting. Attribute coverage: {'era': 0.5231287845683251, 'art_style': 1.0, 'genre': 1.0, 'artist': 1.0}.

| model | contrast | contrast_fold_std | gain_vs_prior | gain_fold_std | log_loss | log_loss_fold_std |
|---|---|---|---|---|---|---|
| language_prior | 0.3327 | 0.0164 | 0.0000 | 0.0000 | 0.8877 | 0.0153 |
| language x era (m=10) | 0.3419 | 0.0167 | 0.0092 | 0.0123 | 0.8840 | 0.0105 |
| language x era (m=25) | 0.3435 | 0.0170 | 0.0108 | 0.0112 | 0.8833 | 0.0111 |
| language x era (m=50) | 0.3451 | 0.0174 | 0.0124 | 0.0099 | 0.8829 | 0.0118 |
| language x era (m=100) | 0.3465 | 0.0178 | 0.0138 | 0.0081 | 0.8827 | 0.0125 |
| language x art_style (m=10) | 0.3488 | 0.0187 | 0.0161 | 0.0111 | 0.8660 | 0.0146 |
| language x art_style (m=25) | 0.3536 | 0.0190 | 0.0208 | 0.0105 | 0.8648 | 0.0160 |
| language x art_style (m=50) | 0.3572 | 0.0195 | 0.0245 | 0.0100 | 0.8653 | 0.0168 |
| language x art_style (m=100) | 0.3588 | 0.0200 | 0.0261 | 0.0091 | 0.8673 | 0.0172 |
| language x genre (m=10) | 0.3695 | 0.0212 | 0.0368 | 0.0095 | 0.8621 | 0.0156 |
| language x genre (m=25) | 0.3714 | 0.0219 | 0.0386 | 0.0096 | 0.8615 | 0.0157 |
| language x genre (m=50) | 0.3733 | 0.0222 | 0.0405 | 0.0093 | 0.8614 | 0.0157 |
| language x genre (m=100) | 0.3749 | 0.0223 | 0.0422 | 0.0086 | 0.8622 | 0.0156 |
| language x artist (m=10) | 0.3049 | 0.0116 | -0.0278 | 0.0099 | 0.8778 | 0.0147 |
| language x artist (m=25) | 0.3282 | 0.0141 | -0.0046 | 0.0078 | 0.8729 | 0.0155 |
| language x artist (m=50) | 0.3369 | 0.0156 | 0.0042 | 0.0062 | 0.8746 | 0.0156 |
| language x artist (m=100) | 0.3390 | 0.0162 | 0.0063 | 0.0044 | 0.8780 | 0.0155 |
