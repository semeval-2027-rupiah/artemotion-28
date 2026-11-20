# ST3 scores (baseline)

849 val pairs (>=5 annotators), 85 paintings; painting bootstrap n=1000, seed=0.

| system | languages | paintings | contrast | contrast CI | log loss | log loss CI |
|---|---|---|---|---|---|---|
| clip_lr | all | 82 | 0.4462 | [0.3856, 0.5062] | 0.7523 | [0.6774, 0.8271] |
| clip_lr | test_proxy | 79 | 0.4228 | [0.3573, 0.4902] | 0.7759 | [0.7015, 0.8509] |
| language_prior | all | 82 | 0.3644 | [0.3002, 0.4262] | 0.8678 | [0.8126, 0.9207] |
| language_prior | test_proxy | 79 | 0.3283 | [0.2583, 0.3953] | 0.9038 | [0.8482, 0.9590] |


## paired_contrast (b - a)

| a | b | point | ci_low | ci_high | p_b_better |
|---|---|---|---|---|---|
| clip_lr@all | language_prior@all | -0.0818 | -0.1225 | -0.0417 | 0.0000 |


## paired_contrast_test_proxy (b - a)

| a | b | point | ci_low | ci_high | p_b_better |
|---|---|---|---|---|---|
| clip_lr@test_proxy | language_prior@test_proxy | -0.0945 | -0.1395 | -0.0526 | 0.0000 |
