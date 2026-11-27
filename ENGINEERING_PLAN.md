# ArtEmotion-28 engineering plan

Use this plan to execute the experiments and prepare submissions for the [research goals](GOALS.md). Add team code under `analysis/` or `scripts/` and use the released files and reference baselines as inputs. Split long remote runs into a launch task and a results review.

## Submission dates

| Subtask | Test inputs available | Submission closes |
| --- | --- | --- |
| ST3 | 10 January 2027 | 16 January 2027, 23:59 UTC. Predictions lock at close. |
| ST1 and ST2 | 17 January 2027 | 31 January 2027, 23:59 UTC |

The [official schedule](README.md#important-dates) lists the system paper deadline as tentative for February 2027. Prioritize the ST3 submission rehearsal because its window closes first.

## 1. Fix the evaluation protocol

1. From the checkout root, check the release with `python3 scripts/verify_dataset.py`. Check the existing metrics with `uv run --no-project --with numpy --with pandas python analysis/test_metrics.py`.
2. Define five fixed, painting-grouped folds on `train` for ST2 and ST3. Store the seed and fold definition so every candidate uses the same held-out paintings. Use `painting` for group membership and `painting` or `image_name` for joins. Numeric `image_id` can collide. Reuse label and language normalization from [`analysis/common.py`](analysis/common.py).
3. Fit priors, retrieval indexes, model parameters, and calibration on training paintings only. Use inner training splits for tuning when reporting outer-fold results. For ST1, use the existing painting-grouped train holdout for checkpoint selection. Reserve labeled `val` for the final comparison after freezing the configurations.
4. Reuse [`analysis/score.py`](analysis/score.py) for labeled ST1 and ST3 development predictions. Add ST2 scoring under `analysis/` using the [official evaluation rules](README.md#evaluation): clip probabilities, calculate count-based log loss in bits, average by source–target language pair, then by target language and across eligible targets. Check the implementation against a small hand-calculated example. Use the [single-source blend](baselines/st2/README.md) as the first scored prediction set.
5. Reproduce the reference models on the same splits as the candidates. For model comparisons, resample paintings and compute paired metric deltas. Report the paired bootstrap interval and use the standard error of the paired deltas for the stopping rules in `GOALS.md`. Keep any language-filtered `val` proxy labeled as a proxy. Record each run's command, configuration, seeds, model revision, and results in `analysis/out/`. Keep predictions, checkpoints, generated datasets, and caches out of Git.

Done when the reference scores can be regenerated, the splits have no shared paintings, and an ST2 score can be checked against the official aggregation.

## 2. Build the ST3 image prior

1. Reproduce the [CLIP ViT-B/32 reference](baselines/st3/README.md) on the fixed folds. Compare it with a language prior and a style-by-language prior. Use [`analysis/era_signal.py`](analysis/era_signal.py) as the starting point for the metadata comparison.
2. Cache one frozen embedding per painting and encoder revision on the remote server. Start with CLIP. Compare SigLIP 2 or DINO once the first comparison runs end to end. Use the same cached features for retrieval and the trained head so the comparison isolates the method.
3. Build two estimators of the eight-class distribution. For retrieval, estimate the language prior from fold-training counts and use neighboring training paintings to estimate how it shifts. For the trained head, use count-weighted annotation targets and regularize a low-rank image-by-language interaction. Compare fitting on all eligible emotion labels with fitting only pairs that have at least five valid annotations. Apply the official minimum of five after dropping ambiguous emotions when building scoring targets.
4. Sum the four positive class probabilities for ST3. Compare within-painting cultural contrast first and log loss second. Apply the stopping rules in [`GOALS.md`](GOALS.md), freeze the selected configuration, and rehearse packaging against `phase_inputs/dev/st3_test.csv`. Retain the CLIP reference as a fallback. When the ST3 test file arrives, generate and submit every requested `(painting, language)` pair before the lock.

Done when the development archive has exactly the requested pairs and the selected system and reference can both generate it without gold labels.

## 3. Build ST1 text and late fusion

1. Reproduce character n-gram TF-IDF with a class-weighted linear SVM as the cheap text comparison. Use [`analysis/train_st1.py`](analysis/train_st1.py) as the starting point for the neural runs. Compare class-weighted mmBERT base and XLM-R large as budget allows. Use official macro-F1 for early stopping and a fresh run tag after changing settings. Add frozen caption embeddings only if they answer a question the first comparison leaves open.
2. Combine the selected text expert with the ST3 eight-class image prior. Tune fusion weights on train paintings that neither expert used for fitting. Include weight zero as the text-only reference. Report overall and per-language multimodal gains on the same labeled `val` rows.
3. Extend [`analysis/decoy_check.py`](analysis/decoy_check.py) to accept fused predictions while preserving its reference-checkpoint path. Run the same text on the real and swapped images. Confirm text-only agreement is at least 99% and the fused model passes the one-sided McNemar check. Use joint training with modality dropout as a comparison only if the core runs leave time.
4. Rehearse packaging against `phase_inputs/dev/st1_test.csv`. Verify that each mode predicts every opaque ID exactly once, including decoys. Keep repeated captions as separate rows. Use the image specified for each row when computing the multimodal prior.

Done when both modes produce complete development submissions and the report records whether fusion met the gain and decoy targets.

## 4. Build ST2 source models and test the image

1. Construct held-out inputs by splitting each evaluation painting's eligible languages into revealed sources and hidden targets. Match the released input structure and scoring eligibility. Keep all hidden response counts out of features, adjustment, and calibration.
2. Compare the official single-source blend, an all-source mean shrunk toward the target-language prior, and regularized regression on all revealed sources. Include the number and identity of sources where they help the source-only model. Fit calibration within train.
3. Add the ST3 prior to the strongest source-only model. Generate the prior out of fold for ST2 training examples and from outer-fold training paintings for held-out examples. Compare the paired loss difference under identical revealed/hidden splits. Keep the image only if the gain meets the research target.
4. Rehearse packaging against `phase_inputs/dev/st2_test.csv`. Verify one finite probability in `[0, 1]` for every requested `(painting, source_language, target_language)` triple.

Done when the comparison isolates what the image added and the selected model can produce a complete submission. When the ST1 and ST2 test files arrive, rerun the frozen pipelines and submit before the deadline.

## Optional experiments

Once the required runs and submission rehearsals are ready, use remaining time for a targeted comparison. Options from the source plan are TabPFN on ST2, LoRA on the vision tower, or LoRA on Qwen3-VL 2B. Check the [organizers' FAQ](README.md#faqs) for guidance before using external captions or running served zero-shot or few-shot comparisons. Check language coverage before committing GPU time to a model. For the persona probe, make the language prior the reference and report the score even if it contradicts the hypothesis.
