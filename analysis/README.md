# analysis/

Evaluation and split tooling for our ArtEmotion-28 entry. The official folders
(`annotations/`, `baselines/`, `phase_inputs/`, ...) are left untouched.

| Script | GPU | What it does | Output |
|---|---|---|---|
| `audit_splits.py` | no | split leakage, image near-duplicates, artist overlap, language spellings, year/era/style shift of val | `out/audit.*` |
| `score.py st1` | no | official ST1 metric + test-proxy (no EN/AR/ZH), painting-bootstrap CIs, paired deltas, per-language/class, `--by era art_style` | `out/<tag>_st1.*` |
| `score.py st3` | no | official ST3 contrast + log loss, CIs, train language-prior reference | `out/<tag>_st3.*` |
| `era_signal.py` | no | does era/style/genre/artist predict a language's p_positive beyond the prior? (train, grouped CV) | `out/era_signal.*` |
| `decoy_check.py` | yes | local replica of the Codabench decoy checks (McNemar for multimodal, >=99% agreement for text-only) | `out/<tag>_decoy.*` |
| `train_st1.py` | yes | ST1 training with early stopping on a painting-grouped train holdout and the official metric | `out/runs/<tag>.json` |
| `st1_submission.py` | no | checks archive contents, labels, and exact ID coverage against the phase input | stdout |
| `st1_dinov3.py` | yes | tests a frozen DINOv3 image prior with a fixed XLM-R checkpoint and packages both dev runs | `out/<tag>_st1.*`, `out/<tag>_fusion.*`, dev ZIP |
| `test_metrics.py` | no | checks the vectorised metrics against reference loops | stdout |

Conventions: bootstrap n=1000, seed 0; the resampling unit is the painting.
Only the `json`/`md` summaries in `out/` are committed; per-row predictions and
checkpoints stay local (see `.gitignore`).

See the [Molmo visual-caption experiment](../scripts/VISUAL_CAPTIONS.md#molmo-7b-o-validation-results) for the overall and English-subset ST1 findings and links to the scoring reports.

## ST1 development submission

On the GPU server, run `uv sync --locked` from the checkout root. In `baselines/st1/`,
reuse the two official checkpoints or train the missing modes with
`uv run --locked bash run_all.sh text_only` or `uv run --locked bash run_all.sh multimodal`.
Still in `baselines/st1/`, generate the archive with `uv run --locked python predict_phase.py --phase dev`.
`run_all.sh` skips a mode if its validation prediction CSV already exists. If
you have that CSV but lost its checkpoint, use `finetune_st1.py --mode <mode>` directly.
For multimodal training, run `precompute_image_pixels.py` first if the pixel cache is missing.

From the checkout root, check the ZIP before uploading it:

```bash
uv run --locked python analysis/st1_submission.py baselines/st1/results/st1_dev_submission.zip
```

Upload it to ST1 development on Codabench. Record both macro-F1 scores and
both decoy-check outcomes. The local archive checks verify the submission
format. Codabench verifies the actual decoy pairs.

### Test the DINOv3 image prior

Start after the baseline upload passes. The
[DINOv3 checkpoint](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
requires access on Hugging Face. Confirm that the GPU server can download it.
From the checkout root, reuse a text-only checkpoint and training report from
`analysis/train_st1.py` with `--select-on train_holdout`, or train one:

```bash
uv run --locked python analysis/train_st1.py --mode text_only --tag text_holdout_s42
uv run --locked --extra dinov3 python analysis/st1_dinov3.py --text-tag text_holdout_s42 --tag dinov3_prior_s42
```

The experiment uses a count-weighted logistic head on frozen image embeddings
and language indicators. It fits the head outside the text run's holdout and
chooses the fusion weight on that holdout. Weight zero is the same XLM-R
without the image prior. Development labels are used for scoring and the local
decoy rehearsal after the weight is fixed.

Read `analysis/out/dinov3_prior_s42_fusion.md` for the paired gain and local
decoy outcome. The detailed scores and per-language results are in
`analysis/out/dinov3_prior_s42_st1.md`. Upload the second archive only if the
local decoy checks pass and the paired 95% interval is above zero. Find the ZIP
at `baselines/st1/results/runs/dinov3_prior_s42/st1_dev_submission.zip`.
