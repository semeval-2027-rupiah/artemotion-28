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
| `test_metrics.py` | no | checks the vectorised metrics against reference loops | stdout |

Conventions: bootstrap n=1000, seed 0; the resampling unit is the painting.
Only the `json`/`md` summaries in `out/` are committed; per-row predictions and
checkpoints stay local (see `.gitignore`).

See the [Molmo visual-caption experiment](../scripts/VISUAL_CAPTIONS.md#molmo-7b-o-validation-results) for the overall and English-subset ST1 findings and links to the scoring reports.
