# Generate image-only captions for ST1

From the `artemotion-28` checkout on the GPU server, with `uv` installed and Hugging Face download access:

```sh
nvidia-smi
export CUDA_VISIBLE_DEVICES=0  # change to an allocated GPU
uv run --python 3.11 scripts/generate_visual_captions.py --backend molmo --limit 12
```

Review `runs/visual-captions/molmo/sample-12.csv` alongside the paintings. Check unsupported objects in abstract art and unsupported mood judgments. Only image pixels and the fixed prompt are used; the human captions and labels are never model inputs. The first run installs isolated dependencies and downloads the model. Molmo loads pinned custom Hugging Face code and must fit on one device. If GPU memory is limited, try `--backend blip`.

For a full build, run inside a persistent terminal session:

```sh
uv run --python 3.11 scripts/generate_visual_captions.py --backend molmo
# Optional cheaper baseline, separate cache and output:
uv run --python 3.11 scripts/generate_visual_captions.py --backend blip
```

Each backend writes `runs/visual-captions/<backend>/augmented.csv` with all input columns plus `visual_caption_en`, preserving row order and split. A pilot includes only rows from its sampled paintings. A local SQLite cache saves each completed image immediately: rerun the command to resume. If you replace image contents under unchanged filenames, use a fresh `--out-dir`. Do not commit generated CSVs or caches. Time the pilot after model download to estimate runtime on your GPU.

## Score the completed build

Wait for the build to report **173,745 annotation rows and 1,658 paintings** for the public release. The evaluator should report **8,057 eligible validation rows**. Use the full `augmented.csv`; the 12-painting pilot contains no validation paintings.

```sh
run_dir=runs/visual-captions/molmo
preds="$run_dir/val_predictions.csv"
uv run --locked python scripts/evaluate_visual_captions.py "$run_dir/augmented.csv" \
  --seed 42 --predictions-out "$preds" > "$run_dir/val_per_language.csv" &&
uv run --locked python analysis/score.py st1 --tag molmo_visual_captions_s42 \
  --preds "$preds:pred_text" "$preds:pred_swapped" "$preds:pred_aligned" \
  --names text_only swapped aligned --seed 0
```

In `analysis/out/molmo_visual_captions_s42_st1.md`, read `aligned − text_only` for the effect of adding captions and `aligned − swapped` for the effect of using the correct painting's caption. The paired 95% intervals resample paintings; a delta whose interval includes zero is inconclusive in this run. This probe uses one swap assignment and does not test the official decoy criterion. Keep the per-row predictions local.

## Molmo-7B-O validation results

Correct Molmo descriptions improved ST1 scores relative to wrong-painting descriptions, but concatenating them with the human captions scored below text-only overall and on English.

Molmo generated one English description per painting from the image alone. The [evaluator](evaluate_visual_captions.py) trained character n-gram classifiers on the multilingual training split. **Text-only** used human captions alone. **Aligned** appended the correct painting's Molmo description. **Swapped** used the same classifier as aligned and changed only the appended description at validation time, taking it from a different painting. The classifier and swap used seed 42.

| Validation scope | Text-only | Aligned | Swapped |
|---|---:|---:|---:|
| All 28 languages (8,057 rows, 86 paintings) | **0.3928** | 0.2738 | 0.2104 |
| English subset (308 rows) | **0.3037** | 0.2015 | 0.1025 |

Scores are official ST1 macro-F1, averaged equally across languages for the full validation set.

| Comparison (macro-F1 delta) | All languages | Paired 95% interval | English subset |
|---|---:|---:|---:|
| Aligned − text-only | −0.1190 | [−0.1422, −0.0930] | −0.1023 |
| Aligned − swapped | +0.0634 | [+0.0263, +0.0941] | +0.0990 |

The overall intervals use 1,000 paired painting bootstrap resamples (seed 0). Aligned scored above swapped in all 28 languages and below text-only in 27. The English subset shows that the drop also occurs when both captions are in English. Its deltas are point estimates from the multilingual models; no English-only model was trained and no English-specific paired intervals were computed.

The result applies to this single-seed concatenation setup. It does not isolate caption quality or establish whether the system would pass the official decoy check.

Run outputs: [full scoring report](../analysis/out/molmo_visual_captions_s42_st1.md) and [machine-readable metrics](../analysis/out/molmo_visual_captions_s42_st1.json), recorded in result commit `54c14a0`. The commands above reproduce the scoring.
