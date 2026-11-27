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
