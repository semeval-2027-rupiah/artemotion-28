"""ST1 training with an unbiased protocol. GPU. Wraps baselines/st1 (left untouched).

Differences from baselines/st1/finetune_st1.py:
  - early stopping / best-checkpoint selection on a painting-grouped holdout of
    *train* (default 10%, fixed --holdout-seed so every run shares it), so val
    stays a clean held-out set; --select-on val reproduces the old protocol;
  - selection uses the official metric (per-language F1-Macro, then mean),
    not pooled sklearn macro-F1;
  - the multimodal image-use check pairs every caption with a *different*
    painting's image (as the Codabench decoys do) instead of a random shuffle.

    uv run python analysis/train_st1.py --mode text_only --seed 1 --tag text_s1

Writes baselines/st1/results/runs/<tag>/predictions.csv (val rows; not committed),
checkpoints to baselines/st1/checkpoints/runs/<tag>/, and analysis/out/runs/<tag>.json
(config + val scores). Needs results/image_pixel_values.pt from the baseline
(precompute_image_pixels.py) for the multimodal mode.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

from common import OUT, REPO, normalise_language, st1_train_holdout
from decoy_check import derangement
from metrics import language_macro_f1

ST1_DIR = REPO / "baselines" / "st1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["text_only", "multimodal", "image_only"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select-on", choices=["train_holdout", "val"], default="train_holdout")
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--decoy-seed", type=int, default=0)
    parser.add_argument("--num-train-epochs", type=float, default=8.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-train-samples", type=int, default=None, help="smoke test only")
    args = parser.parse_args()

    sys.path.insert(0, str(ST1_DIR))
    os.chdir(ST1_DIR)  # the baseline's data/pixel paths are relative to baselines/st1
    import torch
    from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint
    from finetune_st1 import DATA, EMOTIONS, PIXEL_PATHS, PIXELS_PATH, EmotionDataset, load_frame
    from model import TEXT_MODEL_NAME, MultimodalSentimentModel

    df = load_frame(DATA)
    df["language"] = normalise_language(df["language"]).to_numpy()
    train_df, val_df = df[df["split"] == "train"], df[df["split"] == "val"].reset_index(drop=True)

    if args.select_on == "train_holdout":
        train_df, select_df = st1_train_holdout(train_df, args.holdout_frac, args.holdout_seed)
    else:
        select_df = val_df
    if args.max_train_samples:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=0)
    print(f"mode={args.mode} seed={args.seed} train={len(train_df)} "
          f"select({args.select_on})={len(select_df)} [{select_df['painting'].nunique()} paintings] val={len(val_df)}")

    pixels, path_index = None, None
    if args.mode != "text_only":
        pixels = torch.load(PIXELS_PATH)
        path_index = {p: i for i, p in enumerate(np.load(PIXEL_PATHS, allow_pickle=True))}
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    def dataset(frame):
        return EmotionDataset(frame, tokenizer, args.mode, pixels, path_index)

    select_languages = select_df["language"].to_numpy()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"f1_lang_macro": language_macro_f1(labels, np.argmax(logits, -1), select_languages)}

    run_dir = ST1_DIR / "checkpoints" / "runs" / args.tag
    training_args = TrainingArguments(
        output_dir=str(run_dir), seed=args.seed,
        per_device_train_batch_size=args.train_batch_size, per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs, learning_rate=args.learning_rate,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="f1_lang_macro", greater_is_better=True,
        logging_steps=50, fp16=torch.cuda.is_available(), report_to=[],
        dataloader_num_workers=2 if torch.cuda.is_available() else 0)
    trainer = Trainer(model=MultimodalSentimentModel(mode=args.mode, num_labels=len(EMOTIONS)),
                      args=training_args, train_dataset=dataset(train_df), eval_dataset=dataset(select_df),
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)])
    resume = get_last_checkpoint(run_dir) if run_dir.is_dir() else None
    # Re-running a finished tag resumes from its last checkpoint: no steps are left, so
    # Trainer only reloads the best checkpoint and we go straight to prediction.
    trainer.train(resume_from_checkpoint=resume)
    print(f"best checkpoint: {trainer.state.best_model_checkpoint} (select metric {trainer.state.best_metric})")

    # compute_metrics is bound to select_df's languages; predict() on val/decoy rows must not call it.
    trainer.compute_metrics = None
    out = val_df[["row_id", "painting", "language", "emotion", "label"]].copy()
    logits = trainer.predict(dataset(val_df)).predictions
    out["pred"] = logits.argmax(-1)
    out["confidence"] = torch.softmax(torch.tensor(logits), -1).max(-1).values.numpy()
    scores = {"val_f1_lang_macro": language_macro_f1(out["label"], out["pred"], out["language"])}
    if args.mode == "multimodal":
        image_of = val_df.groupby("painting")["image_path"].first()
        decoy_of = derangement(sorted(image_of.index), args.decoy_seed)
        decoy = val_df.assign(image_path=val_df["painting"].map(decoy_of).map(image_of))
        out["pred_decoy"] = trainer.predict(dataset(decoy)).predictions.argmax(-1)
        scores["val_f1_lang_macro_decoy"] = language_macro_f1(out["label"], out["pred_decoy"], out["language"])

    pred_dir = ST1_DIR / "results" / "runs" / args.tag
    pred_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(pred_dir / "predictions.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    record = {"config": vars(args), "git_commit": commit, "train_rows": len(train_df),
              "select_rows": len(select_df), "select_paintings": int(select_df["painting"].nunique()),
              "best_checkpoint": trainer.state.best_model_checkpoint,
              "best_select_metric": trainer.state.best_metric, **scores,
              "predictions": str((pred_dir / "predictions.csv").relative_to(REPO))}
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "runs" / f"{args.tag}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
