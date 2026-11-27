"""Local replica of the Codabench ST1 image checks, on the val split. GPU.

Every val caption is paired with a *different* painting's image (a derangement
over paintings), mirroring the decoy rows of the test file. Then:
  - multimodal: one-sided exact McNemar on pairs labelled correctly with only
    one of the two images -- passes if real-image wins significantly (p < 0.05)
    outnumber decoy-image wins;
  - text-only: share of captions labelled the same with both images (must be >= 99%).

    uv run python analysis/decoy_check.py --tag baseline \
        [--text-ckpt baselines/st1/checkpoints/text_only] [--multi-ckpt baselines/st1/checkpoints/multimodal]

Reuses baselines/st1 (model, predict_phase.predict, predict_phase.image_tensors).
Writes analysis/out/<tag>_decoy.json/.md; per-row predictions go to
baselines/st1/results/decoy_val_<tag>.csv (not committed).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import EMOTIONS, REPO, load_annotations, md_table, st1_rows, write_outputs
from metrics import language_macro_f1

ST1_DIR = REPO / "baselines" / "st1"


def derangement(items, seed):
    """A permutation of `items` with no fixed point."""
    rng = np.random.default_rng(seed)
    items = np.asarray(items)
    if len(items) < 2:
        raise ValueError("Decoy check needs at least two paintings")
    while True:
        perm = rng.permutation(len(items))
        if not (perm == np.arange(len(items))).any():
            return dict(zip(items, items[perm]))


def best_checkpoint(root):
    """The checkpoint the Trainer marked best (as predict_phase.best_checkpoint, any root)."""
    root = Path(root)
    checkpoints = [d for d in root.glob("checkpoint-*") if d.name.removeprefix("checkpoint-").isdigit()
                   and (d / "trainer_state.json").is_file()]
    if not checkpoints:
        raise ValueError(f"No Trainer checkpoints in {root}")
    latest = max(checkpoints, key=lambda d: int(d.name.removeprefix("checkpoint-")))
    best = json.loads((latest / "trainer_state.json").read_text())["best_model_checkpoint"]
    if not best or not (root / Path(best).name / "model.safetensors").is_file():
        raise ValueError(f"Missing best checkpoint weights in {root}")
    return root / Path(best).name


def load_model(mode, root, device):
    from safetensors.torch import load_file
    from model import MultimodalSentimentModel
    path = best_checkpoint(root)
    model = MultimodalSentimentModel(mode=mode, num_labels=len(EMOTIONS))
    model.load_state_dict(load_file(path / "model.safetensors"))
    print(f"{mode}: loaded {path}")
    return model.to(device).eval()


def decoy_scores(gold, text_real, text_decoy, multimodal_real, multimodal_decoy):
    real_ok = multimodal_real == gold
    decoy_ok = multimodal_decoy == gold
    real_wins = int((real_ok & ~decoy_ok).sum())
    decoy_wins = int((~real_ok & decoy_ok).sum())
    discordant = real_wins + decoy_wins
    pvalue = float(binomtest(real_wins, discordant, 0.5, alternative="greater").pvalue) if discordant else 1.0
    agreement = float(np.mean(text_real == text_decoy))
    return {
        "multimodal": {"real_wins": real_wins, "decoy_wins": decoy_wins,
                       "mcnemar_one_sided_p": pvalue, "passes": pvalue < 0.05},
        "text_only": {"agreement": agreement, "passes": agreement >= 0.99},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--text-ckpt", default=str(ST1_DIR / "checkpoints" / "text_only"))
    parser.add_argument("--multi-ckpt", default=str(ST1_DIR / "checkpoints" / "multimodal"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    text_ckpt, multi_ckpt = Path(args.text_ckpt).resolve(), Path(args.multi_ckpt).resolve()

    sys.path.insert(0, str(ST1_DIR))
    os.chdir(ST1_DIR)  # predict_phase resolves images as ../../images
    import torch
    from transformers import AutoTokenizer
    from model import TEXT_MODEL_NAME
    from predict_phase import image_tensors, predict

    val = st1_rows(load_annotations())
    val = val[val["split"] == "val"].reset_index(drop=True)
    image_of = val.groupby("painting")["image_name"].first()
    decoy_painting = derangement(sorted(image_of.index), args.seed)
    real = val.assign(image=val["image_name"])
    decoy = val.assign(image=val["painting"].map(decoy_painting).map(image_of))
    assert (real["image"] != decoy["image"]).all()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    unique_images = sorted(real["image"].unique())
    pixels = image_tensors(unique_images)
    image_index = {path: i for i, path in enumerate(unique_images)}
    to_label = {name: i for i, name in enumerate(EMOTIONS)}

    out = val[["painting", "language", "label"]].copy()
    for mode, root in (("text_only", text_ckpt), ("multimodal", multi_ckpt)):
        model = load_model(mode, root, device)
        for kind, frame in (("real", real), ("decoy", decoy)):
            out[f"{mode}_{kind}"] = [to_label[e] for e in predict(model, frame, tokenizer, pixels, image_index, device)]
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    gold = out["label"].to_numpy()
    columns = ("text_only_real", "text_only_decoy", "multimodal_real", "multimodal_decoy")
    checks = decoy_scores(gold, *(out[name].to_numpy() for name in columns))
    real_wins = checks["multimodal"]["real_wins"]
    decoy_wins = checks["multimodal"]["decoy_wins"]
    mcnemar_p = checks["multimodal"]["mcnemar_one_sided_p"]
    agreement = checks["text_only"]["agreement"]
    f1 = {f"{mode}_{kind}": language_macro_f1(gold, out[f"{mode}_{kind}"], out["language"])
          for mode in ("text_only", "multimodal") for kind in ("real", "decoy")}

    payload = {"seed": args.seed, "rows": len(out), "paintings": int(out["painting"].nunique()),
               "checkpoints": {"text_only": str(best_checkpoint(text_ckpt)),
                               "multimodal": str(best_checkpoint(multi_ckpt))},
               **checks,
               "language_macro_f1": f1}
    rows_path = ST1_DIR / "results" / f"decoy_val_{args.tag}.csv"
    out.to_csv(rows_path, index=False)
    print(f"Per-row predictions: {rows_path} (do not commit)")
    markdown = (f"# Decoy check on val ({args.tag})\n\n{len(out):,} captions, decoy = different painting "
                f"(derangement, seed {args.seed}).\n\n"
                f"- Multimodal: real-image wins {real_wins}, decoy-image wins {decoy_wins}, "
                f"one-sided McNemar p = {mcnemar_p:.3g} -> **{'pass' if mcnemar_p < 0.05 else 'FAIL'}**\n"
                f"- Text-only agreement real vs. decoy: {agreement:.4f} -> "
                f"**{'pass' if agreement >= 0.99 else 'FAIL'}** (needs >= 0.99)\n\n"
                + md_table(pd.DataFrame([f]).T.reset_index().set_axis(["run", "language-macro F1"], axis=1)))
    write_outputs(f"{args.tag}_decoy", payload, markdown)


if __name__ == "__main__":
    main()
