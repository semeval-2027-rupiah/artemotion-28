"""Run the frozen DINOv3 image-prior experiment and package ST1 dev predictions."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import (EMOTIONS, IMAGES, LANGUAGES, OUT, REPO, load_annotations,
                    normalise_language, st1_rows, st1_train_holdout, write_outputs)
from decoy_check import ST1_DIR, best_checkpoint, decoy_scores, derangement, load_model
from metrics import language_macro_f1
from score import score_st1
from st1_submission import DEV_INPUTS, write_submission

DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINO_REVISION = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
FUSION_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)
DECOY_SEED = 0
BOOTSTRAP_SEED = 0
BOOTSTRAP_DRAWS = 1000


def encode_images(images, device, batch_size):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    from transformers.utils import is_torchvision_available

    if not is_torchvision_available():
        raise ValueError("The DINOv3 image processor needs TorchVision. Run with uv run --locked --extra dinov3.")

    processor = AutoImageProcessor.from_pretrained(DINO_MODEL, revision=DINO_REVISION)
    model = AutoModel.from_pretrained(DINO_MODEL, revision=DINO_REVISION).to(device).eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = []
            for name in images[start:start + batch_size]:
                with Image.open(IMAGES / name) as image:
                    batch.append(image.convert("RGB"))
            inputs = processor(images=batch, return_tensors="pt").to(device)
            batches.append(model(**inputs).pooler_output.float().cpu().numpy())
    return np.concatenate(batches)


def cached_embeddings(images, cache_path, device, batch_size):
    if cache_path.is_file():
        with np.load(cache_path) as cache:
            if np.array_equal(cache["images"], images):
                embeddings = cache["embeddings"]
                if embeddings.ndim == 2 and len(embeddings) == len(images) and np.isfinite(embeddings).all():
                    return embeddings

    print(f"Encoding {len(images):,} images with {DINO_MODEL}", flush=True)
    embeddings = encode_images(images, device, batch_size)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("DINOv3 produced a nonfinite or zero embedding")
    embeddings /= norms
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_path.parent) as directory:
        staged = Path(directory) / cache_path.name
        np.savez(staged, images=images, embeddings=embeddings)
        staged.replace(cache_path)
    return embeddings


class ImagePrior:
    def __init__(self, images, embeddings, train):
        self.images = pd.Index(images)
        self.embeddings = embeddings
        self.languages = pd.Index(LANGUAGES)
        self.language_features = np.eye(len(LANGUAGES), dtype=embeddings.dtype)

        counts = train.groupby(["image", "language", "label"], as_index=False).size()
        features, language_ids = self.features(counts)
        labels, weights = counts["label"].to_numpy(), counts["size"].to_numpy()
        self.head = LogisticRegression(max_iter=300).fit(features, labels, sample_weight=weights)
        if not np.array_equal(self.head.classes_, np.arange(len(EMOTIONS))):
            raise ValueError("The fitting split must cover all ST1 emotions")

        prior_counts = np.ones((len(LANGUAGES), len(EMOTIONS)))
        np.add.at(prior_counts, (language_ids, labels), weights)
        self.log_language_prior = np.log(prior_counts / prior_counts.sum(axis=1, keepdims=True))

    def features(self, rows):
        image_ids = self.images.get_indexer(rows["image"])
        language_ids = self.languages.get_indexer(rows["language"])
        if (image_ids < 0).any():
            missing = rows.loc[image_ids < 0, "image"].iloc[0]
            raise ValueError(f"No cached embedding for {missing!r}")
        if (language_ids < 0).any():
            unknown = rows.loc[language_ids < 0, "language"].iloc[0]
            raise ValueError(f"Unknown ST1 language {unknown!r}")
        features = np.column_stack((self.embeddings[image_ids], self.language_features[language_ids]))
        return features, language_ids

    def evidence(self, rows):
        features, language_ids = self.features(rows)
        return self.head.predict_log_proba(features) - self.log_language_prior[language_ids]


def text_logits(captions, model, tokenizer, device, max_length, batch_size):
    import torch

    batches = []
    with torch.inference_mode():
        for start in range(0, len(captions), batch_size):
            batch = captions.iloc[start:start + batch_size].astype(str).tolist()
            inputs = tokenizer(batch, truncation=True, max_length=max_length, padding="max_length",
                               return_tensors="pt").to(device)
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
            batches.append(logits.float().cpu().numpy())
    return np.concatenate(batches)


def fused_labels(logits, image_evidence, weight):
    return (logits + weight * image_evidence).argmax(axis=1)


def text_training_run(tag, annotations):
    report_path = OUT / "runs" / f"{tag}.json"
    if not report_path.is_file():
        raise ValueError(f"No training report {report_path}; run analysis/train_st1.py with --tag {tag}")
    report = json.loads(report_path.read_text())
    config = report["config"]
    if config["mode"] != "text_only" or config["select_on"] != "train_holdout" or config["tag"] != tag:
        raise ValueError("Use a text-only run from analysis/train_st1.py selected on the train holdout")

    train, holdout = st1_train_holdout(annotations[annotations["split"] == "train"],
                                      config["holdout_frac"], config["holdout_seed"])
    if len(holdout) != report["select_rows"] or holdout["painting"].nunique() != report["select_paintings"]:
        raise ValueError("The text training report does not match the released holdout")
    checkpoint_root = ST1_DIR / "checkpoints" / "runs" / tag
    checkpoint = best_checkpoint(checkpoint_root)
    if checkpoint.name != Path(report["best_checkpoint"]).name:
        raise ValueError("The text checkpoint does not match its training report")
    return train, holdout, config, checkpoint


def run(args):
    annotations = st1_rows(load_annotations()).rename(columns={"image_name": "image"})
    train, holdout, text_config, checkpoint = text_training_run(args.text_tag, annotations)
    val = annotations[annotations["split"] == "val"].reset_index(drop=True)
    phase = pd.read_csv(DEV_INPUTS, dtype={"id": str})
    phase["language"] = normalise_language(phase["language"]).to_numpy()
    images = sorted(set(annotations["image"]) | set(phase["image"]))

    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ST1_DIR))
    from finetune_st1 import MAX_TEXT_LENGTH
    from model import TEXT_MODEL_NAME

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_path = ST1_DIR / "results" / "dinov3" / f"{DINO_REVISION}_cls_l2.npz"
    embeddings = cached_embeddings(images, cache_path, device, args.image_batch_size)
    prior = ImagePrior(images, embeddings, train)
    model = load_model("text_only", checkpoint.parent, device)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    holdout_logits, val_logits, phase_logits = (
        text_logits(rows["caption"], model, tokenizer, device, MAX_TEXT_LENGTH, args.text_batch_size)
        for rows in (holdout, val, phase)
    )
    holdout_evidence = prior.evidence(holdout)
    fusion_scores = [
        language_macro_f1(holdout["label"], fused_labels(holdout_logits, holdout_evidence, weight),
                          holdout["language"])
        for weight in FUSION_WEIGHTS
    ]
    weight = FUSION_WEIGHTS[int(np.argmax(fusion_scores))]

    image_of = val.groupby("painting")["image"].first()
    decoy_of = derangement(sorted(image_of.index), DECOY_SEED)
    swapped = val.assign(image=val["painting"].map(decoy_of).map(image_of))
    if (swapped["image"] == val["image"]).any():
        raise ValueError("Some local decoy rows did not change image")

    out = val[["painting", "language", "label"]].copy()
    out["text"] = val_logits.argmax(axis=1)
    out["multimodal"] = fused_labels(val_logits, prior.evidence(val), weight)
    out["multimodal_decoy"] = fused_labels(val_logits, prior.evidence(swapped), weight)
    checks = decoy_scores(out["label"].to_numpy(), out["text"].to_numpy(), out["text"].to_numpy(),
                          out["multimodal"].to_numpy(), out["multimodal_decoy"].to_numpy())

    run_dir = ST1_DIR / "results" / "runs" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    val_path = run_dir / "dinov3_val.csv"
    out.to_csv(val_path, index=False)
    scores = score_st1(argparse.Namespace(
        tag=args.tag, preds=[f"{val_path.relative_to(REPO)}:{column}"
                            for column in ("text", "multimodal", "multimodal_decoy")],
        names=["text", "multimodal", "multimodal_decoy"], by=[], n=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED))
    gain = next(row for row in scores["paired"] if (row["a"], row["b"]) == ("text", "multimodal"))
    upload_candidate = all(check["passes"] for check in checks.values()) and gain["ci_low"] > 0

    archive = run_dir / "st1_dev_submission.zip"
    count = write_submission(archive, DEV_INPUTS, phase["id"], phase_logits.argmax(axis=1),
                             fused_labels(phase_logits, prior.evidence(phase), weight))
    record = {
        "config": vars(args), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                                    cwd=REPO, text=True).strip(),
        "text_checkpoint": str(checkpoint.relative_to(REPO)), "text_training_config": text_config,
        "vision_model": DINO_MODEL, "vision_revision": DINO_REVISION,
        "fit_paintings": int(train["painting"].nunique()), "holdout_paintings": int(holdout["painting"].nunique()),
        "validation_paintings": int(val["painting"].nunique()),
        "fusion_weight": weight, "holdout_grid": dict(zip(FUSION_WEIGHTS, fusion_scores, strict=True)),
        "decoy_seed": DECOY_SEED, "local_decoy": checks, "paired_gain": gain, "upload_candidate": upload_candidate,
        "submission": str(archive.relative_to(REPO)), "submission_rows_per_csv": count,
    }
    pvalue = checks["multimodal"]["mcnemar_one_sided_p"]
    write_outputs(f"{args.tag}_fusion", record, "\n".join([
        f"# ST1 DINOv3 image prior ({args.tag})", "",
        f"Train-holdout fusion weight: {weight:g}.",
        f"Paired validation macro-F1 gain: {gain['point']:+.4f} "
        f"(95% interval [{gain['ci_low']:+.4f}, {gain['ci_high']:+.4f}]).",
        f"Local multimodal decoy check: p = {pvalue:.3g}.",
        f"Recommended for a second development upload: {'yes' if upload_candidate else 'no'}.",
        f"See [{args.tag}_st1.md]({args.tag}_st1.md) for the per-language breakdown.", "",
        f"Archive: `{record['submission']}` ({count:,} IDs per CSV).",
    ]))
    print(f"{archive}: {count:,} IDs per CSV; upload candidate: {'yes' if upload_candidate else 'no'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-tag", required=True, help="Text-only run from analysis/train_st1.py")
    parser.add_argument("--tag", required=True, help="Name for this experiment's results and reports")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=64)
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag) for tag in (args.tag, args.text_tag)):
        parser.error("Run tags must start with a letter or digit and use only letters, digits, dots, hyphens, or underscores")
    if min(args.image_batch_size, args.text_batch_size) < 1:
        parser.error("Batch sizes must be positive")
    try:
        run(args)
    except ValueError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
