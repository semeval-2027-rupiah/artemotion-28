"""Do painting metadata (era, art_style, genre, artist) predict a language's p_positive
beyond the language prior? A diagnostic for ST3, using the train split only.

    uv run python analysis/era_signal.py [--folds 5] [--shrink 10 25 50 100]

5-fold cross-validation grouped by painting. Each model predicts
    q(language, value) = (k_lv + m * prior_l) / (n_lv + m)
i.e. the language x attribute positive share, shrunk toward the language prior
with m pseudo-annotators. Scored with the official ST3 contrast and log loss.
Fold std is reported; a gain smaller than it is noise.

Whether metadata parsed from the painting id may be used as an ST3 *feature* is a
rules question for the organisers -- this script only measures the signal.
Writes analysis/out/era_signal.json and .md.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import ANNOTATIONS, load_annotations, md_table, painting_table, sentiment_pairs, write_outputs
from metrics import ST3Stats

ATTRIBUTES = ("era", "art_style", "genre", "artist")


def fit_predict(train, test, attribute, m):
    totals = train.groupby("language")[["k", "n"]].sum()
    prior = totals["k"] / totals["n"]
    q_prior = test["language"].map(prior).fillna(train["k"].sum() / train["n"].sum())
    if attribute is None:
        return q_prior.to_numpy()
    cell = train.groupby(["language", attribute])[["k", "n"]].sum()
    keys = pd.MultiIndex.from_frame(test[["language", attribute]])
    k = cell["k"].reindex(keys).fillna(0).to_numpy()
    n = cell["n"].reindex(keys).fillna(0).to_numpy()
    return (k + m * q_prior.to_numpy()) / (n + m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default=ANNOTATIONS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--shrink", type=float, nargs="+", default=[10, 25, 50, 100])
    args = parser.parse_args()

    df = load_annotations(args.annotations)
    meta = painting_table(df).set_index("painting")
    pairs = sentiment_pairs(df, split="train")
    for attribute in ATTRIBUTES:
        pairs[attribute] = pairs["painting"].map(meta[attribute]).astype(str)

    models = [("language_prior", None, 0.0)] + [
        (f"language x {a} (m={m:g})", a, m) for a in ATTRIBUTES for m in args.shrink]
    results = {name: {"contrast": [], "log_loss": []} for name, _, _ in models}
    folds = GroupKFold(n_splits=args.folds)
    for train_idx, test_idx in folds.split(pairs, groups=pairs["painting"]):
        train, test = pairs.iloc[train_idx], pairs.iloc[test_idx].reset_index(drop=True)
        paintings, languages = sorted(test["painting"].unique()), sorted(pairs["language"].unique())
        ones = np.ones((1, len(paintings)))
        for name, attribute, m in models:
            stats = ST3Stats(test.assign(q=fit_predict(train, test, attribute, m)), paintings, languages)
            results[name]["contrast"].append(float(stats.contrast(ones)[0]))
            results[name]["log_loss"].append(float(stats.log_loss(ones)[0]))

    rows = []
    base_c = np.array(results["language_prior"]["contrast"])
    for name, _, _ in models:
        c, l = np.array(results[name]["contrast"]), np.array(results[name]["log_loss"])
        rows.append({"model": name, "contrast": c.mean(), "contrast_fold_std": c.std(ddof=1),
                     "gain_vs_prior": (c - base_c).mean(), "gain_fold_std": (c - base_c).std(ddof=1),
                     "log_loss": l.mean(), "log_loss_fold_std": l.std(ddof=1)})
    table = pd.DataFrame(rows)
    coverage = {a: float((pairs[a] != "unknown").mean()) for a in ATTRIBUTES}
    payload = {"folds": args.folds, "train_pairs": len(pairs), "attribute_coverage": coverage,
               "results": table.to_dict(orient="records"), "per_fold": results}
    markdown = ("# Metadata signal for ST3 (train split, grouped CV)\n\n"
                f"{len(pairs):,} train pairs, {args.folds} folds grouped by painting. "
                f"Attribute coverage: {coverage}.\n\n" + md_table(table))
    write_outputs("era_signal", payload, markdown)


if __name__ == "__main__":
    main()
