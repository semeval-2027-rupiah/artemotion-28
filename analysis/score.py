"""Score ST1 / ST3 val predictions with official metrics and painting-bootstrap CIs.

ST1 (per-row prediction CSVs with painting, language, label and a prediction column,
e.g. the baseline's results/predictions_<mode>.csv; FILE:COL picks the column):
    uv run python analysis/score.py st1 --tag baseline \
        --preds baselines/st1/results/predictions_text_only.csv \
                baselines/st1/results/predictions_multimodal.csv \
                baselines/st1/results/predictions_multimodal.csv:pred_shuffled_images \
        --names text multi multi_shuffled --by era art_style

ST3 (painting, language, p_positive for the val pairs, e.g. phase_inputs/dev predictions;
gold is rebuilt from the val annotations; a train language prior is added as a reference):
    uv run python analysis/score.py st3 --tag baseline \
        --preds baselines/st3/results/dev/predictions.csv --names clip_lr

Writes analysis/out/<tag>_<subtask>.json/.md. "test_proxy" drops English, Arabic and
Chinese, which the evaluation phase never scores.
"""
import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import (ANNOTATIONS, EMOTIONS, LANGUAGES, TEST_UNSCORED, load_annotations,
                    md_table, normalise_language, painting_table, sentiment_pairs, write_outputs)
from metrics import ST1Counts, ST3Stats, bootstrap_weights, summarise

warnings.filterwarnings("ignore", message="Mean of empty slice")


def parse_spec(spec, default_col):
    path, _, col = spec.partition(":")
    return Path(path), (col or default_col)


def system_names(args, default_col):
    specs = [parse_spec(s, default_col) for s in args.preds]
    names = args.names or [f"{p.stem}:{c}" for p, c in specs]
    if len(names) != len(specs) or len(set(names)) != len(names):
        raise SystemExit("--names must give one unique name per --preds entry")
    return specs, names


def paired_deltas(samples, points, names):
    """Delta (b - a) for every pair of systems, on the same painting resamples."""
    rows = []
    for a, b in itertools.combinations(names, 2):
        delta = samples[b] - samples[a]
        rows.append({"a": a, "b": b, **summarise(delta, points[b] - points[a]),
                     "p_b_better": float(np.mean(delta > 0))})
    return rows


# ---------------------------------------------------------------- ST1

def load_st1(specs, names):
    frames = {}
    for (path, col), name in zip(specs, names):
        df = pd.read_csv(path)
        missing = {"painting", "language", "label", col} - set(df.columns)
        if missing:
            raise SystemExit(f"{path}: missing columns {sorted(missing)}")
        if "row_id" in df:
            df = df.sort_values("row_id")
        df = df.reset_index(drop=True)
        df["painting"] = df["painting"].astype(str).str.strip()
        df["language_raw"] = df["language"].astype(str).str.strip()
        df["language"] = normalise_language(df["language"]).to_numpy()
        df["pred_"] = df[col].astype(int)
        frames[name] = df
    reference = frames[names[0]]
    for name in names[1:]:
        same = (len(frames[name]) == len(reference)
                and (frames[name][["painting", "language_raw", "label"]].to_numpy()
                     == reference[["painting", "language_raw", "label"]].to_numpy()).all())
        if not same:
            raise SystemExit(f"{name}: rows differ from {names[0]}; systems must score the same val rows")
    return frames


def score_st1(args):
    specs, names = system_names(args, "pred")
    frames = load_st1(specs, names)
    base = frames[names[0]]
    paintings = sorted(base["painting"].unique())
    languages = sorted(base["language"].unique())
    raw_languages = sorted(base["language_raw"].unique())
    variants = sorted(set(raw_languages) - set(LANGUAGES))
    W = bootstrap_weights(len(paintings), args.n, args.seed)
    ones = np.ones((1, len(paintings)))
    proxy = set(languages) - TEST_UNSCORED

    payload = {"subtask": "st1", "n_boot": args.n, "seed": args.seed,
               "n_paintings": len(paintings), "n_rows": len(base),
               "languages": len(languages), "raw_language_spellings": len(raw_languages),
               "non_canonical_spellings": variants, "systems": {}}
    samples, points, per_language = {}, {}, {}
    for name in names:
        counts = ST1Counts(frames[name].assign(pred=frames[name]["pred_"]), "pred", paintings, languages)
        raw = ST1Counts(frames[name].assign(pred=frames[name]["pred_"]), "pred", paintings,
                        raw_languages, group_col="language_raw")
        boot_all, boot_proxy = counts.headline(W), counts.headline(W, keep=proxy)
        point_all, point_proxy = counts.headline(ones)[0], counts.headline(ones, keep=proxy)[0]
        samples[name], points[name] = boot_all, point_all
        samples[name + "@proxy"], points[name + "@proxy"] = boot_proxy, point_proxy
        lang_boot, lang_point = counts.group_f1(W), counts.group_f1(ones)[0]
        per_language[name] = (lang_point, np.nanpercentile(lang_boot, 2.5, axis=0),
                              np.nanpercentile(lang_boot, 97.5, axis=0))
        payload["systems"][name] = {
            "file": str(specs[names.index(name)][0]), "column": specs[names.index(name)][1],
            "headline": summarise(boot_all, point_all),
            "test_proxy": summarise(boot_proxy, point_proxy),
            "headline_raw_spelling": float(raw.headline(ones)[0]),  # what summarize.py prints
            "per_class_f1": per_class_f1(frames[name]),
            "confusion": confusion(frames[name]).tolist(),
        }
        for column in args.by:
            payload["systems"][name].setdefault("by", {})[column] = by_group(
                frames[name], column, paintings, W, ones, args)

    payload["paired"] = paired_deltas(samples, points, names)
    payload["paired_test_proxy"] = paired_deltas(samples, points, [n + "@proxy" for n in names])

    lang_table = pd.DataFrame({"language": languages,
                               "n": [int((base["language"] == l).sum()) for l in languages]})
    for name in names:
        point, low, high = per_language[name]
        lang_table[name] = point
        lang_table[f"{name}_ci"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(low, high)]
    payload["per_language"] = lang_table.to_dict(orient="records")

    write_outputs(f"{args.tag}_st1", payload, st1_markdown(payload, names, lang_table, args))


def per_class_f1(df):
    gold, pred = df["label"].to_numpy(int), df["pred_"].to_numpy(int)
    out = {}
    for i, emotion in enumerate(EMOTIONS):
        tp = ((gold == i) & (pred == i)).sum()
        denom = (gold == i).sum() + (pred == i).sum()
        out[emotion] = float(2 * tp / denom) if denom else float("nan")
    return out


def confusion(df):
    matrix = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=int)
    np.add.at(matrix, (df["label"].to_numpy(int), df["pred_"].to_numpy(int)), 1)
    return matrix  # rows gold, columns predicted


def by_group(df, column, paintings, W, ones, args):
    """Pooled F1-Macro within each value of a painting attribute (era, art_style, ...)."""
    meta = painting_table(load_annotations(args.annotations)).set_index("painting")
    if column not in meta:
        raise SystemExit(f"--by {column}: not a painting attribute ({list(meta.columns)})")
    frame = df.assign(group=df["painting"].map(meta[column]).astype(str), pred=df["pred_"])
    groups = sorted(frame["group"].unique())
    counts = ST1Counts(frame, "pred", paintings, groups, group_col="group")
    boot, point = counts.group_f1(W), counts.group_f1(ones)[0]
    n_paintings = frame.groupby("group")["painting"].nunique()
    return [{"group": g, "paintings": int(n_paintings[g]), "rows": int((frame["group"] == g).sum()),
             **summarise(boot[:, i], point[i])} for i, g in enumerate(groups)]


def st1_markdown(payload, names, lang_table, args):
    lines = [f"# ST1 scores ({args.tag})\n",
             f"{payload['n_rows']:,} val rows, {payload['n_paintings']} paintings, "
             f"{payload['languages']} languages; painting bootstrap n={args.n}, seed={args.seed}.\n"]
    if payload["non_canonical_spellings"]:
        lines.append(f"**Non-canonical language spellings in the file:** {payload['non_canonical_spellings']} "
                     f"({payload['raw_language_spellings']} spellings scored as separate languages by "
                     f"`summarize.py`). `headline` uses the normalised names.\n")
    rows = []
    for name in names:
        s = payload["systems"][name]
        rows.append({"system": name,
                     "headline": s["headline"]["point"],
                     "95% CI": f"[{s['headline']['ci_low']:.4f}, {s['headline']['ci_high']:.4f}]",
                     "boot std": s["headline"]["std"],
                     "test-proxy": s["test_proxy"]["point"],
                     "proxy CI": f"[{s['test_proxy']['ci_low']:.4f}, {s['test_proxy']['ci_high']:.4f}]",
                     "summarize.py": s["headline_raw_spelling"]})
    lines += ["## Headline\n", md_table(pd.DataFrame(rows))]
    for key, title in (("paired", "Paired deltas (b - a)"), ("paired_test_proxy", "Paired deltas, test-proxy")):
        if payload[key]:
            frame = pd.DataFrame(payload[key])[["a", "b", "point", "ci_low", "ci_high", "p_b_better"]]
            lines += [f"\n## {title}\n", md_table(frame)]
    lines += ["\n## Per language (95% CI)\n", md_table(lang_table, "{:.3f}")]
    classes = pd.DataFrame({name: payload["systems"][name]["per_class_f1"] for name in names})
    lines += ["\n## Per-class F1 (pooled over val)\n", md_table(classes.reset_index(names="emotion"), "{:.3f}")]
    for column in args.by:
        lines.append(f"\n## By {column} (pooled F1-Macro, 95% CI)\n")
        for name in names:
            frame = pd.DataFrame(payload["systems"][name]["by"][column])
            lines += [f"\n**{name}**\n\n", md_table(frame[["group", "paintings", "rows", "point", "ci_low", "ci_high"]], "{:.3f}")]
    return "\n".join(lines)


# ---------------------------------------------------------------- ST3

def score_st3(args):
    specs, names = system_names(args, "p_positive")
    annotations = load_annotations(args.annotations)
    gold = sentiment_pairs(annotations, split="val")
    train = sentiment_pairs(annotations, split="train")
    prior = train.groupby("language")[["k", "n"]].sum()
    prior = (prior["k"] / prior["n"]).to_dict()

    systems = {}
    for (path, col), name in zip(specs, names):
        df = pd.read_csv(path)
        df["painting"] = df["painting"].astype(str).str.strip()
        df["language"] = normalise_language(df["language"]).to_numpy()
        merged = gold.merge(df[["painting", "language", col]].rename(columns={col: "q"}),
                            on=["painting", "language"], how="left")
        missing = int(merged["q"].isna().sum())
        if missing:
            raise SystemExit(f"{path}: {missing} of {len(gold)} gold val pairs have no prediction")
        systems[name] = merged
    if not args.no_prior:
        systems["language_prior"] = gold.assign(q=gold["language"].map(prior))
        names = names + ["language_prior"]

    paintings = sorted(gold["painting"].unique())
    languages = sorted(gold["language"].unique())
    W = bootstrap_weights(len(paintings), args.n, args.seed)
    ones = np.ones((1, len(paintings)))

    payload = {"subtask": "st3", "n_boot": args.n, "seed": args.seed, "pairs": len(gold),
               "paintings": len(paintings), "languages": len(languages), "systems": {}}
    samples, points = {}, {}
    for name in names:
        entry = {}
        for subset, frame in (("all", systems[name]),
                              ("test_proxy", systems[name][~systems[name]["language"].isin(TEST_UNSCORED)])):
            stats = ST3Stats(frame, paintings, languages)
            boot_c, point_c = stats.contrast(W), stats.contrast(ones)[0]
            entry[subset] = {"contrast": summarise(boot_c, point_c),
                             "log_loss": summarise(stats.log_loss(W), stats.log_loss(ones)[0]),
                             "paintings_scored": int(stats.valid.sum())}
            samples[f"{name}@{subset}"], points[f"{name}@{subset}"] = boot_c, point_c
        entry["file"] = "train prior" if name == "language_prior" else str(specs[names.index(name)][0])
        payload["systems"][name] = entry
    payload["paired_contrast"] = paired_deltas(samples, points, [f"{n}@all" for n in names])
    payload["paired_contrast_test_proxy"] = paired_deltas(samples, points, [f"{n}@test_proxy" for n in names])

    rows = []
    for name in names:
        for subset in ("all", "test_proxy"):
            e = payload["systems"][name][subset]
            rows.append({"system": name, "languages": subset, "paintings": e["paintings_scored"],
                         "contrast": e["contrast"]["point"],
                         "contrast CI": f"[{e['contrast']['ci_low']:.4f}, {e['contrast']['ci_high']:.4f}]",
                         "log loss": e["log_loss"]["point"],
                         "log loss CI": f"[{e['log_loss']['ci_low']:.4f}, {e['log_loss']['ci_high']:.4f}]"})
    lines = [f"# ST3 scores ({args.tag})\n",
             f"{len(gold):,} val pairs (>=5 annotators), {len(paintings)} paintings; "
             f"painting bootstrap n={args.n}, seed={args.seed}.\n",
             md_table(pd.DataFrame(rows))]
    for key in ("paired_contrast", "paired_contrast_test_proxy"):
        if payload[key]:
            frame = pd.DataFrame(payload[key])[["a", "b", "point", "ci_low", "ci_high", "p_b_better"]]
            lines += [f"\n## {key} (b - a)\n", md_table(frame)]
    write_outputs(f"{args.tag}_st3", payload, "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subtask", choices=["st1", "st3"])
    parser.add_argument("--preds", nargs="+", required=True, help="FILE or FILE:COLUMN")
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--tag", required=True, help="output stem: analysis/out/<tag>_<subtask>.*")
    parser.add_argument("--by", nargs="*", default=[], help="ST1: painting attributes, e.g. era art_style genre")
    parser.add_argument("--no-prior", action="store_true", help="ST3: skip the train language-prior reference")
    parser.add_argument("--annotations", default=ANNOTATIONS)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    (score_st1 if args.subtask == "st1" else score_st3)(args)


if __name__ == "__main__":
    main()
