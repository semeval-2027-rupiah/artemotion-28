"""Shared loading and constants for the ArtEmotion-28 analysis scripts.

Everything here follows the official README: the eight ST1 emotions, the
Russell (1980) valence mapping used by ST2/ST3, and the >=5-annotator rule.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ANNOTATIONS = REPO / "annotations" / "artelingo28_train_val.csv"
IMAGES = REPO / "images"
PHASE_INPUTS = REPO / "phase_inputs"
OUT = REPO / "analysis" / "out"

# Same order as baselines/st1/finetune_st1.py: label index i means EMOTIONS[i].
EMOTIONS = ["amusement", "anger", "awe", "contentment",
            "disgust", "excitement", "fear", "sadness"]
EMOTION_INDEX = {name: i for i, name in enumerate(EMOTIONS)}
POSITIVE = {"amusement", "awe", "contentment", "excitement"}
NEGATIVE = {"anger", "disgust", "fear", "sadness"}
MIN_ANNOTATORS = 5

LANGUAGES = [
    "Arabic", "Burmese", "Chinese", "Darija", "Emakhuwa", "English", "Hausa",
    "Hindi", "Igbo", "Indonesian", "IsiNdebele", "IsiXhosa", "IsiZulu", "Kazakh",
    "Kinyarwanda", "Korean", "Kyrgyz", "Malay", "Setswana", "Swahili", "Tagalog",
    "Tamil", "Thai", "Turkish", "Urdu", "Uzbek", "Vietnamese", "Yoruba",
]
# Public in the original ArtELingo release, so never scored in the evaluation phase.
TEST_UNSCORED = {"English", "Arabic", "Chinese"}
_CANONICAL = {name.lower(): name for name in LANGUAGES}

YEAR_RE = re.compile(r"-(\d{4})(?:\(\d+\))?$")
YEAR_MIN, YEAR_MAX = 1200, 2026
ERA_BINS = [YEAR_MIN, 1600, 1800, 1870, 1910, 1945, YEAR_MAX + 1]
ERA_LABELS = ["<1600", "1600-1799", "1800-1869", "1870-1909", "1910-1944", "1945+"]


def normalise_language(values):
    """Strip and map case variants ('arabic') onto the official spelling ('Arabic')."""
    stripped = pd.Series(values).astype(str).str.strip()
    return stripped.str.lower().map(_CANONICAL).fillna(stripped)


def parse_year(slug):
    """Year from a WikiArt slug ending in '-YYYY' or '-YYYY(n)'; None if absent or implausible."""
    match = YEAR_RE.search(str(slug).strip())
    if not match:
        return None
    year = int(match.group(1))
    return year if YEAR_MIN <= year <= YEAR_MAX else None


def era_of(years):
    """Bin years into eras; missing years become 'unknown'."""
    eras = pd.cut(pd.Series(years, dtype="float"), bins=ERA_BINS, labels=ERA_LABELS, right=False)
    return eras.astype(object).where(eras.notna(), "unknown")


def artist_of(slug):
    return str(slug).strip().split("_", 1)[0]


def load_annotations(path=ANNOTATIONS):
    """All annotations, cleaned. `language` is normalised; `language_raw` keeps the file's spelling."""
    df = pd.read_csv(path)
    for column in ("painting", "image_name", "split", "art_style", "genre"):
        if column in df:
            df[column] = df[column].astype(str).str.strip()
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df["language_raw"] = df["language"].astype(str).str.strip()
    df["language"] = normalise_language(df["language"]).to_numpy()
    df["artist"] = df["painting"].map(artist_of)
    df["year"] = df["painting"].map(parse_year).astype("Int64")
    df["era"] = era_of(df["year"]).to_numpy()
    return df


def st1_rows(df):
    """ST1 instances: the eight emotions only ('something else' / 'other' dropped)."""
    out = df[df["emotion"].isin(EMOTION_INDEX)].copy()
    out["label"] = out["emotion"].map(EMOTION_INDEX).astype(int)
    return out


def st1_train_holdout(train, fraction, seed):
    paintings = np.array(sorted(train["painting"].unique()))
    size = int(round(fraction * len(paintings)))
    if not 0 < size < len(paintings):
        raise ValueError("ST1 holdout must leave at least one painting on each side")
    held = np.random.default_rng(seed).choice(paintings, size, replace=False)
    select = train["painting"].isin(held)
    return train[~select], train[select].reset_index(drop=True)


def sentiment_pairs(df, split=None, min_annotators=MIN_ANNOTATORS):
    """(painting, language) pairs with k positive of n binary-mappable annotators."""
    rows = df[df["emotion"].isin(POSITIVE | NEGATIVE)]
    if split is not None:
        rows = rows[rows["split"] == split]
    rows = rows.assign(positive=rows["emotion"].isin(POSITIVE).astype(int))
    pairs = rows.groupby(["painting", "language"])["positive"].agg(k="sum", n="size").reset_index()
    return pairs[pairs["n"] >= min_annotators].reset_index(drop=True)


def painting_table(df):
    """One row per painting with its split and metadata."""
    columns = [c for c in ("painting", "image_name", "split", "art_style", "genre",
                           "artist", "year", "era") if c in df]
    return df[columns].drop_duplicates("painting").reset_index(drop=True)


def write_outputs(stem, payload, markdown):
    """Write analysis/out/<stem>.json and .md (summaries only -- never per-row data)."""
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{stem}.json", "w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    (OUT / f"{stem}.md").write_text(markdown)
    print(f"Wrote {OUT / stem}.json and .md")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is pd.NA:
        return None
    raise TypeError(f"not JSON serialisable: {type(value)}")


def md_table(frame, floatfmt="{:.4f}"):
    """Minimal Markdown table (avoids a tabulate dependency)."""
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else floatfmt.format(value)
        return str(value)
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"
