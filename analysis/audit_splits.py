"""Audit the released train/val split (and the phase input files, IDs only).

    uv run python analysis/audit_splits.py [--n-perm 1000] [--seed 0]

Checks: column list; language-name spellings; painting overlap between splits;
exact and near-duplicate images across splits (difference hash); artist overlap;
per-split sizes and distributions; year coverage from painting slugs; and whether
val's art_style / era / genre mix is unusual for a random draw of the same number
of paintings (total-variation distance against a permutation null).
Writes analysis/out/audit.json and audit.md. Nothing per-row is written.
"""
import argparse
import hashlib

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import ks_2samp

from common import (ANNOTATIONS, IMAGES, LANGUAGES, PHASE_INPUTS, TEST_UNSCORED, artist_of, era_of,
                    load_annotations, md_table, painting_table, parse_year, sentiment_pairs,
                    st1_rows, write_outputs)

NEAR_DUPLICATE_BITS = 6  # dHash Hamming distance at or below this counts as a near duplicate


def dhash(path, size=8):
    """64-bit difference hash: robust to rescaling and recompression."""
    pixels = np.asarray(Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS), dtype=float)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def file_sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def image_matches(names_a, names_b):
    """Exact (sha1) and near (dHash) duplicates between two sets of image files."""
    def hashes(names):
        out = {}
        for name in names:
            path = IMAGES / name
            if path.is_file():
                out[name] = (file_sha1(path), dhash(path))
        return out
    a, b = hashes(names_a), hashes(names_b)
    exact = sorted({(x, y) for x, (sx, _) in a.items() for y, (sy, _) in b.items() if sx == sy})
    ha = np.array([h for _, h in a.values()], dtype=np.uint64)
    hb = np.array([h for _, h in b.values()], dtype=np.uint64)
    near = []
    if len(ha) and len(hb):
        xor = ha[:, None] ^ hb[None, :]
        distance = np.unpackbits(xor.view(np.uint8).reshape(len(ha), len(hb), 8), axis=-1).sum(-1)
        keys_a, keys_b = list(a), list(b)
        for i, j in zip(*np.nonzero(distance <= NEAR_DUPLICATE_BITS)):
            near.append({"a": keys_a[i], "b": keys_b[j], "bits": int(distance[i, j])})
    return {"hashed_a": len(a), "hashed_b": len(b), "missing_files": len(names_a) + len(names_b) - len(a) - len(b),
            "exact": [list(p) for p in exact], "near": sorted(near, key=lambda d: d["bits"])}


def tvd(a, b):
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    return 0.5 * pa.subtract(pb, fill_value=0).abs().sum()


def shift_test(paintings, column, n_perm, seed):
    """Is val's mix of `column` unusual for a random set of the same number of paintings?"""
    is_val = (paintings["split"] == "val").to_numpy()
    values = paintings[column].astype(str).reset_index(drop=True)
    observed = tvd(values[~is_val], values[is_val])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        mask = np.zeros(len(values), dtype=bool)
        mask[rng.choice(len(values), is_val.sum(), replace=False)] = True
        null[i] = tvd(values[~mask], values[mask])
    return {"tvd": float(observed), "null_median": float(np.median(null)),
            "p_value": float((1 + (null >= observed).sum()) / (1 + n_perm))}


def split_summary(df, st1, paintings, split):
    rows, rows_st1 = df[df["split"] == split], st1[st1["split"] == split]
    per_painting = rows_st1.groupby("painting").size()
    pairs = sentiment_pairs(df, split=split)
    contrast_paintings = (pairs.groupby("painting").size() >= 3).sum()
    pts = paintings[paintings["split"] == split]
    return {
        "paintings": int(len(pts)),
        "annotations": int(len(rows)),
        "st1_instances": int(len(rows_st1)),
        "st1_per_painting": {"min": int(per_painting.min()), "median": float(per_painting.median()),
                             "max": int(per_painting.max())},
        "languages_per_painting_median": float(rows_st1.groupby("painting")["language"].nunique().median()),
        "st3_pairs": int(len(pairs)),
        "st3_paintings_with_3plus_languages": int(contrast_paintings),
        "emotion_share": rows_st1["emotion"].value_counts(normalize=True).round(4).to_dict(),
        "year_coverage": float(pts["year"].notna().mean()),
        "year_quantiles": ({q: int(pts["year"].dropna().quantile(q)) for q in (0.1, 0.5, 0.9)}
                           if pts["year"].notna().any() else {}),
        "era_counts": pts["era"].value_counts().to_dict(),
    }


def phase_input_check(df, paintings, phase):
    """IDs only: which paintings the phase files cover and how they relate to train."""
    out = {}
    train_paintings = set(paintings.loc[paintings["split"] == "train", "painting"])
    train_artists = set(paintings.loc[paintings["split"] == "train", "artist"])
    for subtask in ("st1", "st2", "st3"):
        path = PHASE_INPUTS / phase / f"{subtask}_test.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        entry = {"rows": int(len(frame)), "columns": list(frame.columns)}
        if "painting" in frame:
            pts = pd.Series(frame["painting"].astype(str).str.strip().unique())
            years = pts.map(parse_year)
            entry.update({
                "paintings": int(len(pts)),
                "paintings_in_train": int(pts.isin(train_paintings).sum()),
                "paintings_with_train_artist": int(pts.map(artist_of).isin(train_artists).sum()),
                "year_coverage": float(years.notna().mean()),
                "era_counts": era_of(years).value_counts().to_dict(),
            })
        if "language" in frame:
            entry["languages"] = sorted(frame["language"].astype(str).str.strip().unique())
        out[subtask] = entry
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default=ANNOTATIONS)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    df = load_annotations(args.annotations)
    st1 = st1_rows(df)
    paintings = painting_table(df)
    split_of = df.groupby("painting")["split"].nunique()
    image_split = df.groupby("image_name")["split"].nunique()
    spellings = df.groupby("language")["language_raw"].unique()

    report = {
        "columns": list(pd.read_csv(args.annotations, nrows=0).columns),
        "emotion_values": df["emotion"].value_counts().to_dict(),
        "language_spellings": {lang: sorted(v) for lang, v in spellings.items() if len(v) > 1 or v[0] != lang},
        "unknown_languages": sorted(set(df["language"]) - set(LANGUAGES)),
        "rows_by_raw_language_variant": df[df["language_raw"] != df["language"]]
            .groupby(["language_raw", "split"]).size().rename("rows").reset_index().to_dict(orient="records"),
        "paintings_in_multiple_splits": int((split_of > 1).sum()),
        "images_in_multiple_splits": int((image_split > 1).sum()),
        "painting_to_image_not_1to1": int((df.groupby("painting")["image_name"].nunique() > 1).sum()),
        "splits": {s: split_summary(df, st1, paintings, s) for s in sorted(df["split"].unique())},
    }

    train_artists = set(paintings.loc[paintings["split"] == "train", "artist"])
    val = paintings[paintings["split"] == "val"]
    report["artists"] = {"train": len(train_artists), "val": int(val["artist"].nunique()),
                         "val_paintings_with_train_artist": int(val["artist"].isin(train_artists).sum())}

    report["val_shift"] = {c: shift_test(paintings, c, args.n_perm, args.seed)
                           for c in ("art_style", "genre", "era", "artist") if c in paintings}
    years = paintings.dropna(subset=["year"])
    if (years["split"] == "val").any():
        ks = ks_2samp(years.loc[years["split"] == "train", "year"].astype(float),
                      years.loc[years["split"] == "val", "year"].astype(float))
        report["val_shift"]["year_ks"] = {"statistic": float(ks.statistic), "p_value": float(ks.pvalue)}

    if not args.skip_images:
        train_imgs = paintings.loc[paintings["split"] == "train", "image_name"]
        val_imgs = paintings.loc[paintings["split"] == "val", "image_name"]
        report["image_duplicates_train_val"] = image_matches(list(val_imgs), list(train_imgs))
        test_path = PHASE_INPUTS / "test" / "st3_test.csv"
        if test_path.is_file():
            test_imgs = pd.read_csv(test_path)["image"].astype(str).str.strip().unique()
            report["image_duplicates_test_trainval"] = image_matches(
                list(test_imgs), list(paintings["image_name"]))

    report["phase_inputs"] = {phase: phase_input_check(df, paintings, phase) for phase in ("dev", "test")}
    dev3 = PHASE_INPUTS / "dev" / "st3_test.csv"
    if dev3.is_file():
        dev_pairs = pd.read_csv(dev3)[["painting", "language"]].astype(str).apply(lambda c: c.str.strip())
        gold = sentiment_pairs(df, split="val")[["painting", "language"]]
        report["dev_st3_vs_val_gold"] = {
            "dev_pairs": len(dev_pairs), "gold_pairs": len(gold),
            "in_both": len(dev_pairs.merge(gold, on=["painting", "language"]))}

    write_outputs("audit", report, markdown(report))


def markdown(r):
    lines = ["# Split audit\n", f"Columns: `{', '.join(r['columns'])}`\n"]
    flags = []
    if r["language_spellings"]:
        flags.append(f"Language spelling variants: {r['language_spellings']}. Scoring that does not "
                     "normalise names treats them as separate languages.")
    if r["unknown_languages"]:
        flags.append(f"Languages outside the official 28: {r['unknown_languages']}")
    for key in ("paintings_in_multiple_splits", "images_in_multiple_splits", "painting_to_image_not_1to1"):
        if r[key]:
            flags.append(f"{key}: {r[key]}")
    for key, label in (("image_duplicates_train_val", "Val images duplicating train images"),
                       ("image_duplicates_test_trainval", "Test images duplicating train/val images")):
        dup = r.get(key, {})
        if dup.get("exact") or dup.get("near"):
            flags.append(f"{label}: {len(dup['exact'])} exact, "
                         f"{len(dup['near'])} near (dHash <= {NEAR_DUPLICATE_BITS} bits)")
    lines += ["## Flags\n", *(f"- {f}" for f in flags or ["None."]), ""]

    summary = pd.DataFrame({s: {k: v for k, v in d.items() if not isinstance(v, dict)}
                            for s, d in r["splits"].items()}).reset_index(names="metric")
    lines += ["## Splits\n", md_table(summary, "{:.4g}")]
    lines += ["\n## Era counts (paintings)\n",
              md_table(pd.DataFrame({s: d["era_counts"] for s, d in r["splits"].items()})
                       .fillna(0).astype(int).reset_index(names="era"))]
    lines += ["\n## Artists\n", f"{r['artists']}\n"]
    shift = pd.DataFrame(r["val_shift"]).T.reset_index(names="attribute")
    lines += ["\n## Is val unusual? (TVD vs. random painting draws)\n", md_table(shift)]
    lines += ["\n## Phase inputs (IDs only)\n", f"```\n{r['phase_inputs']}\n```\n"]
    if "dev_st3_vs_val_gold" in r:
        lines.append(f"Dev ST3 file vs. val gold pairs: {r['dev_st3_vs_val_gold']}\n")
    lines.append(f"\nTest-phase unscored languages: {sorted(TEST_UNSCORED)}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
