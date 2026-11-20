"""Check the vectorised metrics against straightforward reference implementations.

    uv run python analysis/test_metrics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import ST1Counts, ST3Stats, bootstrap_weights, language_macro_f1, macro_f1  # noqa: E402


def summarize_py_macro_f1(gold, pred):  # verbatim rule from baselines/st1/summarize.py
    scores = []
    for target in range(8):
        if not (gold == target).any():
            continue
        tp = ((gold == target) & (pred == target)).sum()
        fp = ((gold != target) & (pred == target)).sum()
        fn = ((gold == target) & (pred != target)).sum()
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


rng = np.random.default_rng(0)
n = 3000
df = pd.DataFrame({"painting": rng.integers(0, 40, n).astype(str),
                   "language": rng.choice(list("ABCDE"), n),
                   "label": rng.choice(8, n, p=[.3, .02, .2, .2, .03, .1, .05, .1])})
df["pred"] = np.where(rng.random(n) < .5, df.label, rng.integers(0, 8, n))
paintings, languages = sorted(df.painting.unique()), sorted(df.language.unique())

# 1. point estimate equals the reference per-language loop
counts = ST1Counts(df, "pred", paintings, languages)
point = counts.headline(np.ones((1, len(paintings))))[0]
ref = np.mean([summarize_py_macro_f1(g.label.to_numpy(), g.pred.to_numpy()) for _, g in df.groupby("language")])
assert abs(point - ref) < 1e-12, (point, ref)
assert abs(language_macro_f1(df.label, df.pred, df.language) - ref) < 1e-12
assert abs(macro_f1(df.label, df.pred) - summarize_py_macro_f1(df.label.to_numpy(), df.pred.to_numpy())) < 1e-12

# 2. a weighted resample equals the metric on explicitly duplicated paintings
W = bootstrap_weights(len(paintings), 5, seed=3)
for w in W:
    dup = pd.concat([df[df.painting == p] for p, k in zip(paintings, w) for _ in range(int(k))])
    ref = np.mean([summarize_py_macro_f1(g.label.to_numpy(), g.pred.to_numpy()) for _, g in dup.groupby("language")])
    got = counts.headline(w[None, :])[0]
    assert abs(got - ref) < 1e-12, (got, ref)
assert W.sum(1).tolist() == [len(paintings)] * 5

# 3. keep= restricts the language average
keep = {"A", "B"}
ref = np.mean([summarize_py_macro_f1(g.label.to_numpy(), g.pred.to_numpy())
               for lang, g in df.groupby("language") if lang in keep])
assert abs(counts.headline(np.ones((1, len(paintings))), keep=keep)[0] - ref) < 1e-12

# 4. ST3 contrast and log loss against a direct implementation
pairs = pd.DataFrame([(p, l) for p in range(30) for l in "ABCDEF" if rng.random() < .7], columns=["painting", "language"])
pairs["n"] = rng.integers(5, 9, len(pairs))
pairs["k"] = rng.binomial(pairs.n, .6)
pairs["q"] = rng.random(len(pairs))
pairs.loc[pairs.painting == 0, "q"] = 0.5                  # constant prediction -> r = 0
pairs.loc[pairs.painting == 1, "k"] = pairs.loc[pairs.painting == 1, "n"]  # all observed equal -> skipped
stats = ST3Stats(pairs, range(30), list("ABCDEF"))
rs = []
for p, g in pairs.groupby("painting"):
    obs = g.k / g.n
    if len(g) < 3 or obs.nunique() == 1:
        continue
    rs.append(0.0 if g.q.nunique() == 1 else np.corrcoef(g.q, obs)[0, 1])
ones = np.ones((1, 30))
assert abs(stats.contrast(ones)[0] - np.mean(rs)) < 1e-12
q = pairs.q.clip(.01, .99)
pairs["loss"] = -(pairs.k * np.log2(q) + (pairs.n - pairs.k) * np.log2(1 - q)) / pairs.n
per_lang = pairs.groupby("language").loss.agg(["mean", "size"])
assert abs(stats.log_loss(ones)[0] - per_lang.loc[per_lang["size"] >= 20, "mean"].mean()) < 1e-12
assert stats.r[0] == 0.0 and np.isnan(stats.r[1])
print("all metric checks passed")
