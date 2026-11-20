"""Official ArtEmotion-28 metrics with a painting-cluster bootstrap.

Paintings are the resampling unit: all of a painting's annotations, in every
language, move together. Each resample is a vector of painting multiplicities
w (from a multinomial draw), so a resampled score is just a weighted sum of
per-painting sufficient statistics -- no Python loop per resample.
"""
import numpy as np
import pandas as pd

N_CLASSES = 8
LOG_LOSS_CLIP = (0.01, 0.99)
ST3_MIN_LANGUAGES = 3
ST3_MIN_PAIRS_PER_LANGUAGE = 20


def macro_f1(gold, pred, n_classes=N_CLASSES):
    """Codabench ST1 rule: average F1 over the classes present in gold."""
    gold, pred = np.asarray(gold), np.asarray(pred)
    scores = []
    for target in range(n_classes):
        in_gold = gold == target
        if not in_gold.any():
            continue
        in_pred = pred == target
        tp = (in_gold & in_pred).sum()
        fp = (~in_gold & in_pred).sum()
        fn = (in_gold & ~in_pred).sum()
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def language_macro_f1(gold, pred, languages):
    """Official ST1 headline: macro-F1 per language, unweighted mean over languages."""
    gold, pred, languages = map(np.asarray, (gold, pred, languages))
    return float(np.mean([macro_f1(gold[languages == lang], pred[languages == lang])
                          for lang in np.unique(languages)]))


def bootstrap_weights(n_paintings, n_boot, seed):
    """[n_boot, n_paintings] multiplicities of a painting resample with replacement."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_paintings, np.full(n_paintings, 1.0 / n_paintings), size=n_boot).astype(float)


def summarise(samples, point):
    """Point estimate with the bootstrap std and 95% percentile interval."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[~np.isnan(samples)]
    if len(samples) < 2:
        nan = float("nan")
        return {"point": float(point), "std": nan, "ci_low": nan, "ci_high": nan}
    return {"point": float(point), "std": float(samples.std(ddof=1)),
            "ci_low": float(np.percentile(samples, 2.5)), "ci_high": float(np.percentile(samples, 97.5))}


class ST1Counts:
    """tp/fp/fn per (painting, group, class); `group` is usually the language.

    score(W) gives the F1-Macro of every group for every row of W and is exact
    for W = ones (it matches macro_f1 on the same rows).
    """

    def __init__(self, df, pred_col, paintings, groups, group_col="language", n_classes=N_CLASSES):
        self.paintings, self.groups = list(paintings), list(groups)
        p = pd.Index(self.paintings).get_indexer(df["painting"])
        g = pd.Index(self.groups).get_indexer(df[group_col])
        if (p < 0).any() or (g < 0).any():
            raise ValueError("rows with a painting or group outside the index")
        gold = df["label"].to_numpy(int)
        pred = df[pred_col].to_numpy(int)
        hit = (gold == pred).astype(float)
        counts = np.zeros((len(self.paintings), len(self.groups), n_classes, 3))
        np.add.at(counts, (p, g, gold, 0), hit)          # tp, at the gold class
        np.add.at(counts, (p, g, pred, 1), 1.0 - hit)    # fp, at the predicted class
        np.add.at(counts, (p, g, gold, 2), 1.0 - hit)    # fn, at the gold class
        self.counts = counts
        self.rows = np.zeros((len(self.paintings), len(self.groups)))
        np.add.at(self.rows, (p, g), 1.0)

    def group_f1(self, W):
        """[B, G] F1-Macro per group; NaN where the resample has no rows of that group."""
        agg = np.einsum("bp,pgkc->bgkc", W, self.counts)
        tp, fp, fn = agg[..., 0], agg[..., 1], agg[..., 2]
        present = (tp + fn) > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
            per_group = (f1 * present).sum(-1) / present.sum(-1)
        per_group[(W @ self.rows) == 0] = np.nan
        return per_group

    def headline(self, W, keep=None):
        """[B] mean over groups (optionally only those in `keep`) of the per-group F1-Macro."""
        per_group = self.group_f1(W)
        if keep is not None:
            per_group = per_group[:, [i for i, g in enumerate(self.groups) if g in keep]]
        return np.nanmean(per_group, axis=1)


class ST3Stats:
    """Within-painting cultural contrast and log loss for (painting, language, k, n, q) pairs."""

    def __init__(self, pairs, paintings, languages):
        self.paintings, self.languages = list(paintings), list(languages)
        p = pd.Index(self.paintings).get_indexer(pairs["painting"])
        l = pd.Index(self.languages).get_indexer(pairs["language"])
        if (p < 0).any() or (l < 0).any():
            raise ValueError("pairs with a painting or language outside the index")
        k, n = pairs["k"].to_numpy(float), pairs["n"].to_numpy(float)
        q = np.clip(pairs["q"].to_numpy(float), *LOG_LOSS_CLIP)
        loss = -(k * np.log2(q) + (n - k) * np.log2(1 - q)) / n
        self.loss_sum = np.zeros((len(self.paintings), len(self.languages)))
        self.pair_count = np.zeros_like(self.loss_sum)
        np.add.at(self.loss_sum, (p, l), loss)
        np.add.at(self.pair_count, (p, l), 1.0)

        self.r = np.full(len(self.paintings), np.nan)
        frame = pd.DataFrame({"p": p, "obs": k / n, "q": pairs["q"].to_numpy(float)})
        for painting, group in frame.groupby("p"):
            if len(group) < ST3_MIN_LANGUAGES or np.ptp(group["obs"]) == 0:
                continue  # official rule: skip paintings whose observed proportions are all equal
            # A prediction constant across the painting's languages scores 0.
            self.r[painting] = 0.0 if np.ptp(group["q"]) == 0 else np.corrcoef(group["q"], group["obs"])[0, 1]
        self.valid = ~np.isnan(self.r)

    def contrast(self, W):
        with np.errstate(invalid="ignore", divide="ignore"):
            return (W @ np.where(self.valid, self.r, 0.0)) / (W @ self.valid.astype(float))

    def log_loss(self, W):
        loss, count = W @ self.loss_sum, W @ self.pair_count
        with np.errstate(invalid="ignore", divide="ignore"):
            per_language = np.where(count >= ST3_MIN_PAIRS_PER_LANGUAGE, loss / count, np.nan)
        return np.nanmean(per_language, axis=1)
