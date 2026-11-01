"""H5 with one ranking rule for every model: leave-one-word-out importance (D43).

Remove each word of a sentence in turn and measure how far the model's P(positive) moves. The
model is a black box, so the same rule ranks words for logistic regression, Naive Bayes, the
forests and DistilBERT alike, unlike weights / impurity importance / IG (D16). The live demo uses
the same function (D41).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from mirage.explain.overlap import jaccard, rbo


def leave_one_out(text: str) -> tuple[list[str], list[str]]:
    """The sentence's words, and the sentence with each word removed in turn."""
    words = text.split()
    return words, [" ".join(words[:i] + words[i + 1 :]) for i in range(len(words))]


def occlusion_table(
    predict_proba: Callable[[list[str]], np.ndarray], texts: list[str], chunk: int = 4096
) -> pd.DataFrame:
    """One row per (sentence, word position): delta = p_pos(full) - p_pos(without the word).

    delta > 0 means the word pushed the prediction toward positive. All variants are scored in
    chunks of `chunk` sentences, so large sets stay within memory.
    """
    batch, owner = [], []  # owner: (sentence index, position or -1 for the full sentence)
    words_of = []
    for i, t in enumerate(texts):
        words, dropped = leave_one_out(t)
        words_of.append(words)
        batch += [t, *dropped]
        owner += [(i, -1)] + [(i, j) for j in range(len(words))]
    p = (
        np.concatenate(
            [predict_proba(batch[k : k + chunk])[:, 1] for k in range(0, len(batch), chunk)]
        )
        if batch
        else np.zeros(0)
    )
    full = {i: p[n] for n, (i, j) in enumerate(owner) if j == -1}
    rows = [
        {"sent": i, "position": j, "word": words_of[i][j], "delta": float(full[i] - p[n])}
        for n, (i, j) in enumerate(owner)
        if j != -1
    ]
    return pd.DataFrame(rows, columns=["sent", "position", "word", "delta"])


def word_importance(occ: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    """Per word: occurrences, mean |delta| (the ranking key) and mean signed delta.

    Only words seen at least `min_count` times and containing a letter are ranked: rarer words
    are too noisy, and punctuation isn't a "shortcut word" (it is still removed and scored).
    """
    eligible = occ[occ["word"].map(lambda w: any(c.isalpha() for c in w))]
    agg = eligible.groupby("word")["delta"].agg(
        n="size", mean_abs=lambda d: float(np.abs(d).mean()), mean_signed="mean"
    )
    agg = agg[agg["n"] >= min_count].sort_values(["mean_abs", "n"], ascending=[False, False])
    return agg.reset_index().assign(rank=lambda d: np.arange(1, len(d) + 1))


def split_half_overlap(occ: pd.DataFrame, min_count: int, k: int, seed: int) -> dict:
    """Noise ceiling: the model's top-k from one random half of the sentences vs the other half.

    Both halves rank the *same* words: those eligible in the full data (seen >= min_count times)
    and present in both halves. Ranking each half with its own vocabulary let words seen two or
    three times dominate, so a model "agreed with itself" less than with other models (D43).
    """
    eligible = set(word_importance(occ, min_count)["word"])
    sents = occ["sent"].unique()
    rng = np.random.default_rng(seed)
    in_a = occ["sent"].isin(set(rng.choice(sents, len(sents) // 2, replace=False)))
    a, b = (word_importance(h[h["word"].isin(eligible)], 1) for h in (occ[in_a], occ[~in_a]))
    both = set(a["word"]) & set(b["word"])
    ra = [w for w in a["word"] if w in both]
    rb = [w for w in b["word"] if w in both]
    return {"jaccard": jaccard(ra, rb, k), "rbo": rbo(ra[:k], rb[:k])}


def chance_jaccard(n_words: int, k: int) -> float:
    """Expected Jaccard@k of two random top-k lists drawn from the same n_words."""
    if n_words <= k:
        return 1.0
    shared = k * k / n_words
    return shared / (2 * k - shared)


def word_effects(occ_by_model: dict[str, pd.DataFrame], min_count: int) -> pd.DataFrame:
    """Words x models table of each word's mean signed effect (> 0 pushes positive), on the words
    every model can rank (the same sentences, so in practice the same words)."""
    cols = {
        m: word_importance(o, min_count).set_index("word")["mean_signed"]
        for m, o in occ_by_model.items()
    }
    return pd.DataFrame(cols).dropna()


def split_half_reliability(occ: pd.DataFrame, words, n_splits: int, seed: int) -> float:
    """How reproducible a model's word effects are: the Spearman correlation between two random
    halves of the sentences, stepped up to full length (Spearman-Brown), averaged over splits."""
    from scipy.stats import spearmanr

    words = pd.Index(words)
    sents = occ["sent"].unique()
    out = []
    for s in range(n_splits):
        rng = np.random.default_rng(seed + s)
        in_a = occ["sent"].isin(set(rng.choice(sents, len(sents) // 2, replace=False)))
        ha = occ[in_a].groupby("word")["delta"].mean().reindex(words)
        hb = occ[~in_a].groupby("word")["delta"].mean().reindex(words)
        ok = ha.notna() & hb.notna()
        r = spearmanr(ha[ok], hb[ok]).statistic
        out.append(2 * r / (1 + r))
    return float(np.mean(out))


def effect_agreement(
    effects: pd.DataFrame, reliability: dict[str, float], n_resamples: int, seed: int
) -> pd.DataFrame:
    """H5's main measure (D43): for every pair of models, the Spearman correlation of their word
    effects (bootstrap CI over words), and the same correlation corrected for each model's own
    reliability (r / sqrt(rel_a * rel_b)), so noise can't make two models look different."""
    import itertools

    from scipy.stats import rankdata

    rng = np.random.default_rng(seed)
    ranks = effects.apply(rankdata)
    idx = rng.integers(0, len(effects), size=(n_resamples, len(effects)))
    rows = []
    for a, b in itertools.combinations(effects.columns, 2):
        x, y = ranks[a].to_numpy(), ranks[b].to_numpy()
        rho = float(np.corrcoef(x, y)[0, 1])
        boot = [
            np.corrcoef(rankdata(effects[a].to_numpy()[i]), rankdata(effects[b].to_numpy()[i]))[
                0, 1
            ]
            for i in idx
        ]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "n_words": len(effects),
                "rho": rho,
                "rho_lo": float(lo),
                "rho_hi": float(hi),
                # undefined if either model's word effects aren't reproducible at all
                "corrected": rho / np.sqrt(reliability[a] * reliability[b])
                if min(reliability[a], reliability[b]) > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
