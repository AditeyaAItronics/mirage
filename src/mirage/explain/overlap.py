"""H5: do model families rely on the same shortcut words?

Rankings are compared on *unigram* words only, with the NOT_ prefix kept (it is a different
feature). IG yields words, so restricting to unigrams keeps every model comparable.
"""

from __future__ import annotations

import itertools

import pandas as pd


def unigram_ranking(features: list[str]) -> list[str]:
    seen, out = set(), []
    for f in features:
        if " " not in f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def jaccard(a: list[str], b: list[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    return len(sa & sb) / len(sa | sb) if sa | sb else float("nan")


def rbo(a: list[str], b: list[str], p: float = 0.9, k: int | None = None) -> float:
    """Truncated rank-biased overlap (Webber et al., 2010): top-weighted, in [0, 1]."""
    k = k or min(len(a), len(b))
    score, sa, sb = 0.0, set(), set()
    for d in range(1, k + 1):
        sa.add(a[d - 1])
        sb.add(b[d - 1])
        score += p ** (d - 1) * len(sa & sb) / d
    return (1 - p) * score


def overlap_matrix(rankings: dict[str, list[str]], k: int = 50) -> pd.DataFrame:
    """Long table of pairwise Jaccard@k and RBO between every pair of models."""
    rows = []
    for a, b in itertools.combinations(sorted(rankings), 2):
        ra, rb = rankings[a], rankings[b]
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "jaccard": jaccard(ra, rb, k),
                "rbo": rbo(ra[:k], rb[:k]),
                "shared": ", ".join(sorted(set(ra[:k]) & set(rb[:k]))),
            }
        )
    return pd.DataFrame(rows)
