"""Uncertainty: bootstrap confidence intervals and McNemar's test.

Every bootstrap here resamples *items* (indices) and evaluates a statistic on the resampled
indices. Paired comparisons (model A vs B, base vs negmark) pass aligned arrays so both sides
are resampled together - this is what makes the gap CIs paired (majordecisions.md D12).

Comparisons also get a two-sided bootstrap p-value against a null value, and `holm` adjusts a
family of p-values for multiple testing (majordecisions.md D28).

Clustered items (G7, D52): behavioral cases built on the same adjective, and IMDb reviews of the
same film, pass or fail together. Every routine takes an optional `clusters` array (one label per
item); the bootstrap then resamples whole clusters, so the interval reflects how many independent
units there really are. Without `clusters` the draws are exactly the item bootstrap of before.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from mirage.metrics import auroc


def _cluster_groups(clusters, n_items: int) -> list[np.ndarray] | None:
    """Item indices per cluster, or None for the plain item bootstrap."""
    if clusters is None:
        return None
    labels = pd.Series(np.asarray(clusters, dtype=object))
    if len(labels) != n_items:
        raise ValueError("clusters must have one label per item")
    codes = pd.factorize(labels)[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    return np.split(order, bounds)


def _draw(rng: np.random.Generator, n_items: int, groups) -> np.ndarray:
    """One resample of item indices: items with replacement, or whole clusters with replacement."""
    if groups is None:
        return rng.integers(0, n_items, n_items)
    picks = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[k] for k in picks])


def bootstrap(
    stat: Callable[[np.ndarray], float],
    n_items: int,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    null: float | None = None,
    clusters=None,
) -> dict:
    """Percentile bootstrap. `stat(idx)` computes the statistic on item indices `idx`.

    With `null`, also returns a two-sided `p_value` from the bootstrap standard error
    (z = (estimate - null) / SE). Counting resamples past the null instead caps p at
    2 / (n_resamples + 1); after Holm over a family of 20 that floor is already 0.04, so nothing
    could be significant in a larger family however strong the effect (D28).

    With `clusters`, whole clusters are resampled (G7, D52); `n` stays the number of items.
    """
    rng = np.random.default_rng(seed)
    groups = _cluster_groups(clusters, n_items)
    point = stat(np.arange(n_items))
    draws = np.array([stat(_draw(rng, n_items, groups)) for _ in range(n_resamples)])
    return _summarize(point, draws, n_items, alpha, null)


def _summarize(point: float, draws: np.ndarray, n, alpha: float, null: float | None) -> dict:
    """Percentile CI and (with `null`) the SE-based two-sided p-value from bootstrap draws."""
    draws = draws[~np.isnan(draws)]
    out = {"estimate": float(point), "ci_low": np.nan, "ci_high": np.nan, "n": n}
    if null is not None:
        out["p_value"] = np.nan
    if len(draws) == 0:
        return out
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    out |= {"ci_low": float(lo), "ci_high": float(hi)}
    if null is not None:
        se = draws.std(ddof=1) if len(draws) > 1 else 0.0
        if se > 0:
            out["p_value"] = float(2 * norm.sf(abs(point - null) / se))
        else:  # every resample identical: no evidence either way unless it differs from null
            out["p_value"] = 1.0 if np.isclose(point, null) else 0.0
    return out


def diff_independent_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    clusters_a=None,
    clusters_b=None,
) -> dict:
    """CI and p-value for mean(a) - mean(b) when a and b are *different* items (H1, D47):
    each group is resampled on its own (by its own clusters, if given), so no pairing is
    assumed. `n` is (len(a), len(b))."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    ga, gb = _cluster_groups(clusters_a, len(a)), _cluster_groups(clusters_b, len(b))
    draws = np.array(
        [
            a[_draw(rng, len(a), ga)].mean() - b[_draw(rng, len(b), gb)].mean()
            for _ in range(n_resamples)
        ]
    )
    return _summarize(a.mean() - b.mean(), draws, (len(a), len(b)), alpha, null=0.0)


def rate_ci(flags: np.ndarray, **kw) -> dict:
    """CI for a proportion (e.g. an error or failure rate) from a boolean array."""
    flags = np.asarray(flags, dtype=float)
    return bootstrap(lambda i: flags[i].mean(), len(flags), **kw)


def paired_diff_ci(a: np.ndarray, b: np.ndarray, **kw) -> dict:
    """CI for mean(a) - mean(b), with a and b aligned item-by-item."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must be aligned")
    return bootstrap(lambda i: a[i].mean() - b[i].mean(), len(a), null=0.0, **kw)


def gap_closed_ci(err_base: np.ndarray, err_negmark: np.ndarray, err_ref: np.ndarray, **kw) -> dict:
    """H3: fraction of the (classical - DistilBERT) error gap removed by negation marking.

    closed = (gap_base - gap_negmark) / gap_base, with gap = mean(err_model) - mean(err_ref).
    NaN for resamples where the base gap is ~0 (the ratio is undefined there).
    """
    e0, e1, r = (np.asarray(x, dtype=float) for x in (err_base, err_negmark, err_ref))

    def stat(i):
        g0 = e0[i].mean() - r[i].mean()
        g1 = e1[i].mean() - r[i].mean()
        return (g0 - g1) / g0 if abs(g0) > 1e-9 else np.nan

    return bootstrap(stat, len(e0), null=0.0, **kw)


def paired_auroc_diff(
    target: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
    clusters=None,
) -> dict:
    """AUROC(s1) - AUROC(s2) on the same items, resampled together (H4: distance vs confidence)."""
    y, a, b = (np.asarray(x, dtype=float) for x in (target, s1, s2))
    ok = ~(np.isnan(a) | np.isnan(b))
    y, a, b = y[ok], a[ok], b[ok]
    c = None if clusters is None else np.asarray(clusters, dtype=object)[ok]
    return bootstrap(
        lambda i: auroc(y[i], a[i]) - auroc(y[i], b[i]),
        len(y),
        n_resamples,
        seed,
        null=0.0,
        clusters=c,
    )


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values for one family of tests; NaNs are left out and kept."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """McNemar on paired correctness. Exact binomial when discordant pairs < 25."""
    a, b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    table = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    discordant = table[0][1] + table[1][0]
    res = mcnemar(table, exact=discordant < 25, correction=True)
    return {
        "a_only_correct": table[0][1],
        "b_only_correct": table[1][0],
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "exact": discordant < 25,
    }
