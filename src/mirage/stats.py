"""Uncertainty: bootstrap confidence intervals and McNemar's test.

Every bootstrap here resamples *items* (indices) and evaluates a statistic on the resampled
indices. Paired comparisons (model A vs B, base vs negmark) pass aligned arrays so both sides
are resampled together - this is what makes the gap CIs paired (majordecisions.md D12).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def bootstrap(
    stat: Callable[[np.ndarray], float],
    n_items: int,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap. `stat(idx)` computes the statistic on item indices `idx`."""
    rng = np.random.default_rng(seed)
    point = stat(np.arange(n_items))
    draws = np.array([stat(rng.integers(0, n_items, n_items)) for _ in range(n_resamples)])
    draws = draws[~np.isnan(draws)]
    if len(draws) == 0:
        return {"estimate": point, "ci_low": np.nan, "ci_high": np.nan, "n": n_items}
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(point), "ci_low": float(lo), "ci_high": float(hi), "n": n_items}


def rate_ci(flags: np.ndarray, **kw) -> dict:
    """CI for a proportion (e.g. an error or failure rate) from a boolean array."""
    flags = np.asarray(flags, dtype=float)
    return bootstrap(lambda i: flags[i].mean(), len(flags), **kw)


def paired_diff_ci(a: np.ndarray, b: np.ndarray, **kw) -> dict:
    """CI for mean(a) - mean(b), with a and b aligned item-by-item."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must be aligned")
    return bootstrap(lambda i: a[i].mean() - b[i].mean(), len(a), **kw)


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

    return bootstrap(stat, len(e0), **kw)


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
