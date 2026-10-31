"""H4: does Gaussian distance predict a model's IMDb errors better than its own confidence?

Error predictors per classifier (target = 1 if the model is wrong):
  distance     - how unfamiliar the review's embedding is (configs/warning.yaml, D25)
  low_conf     - 1 - max predicted probability (Hendrycks & Gimpel, 2017)
  combined     - logistic regression on [distance, max prob], out-of-fold over 5 folds
  length       - number of words: a baseline, since per-sentence max distance grows with length
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from mirage.metrics import auroc, confidence
from mirage.stats import bootstrap


def oof_combined(
    distance: np.ndarray, conf: np.ndarray, target: np.ndarray, folds: int, seed: int
) -> np.ndarray:
    """Out-of-fold scores: each review is scored by a combiner that never saw it.

    Confidence enters as log-odds, and both features are rank-transformed to a normal shape:
    Naive Bayes probabilities on long IMDb reviews pile up at 1.0 (half of them exactly 1.0 in
    float64), and a linear fit on raw or standardised log-odds let a useless feature outweigh
    confidence. With a pure-noise distance, MNB's combined AUROC was 0.637 against 0.688 for
    confidence alone; with this transform it is 0.676 (majordecisions.md D26).
    """
    c = np.clip(conf, 1e-15, 1 - 1e-15)
    X = np.c_[distance, np.log(c / (1 - c))]
    out = np.full(len(target), np.nan)
    n_splits = min(folds, int(target.sum()), int((1 - target).sum()))
    if n_splits < 2:
        return out
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, target):
        ranks = QuantileTransformer(
            n_quantiles=min(1000, len(tr)), output_distribution="normal", random_state=seed
        )
        comb = make_pipeline(ranks, LogisticRegression()).fit(X[tr], target[tr])
        out[te] = comb.predict_proba(X[te])[:, 1]
    return out


def compare_predictors(
    preds: pd.DataFrame,
    distance: np.ndarray,
    folds: int = 5,
    seed: int = 42,
    n_resamples: int = 1000,
    length: np.ndarray | None = None,
) -> pd.DataFrame:
    """`preds` = one model's IMDb predictions, row-aligned with `distance` (and `length`)."""
    target = (~preds["correct"].astype(bool)).to_numpy().astype(int)
    conf = confidence(preds["p_pos"].to_numpy())
    scores = {
        "distance": distance,
        "low_conf": 1 - conf,
        "combined": oof_combined(distance, conf, target, folds, seed),
    }
    if length is not None:
        scores["length"] = np.asarray(length, dtype=float)
    rows = []
    for name, s in scores.items():
        ci = bootstrap(lambda i, s=s: auroc(target[i], s[i]), len(target), n_resamples, seed)
        rows.append(
            {
                "predictor": name,
                "auroc": ci["estimate"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "n": len(target),
                "n_errors": int(target.sum()),
            }
        )
    return pd.DataFrame(rows)


def ood_sanity(dist_in: np.ndarray, dist_out: np.ndarray) -> float:
    """AUROC for telling SST-2 val (in) from IMDb (out) by distance; expected close to 1."""
    target = np.r_[np.zeros(len(dist_in)), np.ones(len(dist_out))]
    return auroc(target, np.r_[dist_in, dist_out])


def error_by_decile(preds: pd.DataFrame, score: np.ndarray, label: str) -> pd.DataFrame:
    """Error rate per decile of a score - for the 'error vs distance' figure."""
    err = (~preds["correct"].astype(bool)).to_numpy()
    dec = pd.qcut(pd.Series(score).rank(method="first"), 10, labels=False)
    return (
        pd.DataFrame({"decile": dec, "error": err})
        .groupby("decile")["error"]
        .mean()
        .reset_index()
        .assign(score=label)
    )
