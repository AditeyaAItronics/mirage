"""Point metrics computed from prediction frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, average="macro", zero_division=0))


def confidence(p_pos: np.ndarray) -> np.ndarray:
    """Max predicted probability for a binary classifier."""
    p = np.asarray(p_pos, dtype=float)
    return np.maximum(p, 1 - p)


def reliability_bins(p_pos: np.ndarray, y: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Equal-width bins on confidence. Returns per-bin count, mean confidence and accuracy."""
    p = np.asarray(p_pos, dtype=float)
    conf = confidence(p)
    correct = ((p >= 0.5).astype(int) == np.asarray(y)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-inclusive bins, so confidence 1.0 lands in the last bin
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append(
            {
                "bin": b,
                "lo": edges[b],
                "hi": edges[b + 1],
                "count": int(m.sum()),
                "confidence": float(conf[m].mean()) if m.any() else np.nan,
                "accuracy": float(correct[m].mean()) if m.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ece(p_pos: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error (Guo et al., 2017), equal-width bins on max probability."""
    bins = reliability_bins(p_pos, y, n_bins)
    bins = bins[bins["count"] > 0]
    weights = bins["count"] / bins["count"].sum()
    return float((weights * (bins["accuracy"] - bins["confidence"]).abs()).sum())


def auroc(target: np.ndarray, score: np.ndarray) -> float:
    """AUROC, or NaN when only one class is present (e.g. a bootstrap resample with no errors)."""
    target = np.asarray(target)
    if target.min() == target.max():
        return float("nan")
    return float(roc_auc_score(target, score))


def summarize(preds: pd.DataFrame, n_bins: int = 15) -> dict:
    y, pred, p = preds["label"].to_numpy(), preds["pred"].to_numpy(), preds["p_pos"].to_numpy()
    return {
        "n": len(preds),
        "accuracy": accuracy(y, pred),
        "macro_f1": macro_f1(y, pred),
        "error": 1 - accuracy(y, pred),
        "ece": ece(p, y, n_bins),
    }
