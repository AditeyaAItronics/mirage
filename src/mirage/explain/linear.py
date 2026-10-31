"""Explanations read directly off linear and Naive Bayes models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mirage.models.base import SklearnClassifier


def _vec_and_clf(clf: SklearnClassifier):
    steps = clf.pipeline.named_steps
    return steps["vec"], steps["clf"]


def _top(names: np.ndarray, scores: np.ndarray, k: int, model: str) -> pd.DataFrame:
    order = np.argsort(scores)
    rows = [(names[i], scores[i], "positive") for i in order[::-1][:k]]
    rows += [(names[i], scores[i], "negative") for i in order[:k]]
    return pd.DataFrame(rows, columns=["feature", "score", "direction"]).assign(model=model)


def top_lr_weights(clf: SklearnClassifier, k: int = 25) -> pd.DataFrame:
    vec, lr = _vec_and_clf(clf)
    return _top(vec.get_feature_names_out(), lr.coef_[0], k, clf.name)


def top_mnb_ratios(clf: SklearnClassifier, k: int = 25) -> pd.DataFrame:
    """log P(w | pos) - log P(w | neg): the Naive Bayes analogue of a weight."""
    vec, nb = _vec_and_clf(clf)
    pos, neg = list(nb.classes_).index(1), list(nb.classes_).index(0)
    ratio = nb.feature_log_prob_[pos] - nb.feature_log_prob_[neg]
    return _top(vec.get_feature_names_out(), ratio, k, clf.name)


def lr_word_contributions(clf: SklearnClassifier, text: str) -> dict[str, float]:
    """Per-word contribution (tf-idf x coef) to the positive logit for one text.

    A bigram's contribution is split equally between its two words, so words are comparable with
    DistilBERT's word-level IG attributions in the deletion check.
    """
    vec, lr = _vec_and_clf(clf)
    x = vec.transform([text]).tocoo()
    names = vec.get_feature_names_out()
    out: dict[str, float] = {}
    for j, v in zip(x.col, x.data, strict=True):
        words = names[j].split()
        for w in words:
            out[w] = out.get(w, 0.0) + float(v * lr.coef_[0][j]) / len(words)
    return out
