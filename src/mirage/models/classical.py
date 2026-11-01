"""The five classical models and their held-out grid search.

Models are trained on the 95% split and selected on the 5% held-out split. They are *not* refit on
train + held-out, so classical models and DistilBERT see exactly the same training data
(majordecisions.md D10).
"""

from __future__ import annotations

import time

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from mirage.features import build_counts, build_tfidf
from mirage.models.base import CLASSICAL_MODELS, SklearnClassifier

# Variant name -> what the vectorizer's Preprocessor does with negation (D31).
VARIANT_NEGMARK: dict[str, bool | str] = {
    "base": False,
    "negmark": True,  # Pang et al. scope, up to punctuation
    "negmark_w3": "window3",  # 3-token scope with fixed-phrase exceptions
}


def build_pipeline(name: str, cfg: dict, negmark: bool | str, params: dict) -> Pipeline:
    seed = cfg["seed"]
    vec_cfg = cfg["vectorizer"]
    if name == "logreg":
        # L2 is the default; passing penalty= is deprecated from scikit-learn 1.8.
        clf = LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed)
        steps = [("vec", build_tfidf(vec_cfg, negmark)), ("clf", clf)]
    elif name == "dtree":
        steps = [
            ("vec", build_tfidf(vec_cfg, negmark)),
            ("clf", DecisionTreeClassifier(criterion="gini", random_state=seed)),
        ]
    elif name == "rf":
        rf = RandomForestClassifier(
            n_estimators=cfg["models"]["rf"]["n_estimators"],
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
        steps = [("vec", build_tfidf(vec_cfg, negmark)), ("clf", rf)]
    elif name == "mnb":
        steps = [("vec", build_counts(vec_cfg, negmark)), ("clf", MultinomialNB())]
    elif name == "gnb":
        svd = TruncatedSVD(n_components=cfg["models"]["gnb"]["svd_components"], random_state=seed)
        steps = [("vec", build_tfidf(vec_cfg, negmark)), ("svd", svd), ("clf", GaussianNB())]
    else:
        raise ValueError(f"unknown classical model {name!r}")
    pipe = Pipeline(steps)
    pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
    return pipe


def tune(
    name: str,
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    cfg: dict,
    variant: str = "base",
) -> tuple[SklearnClassifier, pd.DataFrame]:
    """Fit every grid point on train, score on held-out, return the best model and the log."""
    grid = cfg["models"][name].get("grid") or {}
    rows, best, best_acc, best_idx = [], None, -1.0, -1
    for i, params in enumerate(ParameterGrid(grid)):
        pipe = build_pipeline(name, cfg, VARIANT_NEGMARK[variant], params)
        t0 = time.perf_counter()
        pipe.fit(train["text"].tolist(), train["label"].to_numpy())
        fit_s = time.perf_counter() - t0
        acc = float((pipe.predict(heldout["text"].tolist()) == heldout["label"].to_numpy()).mean())
        rows.append(
            {
                "model": name,
                "variant": variant,
                "params": str(params),
                "heldout_acc": round(acc, 4),
                "fit_seconds": round(fit_s, 1),
            }
        )
        if acc > best_acc:  # strict: ties keep the earlier (simpler-first) grid point
            best, best_acc, best_idx = pipe, acc, i
    log = pd.DataFrame(rows)
    log["selected"] = log.index == best_idx
    return SklearnClassifier(name, best), log


def train_all(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    cfg: dict,
    variant: str = "base",
    models: tuple[str, ...] = CLASSICAL_MODELS,
) -> tuple[dict[str, SklearnClassifier], pd.DataFrame]:
    fitted, logs = {}, []
    for name in models:
        clf, log = tune(name, train, heldout, cfg, variant)
        fitted[name] = clf
        logs.append(log)
    return fitted, pd.concat(logs, ignore_index=True)
