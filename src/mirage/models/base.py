"""The single interface every classifier implements, plus save/load dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import numpy as np

CLASSICAL_MODELS = ("logreg", "dtree", "rf", "mnb", "gnb")
MODEL_NAMES = (*CLASSICAL_MODELS, "distilbert")


@runtime_checkable
class Classifier(Protocol):
    """Raw text in, class probabilities out. Normalization happens inside the model."""

    name: str

    def predict_proba(self, texts: list[str], long_text: bool = False) -> np.ndarray:
        """Return an (n, 2) array: column 0 = P(negative), column 1 = P(positive).

        `long_text` lets models with a token budget (DistilBERT) use a longer limit for IMDb.
        """
        ...

    def save(self, path: Path) -> None: ...


class SklearnClassifier:
    """Wraps a fitted sklearn Pipeline whose first step takes raw text."""

    def __init__(self, name: str, pipeline) -> None:
        self.name = name
        self.pipeline = pipeline

    def predict_proba(self, texts: list[str], long_text: bool = False) -> np.ndarray:
        proba = self.pipeline.predict_proba(list(texts))
        classes = list(self.pipeline.classes_)
        if classes != [0, 1]:  # guarantee column order regardless of sklearn internals
            proba = proba[:, [classes.index(0), classes.index(1)]]
        return proba

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "model.joblib")

    @classmethod
    def load(cls, path: Path) -> SklearnClassifier:
        return joblib.load(path / "model.joblib")


def load_classifier(path: Path) -> Classifier:
    """Load either kind of saved model from its directory."""
    if (path / "model.joblib").exists():
        return SklearnClassifier.load(path)
    if (path / "config.json").exists():
        from mirage.config import load_config
        from mirage.models.distilbert import DistilBertClassifier  # lazy: needs torch

        cfg = load_config("distilbert")
        return DistilBertClassifier.load(path, cfg["max_length"], cfg["imdb_max_length"])
    raise FileNotFoundError(f"no saved model in {path}")
