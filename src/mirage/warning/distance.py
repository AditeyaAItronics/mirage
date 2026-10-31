"""Choose and apply the early-warning distance (majordecisions.md D25).

`distance` picks how unfamiliar an embedding is; `unit` picks what gets embedded:
- review:        the whole text as one embedding
- sentence_mean: each sentence embedded separately, distances averaged per text
- sentence_max:  the same, but the text's most unusual sentence counts
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mirage.warning.embed import cached_embeddings, split_sentences
from mirage.warning.gaussian import GaussianDensity
from mirage.warning.knn import KnnDistance

DISTANCES = ("gaussian", "class_gaussian", "knn_cosine")
UNITS = ("review", "sentence_mean", "sentence_max")


def make_distance(cfg: dict):
    kind = cfg["distance"]
    if kind == "gaussian":
        return GaussianDensity()
    if kind == "class_gaussian":
        return GaussianDensity(class_conditional=True)
    if kind == "knn_cosine":
        return KnnDistance(k=cfg.get("knn_k", 10))
    raise ValueError(f"unknown distance {kind!r}; expected one of {DISTANCES}")


def text_distance(scorer, df: pd.DataFrame, name: str, cfg: dict) -> pd.Series:
    """Distance per row of `df`, indexed by item_id."""
    model, batch = cfg["embedding_model"], cfg["embedding_batch_size"]
    unit = cfg.get("unit", "review")
    if unit == "review":
        emb = cached_embeddings(df, name, model, batch)
        return pd.Series(scorer.score(emb), index=df["item_id"].to_numpy())
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}; expected one of {UNITS}")
    sents = split_sentences(df)
    d = pd.Series(scorer.score(cached_embeddings(sents, f"{name}_sentences", model, batch)))
    agg = d.groupby(sents["owner"].to_numpy()).agg("max" if unit == "sentence_max" else "mean")
    return agg.reindex(df["item_id"].to_numpy()).astype(float)


def fit_distance(train: pd.DataFrame, cfg: dict):
    emb = cached_embeddings(
        train, "sst2_train", cfg["embedding_model"], cfg["embedding_batch_size"]
    )
    return make_distance(cfg).fit(emb, np.asarray(train["label"]))
