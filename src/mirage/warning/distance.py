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


def sentence_distances(scorer, df: pd.DataFrame, name: str, cfg: dict) -> pd.DataFrame:
    """One row per sentence of each text: its owner's item_id and its distance."""
    sents = split_sentences(df)
    emb = cached_embeddings(
        sents, f"{name}_sentences", cfg["embedding_model"], cfg["embedding_batch_size"]
    )
    return pd.DataFrame({"owner": sents["owner"].to_numpy(), "distance": scorer.score(emb)})


def text_distance(scorer, df: pd.DataFrame, name: str, cfg: dict) -> pd.Series:
    """Distance per row of `df`, indexed by item_id."""
    unit = cfg.get("unit", "review")
    if unit == "review":
        emb = cached_embeddings(df, name, cfg["embedding_model"], cfg["embedding_batch_size"])
        return pd.Series(scorer.score(emb), index=df["item_id"].to_numpy())
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}; expected one of {UNITS}")
    sd = sentence_distances(scorer, df, name, cfg)
    agg = sd.groupby("owner")["distance"].agg("max" if unit == "sentence_max" else "mean")
    return agg.reindex(df["item_id"].to_numpy()).astype(float)


def one_sentence_each(sd: pd.DataFrame, seed: int) -> pd.Series:
    """One randomly chosen sentence's distance per text: a length-matched sample (F4)."""
    pick = sd.sample(frac=1.0, random_state=seed).drop_duplicates("owner")
    return pick.set_index("owner")["distance"]


def length_adjusted(distance: np.ndarray, n_sentences: np.ndarray) -> np.ndarray:
    """Distance minus what log(number of sentences) predicts (OLS residual, no labels used).

    The max over more sentences is larger by chance; the residual keeps only the part of the
    distance that length doesn't explain (shortfalls.md F4, D30).
    """
    x = np.log1p(np.asarray(n_sentences, dtype=float))
    d = np.asarray(distance, dtype=float)
    slope, intercept = np.polyfit(x, d, 1)
    return d - (intercept + slope * x)


def fit_distance(train: pd.DataFrame, cfg: dict):
    emb = cached_embeddings(
        train, "sst2_train", cfg["embedding_model"], cfg["embedding_batch_size"]
    )
    return make_distance(cfg).fit(emb, np.asarray(train["label"]))
