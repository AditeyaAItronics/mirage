"""Sentence embeddings with an on-disk cache keyed by item_id."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mirage import paths
from mirage.text import sentences


def encode(texts: list[str], model_name: str, batch_size: int = 128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # lazy: optional dependency

    model = SentenceTransformer(model_name)
    # Raw text on purpose: MiniLM has its own tokenizer and was trained on normal English.
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)


def cached_embeddings(
    df: pd.DataFrame, name: str, model_name: str, batch_size: int = 128
) -> np.ndarray:
    """Embed df["text"], reusing artifacts/emb/<name>.npz when its item_ids match."""
    path: Path = paths.EMBEDDINGS / f"{name}.npz"
    ids = df["item_id"].to_numpy().astype(str)
    if path.exists():
        cache = np.load(path, allow_pickle=False)
        if np.array_equal(cache["ids"], ids):
            return cache["emb"]
    emb = encode(df["text"].tolist(), model_name, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=ids, emb=emb)
    return emb


def split_sentences(df: pd.DataFrame, min_words: int = 3) -> pd.DataFrame:
    """One row per sentence (>= min_words words), keeping its review's item_id in `owner`."""
    rows = []
    for item_id, text in zip(df["item_id"], df["text"], strict=True):
        for j, s in enumerate(
            sentences(text, min_words) or [text]
        ):  # a review with no full sentence counts as one
            rows.append({"item_id": f"{item_id}#{j}", "owner": item_id, "text": s})
    return pd.DataFrame(rows)
