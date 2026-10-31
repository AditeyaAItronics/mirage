"""k-nearest-neighbour cosine distance to the training embeddings (Sun et al., 2022).

Makes no Gaussian assumption: an input is unfamiliar if even its closest training phrases are far
away. Embeddings are L2-normalised, so cosine similarity is a dot product.
"""

from __future__ import annotations

import numpy as np


def _l2(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


class KnnDistance:
    def __init__(self, k: int = 10, chunk: int = 512) -> None:
        self.k = k
        self.chunk = chunk

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> KnnDistance:
        self.train_ = _l2(X).astype(np.float32)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """1 - mean cosine similarity to the k nearest training items. Higher = less familiar."""
        X = _l2(X).astype(np.float32)
        k = min(self.k, len(self.train_))
        out = []
        for i in range(0, len(X), self.chunk):
            sim = X[i : i + self.chunk] @ self.train_.T
            out.append(1 - np.partition(sim, -k, axis=1)[:, -k:].mean(1))
        return np.concatenate(out) if out else np.zeros(0)
