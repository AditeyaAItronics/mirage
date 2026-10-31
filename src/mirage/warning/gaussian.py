"""Gaussian density model on sentence embeddings (Lee et al., 2018).

Fits a mean and a Ledoit-Wolf shrinkage covariance to the SST-2 training embeddings and scores
inputs by Mahalanobis distance. Shrinkage keeps the 384x384 covariance well-conditioned.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


class GaussianDensity:
    def __init__(self, class_conditional: bool = False) -> None:
        self.class_conditional = class_conditional

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> GaussianDensity:
        if self.class_conditional:
            if y is None:
                raise ValueError("class_conditional=True needs labels")
            # Shared covariance on class-centred data, one mean per class (Lee et al.).
            self.means_ = np.stack([X[y == c].mean(0) for c in np.unique(y)])
            centred = X - self.means_[np.searchsorted(np.unique(y), y)]
            lw = LedoitWolf(assume_centered=True).fit(centred)
        else:
            lw = LedoitWolf().fit(X)
            self.means_ = lw.location_[None, :]
        self.precision_ = lw.precision_
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Mahalanobis distance (min over class means if class-conditional). Higher = less familiar."""
        d = []
        for mu in self.means_:
            diff = X - mu
            d.append(np.sqrt(np.einsum("ij,jk,ik->i", diff, self.precision_, diff)))
        return np.min(np.stack(d), axis=0)
