"""Faithfulness check: does deleting the top-attributed words change the prediction more than
deleting the same number of random words?
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _drop(words: list[str], idx) -> str:
    drop = set(int(i) for i in idx)
    return " ".join(w for i, w in enumerate(words) if i not in drop)


def deletion_check(
    predict_proba: Callable[[list[str]], np.ndarray],
    words_list: list[list[str]],
    attrs_list: list[list[float]],
    ks=(1, 2, 3),
    n_random: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """For each text and k: drop in P(original predicted class) after removing the k words with the
    highest attribution toward that class, vs the mean drop over `n_random` random k-word deletions.

    Words are normalized tokens; re-joining them is safe because normalize() is idempotent.
    """
    rng = np.random.default_rng(seed)
    originals = predict_proba([" ".join(w) for w in words_list])
    rows = []
    for t, (words, attrs) in enumerate(zip(words_list, attrs_list, strict=True)):
        cls = int(originals[t].argmax())
        p0 = originals[t, cls]
        order = np.argsort(attrs)[::-1]  # attributions are already toward the predicted class
        for k in ks:
            if k >= len(words):
                continue
            top_text = _drop(words, order[:k])
            rand_texts = [
                _drop(words, rng.choice(len(words), k, replace=False)) for _ in range(n_random)
            ]
            probs = predict_proba([top_text, *rand_texts])[:, cls]
            rows.append(
                {
                    "text_idx": t,
                    "k": k,
                    "drop_top": float(p0 - probs[0]),
                    "drop_random": float(p0 - probs[1:].mean()),
                }
            )
    return pd.DataFrame(rows)
