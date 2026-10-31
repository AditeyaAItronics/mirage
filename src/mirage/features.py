"""Vectorizers shared by the classical models.

The Preprocessor (normalize [+ negation_mark]) runs *inside* the vectorizer, so fitted pipelines
take raw text. The default sklearn token_pattern is replaced: it drops "n't" and punctuation, which
would silently delete the negation features this project measures (majordecisions.md D8).
"""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from mirage.text import Preprocessor, tokens


def _common(cfg: dict, negmark: bool) -> dict:
    return {
        "preprocessor": Preprocessor(negmark=negmark),
        "tokenizer": tokens,
        "token_pattern": None,
        "lowercase": False,  # normalize() already lowercased; keeps the NOT_ prefix distinct
        "stop_words": None,  # stop-word lists contain "not"
        "ngram_range": tuple(cfg["ngram_range"]),
        "min_df": cfg["min_df"],
    }


def build_tfidf(cfg: dict, negmark: bool = False) -> TfidfVectorizer:
    return TfidfVectorizer(sublinear_tf=cfg["sublinear_tf"], **_common(cfg, negmark))


def build_counts(cfg: dict, negmark: bool = False) -> CountVectorizer:
    return CountVectorizer(**_common(cfg, negmark))
