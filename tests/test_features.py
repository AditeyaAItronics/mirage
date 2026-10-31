"""Regression guard for the token-pattern trap (majordecisions.md D8)."""

import pickle

from mirage.features import build_counts, build_tfidf

CFG = {"ngram_range": [1, 2], "min_df": 1, "sublinear_tf": True}
DOCS = ["It isn't good.", "Not good at all!", "A good film", "Never boring"]


def test_vocabulary_keeps_negation_tokens():
    vocab = build_tfidf(CFG).fit(DOCS).vocabulary_
    for tok in ["n't", "not", "never", "not good", "n't good", "a", "!"]:
        assert tok in vocab, tok


def test_negmark_features_are_distinct_and_uppercase():
    vocab = build_counts(CFG, negmark=True).fit(DOCS).vocabulary_
    assert "NOT_good" in vocab
    assert "good" in vocab  # un-negated "good" in "a good film"
    assert "not_good" not in vocab


def test_vectorizer_pickles():
    vec = build_tfidf(CFG, negmark=True).fit(DOCS)
    clone = pickle.loads(pickle.dumps(vec))
    assert (clone.transform(DOCS) != vec.transform(DOCS)).nnz == 0
