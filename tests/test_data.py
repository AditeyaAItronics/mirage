import pandas as pd

from mirage.data import build_splits, heldout_mask, negation_subset, split_heldout

CFG = {"seed": 42, "heldout_frac": 0.05, "imdb_sample_size": 100}


def test_heldout_is_disjoint(sst):
    df = sst(1000)
    df["uid"] = range(len(df))
    tr, ho = split_heldout(df, 0.05, 42)
    assert len(ho) >= 50 and len(tr) + len(ho) == 1000
    assert set(tr["uid"]).isdisjoint(ho["uid"])
    # identical texts share a fate, so a held-out text never also appears in training
    assert set(tr["text"]).isdisjoint(ho["text"])


def test_heldout_keeps_sentences_with_their_phrases():
    """SST-2 train has a row per sub-phrase: they must follow their sentence (shortfalls.md B1)."""
    s1 = ["the plot is not good", "the plot", "not good", "is not good"]
    s2 = ["a truly great cast", "truly great", "great cast"]
    s3 = ["great fun , not boring", "not boring", "great fun"]
    shared = ["great"]  # inside s2 and s3: held out only if both sentences are
    texts = s1 + s2 + s3 + shared
    df = pd.DataFrame({"text": texts, "label": [0] * 4 + [1] * 3 + [1] * 3 + [1]})
    for seed in range(20):
        mask = heldout_mask(df["text"].tolist(), 0.3, seed)
        held = {t for t, m in zip(texts, mask, strict=True) if m}
        train = set(texts) - held
        for sent in (s1, s2, s3):
            assert set(sent) <= held or not set(sent) & held  # all-or-nothing per sentence
        assert ("great" in held) == (set(s2) | set(s3) <= held)
        assert not any(f" {h} " in f" {t} " for h in held for t in train)  # no leakage


def test_negation_subset_matches_normalized_text():
    val = pd.DataFrame(
        {"text": ["It isn't good", "a notable film", "never again", "great"], "label": [0, 1, 0, 1]}
    )
    assert negation_subset(val)["text"].tolist() == ["It isn't good", "never again"]


def test_build_splits_ids_and_sizes(sst):
    splits = build_splits(sst(400), sst(120, 1), sst(300, 2), CFG)
    assert len(splits["imdb"]) == 100
    for name, df in splits.items():
        assert df["item_id"].is_unique, name
        assert set(df["label"]) <= {0, 1}
    # the negation subset keeps validation ids so items can be joined back
    assert set(splits["sst2_neg"]["item_id"]) <= set(splits["sst2_val"]["item_id"])
    assert len(splits["sst2_neg"]) > 0
