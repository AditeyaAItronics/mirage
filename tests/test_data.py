import pandas as pd

from mirage.data import (
    build_splits,
    drop_seen,
    head_tail,
    heldout_mask,
    negation_subset,
    split_heldout,
)

CFG = {
    "seed": 42,
    "heldout_frac": 0.05,
    "imdb_sample_size": 100,
    "imdb_head_words": 2,
    "imdb_tail_words": 1,
}


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
    test = sst(120, 3).assign(text=lambda d: d["text"] + " again")  # unseen sentences
    test = pd.concat([test, pd.DataFrame({"text": ["A great  film movie"], "label": [1]})])
    splits = build_splits(sst(400), sst(120, 1), test, sst(300, 2), CFG)
    assert len(splits["imdb"]) == 100
    for name, df in splits.items():
        assert df["item_id"].is_unique, name
        assert set(df["label"]) <= {0, 1}
    # the negation subset keeps validation ids so items can be joined back
    ids = set(splits["sst2_val"]["item_id"]) | set(splits["sst2_test"]["item_id"])
    assert set(splits["sst2_neg"]["item_id"]) <= ids
    assert {i.split("-")[0] for i in splits["sst2_neg"]["item_id"]} == {"sst2_val", "sst2_test"}
    # test sentences already seen in train/val are dropped (case- and space-insensitive)
    seen = {
        " ".join(t.lower().split())
        for t in pd.concat(
            [
                splits["sst2_train"]["text"],
                splits["sst2_heldout"]["text"],
                splits["sst2_val"]["text"],
            ]
        )
    }
    assert not any(" ".join(t.lower().split()) in seen for t in splits["sst2_test"]["text"])
    # every IMDb review was capped to head + tail words, and the cut is recorded
    assert splits["imdb"]["text"].str.split().str.len().max() <= 3
    assert splits["imdb"]["capped"].any()
    assert len(splits["sst2_neg"]) > 0
    # imdb_edges pairs item-for-item with imdb (B4)
    assert splits["imdb_edges"]["item_id"].tolist() == splits["imdb"]["item_id"].tolist()


def test_head_tail_keeps_the_ending():
    assert head_tail("a b c d e f", 2, 1) == "a b f"
    assert head_tail("a b c", 2, 1) == "a b c"


def test_drop_seen_ignores_case_and_spacing():
    df = pd.DataFrame({"text": ["Great  Film", "new one"], "label": [1, 0]})
    assert drop_seen(df, pd.Series(["great film"]))["text"].tolist() == ["new one"]


def test_imdb_edges_keep_first_and_last_sentence():
    from mirage.data import build_imdb_edges

    sample = pd.DataFrame(
        {
            "item_id": ["imdb-0", "imdb-1"],
            "text": [
                "I went in with hopes. The middle dragged on. In the end it was a fine film.",
                "Short and sweet review here",
            ],
            "label": [1, 0],
        }
    )
    e = build_imdb_edges(sample)
    assert e["text"].tolist() == [
        "I went in with hopes. In the end it was a fine film.",
        "Short and sweet review here",  # no sentence break -> whole text
    ]
    assert e["item_id"].tolist() == ["imdb-0", "imdb-1"] and e["label"].tolist() == [1, 0]
