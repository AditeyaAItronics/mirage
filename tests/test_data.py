import pandas as pd

from mirage.data import build_splits, negation_subset, split_heldout

CFG = {"seed": 42, "heldout_frac": 0.05, "imdb_sample_size": 100}


def test_heldout_is_stratified_and_disjoint(sst):
    df = sst(1000)
    df["uid"] = range(len(df))
    tr, ho = split_heldout(df, 0.05, 42)
    assert len(ho) == 50 and len(tr) == 950
    assert set(tr["uid"]).isdisjoint(ho["uid"])
    assert abs(ho["label"].mean() - df["label"].mean()) < 0.03


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
