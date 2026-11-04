"""H8 (D52): extra seeds and variants never change a headline table."""

import numpy as np
import pandas as pd
import pytest

from mirage import evaluate


def _preds(model, variant, seed, acc, n=40, dataset="sst2_val"):
    correct = np.arange(n) < round(acc * n)
    return pd.DataFrame(
        {
            "model": model,
            "variant": variant,
            "seed": seed,
            "dataset": dataset,
            "item_id": [f"{dataset}-{i}" for i in range(n)],
            "label": 1,
            "p_pos": np.where(correct, 0.9, 0.1),
            "pred": correct.astype(int),
            "correct": correct,
        }
    )


@pytest.fixture
def headline():
    rows = []
    for ds in ("sst2_val", "sst2_test"):
        rows += [
            _preds("rf", "base", 42, 0.75, dataset=ds),
            _preds("logreg", "base", 42, 0.8, dataset=ds),
        ]
        rows += [
            _preds("distilbert", "base", s, a, dataset=ds)
            for s, a in ((13, 0.9), (42, 0.925), (87, 0.95))
        ]
    return pd.concat(rows, ignore_index=True)


def test_extra_seeds_and_variants_leave_main_table_unchanged(headline):
    extra = pd.concat(
        [
            headline,
            _preds("rf", "base", 13, 0.2),  # a robustness seed of the forest
            _preds("rf", "base", 13, 0.2, dataset="sst2_test"),
            _preds("distilbert", "base", 7, 0.1),  # a DistilBERT robustness seed
            _preds("distilbert", "base", 7, 0.1, dataset="sst2_test"),
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(evaluate.main_table(headline), evaluate.main_table(extra))


def test_median_seed_ignores_robustness_seeds_and_other_variants(headline):
    assert evaluate.median_seed(headline) == 42
    extra = pd.concat(
        [headline, _preds("distilbert", "base", 7, 0.1), _preds("distilbert", "tuned", 13, 0.99)],
        ignore_index=True,
    )
    assert evaluate.median_seed(extra) == 42
    assert evaluate.median_seed(extra, variant="tuned") == 13


def test_canonical_keeps_one_seed_per_model_and_variant(headline):
    extra = pd.concat(
        [headline, _preds("rf", "base", 13, 0.2), _preds("distilbert", "tuned", 13, 0.99)],
        ignore_index=True,
    )
    c = evaluate.canonical(extra)
    seeds = c.groupby(["model", "variant"])["seed"].unique().map(list).to_dict()
    assert seeds == {
        ("distilbert", "base"): [42],
        ("distilbert", "tuned"): [13],
        ("logreg", "base"): [42],
        ("rf", "base"): [42],
    }


def test_single_seed_data_passes_through():
    p = _preds("rf", "base", 0, 0.7)  # fixtures with seed 0: no headline seed present
    assert len(evaluate.headline_pool(p)) == len(p)
