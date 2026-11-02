"""D47: the formal H1 test (negation / typo error vs non-negated SST-2 test sentences)."""

import numpy as np
import pandas as pd
import pytest

from mirage.evaluate import h1_test
from mirage.stats import diff_independent_ci


def test_diff_independent_ci():
    a = np.r_[np.ones(40), np.zeros(60)]  # 40% error
    b = np.r_[np.ones(10), np.zeros(190)]  # 5% error, a different group size
    r = diff_independent_ci(a, b, n_resamples=500)
    assert r["estimate"] == pytest.approx(0.35)
    assert r["ci_low"] < 0.35 < r["ci_high"] and r["ci_low"] > 0
    assert r["p_value"] < 0.001 and r["n"] == (100, 200)
    same = diff_independent_ci(a, a.copy(), n_resamples=500)
    assert same["estimate"] == 0 and same["p_value"] > 0.5


def _rows(dataset, ids, wrong, model="logreg"):
    return pd.DataFrame(
        {
            "model": model,
            "variant": "base",
            "seed": 42,
            "dataset": dataset,
            "item_id": ids,
            "correct": ~np.isin(ids, wrong),
        }
    )


def test_h1_compares_with_non_negated_test_sentences():
    test_ids = [f"sst2_test-{i}" for i in range(100)]
    neg_ids = test_ids[:20] + [f"sst2_val-{i}" for i in range(20)]  # 20 negated test sentences
    # every negated sentence is wrong; 4 of the 80 plain test sentences are wrong
    preds = pd.concat(
        [
            _rows("sst2_test", test_ids, neg_ids[:20] + test_ids[20:24]),
            _rows("sst2_neg", neg_ids, neg_ids),
        ]
    )
    r = h1_test(preds, tests=("sst2_neg",), n_resamples=300).iloc[0]
    assert r["n_plain"] == 80  # the negated test sentences are excluded from the comparison
    assert r["error_plain"] == pytest.approx(4 / 80)
    assert r["error_all_sst2_test"] == pytest.approx(24 / 100)
    assert r["diff"] == pytest.approx(1 - 4 / 80)
    assert bool(r["clearly_higher"]) and r["family"] == "ours"
