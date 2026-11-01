import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.contingency_tables import mcnemar

from mirage.metrics import auroc, ece, reliability_bins
from mirage.stats import bootstrap, gap_closed_ci, mcnemar_test, paired_diff_ci, rate_ci


def test_ece_zero_when_calibrated():
    # 80% confident and 80% right -> perfectly calibrated
    p = np.full(100, 0.8)
    y = np.r_[np.ones(80), np.zeros(20)].astype(int)
    assert ece(p, y) == pytest.approx(0.0)


def test_ece_hand_computed():
    # all predicted positive at 0.9 but only half right -> ECE = |0.5 - 0.9| = 0.4
    p = np.full(10, 0.9)
    y = np.r_[np.ones(5), np.zeros(5)].astype(int)
    assert ece(p, y) == pytest.approx(0.4)


def test_confidence_one_lands_in_last_bin():
    bins = reliability_bins(np.array([1.0, 0.0]), np.array([1, 0]))
    assert bins.iloc[-1]["count"] == 2


def test_auroc_nan_on_single_class():
    assert np.isnan(auroc(np.zeros(5), np.arange(5)))


def test_bootstrap_contains_estimate_and_is_deterministic():
    flags = np.r_[np.ones(30), np.zeros(70)]
    a, b = rate_ci(flags, n_resamples=300), rate_ci(flags, n_resamples=300)
    assert a == b
    assert a["ci_low"] <= 0.3 <= a["ci_high"]


def test_paired_diff_tighter_than_unpaired():
    rng = np.random.default_rng(0)
    hard = rng.random(500) < 0.5  # shared item difficulty
    a = hard | (rng.random(500) < 0.1)
    b = hard
    paired = paired_diff_ci(a, b, n_resamples=500)
    rng2 = np.random.default_rng(1)
    unpaired = bootstrap(
        lambda i: a[i].mean() - b[rng2.integers(0, 500, 500)].mean(), 500, n_resamples=500
    )
    assert (paired["ci_high"] - paired["ci_low"]) < (unpaired["ci_high"] - unpaired["ci_low"])


def test_gap_closed_point_estimate():
    ref = np.zeros(100)
    base = np.r_[np.ones(40), np.zeros(60)]  # gap 0.4
    neg = np.r_[np.ones(10), np.zeros(90)]  # gap 0.1 -> 75% closed
    assert gap_closed_ci(base, neg, ref, n_resamples=50)["estimate"] == pytest.approx(0.75)


def test_mcnemar_matches_statsmodels():
    rng = np.random.default_rng(0)
    a, b = rng.random(200) < 0.8, rng.random(200) < 0.7
    ours = mcnemar_test(a, b)
    table = [[(a & b).sum(), (a & ~b).sum()], [(~a & b).sum(), (~a & ~b).sum()]]
    assert ours["p_value"] == pytest.approx(
        mcnemar(table, exact=ours["exact"], correction=True).pvalue
    )


def test_bootstrap_p_value_separates_real_and_null_effects():
    rng = np.random.default_rng(0)
    a = rng.random(400) < 0.4
    b = rng.random(400) < 0.2
    real = paired_diff_ci(a, b, n_resamples=500)
    same = paired_diff_ci(a, a.copy(), n_resamples=500)
    assert real["p_value"] < 1e-6  # no 2 / (n_resamples + 1) floor (D28)
    assert same["p_value"] == 1.0  # all resamples sit exactly at the null
    assert 0 < real["p_value"] <= 1


def test_holm_matches_textbook_and_keeps_nans():
    from mirage.stats import holm

    out = holm([0.01, 0.04, 0.03, np.nan])
    # sorted 0.01, 0.03, 0.04 -> x3, x2, x1 -> 0.03, 0.06, 0.06 (monotone)
    assert out[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert np.isnan(out[3])
    assert np.isnan(holm([np.nan])).all()


def test_paired_auroc_diff_detects_the_better_score():
    from mirage.stats import paired_auroc_diff

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    good = y + rng.normal(0, 0.5, 300)
    noise = rng.normal(0, 1, 300)
    d = paired_auroc_diff(y, good, noise, n_resamples=300)
    assert d["estimate"] > 0.3 and d["ci_low"] > 0 and d["p_value"] < 0.01
    assert paired_auroc_diff(y, good, good, n_resamples=100)["p_value"] == 1.0


def test_length_table_bins_imdb_reviews():
    from mirage.evaluate import length_table

    n_words = [50, 150, 150, 300, 500, 500]
    preds = pd.DataFrame(
        {
            "model": "logreg",
            "variant": "base",
            "seed": 42,
            "dataset": "imdb",
            "item_id": [f"imdb-{i}" for i in range(6)],
            "n_words": n_words,
            "correct": [True, True, False, True, False, False],
        }
    )
    t = length_table(preds, n_resamples=20).set_index("length")
    assert t.loc["<=100", "n"] == 1 and t.loc["101-200", "error"] == 0.5
    assert t.loc[">360 (capped)", "error"] == 1.0


def test_name_effect_measures_the_shift_toward_the_positive_name():
    from mirage.evaluate import name_effect

    rows = []
    for i in range(40):
        for role, p in (("original", 0.7), ("perturbed", 0.4 if i % 2 else 0.6)):
            rows.append(
                {
                    "model": "logreg",
                    "variant": "base",
                    "seed": 42,
                    "dataset": "behavioral",
                    "test_type": "name_shortcut",
                    "case_id": f"name_shortcut-{i}",
                    "role": role,
                    "p_pos": p,
                    "pred": int(p >= 0.5),
                }
            )
    out = name_effect(pd.DataFrame(rows), n_resamples=50).iloc[0]
    assert out["shift"] == pytest.approx(0.2)  # (0.3 + 0.1) / 2
    assert out["flip_rate"] == 0.5 and out["shift_p"] < 0.001
