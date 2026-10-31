import numpy as np
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
