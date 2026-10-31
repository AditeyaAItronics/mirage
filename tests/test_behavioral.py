import numpy as np
import pytest

from mirage.behavioral.suite import build_suite, failure_rates
from mirage.behavioral.templates import make_typo
from mirage.config import load_config


@pytest.fixture(scope="module")
def suite():
    return build_suite()


def test_counts_match_config(suite):
    c = load_config("behavioral")["counts"]
    cases = suite.groupby("test_type")["case_id"].nunique()
    assert cases["simple"] == c["simple"]
    assert cases["negation"] == c["negation_pos"]
    assert cases["negation_neg"] == c["negation_neg"]
    assert cases["typo"] == c["typo"]
    assert cases["name_swap"] == c["name_swap"]


def test_deterministic(suite):
    assert build_suite().equals(suite)


def test_ids_and_pairs(suite):
    assert suite["item_id"].is_unique
    inv = suite[suite["kind"] == "INV"]
    sizes = inv.groupby("case_id")["role"].agg(sorted)
    assert all(r == ["original", "perturbed"] for r in sizes)
    # both rows of a pair share the intended label
    assert (inv.groupby("case_id")["label"].nunique() == 1).all()


def test_typo_changes_exactly_one_word(suite):
    inv = suite[suite["test_type"] == "typo"]
    for _, g in inv.groupby("case_id"):
        a = g[g.role == "original"].text.iloc[0].split()
        b = g[g.role == "perturbed"].text.iloc[0].split()
        assert len(a) == len(b)
        assert sum(x != y for x, y in zip(a, b, strict=True)) == 1


def test_make_typo_never_identity():
    rng = np.random.default_rng(0)
    for w in ["good", "great", "boring", "terrible", "dull"]:
        for _ in range(50):
            assert make_typo(w, rng) != w


def test_scoring_with_constant_model(suite):
    p = suite.copy()
    p["pred"] = 1
    p["correct"] = p["pred"] == p["label"]
    r = failure_rates(p).set_index("test_type")["fail_rate"]
    assert r["negation"] == 1.0  # all "not <positive>" should be negative
    assert r["simple"] == 0.5
    assert r["typo"] == 0.0 and r["name_swap"] == 0.0  # constant predictions never change
