import numpy as np
import pytest

from mirage.behavioral.suite import build_suite, case_failures, dir_cases, failure_rates
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
    assert cases["negation_dir"] == c["negation_dir"]
    assert cases["negation_dir_neg"] == c["negation_dir_neg"]
    assert cases["name_shortcut"] == c["name_shortcut"]
    assert cases["typo_noun"] == c["typo_noun"]


def test_deterministic(suite):
    assert build_suite().equals(suite)


def test_ids_and_pairs(suite):
    assert suite["item_id"].is_unique
    inv = suite[suite["kind"].isin(["INV", "DIR"])]
    sizes = inv.groupby("case_id")["role"].agg(sorted)
    assert all(r == ["original", "perturbed"] for r in sizes)
    # INV pairs share the intended label; DIR pairs have opposite labels
    labels = inv.groupby(["kind", "case_id"])["label"].nunique()
    assert (labels.loc["INV"] == 1).all() and (labels.loc["DIR"] == 2).all()


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
    p["p_pos"] = 1.0
    p["correct"] = p["pred"] == p["label"]
    r = failure_rates(p).set_index("test_type")["fail_rate"]
    assert r["negation"] == 1.0  # all "not <positive>" should be negative
    assert r["simple"] == 0.5
    assert r["typo"] == 0.0 and r["name_swap"] == 0.0  # constant predictions never change


def test_dir_pairs_differ_only_by_negation(suite):
    d = suite[suite["kind"] == "DIR"]
    neg_words = {"not", "is", "isn't", "don't", "never", "think", "thought", "always"}
    for _, g in d.groupby("case_id"):
        a = g[g.role == "original"].text.iloc[0].lower().split()
        b = g[g.role == "perturbed"].text.iloc[0].lower().split()
        assert set(a) ^ set(b) <= neg_words  # only negation / frame words differ
        assert g["word"].nunique() == 1 and g["word"].iloc[0] in " ".join(a)


def _predict_constant(suite, label):
    p = suite.copy()
    p["pred"] = label
    p["p_pos"] = float(label)
    p["correct"] = p["pred"] == p["label"]
    return p


def test_dir_is_not_fooled_by_a_negative_leaning_model(suite):
    """The MFT negation test gives an always-negative model a perfect score; DIR doesn't."""
    always_neg = _predict_constant(suite, 0)
    rates = failure_rates(always_neg).set_index("test_type")["fail_rate"]
    assert rates["negation"] == 0.0  # the MFT is fooled
    cf = case_failures(always_neg)
    assert "negation_dir" not in set(cf["test_type"])  # no affirmative right -> nothing to test
    always_pos = case_failures(_predict_constant(suite, 1))
    assert always_pos[always_pos.test_type == "negation_dir"]["failed"].all()  # never flips


def test_dir_cases_scores_a_model_that_reads_negation(suite):
    d = suite[suite["test_type"] == "negation_dir"].copy()
    d["p_pos"] = d["role"].map({"original": 0.9, "perturbed": 0.2})
    d["pred"] = (d["p_pos"] >= 0.5).astype(int)
    d["correct"] = d["pred"] == d["label"]
    dc = dir_cases(d)
    assert dc["applicable"].all() and not dc["failed"].any() and not dc["no_move"].any()
    assert dc["shift"].round(6).eq(0.7).all()


def test_name_shortcut_pairs_differ_only_by_the_name(suite):
    from mirage.config import load_config

    pairs = [tuple(p) for p in load_config("behavioral")["name_pairs"]]
    d = suite[suite["test_type"] == "name_shortcut"]
    for _, g in d.groupby("case_id"):
        orig = g[g.role == "original"].text.iloc[0]
        pert = g[g.role == "perturbed"].text.iloc[0]
        good, bad = next((a, b) for a, b in pairs if a in orig)
        assert orig.replace(good, bad) == pert  # same sentence, only the name swapped
        assert g["label"].nunique() == 1


def test_name_swap_pairs_match_gender():
    from mirage.behavioral import wordlists as W

    female = {
        "Priya",
        "Ananya",
        "Kavya",
        "Meera",
        "Diya",
        "Emily",
        "Sarah",
        "Jessica",
        "Laura",
        "Emma",
    }
    for a, b in zip(W.NAMES_A, W.NAMES_B, strict=True):
        assert (a in female) == (b in female), (a, b)


def test_name_associations_and_pairs():
    import pandas as pd

    from mirage.behavioral.names import name_associations, pick_pairs

    train = pd.DataFrame(
        {
            "text": ["spielberg wins"] * 20 + ["seagal again"] * 20 + ["plain"] * 40,
            "label": [1] * 19 + [0] + [0] * 19 + [1] + [1, 0] * 20,
        }
    )
    a = name_associations(train, candidates=["spielberg", "seagal", "absent"]).set_index("name")
    assert set(a.index) == {"spielberg", "seagal"}  # "absent" is under min_phrases
    assert a.loc["spielberg", "log_odds"] > 0 > a.loc["seagal", "log_odds"]
    assert pick_pairs(a.reset_index(), k=1) == [("Spielberg", "Seagal")]


def test_typo_noun_keeps_the_sentiment_word(suite):
    d = suite[suite["test_type"] == "typo_noun"]
    for _, g in d.groupby("case_id"):
        a = g[g.role == "original"].text.iloc[0].split()
        b = g[g.role == "perturbed"].text.iloc[0].split()
        changed = [(x, y) for x, y in zip(a, b, strict=True) if x != y]
        assert len(changed) == 1  # exactly one word differs ...
        assert g["word"].iloc[0] in " ".join(b)  # ... and it isn't the sentiment adjective
