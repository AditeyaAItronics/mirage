"""D42: the single-word variant and the H2/H3 ladder (single words -> + pairs -> + marking)."""

import numpy as np
import pandas as pd
import pytest

from mirage.evaluate import holm_family, negation_gap, negmark_effect


def test_unigram_variant_has_no_word_pairs(sst):
    from mirage.config import load_config
    from mirage.models.classical import VARIANT_NEGMARK, VARIANT_NGRAMS, build_pipeline

    cfg = load_config("classical")
    df = sst(80)
    feats = {}
    for variant in ("base", "unigram"):
        pipe = build_pipeline(
            "logreg", cfg, VARIANT_NEGMARK[variant], {"C": 1.0}, VARIANT_NGRAMS.get(variant)
        )
        pipe.fit(df["text"], df["label"])
        feats[variant] = set(pipe.named_steps["vec"].vocabulary_)
    assert not any(" " in f for f in feats["unigram"])
    assert any(" " in f for f in feats["base"])  # base keeps word pairs, e.g. "not good"
    assert feats["unigram"] < feats["base"]
    assert cfg["vectorizer"]["ngram_range"] == [1, 2]  # override doesn't mutate the config


def _preds(errors: dict[tuple[str, str], np.ndarray], n: int = 200) -> pd.DataFrame:
    """sst2_val + sst2_neg rows for each (model, variant) with the given error flags on sst2_neg."""
    rows = []
    for (m, v), err in errors.items():
        for ds in ("sst2_val", "sst2_neg"):
            e = err if ds == "sst2_neg" else np.zeros(n, dtype=bool)
            rows.append(
                pd.DataFrame(
                    {
                        "model": m,
                        "variant": v,
                        "seed": 42,
                        "dataset": ds,
                        "item_id": [f"sst2_test-{i}" for i in range(n)],
                        "correct": ~e,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _err(rate: float, n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = np.zeros(n, dtype=bool)
    e[rng.choice(n, int(rate * n), replace=False)] = True
    return e


ERRORS = {
    ("distilbert", "base"): _err(0.10, seed=1),
    ("logreg", "unigram"): _err(0.40, seed=2),
    ("logreg", "base"): _err(0.30, seed=3),
    ("logreg", "negmark"): _err(0.20, seed=4),
}


def test_ladder_columns_and_values():
    g = negation_gap(_preds(ERRORS), tests=("sst2_neg",), n_resamples=200)
    r = g.set_index("model").loc["logreg"]
    assert r["gap_unigram"] == pytest.approx(0.30)
    assert r["gap_base"] == pytest.approx(0.20)
    assert r["gap_negmark"] == pytest.approx(0.10)
    assert r["pairs_closed"] == pytest.approx((0.30 - 0.20) / 0.30)
    assert r["ladder_closed"] == pytest.approx((0.30 - 0.10) / 0.30)
    for col in ("gap_unigram", "pairs_closed", "ladder_closed"):
        assert f"{col}_p_holm" in g and r[f"{col}_lo"] <= r[col] <= r[f"{col}_hi"]


def test_ladder_leaves_existing_columns_unchanged():
    without = {k: v for k, v in ERRORS.items() if k[1] != "unigram"}
    a = negation_gap(_preds(without), tests=("sst2_neg",), n_resamples=200)
    b = negation_gap(_preds(ERRORS), tests=("sst2_neg",), n_resamples=200)
    pd.testing.assert_frame_equal(a, b[a.columns])
    assert "gap_unigram" not in a


def test_negmark_effect_skips_the_ablation():
    out = negmark_effect(_preds(ERRORS), tests=("sst2_neg",), n_resamples=100)
    assert set(out["variant"]) == {"negmark"}


def test_holm_family_with_variants():
    models = pd.Series(["logreg", "logreg", "roberta_large", "mnb"])
    variants = pd.Series(["base", "unigram", "base", "negmark"])
    assert holm_family(models, variant=variants).tolist() == [
        "ours",
        "ablation",
        "external",
        "ours",
    ]
