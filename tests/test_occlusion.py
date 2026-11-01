"""D43: leave-one-word-out importance, the shared ranking rule for H5."""

import numpy as np
import pandas as pd
import pytest

from mirage.explain.occlusion import (
    leave_one_out,
    occlusion_table,
    split_half_overlap,
    word_importance,
)


def word_model(texts):
    """Stand-in: p_pos = 0.5 + 0.3 * (#"good" - #"bad"), clipped; "the" has no effect."""
    p = np.array([0.5 + 0.3 * (t.split().count("good") - t.split().count("bad")) for t in texts])
    p = np.clip(p, 0.01, 0.99)
    return np.column_stack([1 - p, p])


def test_leave_one_out():
    words, dropped = leave_one_out("a good film")
    assert words == ["a", "good", "film"]
    assert dropped == ["good film", "a film", "a good"]


def test_occlusion_table_deltas_and_chunking():
    texts = ["the good film", "the bad film .", ""]
    occ = occlusion_table(word_model, texts)
    assert len(occ) == 3 + 4  # one row per word; the empty sentence contributes none
    d = occ.set_index(["sent", "word"])["delta"]
    assert d[(0, "good")] == pytest.approx(0.3)
    assert d[(1, "bad")] == pytest.approx(-0.3)
    assert d[(0, "the")] == pytest.approx(0.0)
    # chunking must not change anything
    pd.testing.assert_frame_equal(occ, occlusion_table(word_model, texts, chunk=2))


def test_word_importance_ranks_and_filters():
    texts = ["the good film ."] * 6 + ["the bad film ."] * 6 + ["rare good"]
    imp = word_importance(occlusion_table(word_model, texts), min_count=5).set_index("word")
    assert "." not in imp.index  # punctuation is scored but never ranked
    assert "rare" not in imp.index  # seen once < min_count
    assert imp.loc["good", "rank"] in (1, 2) and imp.loc["bad", "rank"] in (1, 2)
    assert imp.loc["good", "mean_signed"] > 0 > imp.loc["bad", "mean_signed"]
    assert imp.loc["the", "mean_abs"] == pytest.approx(0.0)
    assert list(imp["rank"]) == list(range(1, len(imp) + 1))


def test_split_half_is_perfect_for_a_stable_model():
    texts = [f"the good film number{i % 3}" for i in range(40)] + [
        f"the bad plot number{i % 3}" for i in range(40)
    ]
    occ = occlusion_table(word_model, texts)
    half = split_half_overlap(occ, min_count=4, k=3, seed=0)
    assert half["jaccard"] == pytest.approx(1.0)  # same top words in both halves


def test_split_half_ignores_rare_word_noise():
    """Regression (D43): words too rare for the full ranking must not dominate either half.

    Each sentence carries a unique one-off word with a big random effect. Ranked per half with a
    lowered threshold they swamped the top-k, so a model 'disagreed with itself'."""
    rng = np.random.default_rng(0)
    effects = {f"oneoff{i}": rng.uniform(-0.5, 0.5) for i in range(200)}

    def noisy_model(texts):
        p = word_model(texts)[:, 1] + np.array(
            [sum(effects.get(w, 0.0) for w in t.split()) for t in texts]
        )
        p = np.clip(p, 0.01, 0.99)
        return np.column_stack([1 - p, p])

    texts = [f"the {'good' if i % 2 else 'bad'} film oneoff{i}" for i in range(200)]
    half = split_half_overlap(occlusion_table(noisy_model, texts), min_count=5, k=3, seed=1)
    assert half["jaccard"] == pytest.approx(1.0)  # only good / bad / the / film are eligible


def test_chance_jaccard():
    from mirage.explain.occlusion import chance_jaccard

    assert chance_jaccard(50, 50) == 1.0
    assert chance_jaccard(1000, 50) == pytest.approx(2.5 / 97.5)


def _effects_and_tables():
    """Two models that read words the same way, and one that reads them the opposite way."""
    from mirage.explain.occlusion import word_effects

    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(60)]
    weight = dict(zip(vocab, rng.normal(0, 0.2, len(vocab)), strict=True))
    texts = [" ".join(rng.choice(vocab, 6)) for _ in range(400)]

    def model(sign, noise_seed):
        noise = np.random.default_rng(noise_seed)

        def f(ts):
            p = 0.5 + sign * np.array([sum(weight[w] for w in t.split()) for t in ts])
            p = np.clip(p + noise.normal(0, 0.01, len(ts)), 0.01, 0.99)
            return np.column_stack([1 - p, p])

        return f

    tables = {
        "a": occlusion_table(model(1, 1), texts),
        "b": occlusion_table(model(1, 2), texts),
        "flip": occlusion_table(model(-1, 3), texts),
    }
    return word_effects(tables, 5), tables


def test_effect_agreement_and_reliability():
    from mirage.explain.occlusion import effect_agreement, split_half_reliability

    effects, tables = _effects_and_tables()
    rel = {m: split_half_reliability(o, effects.index, 5, 0) for m, o in tables.items()}
    assert all(r > 0.8 for r in rel.values())  # stable stand-in models
    agr = effect_agreement(effects, rel, 200, 0).set_index(["model_a", "model_b"])
    assert agr.loc[("a", "b"), "rho"] > 0.8
    assert agr.loc[("a", "flip"), "rho"] < -0.8  # opposite readings correlate negatively
    r = agr.loc[("a", "b")]
    assert r["rho_lo"] <= r["rho"] <= r["rho_hi"]
    assert r["corrected"] == pytest.approx(r["rho"] / np.sqrt(rel["a"] * rel["b"]))


def test_correction_undefined_without_reliability():
    from mirage.explain.occlusion import effect_agreement

    effects = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0)[::-1]})
    agr = effect_agreement(effects, {"a": 0.9, "b": -0.1}, 50, 0)
    assert agr.loc[0, "rho"] == pytest.approx(-1.0)
    assert np.isnan(agr.loc[0, "corrected"])
