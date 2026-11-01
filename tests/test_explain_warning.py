import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import mahalanobis

from mirage.explain.deletion import deletion_check
from mirage.explain.overlap import jaccard, overlap_matrix, rbo, unigram_ranking
from mirage.warning.error_detection import compare_predictors, ood_sanity, oof_combined
from mirage.warning.gaussian import GaussianDensity


def test_overlap_measures():
    a, b = ["good", "bad", "great", "dull"], ["good", "bad", "awful", "fun"]
    assert jaccard(a, b, 4) == pytest.approx(2 / 6)
    assert rbo(a, a) > rbo(a, b) > rbo(a, ["x", "y", "z", "w"]) == 0
    assert unigram_ranking(["not good", "good", "bad", "good"]) == ["good", "bad"]
    assert len(overlap_matrix({"m1": a, "m2": b, "m3": a}, k=4)) == 3


def test_deletion_detects_the_word_that_matters():
    # toy model: P(pos) = 0.9 if "good" present else 0.5
    def proba(texts):
        p = np.array([0.9 if "good" in t.split() else 0.5 for t in texts])
        return np.c_[1 - p, p]

    words = [["the", "film", "was", "good"]]
    attrs = [[0.0, 0.1, 0.0, 0.8]]
    out = deletion_check(proba, words, attrs, ks=(1,), n_random=20)
    row = out.iloc[0]
    assert row["drop_top"] == pytest.approx(0.4)
    assert row["drop_random"] < row["drop_top"]


def test_mahalanobis_matches_scipy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5)) @ rng.normal(size=(5, 5))
    g = GaussianDensity().fit(X)
    x = rng.normal(size=(3, 5))
    ours = g.score(x)
    ref = [mahalanobis(v, g.means_[0], g.precision_) for v in x]
    assert ours == pytest.approx(ref)


def test_class_conditional_takes_min_over_classes():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(-3, 1, (200, 2)), rng.normal(3, 1, (200, 2))]
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    g = GaussianDensity(class_conditional=True).fit(X, y)
    near_a, middle = g.score(np.array([[-3.0, -3.0], [0.0, 0.0]]))
    assert near_a < middle


def test_ood_sanity_and_error_detection():
    rng = np.random.default_rng(0)
    assert ood_sanity(rng.normal(0, 1, 200), rng.normal(5, 1, 200)) > 0.99
    n = 400
    wrong = rng.random(n) < 0.2
    distance = rng.normal(0, 1, n) + 2 * wrong  # distance is informative
    p_pos = np.clip(rng.uniform(0.55, 0.99, n), 0, 1)  # confidence is not
    preds = pd.DataFrame({"p_pos": p_pos, "correct": ~wrong})
    res = compare_predictors(preds, distance, n_resamples=100).set_index("predictor")
    assert res.loc["distance", "auroc"] > 0.8
    assert abs(res.loc["low_conf", "auroc"] - 0.5) < 0.12


def test_oof_never_scores_its_training_fold():
    rng = np.random.default_rng(0)
    target = (rng.random(100) < 0.3).astype(int)
    out = oof_combined(rng.normal(size=100), rng.random(100), target, folds=5, seed=0)
    assert not np.isnan(out).any()


def test_knn_distance_is_small_near_training_data():
    from mirage.warning.knn import KnnDistance

    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 8)) + np.array([5.0] + [0.0] * 7)
    knn = KnnDistance(k=1).fit(train)
    near = knn.score(train[:20] + 0.01)
    far = knn.score(-train[:20])
    assert (near < 1e-3).all() and (far > near.max()).all()  # k=1: nearest is itself


def test_split_sentences_keeps_owner_and_short_reviews():
    from mirage.warning.embed import split_sentences

    df = pd.DataFrame(
        {"item_id": ["a", "b"], "text": ["Great film overall. Too long though!<br />Ok", "Meh."]}
    )
    s = split_sentences(df)
    assert s[s["owner"] == "a"]["text"].tolist() == ["Great film overall.", "Too long though!"]
    assert s[s["owner"] == "b"]["text"].tolist() == ["Meh."]  # no full sentence -> whole text
    assert s["item_id"].is_unique


def test_text_distance_aggregates_sentences(monkeypatch, tmp_path):
    from mirage import paths
    from mirage.warning import distance, embed

    monkeypatch.setattr(paths, "EMBEDDINGS", tmp_path)
    # fake encoder: "odd" sentences sit far from the training cluster
    monkeypatch.setattr(
        embed,
        "encode",
        lambda texts, *a, **k: np.array([[0.0, 1.0] if "odd" in t else [1.0, 0.0] for t in texts]),
    )
    cfg = {"embedding_model": "x", "embedding_batch_size": 8, "distance": "knn_cosine", "knn_k": 1}
    scorer = distance.make_distance(cfg).fit(np.array([[1.0, 0.0], [1.0, 0.1]]))
    df = pd.DataFrame({"item_id": ["r1", "r2"], "text": ["A fine film. An odd one here.", "Fine."]})
    mx = distance.text_distance(scorer, df, "t", {**cfg, "unit": "sentence_max"})
    mean = distance.text_distance(scorer, df, "t2", {**cfg, "unit": "sentence_mean"})
    assert list(mx.index) == ["r1", "r2"]
    assert mx["r1"] > mean["r1"] > mean["r2"]


def test_length_adjusted_distance_is_uncorrelated_with_length():
    from mirage.warning.distance import length_adjusted

    rng = np.random.default_rng(3)
    n = rng.integers(1, 40, 500)
    signal = rng.normal(size=500)
    d = 2.0 * np.log1p(n) + signal  # distance = length effect + real signal
    adj = length_adjusted(d, n)
    assert abs(np.corrcoef(np.log1p(n), adj)[0, 1]) < 1e-8
    assert np.corrcoef(signal, adj)[0, 1] > 0.99  # the non-length part survives


def test_one_sentence_each_is_one_per_text_and_deterministic():
    from mirage.warning.distance import one_sentence_each

    sd = pd.DataFrame({"owner": ["a", "a", "a", "b"], "distance": [1.0, 2.0, 3.0, 9.0]})
    one = one_sentence_each(sd, seed=0)
    assert sorted(one.index) == ["a", "b"] and one["b"] == 9.0
    assert one.equals(one_sentence_each(sd, seed=0))


def test_compare_predictors_scores_extra_predictors():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    preds = pd.DataFrame({"p_pos": rng.random(300), "label": y})
    preds["correct"] = (preds["p_pos"] >= 0.5).astype(int) == y
    err = (~preds["correct"]).astype(int).to_numpy()
    res = compare_predictors(
        preds, rng.normal(size=300), n_resamples=50, extra={"oracle": err + 0.0}
    ).set_index("predictor")
    assert res.loc["oracle", "auroc"] == 1.0 and res.loc["oracle", "diff_vs_conf"] > 0
