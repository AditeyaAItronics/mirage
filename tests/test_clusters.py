"""G7 (D52): cluster bootstrap, cluster labels, the item-vs-cluster table, and IMDb film matching."""

import io
import tarfile

import numpy as np
import pandas as pd
import pytest

from mirage import evaluate
from mirage.stats import bootstrap, diff_independent_ci, paired_diff_ci, rate_ci


def test_no_clusters_is_the_item_bootstrap():
    rng = np.random.default_rng(0)
    x = rng.random(200) < 0.3
    a = rate_ci(x)
    b = rate_ci(x, clusters=[f"c{i}" for i in range(200)])  # one item per cluster
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])


def test_identical_items_in_a_cluster_widen_the_interval():
    # 10 clusters of 50 identical items: the effective sample is 10, not 500
    x = np.repeat(np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=float), 50)
    clusters = np.repeat(np.arange(10), 50)
    item = rate_ci(x)
    clus = rate_ci(x, clusters=clusters)
    means = x.reshape(10, 50).mean(axis=1)
    by_mean = bootstrap(lambda i: means[i].mean(), 10)
    assert clus["ci_high"] - clus["ci_low"] > 3 * (item["ci_high"] - item["ci_low"])
    assert clus["ci_low"] == pytest.approx(by_mean["ci_low"], abs=0.05)
    assert clus["ci_high"] == pytest.approx(by_mean["ci_high"], abs=0.05)


def test_paired_and_two_group_versions_accept_clusters():
    x = np.repeat([1.0, 0.0, 1.0, 0.0], 25)
    y = np.repeat([0.0, 0.0, 1.0, 0.0], 25)
    c = np.repeat(list("abcd"), 25)
    assert paired_diff_ci(x, y, clusters=c)["estimate"] == pytest.approx(0.25)
    d = diff_independent_ci(x, y, clusters_a=c, clusters_b=c)
    assert d["ci_high"] - d["ci_low"] > diff_independent_ci(x, y)["ci_high"] - d["estimate"]
    with pytest.raises(ValueError):
        rate_ci(x, clusters=c[:10])


def test_cluster_labels_from_words_and_films(monkeypatch):
    monkeypatch.setattr(evaluate, "_films", lambda: pd.Series({"imdb-0": "film:tt1"}))
    preds = pd.DataFrame(
        {
            "dataset": ["behavioral", "behavioral", "imdb"],
            "case_id": ["typo-0", "typo-0", None],
            "word": ["superb", "superb", None],
        }
    )
    cmap = evaluate._cluster_map(preds)
    lab = evaluate._clusters(["typo-0", "imdb-0", "imdb-1"], cmap)
    assert list(lab) == ["word:superb", "film:tt1", "item:imdb-1"]
    with evaluate.item_level_intervals():
        assert evaluate._clusters(["typo-0"], cmap) is None
    assert evaluate._clusters(["sst2_test-1"], cmap) is None  # no cluster info: item bootstrap


def test_cluster_robustness_pairs_up_intervals():
    clustered = {
        "name_effect": pd.DataFrame(
            {
                "model": ["mnb"],
                "variant": ["base"],
                "test": ["name_shortcut"],
                "shift": [0.37],
                "shift_lo": [0.2],
                "shift_hi": [0.55],
                "shift_p_holm": [0.06],
            }
        )
    }
    item = {
        "name_effect": clustered["name_effect"].assign(
            shift_lo=0.34, shift_hi=0.40, shift_p_holm=0.001
        )
    }
    t = evaluate.cluster_robustness(clustered, item)
    r = t.iloc[0]
    assert r["width_ratio"] == pytest.approx(0.35 / 0.06)
    assert bool(r["significance_changed"])


def test_film_matching_on_a_tiny_archive(tmp_path):
    from mirage.data import match_films, read_archive_films

    files = {
        "aclImdb/test/pos/0_9.txt": "A wonderful film about a dog.",
        "aclImdb/test/neg/0_2.txt": "Dreadful from start to finish.",
        "aclImdb/test/urls_pos.txt": "http://www.imdb.com/title/tt0000001/usercomments\n",
        "aclImdb/test/urls_neg.txt": "http://www.imdb.com/title/tt0000002/usercomments\n",
    }
    archive = tmp_path / "aclImdb_v1.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    films = read_archive_films(archive)
    df = pd.DataFrame(
        {
            "item_id": ["imdb-0", "imdb-1", "imdb-2"],
            "text": [
                "A wonderful film  about a dog.",
                "Dreadful from start to finish.",
                "Unknown.",
            ],
        }
    )
    out = match_films(df, films).set_index("item_id")["film"]
    assert out["imdb-0"] == "tt0000001" and out["imdb-1"] == "tt0000002"
    assert pd.isna(out["imdb-2"])


def test_fresh_imdb_never_overlaps_the_main_sample():
    from mirage.data import build_fresh_imdb, stratified_sample

    imdb = pd.DataFrame(
        {"text": [f"review number {i} " * 3 for i in range(400)], "label": [0, 1] * 200}
    )
    cfg = {
        "imdb_sample_size": 100,
        "imdb_fresh_sample_size": 100,
        "seed": 42,
        "fresh_seed": 2026,
        "imdb_head_words": 220,
        "imdb_tail_words": 140,
    }
    used = set(stratified_sample(imdb, 100, 42)["text"])
    fresh = build_fresh_imdb(imdb, cfg)["imdb_fresh"]
    assert len(fresh) == 100 and not used & set(fresh["text"])
    assert fresh["label"].mean() == pytest.approx(0.5)
    assert fresh["item_id"].str.startswith("imdb_fresh-").all()
