"""D58: the descriptive tables behind the README's data and model sections (GitHub issue #1)."""

import pandas as pd
import pytest

from mirage.describe import (
    classical_structure,
    dataset_comparison,
    distinctive_words,
    training_history,
)


def _df(texts, labels):
    return pd.DataFrame({"text": texts, "label": labels})


def test_dataset_comparison_counts_length_vocabulary_and_negation():
    train = _df(["a good film", "a bad film", "not good"], [1, 0, 0])
    other = _df(["a good film . it was not bad at all", "unseen words here"], [1, 0])
    out = dataset_comparison({"sst2_train": train, "other": other}).set_index("dataset")
    assert out.loc["sst2_train", "types_seen_in_train"] == 1.0
    assert out.loc["sst2_train", "tokens_seen_in_train"] == 1.0
    assert out.loc["other", "n"] == 2
    assert out.loc["other", "mean_words"] == pytest.approx((10 + 3) / 2)
    assert out.loc["other", "negation_share"] == 0.5
    # 13 types (". " counts as one); seen in train: a, good, film, not, bad
    assert out.loc["other", "types_seen_in_train"] == pytest.approx(5 / 13, abs=1e-4)
    assert out.loc["other", "mean_sentences"] == 1.5  # "a good film ." + "it was …"; then 1
    assert out.loc["sst2_train", "pos_frac"] == pytest.approx(1 / 3, abs=1e-4)


def test_distinctive_words_point_each_way():
    a = _df(["plot plot plot acting"] * 30, [1] * 30)
    b = _df(["film film film acting"] * 30, [1] * 30)
    out = distinctive_words(a, b, "a", "b", min_count=10, top=1)
    assert list(out["word"]) == ["plot", "film"]
    assert list(out["more_common_in"]) == ["a", "b"]
    assert out.loc[0, "log2_ratio"] > 0 > out.loc[1, "log2_ratio"]


def test_training_history_reads_epochs_and_marks_the_kept_one():
    log = [
        {"loss": 0.6, "epoch": 0.5, "step": 50},
        {"eval_accuracy": 0.90, "eval_loss": 0.30, "epoch": 1.0, "step": 100},
        {"eval_accuracy": 0.92, "eval_loss": 0.32, "epoch": 2.0, "step": 200},
        {"eval_accuracy": 0.91, "eval_loss": 0.35, "epoch": 3.0, "step": 300},
        {"train_runtime": 10.0, "epoch": 3.0, "step": 300},
        # the Trainer's final re-evaluation of the kept model (epoch 2), logged as epoch 3
        {"eval_accuracy": 0.92, "eval_loss": 0.32, "epoch": 3.0, "step": 300},
    ]
    out = training_history({87: {"log_history": log}})
    assert list(out["epoch"]) == [1.0, 2.0, 3.0]
    assert list(out["heldout_accuracy"]) == [0.90, 0.92, 0.91]  # epoch 3 keeps its own score
    assert list(out["selected"]) == [False, True, False]
    assert training_history({}).empty


@pytest.mark.parametrize("name", ["logreg", "mnb", "dtree", "rf", "gnb"])
def test_classical_structure_counts_parameters(name):
    from mirage.models.classical import build_pipeline

    cfg = {
        "seed": 0,
        "vectorizer": {"ngram_range": [1, 1], "min_df": 1, "sublinear_tf": True},
        "models": {"rf": {"n_estimators": 3}, "gnb": {"svd_components": 2}},
    }
    texts = ["good film", "bad film", "great acting", "awful plot", "good plot", "bad acting"]
    labels = [1, 0, 1, 0, 1, 0]
    pipe = build_pipeline(name, cfg, False, {})
    pipe.fit(texts, labels)
    row = classical_structure(name, pipe)
    assert row["n_features"] == 7
    assert row["n_parameters"] > 0
    if name == "logreg":
        assert row["n_parameters"] == 7 + 1
    if name == "rf":
        assert row["n_trees"] == 3


def test_transformer_structure_counts_weights_and_reads_the_kept_epochs(tmp_path):
    import json

    import numpy as np
    from safetensors.numpy import save_file

    from mirage.describe import transformer_structure

    save_file(
        {"a": np.zeros((3, 4), np.float32), "b": np.zeros(5, np.float32)},
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"vocab_size": 10, "n_layers": 2, "dim": 8, "n_heads": 2})
    )
    history = pd.DataFrame(
        {
            "seed": [13, 13, 42, 42],
            "epoch": [1.0, 2.0, 1.0, 2.0],
            "heldout_accuracy": [0.8, 0.9, 0.85, 0.7],
            "selected": [False, True, True, False],
        }
    )
    row = transformer_structure(tmp_path, history)
    assert row["n_parameters"] == 17
    assert row["n_features"] == 10 and row["layers"] == 2
    assert row["selected_params"] == "epoch kept per seed: 13: 2, 42: 1"
    assert row["heldout_acc"] == 0.875
