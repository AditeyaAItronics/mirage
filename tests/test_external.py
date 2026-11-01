"""External (ready-made) models, D39: label mapping, contract and the CLI stage.

Uses a tiny randomly initialised BERT built locally, so no download is needed.
"""

import argparse

import numpy as np
import pandas as pd
import pytest

from mirage.models.external import EXTERNAL_SEED, label_columns
from mirage.predict import make_predictions

pytest.importorskip("transformers")
pytest.importorskip("torch")

WORDS = ["the", "movie", "was", "good", "bad", "not", "great", "film", "."]


def tiny_classifier(tmp_path, id2label, name="tiny"):
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    from mirage.models.external import ExternalClassifier

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]))
    tok = BertTokenizerFast(vocab_file=str(vocab))
    torch.manual_seed(0)
    cfg = BertConfig(
        vocab_size=len(WORDS) + 5,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=2,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    return ExternalClassifier(name, BertForSequenceClassification(cfg), tok, 16, 32, batch_size=2)


def test_label_columns_both_orders():
    assert label_columns({0: "NEGATIVE", 1: "POSITIVE"}) == [0, 1]
    assert label_columns({"0": "positive", "1": "negative"}) == [1, 0]


@pytest.mark.parametrize(
    "bad",
    [
        {0: "LABEL_0", 1: "LABEL_1"},
        {0: "negative", 1: "neutral", 2: "positive"},
        {0: "negative", 1: "negative"},
    ],
)
def test_label_columns_rejects(bad):
    with pytest.raises(ValueError):
        label_columns(bad)


def test_reversed_labels_swap_columns(tmp_path):
    texts = ["the movie was good .", "not great", "bad film"]
    fwd = tiny_classifier(tmp_path, {0: "NEGATIVE", 1: "POSITIVE"})
    rev = tiny_classifier(tmp_path, {0: "POSITIVE", 1: "NEGATIVE"})  # same weights (same seed)
    p_fwd, p_rev = fwd.predict_proba(texts), rev.predict_proba(texts)
    np.testing.assert_allclose(p_fwd[:, ::-1], p_rev, atol=1e-6)
    np.testing.assert_allclose(p_fwd.sum(1), 1, atol=1e-6)


def test_raw_text_and_long_text(tmp_path):
    clf = tiny_classifier(tmp_path, {0: "NEGATIVE", 1: "POSITIVE"})
    assert clf.normalize_input is False
    long = " ".join(["the movie was good ."] * 40)  # > long_max_length tokens: head + tail
    p = clf.predict_proba([long, "bad"], long_text=True)
    assert p.shape == (2, 2)
    df = pd.DataFrame({"item_id": ["a", "b"], "text": [long, "bad"], "label": [1, 0]})
    make_predictions(df, p)  # passes validate_predictions


def test_stage_writes_contract_files(tmp_path, monkeypatch):
    from mirage import cli, paths, predict
    from mirage.models import external

    monkeypatch.setattr(paths, "PREDICTIONS", tmp_path / "pred")
    eval_df = pd.DataFrame(
        {"item_id": ["x1", "x2"], "text": ["the movie was good .", "bad"], "label": [1, 0]}
    )
    monkeypatch.setattr(predict, "load_eval_set", lambda ds: eval_df)
    monkeypatch.setattr(
        external.ExternalClassifier,
        "from_hub",
        classmethod(
            lambda cls, name, spec, cfg: tiny_classifier(tmp_path, {0: "neg", 1: "pos"}, name)
        ),
    )
    args = argparse.Namespace(models=["distilbert_hf"], datasets=["sst2_val"], overwrite=False)
    cli.stage_predict_external(args)
    out = predict.load_predictions(model="distilbert_hf")
    assert set(out["seed"]) == {EXTERNAL_SEED} and set(out["variant"]) == {"base"}
    assert len(out) == 2
    # second run skips: files exist
    monkeypatch.setattr(
        external.ExternalClassifier, "from_hub", classmethod(lambda *a: pytest.fail("reloaded"))
    )
    cli.stage_predict_external(args)
