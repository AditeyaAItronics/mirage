"""D10 / G8 / F2 (D53): frame sets, the second suite, claim scoring, DistilBERT features."""

import numpy as np
import pandas as pd
import pytest

from mirage.behavioral import templates
from mirage.behavioral.suite import build_suite, build_suite_v2


def test_frame_set_a_rebuilds_the_original_suite_exactly():
    a, b = build_suite(), build_suite(frame_set="A")
    pd.testing.assert_frame_equal(a, b)


def test_second_suite_uses_frame_set_b_and_its_own_ids():
    v2 = build_suite_v2()
    assert v2["item_id"].str.startswith("beh2-").all()
    first = v2.drop_duplicates("case_id")
    frames = first.apply(lambda r: templates.frame_of(r["text"], r["test_type"]), axis=1)
    framed = first["test_type"].isin(templates.FRAME_KIND)
    assert frames[framed].str.startswith("B").all()
    assert (frames[~framed] == "").all()  # name_swap keeps its own (unframed) sentences
    assert v2.groupby("test_type").size().equals(build_suite().groupby("test_type").size())


def test_frame_of_reads_both_sets():
    assert templates.frame_of("The movie was superb.", "simple") == "A0"
    assert templates.frame_of("Sadly, the plot was not moving.", "negation") == "B2"
    assert templates.frame_of("I found the film dull.", "negation_dir") == "B0"
    assert templates.frame_of("Kidman's new film was decent.", "name_shortcut") == "B0"
    assert templates.frame_of("Something else entirely.", "simple") == ""


def test_fresh_view_renames_only_the_fresh_sets():
    from mirage.replication import fresh_view

    p = pd.DataFrame(
        {
            "dataset": ["behavioral", "behavioral_v2", "imdb", "imdb_fresh", "sst2_val"],
            "x": range(5),
        }
    )
    v = fresh_view(p)
    assert list(v["dataset"]) == ["behavioral", "imdb", "sst2_val"]
    assert list(v["x"]) == [1, 3, 4]


def test_claims_are_pending_without_fresh_data():
    from mirage.replication import score_claims

    p = pd.DataFrame(
        columns=[
            "model",
            "variant",
            "seed",
            "dataset",
            "item_id",
            "p_pos",
            "pred",
            "correct",
            "label",
        ]
    )
    out = score_claims(p)
    assert list(out["claim"]) == list(range(1, 18))
    assert (out["status"] == "pending").all()


def test_distilbert_features_match_the_classification_head(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from mirage.models.distilbert import DistilBertClassifier

    words = ["the", "movie", "was", "good", "plot", "bad", "."]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]))
    tok = transformers.BertTokenizerFast(vocab_file=str(vocab))
    cfg = transformers.DistilBertConfig(
        vocab_size=len(words) + 5, dim=32, hidden_dim=64, n_layers=1, n_heads=2, num_labels=2
    )
    torch.manual_seed(0)
    model = transformers.DistilBertForSequenceClassification(cfg)
    clf = DistilBertClassifier(model, tok)
    texts = ["the movie was good", "the plot was bad"]
    X = clf.features(texts)
    assert X.shape == (2, 32)
    with torch.no_grad():  # the head applied to the features gives the model's own outputs
        logits = model.classifier(torch.tensor(X)).numpy()
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    np.testing.assert_allclose(p, clf.predict_proba(texts), atol=1e-5)
