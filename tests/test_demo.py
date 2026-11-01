"""Shortcut Explorer (D41): demo logic with stand-in models, plus real-model checks when the
trained models are on disk (skipped otherwise, e.g. on a fresh clone)."""

import numpy as np
import pytest

from mirage import paths
from mirage.demo import (
    check_example,
    leave_one_out,
    load_examples,
    occlusion,
    score,
    trim,
    unknown_words,
)


class WordModel:
    """Stand-in classifier: p_pos = 0.5 + 0.4 * (#"good" - #"bad"), clipped."""

    name = "fake"

    def predict_proba(self, texts, long_text=False):
        p = np.array(
            [0.5 + 0.4 * (t.split().count("good") - t.split().count("bad")) for t in texts]
        )
        p = np.clip(p, 0.01, 0.99)
        return np.column_stack([1 - p, p])


def test_trim():
    assert trim("  a b  c ", 5) == ("a b c", False)
    assert trim("a b c d", 2) == ("a b", True)
    assert trim("", 5) == ("", False)


def test_leave_one_out():
    words, dropped = leave_one_out("the movie was good")
    assert words == ["the", "movie", "was", "good"]
    assert dropped == ["movie was good", "the was good", "the movie good", "the movie was"]


def test_occlusion_signs():
    occ = occlusion(WordModel(), "a good but bad film").set_index("word")["importance"]
    assert occ["good"] > 0 and occ["bad"] < 0 and occ["film"] == pytest.approx(0)
    assert occlusion(WordModel(), "").empty


def test_score_flags_low_confidence():
    sc = score({"fake": WordModel()}, ["good", "neutral words", "bad"], low_confidence=0.7)
    assert sc["label"].tolist() == ["positive", "positive", "negative"]
    assert sc["low_confidence"].tolist() == [False, True, False]


def test_unknown_words_uses_vocabulary(sst):
    from mirage.config import load_config
    from mirage.models.classical import build_pipeline

    pipe = build_pipeline("logreg", load_config("classical"), False, {"C": 1.0})
    df = sst(60)
    pipe.fit(df["text"], df["label"])

    class Clf:
        pipeline = pipe

    assert unknown_words(Clf(), "a great flurbish movie") == [False, False, True, False]
    assert unknown_words(WordModel(), "anything") is None  # no word vocabulary


def test_check_example_reports_broken_expectations():
    models = {"fake": WordModel()}
    ok = {"id": "x", "a": "good", "b": "bad", "expect": {"flips": ["fake"], "stable": ["absent"]}}
    assert check_example(models, ok) == []
    broken = {"id": "y", "a": "good", "expect": {"negative": ["fake"]}}
    assert check_example(models, broken) == ["y: fake expected negative"]


def test_examples_file_is_well_formed():
    ex = load_examples()
    assert len({e["id"] for e in ex}) == len(ex) >= 8
    for e in ex:
        assert {"id", "title", "a", "talking_point", "expect"} <= set(e)
        kinds = set(e["expect"])
        assert kinds <= {"flips", "stable", "positive", "negative"}
        assert ("flips" in kinds or "stable" in kinds) == ("b" in e)  # pair checks need a pair


needs_models = pytest.mark.skipif(
    not paths.model_dir("logreg", "base", 42).exists(), reason="trained models not on disk"
)


@needs_models
def test_curated_examples_behave():
    """Every documented expectation in app/examples.yaml still holds on the real models."""
    pytest.importorskip("transformers")
    from mirage.demo import load_models

    models = load_models(light=True)  # light: keeps the test quick; skips rf / RoBERTa lines
    failures = [f for e in load_examples() for f in check_example(models, e)]
    assert failures == []


@needs_models
def test_app_runs_headless():
    pytest.importorskip("streamlit")
    pytest.importorskip("transformers")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(paths.ROOT / "app" / "shortcut_explorer.py"), default_timeout=300)
    at.run()  # default settings: every available model
    assert not at.exception
    assert any("What changed" in s.value for s in at.subheader)
    at.radio[0].set_value("Your own sentence").run()
    assert not at.exception
