import pytest

from mirage.text import Preprocessor, has_negation, negation_mark, normalize


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("It isn't good!", "it is n't good !"),
        ("I can't believe it won't work", "i ca n't believe it wo n't work"),
        ("The film's ending -- wow.", "the film 's ending -- wow ."),
        ("Great<br />movie", "great movie"),
        ("Great<br>movie", "great movie"),
        ("Wait...", "wait ..."),
        ("it’s fine", "it 's fine"),
        ("  lots   of   space ", "lots of space"),
        # Found by screening real SST-2: abbreviation periods stay attached, as SST-2 has them
        ("Mr. Taylor is not good.", "mr. taylor is not good ."),
        ("L.A. and the P.O.W. film", "l.a. and the p.o.w. film"),
        ("Robert J. Siegel", "robert j. siegel"),
        ("It cost 10,000 dollars, sadly.", "it cost 10,000 dollars , sadly ."),
        ("It was great.The end", "it was great . the end"),  # glued IMDb sentences
        ("version 3.5", "version 3.5"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    "sst",
    [
        "it 's a charming and often affecting journey . ",
        "does n't work , really",
        "ca n't stop , wo n't stop",
    ],
)
def test_normalize_is_idempotent_on_sst_text(sst):
    once = normalize(sst)
    assert normalize(once) == once
    assert once == sst.strip()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("not good movie .", "not NOT_good NOT_movie ."),
        ("not good movie", "not NOT_good NOT_movie"),  # scope runs to end of text (SST phrases)
        ("it is n't good , but fun", "it is n't NOT_good , but fun"),
        ("never boring ! great", "never NOT_boring ! great"),
        ("no fun and not good", "no NOT_fun NOT_and not NOT_good"),  # restart on new negation
        ("a great film", "a great film"),
        ("nothing works ... yes", "nothing NOT_works ... yes"),
    ],
)
def test_negation_mark(text, expected):
    assert negation_mark(text) == expected


def test_has_negation_uses_tokens_not_substrings():
    assert has_negation("it is n't good")
    assert has_negation("not good")
    assert not has_negation("a notable film")  # "not" inside a word is not a negation
    assert not has_negation("knowing nobody")


def test_preprocessor_pipeline():
    assert Preprocessor(negmark=False)("It isn't good.") == "it is n't good ."
    assert Preprocessor(negmark=True)("It isn't good.") == "it is n't NOT_good ."
