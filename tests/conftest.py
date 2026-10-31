import numpy as np
import pandas as pd
import pytest

POS = [
    "a great film",
    "truly wonderful acting",
    "i loved it",
    "good fun",
    "an excellent movie",
    "it 's brilliant",
]
NEG = [
    "a terrible film",
    "truly awful acting",
    "i hated it",
    "bad and boring",
    "it is n't good",
    "not good",
]


def synthetic_sst(n: int, seed: int = 0) -> pd.DataFrame:
    """Small SST-style frame (already normalized text, 0/1 labels)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    nouns = ["movie", "story", "plot", "cast"]
    text = [f"{rng.choice(POS if label else NEG)} {rng.choice(nouns)}" for label in y]
    return pd.DataFrame({"text": text, "label": y})


@pytest.fixture
def sst():
    return synthetic_sst
