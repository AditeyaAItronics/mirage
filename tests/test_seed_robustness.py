"""G2 / D44: per-seed robustness of the results that use DistilBERT's median seed."""

import numpy as np
import pandas as pd
import pytest

from mirage import evaluate

# value per seed for each (quantity) row: stable, sign flip, significance flip, rank change
VALUES = {
    "gap": {13: 0.2, 42: 0.25, 87: 0.18},
    "shift": {13: 0.1, 42: -0.05, 87: 0.2},
    "gap_p_holm": {13: 0.01, 42: 0.2, 87: 0.03},
    "rank": {13: 2.0, 42: 2.0, 87: 3.0},
}


def fake_preds() -> pd.DataFrame:
    """DistilBERT with three seeds; its sst2_val accuracy makes seed 42 the median."""
    acc = {13: 0.90, 42: 0.91, 87: 0.92}
    rows = []
    for s, a in acc.items():
        n_right = int(a * 100)
        rows.append(
            pd.DataFrame(
                {
                    "model": "distilbert",
                    "variant": "base",
                    "seed": s,
                    "dataset": "sst2_val",
                    "item_id": [f"v{i}" for i in range(100)],
                    "correct": np.arange(100) < n_right,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def fake_seed_rows(preds, **_):
    s = int(preds.loc[preds["model"] == "distilbert", "seed"].unique()[0])
    return pd.DataFrame(
        [
            {
                "analysis": "a",
                "model": "m",
                "variant": "base",
                "test": "t",
                "quantity": q,
                "value": v[s],
            }
            for q, v in VALUES.items()
        ]
    )


def test_flags(monkeypatch):
    monkeypatch.setattr(evaluate, "_seed_rows", fake_seed_rows)
    out = evaluate.seed_robustness(fake_preds()).set_index("quantity")
    assert list(out.filter(like="seed").columns[:3]) == ["seed13", "seed42", "seed87"]
    assert out.loc["gap", "same_sign"] == 1
    assert out.loc["shift", "same_sign"] == 0  # one seed flips the sign
    assert out.loc["gap_p_holm", "same_significance"] == 0  # significant on 2 of 3 seeds
    assert np.isnan(out.loc["gap_p_holm", "same_sign"])
    assert out.loc["rank", "same_sign"] == 0  # a rank must be identical, not just positive
    assert out.loc["gap", "min"] == pytest.approx(0.18)
    assert set(out["median_seed"]) == {"seed42"}


def test_single_seed_gives_empty(monkeypatch):
    monkeypatch.setattr(evaluate, "_seed_rows", fake_seed_rows)
    p = fake_preds()
    assert evaluate.seed_robustness(p[p["seed"] == 42]).empty
