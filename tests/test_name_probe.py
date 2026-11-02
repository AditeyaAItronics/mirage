"""D45: the name probe (every film person in the same sentences) and its analysis."""

import numpy as np
import pandas as pd
import pytest

from mirage.behavioral.names import name_probe_suite
from mirage.evaluate import name_probe, name_probe_effects

ASSOC = pd.DataFrame(
    {
        "name": ["kidman", "spielberg", "allen", "chan", "sandler", "seagal"],
        "phrases": [26, 90, 66, 68, 77, 47],
        "pos_rate": [1.0, 0.92, 0.67, 0.35, 0.31, 0.11],
        "log_odds": [3.0, 2.1, 0.4, -0.8, -1.0, -2.2],
    }
)


def test_suite_puts_every_name_in_the_same_sentences():
    s = name_probe_suite(ASSOC, {"kidman", "seagal"}, n_sentences=10, seed=0)
    assert len(s) == 6 * 10 and s["item_id"].is_unique
    per_name = s.groupby("name")["text"].apply(
        lambda t: [x.replace(t.name.capitalize(), "{N}") for x in t]
    )
    assert all(v == per_name.iloc[0] for v in per_name)  # identical apart from the name
    assert set(s.loc[s["tested"], "name"]) == {"kidman", "seagal"}
    assert set(s["label"]) <= {0, 1}
    pd.testing.assert_frame_equal(s, name_probe_suite(ASSOC, {"kidman", "seagal"}, 10, 0))


def _preds(suite: pd.DataFrame, slope: float, model: str) -> pd.DataFrame:
    """A stand-in model whose p_pos follows the label plus `slope` x the name's log-odds."""
    p = np.clip(0.3 + 0.4 * suite["label"] + slope * suite["log_odds"], 0.01, 0.99)
    return suite.assign(
        model=model,
        variant="base",
        seed=42,
        dataset="name_probe",
        p_pos=p,
        pred=(p >= 0.5).astype(int),
        correct=lambda d: d["pred"] == d["label"],
    )


def test_effects_and_correlation():
    suite = name_probe_suite(ASSOC, {"kidman", "seagal"}, n_sentences=10, seed=0)
    preds = pd.concat(
        [_preds(suite, 0.05, "mnb"), _preds(suite, 0.0, "logreg"), _preds(suite, -0.05, "rf")]
    )
    eff = name_probe_effects(preds)
    assert eff.groupby("model")["effect"].sum().abs().max() == pytest.approx(0, abs=1e-9)
    res = name_probe(eff, n_resamples=200).set_index("model")
    assert res.loc["mnb", "rho_new"] == pytest.approx(1.0)  # follows the names' reputations
    assert res.loc["rf", "rho_all"] == pytest.approx(-1.0)
    assert np.isnan(res.loc["logreg", "rho_all"])  # ignores names: no relationship at all
    assert res.loc["mnb", "n_names_new"] == 4 and res.loc["mnb", "n_names_all"] == 6
    assert res.loc["mnb", "new_pos_minus_neg"] > 0
    assert set(res["family"]) == {"ours"}
    assert res.loc["logreg", "effect_sd"] == pytest.approx(0)  # sized, so rho can't mislead
    assert res.loc["mnb", "max_abs_effect"] > 0
