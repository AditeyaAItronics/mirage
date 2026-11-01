"""D40: robustness score, leave-one-out sensitivity, fingerprint table and figures."""

import pandas as pd
import pytest

from mirage.evaluate import (
    ROBUSTNESS_TESTS,
    fingerprint_table,
    mirage_gap,
    mirage_gap_sensitivity,
    pair_accuracy,
)


def preds_frame(spec: dict[str, dict]) -> pd.DataFrame:
    """Prediction rows: sst2_test / sst2_neg accuracy, and behavioral pairs given as
    (n_both_right, n_one_wrong) per test type."""
    rows = []
    for m, sp in spec.items():
        for ds in ("sst2_test", "sst2_neg"):
            n_ok = int(sp[ds] * 10)
            rows += [
                {"model": m, "dataset": ds, "item_id": f"{ds}{i}", "correct": i < n_ok}
                for i in range(10)
            ]
        for tt in ("negation_dir", "typo", "name_shortcut"):
            ok, bad = sp[tt]
            for c in range(ok + bad):
                for role in ("original", "perturbed"):
                    rows.append(
                        {
                            "model": m,
                            "dataset": "behavioral",
                            "item_id": f"{tt}{c}{role}",
                            "test_type": tt,
                            "case_id": f"{tt}{c}",
                            "role": role,
                            "correct": c < ok or role == "original",
                        }
                    )
    return pd.DataFrame(rows).assign(variant="base", seed=0)


SPEC = {
    # looks good, behaves badly
    "mnb": {
        "sst2_test": 0.8,
        "sst2_neg": 0.7,
        "negation_dir": (4, 6),
        "typo": (6, 4),
        "name_shortcut": (5, 5),
    },
    "logreg": {
        "sst2_test": 0.8,
        "sst2_neg": 0.8,
        "negation_dir": (7, 3),
        "typo": (5, 5),
        "name_shortcut": (10, 0),
    },
    "roberta_large": {
        "sst2_test": 0.9,
        "sst2_neg": 0.9,
        "negation_dir": (9, 1),
        "typo": (9, 1),
        "name_shortcut": (10, 0),
    },
}


def test_score_on_hand_made_numbers():
    g = mirage_gap(preds_frame(SPEC)).set_index("model")
    assert g.loc["mnb", "accuracy"] == pytest.approx(0.8)
    assert g.loc["mnb", "pass__beh_negation_dir"] == pytest.approx(0.4)
    assert g.loc["mnb", "robustness"] == pytest.approx((0.4 + 0.7 + 0.6 + 0.5) / 4)
    assert g.loc["mnb", "gap"] == pytest.approx(0.8 - 0.55)
    assert bool(g.loc["roberta_large", "external"]) and not bool(g.loc["mnb", "external"])
    assert g.index[0] == "mnb"  # sorted by gap, largest first


def test_pair_needs_both_sentences_right():
    beh = pd.DataFrame(
        {
            "test_type": "typo",
            "case_id": ["a", "a", "b", "b", "c", "c"],
            "correct": [True, True, True, False, False, False],
        }
    )
    assert pair_accuracy(beh, "typo").to_dict() == {"a": True, "b": False, "c": False}


def test_missing_component_drops_model():
    p = preds_frame(SPEC)
    p = p[~((p["model"] == "mnb") & (p["test_type"] == "typo"))]
    assert "mnb" not in set(mirage_gap(p)["model"])


def test_sensitivity_flags_order_change():
    s = mirage_gap_sensitivity(mirage_gap(preds_frame(SPEC))).set_index("left_out")
    assert set(s.index) == set(ROBUSTNESS_TESTS)
    assert s["same_bottom"].all()  # mnb is worst whatever is left out
    assert s["spearman"].between(-1, 1).all()


def test_fingerprint_merges_calibration_and_warning():
    gap = mirage_gap(preds_frame(SPEC))
    main = pd.DataFrame({"model": list(SPEC), "variant": "base", "ece__imdb": [0.2, 0.05, 0.03]})
    warn = pd.DataFrame(
        {
            "model": ["mnb", "mnb", "logreg"],
            "variant": "base",
            "predictor": ["low_conf", "distance", "low_conf"],
            "auroc": [0.6, 0.5, 0.84],
        }
    )
    fp = fingerprint_table(gap, main, warn).set_index("model")
    assert fp.loc["mnb", "calibration"] == pytest.approx(0.8)
    assert fp.loc["mnb", "warning_auroc"] == pytest.approx(0.6)  # low_conf, not distance
    assert pd.isna(fp.loc["roberta_large", "warning_auroc"])


def test_figures_render(tmp_path):
    from mirage import plots

    gap = mirage_gap(preds_frame(SPEC))
    main = pd.DataFrame({"model": list(SPEC), "variant": "base", "ece__imdb": [0.2, 0.05, 0.03]})
    fp = fingerprint_table(gap, main)
    for slide in (False, True):
        assert plots.mirage_gap_chart(gap, slide=slide, out_dir=tmp_path).exists()
        assert plots.shortcut_fingerprint(fp, slide=slide, out_dir=tmp_path).exists()
