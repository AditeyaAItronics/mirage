import numpy as np
import pandas as pd
import pytest

from mirage.predict import make_predictions, validate_predictions


def frame(n=4):
    return pd.DataFrame(
        {"item_id": [f"x{i}" for i in range(n)], "text": ["t"] * n, "label": [0, 1] * (n // 2)}
    )


def test_make_predictions_builds_valid_frame():
    out = make_predictions(frame(), np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.5, 0.5]]))
    assert out["pred"].tolist() == [0, 1, 1, 1]  # 0.5 counts as positive
    assert out["correct"].tolist() == [True, True, False, True]
    validate_predictions(out)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        make_predictions(frame(), np.zeros((3, 2)))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.drop(columns="p_pos"),
        lambda d: d.assign(p_pos=1.5),
        lambda d: d.assign(pred=1 - d["pred"]),
        lambda d: d.assign(item_id="same"),
        lambda d: d.assign(label=-1),  # hidden SST-2 test labels must never get in
    ],
)
def test_validate_rejects_broken_frames(mutate):
    good = make_predictions(frame(), np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.5, 0.5]]))
    with pytest.raises(ValueError):
        validate_predictions(mutate(good))
