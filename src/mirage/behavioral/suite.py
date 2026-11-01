"""Build, store and score the behavioral test suite.

The suite is saved as results/behavioral_suite.csv - a reusable deliverable on its own.
Scoring works on prediction files (majordecisions.md D7), never on models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mirage import paths
from mirage.behavioral import templates
from mirage.config import load_config

SUITE_PATH = paths.RESULTS / "behavioral_suite.csv"
COLUMNS = ["item_id", "text", "label", "test_type", "kind", "case_id", "role", "scored", "word"]


def build_suite(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config("behavioral")
    rng = np.random.default_rng(cfg["seed"])
    c = cfg["counts"]
    rows = (
        templates.simple(c["simple"], rng)
        + templates.negation(c["negation_pos"], c["negation_neg"], rng)
        + templates.negation_dir(c["negation_dir"], c["negation_dir_neg"], rng)
        + templates.typo(c["typo"], rng)
        + templates.name_swap(c["name_swap"], rng)
        # last, so adding it didn't reshuffle the random draws of the tests above
        + templates.name_shortcut(c["name_shortcut"], [tuple(p) for p in cfg["name_pairs"]], rng)
        + templates.typo_noun(c["typo_noun"], rng)  # appended last: earlier draws unchanged
    )
    df = pd.DataFrame(rows)  # case_id is set by each generator; INV/DIR pairs share one
    df.insert(0, "item_id", [f"beh-{i}" for i in range(len(df))])
    return df[COLUMNS]


def save_suite(df: pd.DataFrame) -> None:
    SUITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUITE_PATH, index=False)


def load_suite() -> pd.DataFrame:
    if not SUITE_PATH.exists():
        raise FileNotFoundError(f"{SUITE_PATH} missing - run `python -m mirage build-suite`")
    return pd.read_csv(SUITE_PATH, keep_default_na=False, dtype={"label": int, "scored": bool})


def dir_cases(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per DIR case: both predictions and probabilities side by side.

    `applicable` = the model gets the affirmative sentence right; only then can negation be
    tested. `failed` (applicable cases only) = the negated twin gets the same prediction.
    `no_move` = p_pos did not move in the expected direction at all (uses every case).
    """
    d = preds[preds["kind"] == "DIR"]
    if d.empty:
        return pd.DataFrame(
            columns=["test_type", "case_id", "scored", "applicable", "failed", "no_move", "shift"]
        )
    wide = d.pivot_table(
        index=["test_type", "case_id", "scored"],
        columns="role",
        values=["pred", "p_pos", "label"],
        aggfunc="first",
    )
    wide.columns = [f"{v}_{r}" for v, r in wide.columns]
    wide = wide.reset_index()
    wide["applicable"] = wide["pred_original"] == wide["label_original"]
    wide["failed"] = wide["pred_perturbed"] == wide["pred_original"]
    toward = wide["label_perturbed"] - wide["label_original"]  # +1: p_pos should rise, -1: fall
    wide["shift"] = (wide["p_pos_perturbed"] - wide["p_pos_original"]) * toward
    wide["no_move"] = wide["shift"] <= 0
    return wide


def case_failures(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per test case with a boolean `failed`. Input: one model's behavioral predictions.

    DIR cases count only when applicable (the affirmative sentence is classified right).
    """
    mft = preds[preds["kind"] == "MFT"]
    out = [
        mft.assign(failed=~mft["correct"].astype(bool))[
            ["test_type", "case_id", "scored", "failed"]
        ]
    ]
    inv = preds[preds["kind"] == "INV"]
    if len(inv):
        wide = inv.pivot_table(
            index=["test_type", "case_id", "scored"], columns="role", values="pred", aggfunc="first"
        ).reset_index()
        wide["failed"] = wide["original"] != wide["perturbed"]
        out.append(wide[["test_type", "case_id", "scored", "failed"]])
    dc = dir_cases(preds)
    if len(dc):
        dc = dc[dc["applicable"]]
        out.append(dc[["test_type", "case_id", "scored", "failed"]].astype({"failed": bool}))
    return pd.concat(out, ignore_index=True)


def failure_rates(preds: pd.DataFrame) -> pd.DataFrame:
    cf = case_failures(preds)
    return (
        cf.groupby(["test_type", "scored"])["failed"]
        .agg(n_cases="size", n_failed="sum", fail_rate="mean")
        .reset_index()
    )
