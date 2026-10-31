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
        + templates.typo(c["typo"], rng)
        + templates.name_swap(c["name_swap"], rng)
    )
    df = pd.DataFrame(rows)  # case_id is set by each generator; INV pairs share one
    df.insert(0, "item_id", [f"beh-{i}" for i in range(len(df))])
    return df[COLUMNS]


def save_suite(df: pd.DataFrame) -> None:
    SUITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUITE_PATH, index=False)


def load_suite() -> pd.DataFrame:
    if not SUITE_PATH.exists():
        raise FileNotFoundError(f"{SUITE_PATH} missing - run `python -m mirage build-suite`")
    return pd.read_csv(SUITE_PATH, keep_default_na=False, dtype={"label": int, "scored": bool})


def case_failures(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per test case with a boolean `failed`. Input: one model's behavioral predictions."""
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
    return pd.concat(out, ignore_index=True)


def failure_rates(preds: pd.DataFrame) -> pd.DataFrame:
    cf = case_failures(preds)
    return (
        cf.groupby(["test_type", "scored"])["failed"]
        .agg(n_cases="size", n_failed="sum", fail_rate="mean")
        .reset_index()
    )
