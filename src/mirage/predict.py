"""The predictions contract (majordecisions.md D7).

One parquet file per (model, variant, seed, dataset). Everything downstream reads these files only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mirage import paths

REQUIRED = ("item_id", "text", "label", "p_pos", "pred", "correct")
EVAL_SETS = (
    "sst2_heldout",
    "sst2_val",
    "sst2_test",
    "sst2_neg",
    "behavioral",
    "imdb",
    "imdb_edges",  # same item_ids as imdb: first + last sentence of each review (D31)
)
# Side sets outside EVAL_SETS: name_probe (D45), and for the confirmatory check (G8, D53) the
# fresh IMDb sample and the second behavioral suite. Each has its own stage.
SIDE_SETS = ("name_probe", "imdb_fresh", "imdb_fresh_edges", "behavioral_v2")
LONG_TEXT_SETS = frozenset({"imdb", "imdb_fresh"})


def predictions_path(model: str, variant: str, seed: int, dataset: str) -> Path:
    return paths.PREDICTIONS / f"{model}__{variant}__seed{seed}__{dataset}.parquet"


def validate_predictions(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"predictions missing columns {missing}")
    if df["item_id"].duplicated().any():
        raise ValueError("duplicate item_id in predictions")
    if not df["p_pos"].between(0, 1).all():
        raise ValueError("p_pos outside [0, 1]")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("labels must be 0/1")
    if not (df["pred"] == (df["p_pos"] >= 0.5).astype(int)).all():
        raise ValueError("pred must equal p_pos >= 0.5")
    if not (df["correct"] == (df["pred"] == df["label"])).all():
        raise ValueError("correct must equal pred == label")


def make_predictions(eval_df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Attach probabilities to an evaluation frame, keeping its extra columns."""
    if proba.shape != (len(eval_df), 2):
        raise ValueError(f"expected proba shape {(len(eval_df), 2)}, got {proba.shape}")
    out = eval_df.reset_index(drop=True).copy()
    out["p_pos"] = proba[:, 1].astype(float)
    out["pred"] = (out["p_pos"] >= 0.5).astype(int)
    out["correct"] = out["pred"] == out["label"]
    validate_predictions(out)
    return out


def write_predictions(df: pd.DataFrame, model: str, variant: str, seed: int, dataset: str) -> Path:
    validate_predictions(df)
    path = predictions_path(model, variant, seed, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_eval_set(dataset: str) -> pd.DataFrame:
    if dataset == "behavioral":
        from mirage.behavioral.suite import load_suite

        return load_suite()
    if dataset == "behavioral_v2":  # D53: the paraphrased second suite (replicate stage)
        from mirage.behavioral.suite import load_suite_v2

        return load_suite_v2()
    if dataset == "name_probe":  # D45: built by the name-probe stage, never part of EVAL_SETS
        from mirage.behavioral.names import PROBE_PATH_NAME

        return pd.read_csv(paths.RESULTS / PROBE_PATH_NAME)
    from mirage.data import load

    return load(dataset)


def run_predictions(clf, variant: str, seed: int, datasets=EVAL_SETS) -> list[Path]:
    written = []
    for ds in datasets:
        eval_df = load_eval_set(ds)
        proba = clf.predict_proba(eval_df["text"].tolist(), long_text=ds in LONG_TEXT_SETS)
        preds = make_predictions(eval_df, proba)
        written.append(write_predictions(preds, clf.name, variant, seed, ds))
    return written


def load_predictions(
    model: str = "*", variant: str = "*", seed: int | str = "*", dataset: str = "*"
) -> pd.DataFrame:
    """Load and stack matching prediction files, adding model/variant/seed/dataset columns."""
    frames = []
    for f in sorted(paths.PREDICTIONS.glob(f"{model}__{variant}__seed{seed}__{dataset}.parquet")):
        m, v, s, d = f.stem.split("__")
        df = pd.read_parquet(f)
        df.insert(0, "dataset", d)
        df.insert(0, "seed", int(s.removeprefix("seed")))
        df.insert(0, "variant", v)
        df.insert(0, "model", m)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"no predictions match {model}/{variant}/seed{seed}/{dataset} in {paths.PREDICTIONS}"
        )
    return pd.concat(frames, ignore_index=True)


def load_clusters() -> pd.Series:
    """Cluster label per item_id for the bootstrap (G7, D52): `film:<imdb id>` for IMDb reviews
    (from `data.build_imdb_films`). Empty if that file hasn't been built; items without a label
    are their own cluster. Behavioral clusters come from the predictions' own `word` column."""
    from mirage.data import FILMS_PATH

    if not FILMS_PATH.exists():
        return pd.Series(dtype=object)
    f = pd.read_parquet(FILMS_PATH).dropna(subset=["film"])
    return ("film:" + f["film"]).set_axis(f["item_id"])
