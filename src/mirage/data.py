"""Dataset loading, splitting and the derived evaluation sets.

Every dataset is stored as a parquet file with columns: item_id, text, label (+ extras).
`prepare()` downloads once; everything else reads the parquet files.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from mirage import paths
from mirage.config import load_config
from mirage.text import has_negation, normalize

DATASETS = ("sst2_train", "sst2_heldout", "sst2_val", "sst2_neg", "imdb")


def _download_sst2(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset  # lazy: optional dependency

    ds = load_dataset(name)
    train = ds["train"].to_pandas().rename(columns={"sentence": "text"})
    val = ds["validation"].to_pandas().rename(columns={"sentence": "text"})
    # The test split is deliberately never loaded: its labels are hidden (-1).
    return train[["text", "label"]], val[["text", "label"]]


def _download_imdb(name: str) -> pd.DataFrame:
    from datasets import load_dataset

    return load_dataset(name, split="test").to_pandas()[["text", "label"]]


def split_heldout(train: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified held-out split used for all tuning and DistilBERT checkpoint selection."""
    tr, ho = train_test_split(train, test_size=frac, stratify=train["label"], random_state=seed)
    return tr.reset_index(drop=True), ho.reset_index(drop=True)


def negation_subset(val: pd.DataFrame) -> pd.DataFrame:
    """Validation sentences containing a negation word (matched on normalized text)."""
    mask = val["text"].map(lambda t: has_negation(normalize(t)))
    return val[mask].reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.reset_index(drop=True)
    sample, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return sample.reset_index(drop=True)


def _with_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df.insert(0, "item_id", [f"{prefix}-{i}" for i in range(len(df))])
    df["label"] = df["label"].astype(int)
    return df


def build_splits(
    train: pd.DataFrame, val: pd.DataFrame, imdb: pd.DataFrame, cfg: dict
) -> dict[str, pd.DataFrame]:
    """Pure function: raw frames -> every evaluation set. Testable without downloads."""
    tr, ho = split_heldout(train, cfg["heldout_frac"], cfg["seed"])
    val = _with_ids(val, "sst2_val")
    neg = negation_subset(val)  # keeps the sst2_val item_ids so items can be joined back
    return {
        "sst2_train": _with_ids(tr, "sst2_train"),
        "sst2_heldout": _with_ids(ho, "sst2_heldout"),
        "sst2_val": val,
        "sst2_neg": neg,
        "imdb": _with_ids(stratified_sample(imdb, cfg["imdb_sample_size"], cfg["seed"]), "imdb"),
    }


def data_stats(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        n_words = df["text"].map(lambda t: len(normalize(t).split()))
        rows.append(
            {
                "dataset": name,
                "n": len(df),
                "pos_frac": round(df["label"].mean(), 4),
                "mean_tokens": round(n_words.mean(), 1),
                "median_tokens": float(n_words.median()),
            }
        )
    return pd.DataFrame(rows)


def prepare() -> pd.DataFrame:
    cfg = load_config("data")
    paths.ensure_dirs()
    train, val = _download_sst2(cfg["sst2_dataset"])
    imdb = _download_imdb(cfg["imdb_dataset"])
    splits = build_splits(train, val, imdb, cfg)
    for name, df in splits.items():
        df.to_parquet(paths.DATA / f"{name}.parquet", index=False)
    stats = data_stats(splits)
    stats.to_csv(paths.RESULTS / "data_stats.csv", index=False)
    return stats


def load(name: str) -> pd.DataFrame:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    path = paths.DATA / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m mirage prepare` first")
    return pd.read_parquet(path)
