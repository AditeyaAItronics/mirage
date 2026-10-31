"""Dataset loading, splitting and the derived evaluation sets.

Every dataset is stored as a parquet file with columns: item_id, text, label (+ extras).
`prepare()` downloads once; everything else reads the parquet files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mirage import paths
from mirage.config import load_config
from mirage.text import has_negation, normalize

DATASETS = ("sst2_train", "sst2_heldout", "sst2_val", "sst2_test", "sst2_neg", "imdb")


def _download_sst2(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset  # lazy: optional dependency

    ds = load_dataset(name)
    train = ds["train"].to_pandas().rename(columns={"sentence": "text"})
    val = ds["validation"].to_pandas().rename(columns={"sentence": "text"})
    # GLUE's test split is never loaded: its labels are hidden (-1). See _download_sst2_test.
    return train[["text", "label"]], val[["text", "label"]]


def _download_sst2_test(name: str) -> pd.DataFrame:
    """SST-2 test sentences with the original public Stanford labels (majordecisions.md D22)."""
    from datasets import load_dataset

    return load_dataset(name, split="test").to_pandas()[["text", "label"]]


def _download_imdb(name: str) -> pd.DataFrame:
    from datasets import load_dataset

    return load_dataset(name, split="test").to_pandas()[["text", "label"]]


def _contained(short: str, long: str) -> bool:
    """`short` occurs in `long` on token boundaries (both already whitespace-canonical)."""
    return f" {short} " in f" {long} "


def _candidates(tokens: list[str], postings: dict[str, set[int]]) -> set[int]:
    """Texts that contain every token of `tokens` (smallest posting list first)."""
    lists = sorted((postings.get(t, set()) for t in set(tokens)), key=len)
    if not lists:
        return set()
    out = set(lists[0])
    for p in lists[1:]:
        out &= p
    return out


def heldout_mask(texts: list[str], frac: float, seed: int) -> np.ndarray:
    """Leak-free held-out rows for SST-2's phrase-level training set (majordecisions.md D21).

    SST-2 train has one row per parse-tree phrase, so a random row split puts most held-out
    phrases inside a training sentence. Instead: find the *roots* (phrases contained in no other
    phrase, i.e. sentences), shuffle them, and hold out the first k. A phrase is held out only if
    *every* root that contains it is held out, so no held-out phrase occurs in any training phrase.
    k is the smallest number of roots giving at least `frac` of the rows.
    """
    canon = [" ".join(t.split()) for t in texts]
    uniq = list(dict.fromkeys(canon))
    toks = [u.split() for u in uniq]
    postings: dict[str, set[int]] = {}
    for i, ts in enumerate(toks):
        for t in set(ts):
            postings.setdefault(t, set()).add(i)

    def is_root(i: int) -> bool:
        return not any(
            len(uniq[j]) > len(uniq[i]) and _contained(uniq[i], uniq[j])
            for j in _candidates(toks[i], postings)
        )

    roots = [i for i in range(len(uniq)) if toks[i] and is_root(i)]
    rank = dict(zip(roots, np.random.default_rng(seed).permutation(len(roots)), strict=True))
    root_postings = {t: p & rank.keys() for t, p in postings.items()}

    # Latest-ranked root containing each phrase: it is held out iff that rank < k.
    last = np.full(len(uniq), len(roots), dtype=np.int64)  # no root (empty text) -> always train
    for i, ts in enumerate(toks):
        if not ts:
            continue
        ranks = [rank[j] for j in _candidates(ts, root_postings) if _contained(uniq[i], uniq[j])]
        last[i] = max(ranks)

    pos = {u: i for i, u in enumerate(uniq)}
    row_last = last[[pos[c] for c in canon]]
    n_heldout = max(int(np.ceil(frac * len(canon))), 1)
    k = min(int(np.sort(row_last)[n_heldout - 1]) + 1, len(roots))
    return row_last < k


def split_heldout(train: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sentence-grouped held-out split used for all tuning and DistilBERT checkpoint selection."""
    mask = heldout_mask(train["text"].tolist(), frac, seed)
    return train[~mask].reset_index(drop=True), train[mask].reset_index(drop=True)


def negation_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Sentences containing a negation word (matched on normalized text)."""
    mask = df["text"].map(lambda t: has_negation(normalize(t)))
    return df[mask].reset_index(drop=True)


def drop_seen(df: pd.DataFrame, seen: pd.Series) -> pd.DataFrame:
    """Remove rows whose (whitespace-canonical, lowercased) text also occurs in `seen`."""
    key = lambda s: s.map(lambda t: " ".join(t.lower().split()))  # noqa: E731
    return df[~key(df["text"]).isin(set(key(seen)))].reset_index(drop=True)


def head_tail(text: str, head: int, tail: int) -> str:
    """Keep the first `head` and last `tail` words: reviews often state the verdict at the end."""
    words = text.split()
    if len(words) <= head + tail:
        return text
    return " ".join(words[:head] + words[-tail:])


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


def build_imdb(imdb: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stratified sample, each review capped to head + tail words so every model reads the same
    text (shortfalls.md C1). `n_words` and `capped` record what was cut."""
    df = _with_ids(stratified_sample(imdb, cfg["imdb_sample_size"], cfg["seed"]), "imdb")
    df["n_words"] = df["text"].str.split().str.len()
    head, tail = cfg["imdb_head_words"], cfg["imdb_tail_words"]
    df["capped"] = df["n_words"] > head + tail
    df["text"] = df["text"].map(lambda t: head_tail(t, head, tail))
    return df


def build_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    imdb: pd.DataFrame,
    cfg: dict,
) -> dict[str, pd.DataFrame]:
    """Pure function: raw frames -> every evaluation set. Testable without downloads.

    sst2_val is used for choices (DistilBERT's median seed); sst2_test is the reported test set.
    The negation subset pools both, keeping their item_ids so items can be joined back.
    """
    tr, ho = split_heldout(train, cfg["heldout_frac"], cfg["seed"])
    val = _with_ids(val, "sst2_val")
    test = _with_ids(drop_seen(test, pd.concat([train["text"], val["text"]])), "sst2_test")
    return {
        "sst2_train": _with_ids(tr, "sst2_train"),
        "sst2_heldout": _with_ids(ho, "sst2_heldout"),
        "sst2_val": val,
        "sst2_test": test,
        "sst2_neg": negation_subset(pd.concat([val, test], ignore_index=True)),
        "imdb": build_imdb(imdb, cfg),
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
    test = _download_sst2_test(cfg["sst2_test_dataset"])
    imdb = _download_imdb(cfg["imdb_dataset"])
    splits = build_splits(train, val, test, imdb, cfg)
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
