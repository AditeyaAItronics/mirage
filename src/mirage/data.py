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
from mirage.text import has_negation, normalize, sentences

DATASETS = (
    "sst2_train",
    "sst2_heldout",
    "sst2_val",
    "sst2_test",
    "sst2_neg",
    "imdb",
    "imdb_edges",
    # G8 (D53): reviews never used before, for the confirmatory check; side sets, not EVAL_SETS
    "imdb_fresh",
    "imdb_fresh_edges",
)
FILMS_PATH = paths.DATA / "imdb_films.parquet"
KEY_WORDS = 40  # a review is identified by its first 40 words (capping keeps the head)


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


def build_imdb(sample: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Each sampled review capped to head + tail words so every model reads the same text
    (shortfalls.md C1). `n_words` and `capped` record what was cut."""
    df = sample.copy()
    df["n_words"] = df["text"].str.split().str.len()
    head, tail = cfg["imdb_head_words"], cfg["imdb_tail_words"]
    df["capped"] = df["n_words"] > head + tail
    df["text"] = df["text"].map(lambda t: head_tail(t, head, tail))
    return df


def build_imdb_edges(sample: pd.DataFrame) -> pd.DataFrame:
    """B4 (D31): each review's first and last sentence only, with the review's label and item_id.

    SST-like length (about 40 words) with IMDb content, so the IMDb drop can be split into a
    length/format part (imdb vs imdb_edges, same reviews) and a content part (imdb_edges vs
    sst2_test). Caveat: the two sentences may not carry the review's verdict, so this set has
    some label noise that the full review doesn't.
    """

    def edges(text: str) -> str:
        s = sentences(text) or [text]
        return s[0] if len(s) == 1 else f"{s[0]} {s[-1]}"

    df = sample[["item_id", "text", "label"]].copy()
    df["text"] = df["text"].map(edges)
    df["n_words"] = df["text"].str.split().str.len()
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
    sample = _with_ids(stratified_sample(imdb, cfg["imdb_sample_size"], cfg["seed"]), "imdb")
    test = _with_ids(drop_seen(test, pd.concat([train["text"], val["text"]])), "sst2_test")
    return {
        "sst2_train": _with_ids(tr, "sst2_train"),
        "sst2_heldout": _with_ids(ho, "sst2_heldout"),
        "sst2_val": val,
        "sst2_test": test,
        "sst2_neg": negation_subset(pd.concat([val, test], ignore_index=True)),
        "imdb": build_imdb(sample, cfg),
        "imdb_edges": build_imdb_edges(sample),
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


def build_fresh_imdb(imdb: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """G8 (D53): `imdb_fresh_sample_size` reviews from the IMDb test split that are *not* in the
    main sample, drawn with `fresh_seed`, capped and cut to edges exactly like `imdb`. The main
    sample is recomputed with the original seed, so the two never overlap."""
    used = stratified_sample(imdb, cfg["imdb_sample_size"], cfg["seed"])["text"]
    rest = imdb[~imdb["text"].isin(set(used))]
    fresh = _with_ids(
        stratified_sample(rest, cfg["imdb_fresh_sample_size"], cfg["fresh_seed"]), "imdb_fresh"
    )
    return {"imdb_fresh": build_imdb(fresh, cfg), "imdb_fresh_edges": build_imdb_edges(fresh)}


def prepare_fresh() -> pd.DataFrame:
    """Build and save the G8 side sets (needs the IMDb download, cached after `prepare`)."""
    cfg = load_config("data")
    sets = build_fresh_imdb(_download_imdb(cfg["imdb_dataset"]), cfg)
    for name, df in sets.items():
        df.to_parquet(paths.DATA / f"{name}.parquet", index=False)
    return data_stats(sets)


def review_key(text: str) -> str:
    return " ".join(str(text).split()[:KEY_WORDS])


def read_archive_films(archive) -> pd.DataFrame:
    """(key, film) for every review in the original IMDb release (aclImdb_v1.tar.gz).

    Line n of `urls_pos.txt` / `urls_neg.txt` is the IMDb page of review file `n_<rating>.txt`;
    the film id is the `tt...` part of that URL. Only text files are read, nothing is extracted.
    """
    import re
    import tarfile

    reviews, urls = {}, {}
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            parts = m.name.split("/")  # aclImdb/<split>/<pos|neg>/<n>_<rating>.txt or urls_*.txt
            if len(parts) == 4 and parts[2] in ("pos", "neg") and parts[3].endswith(".txt"):
                n = int(parts[3].split("_")[0])
                reviews[(parts[1], parts[2], n)] = tar.extractfile(m).read().decode("utf-8")
            elif len(parts) == 3 and parts[2] in ("urls_pos.txt", "urls_neg.txt"):
                lines = tar.extractfile(m).read().decode("utf-8").splitlines()
                urls[(parts[1], parts[2][5:8])] = lines
    rows = []
    for (split, pol, n), text in reviews.items():
        hit = re.search(r"title/(tt\d+)", urls[(split, pol)][n])
        rows.append({"key": review_key(text), "film": hit.group(1) if hit else None})
    return pd.DataFrame(rows).drop_duplicates("key")


def match_films(df: pd.DataFrame, films: pd.DataFrame) -> pd.DataFrame:
    """(item_id, film) for one eval set; reviews that can't be matched get no film (they are
    then their own cluster)."""
    keys = df[["item_id"]].assign(key=df["text"].map(review_key))
    return keys.merge(films, on="key", how="left")[["item_id", "film"]]


def build_imdb_films(sets=("imdb", "imdb_fresh")) -> pd.DataFrame:
    """G7 (D52): film id for every review in `sets`, from the original release, so IMDb results
    can be bootstrapped by film. Downloads the archive (checked against its SHA-256) once."""
    import hashlib
    import urllib.request

    cfg = load_config("data")
    archive = paths.DATA / "raw" / "aclImdb_v1.tar.gz"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(cfg["imdb_archive_url"], archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != cfg["imdb_archive_sha256"]:
        raise ValueError(f"{archive} has SHA-256 {digest}, expected the official release's")
    films = read_archive_films(archive)
    parts = [match_films(load(s), films) for s in sets if (paths.DATA / f"{s}.parquet").exists()]
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(FILMS_PATH, index=False)
    return out


def load(name: str) -> pd.DataFrame:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    path = paths.DATA / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m mirage prepare` first")
    return pd.read_parquet(path)
