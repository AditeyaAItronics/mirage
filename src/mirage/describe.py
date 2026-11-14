"""Descriptive tables for the README and the model cards (D58, GitHub issue #1).

Nothing here tests a hypothesis. It records what the data and the trained models look like:
how SST-2 and IMDb differ, which words set them apart, DistilBERT's held-out score after each
training epoch, and the size and selected settings of every classical model. The `describe`
stage writes these to `results/` so the docs quote files, not hand-typed numbers.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from mirage.text import has_negation, normalize, sentences


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


def dataset_comparison(
    sets: dict[str, pd.DataFrame], reference: str = "sst2_train"
) -> pd.DataFrame:
    """One row per set: size, balance, length, vocabulary and how much of it the reference
    (training) set has seen, the share of texts with a negation word, and sentences per text."""
    ref_vocab = {t for text in sets[reference]["text"] for t in _tokens(text)}
    rows = []
    for name, df in sets.items():
        toks = [_tokens(t) for t in df["text"]]
        n_words = np.array([len(t) for t in toks])
        counts = Counter(t for ts in toks for t in ts)
        n_tokens = int(n_words.sum())
        seen_tokens = sum(c for t, c in counts.items() if t in ref_vocab)
        rows.append(
            {
                "dataset": name,
                "n": len(df),
                "pos_frac": round(float(df["label"].mean()), 4),
                "mean_words": round(float(n_words.mean()), 1),
                "median_words": float(np.median(n_words)),
                "p90_words": float(np.percentile(n_words, 90)),
                "mean_sentences": round(
                    float(np.mean([max(len(sentences(t, min_words=1)), 1) for t in df["text"]])), 2
                ),
                "vocab_types": len(counts),
                "types_seen_in_train": round(
                    sum(t in ref_vocab for t in counts) / max(len(counts), 1), 4
                ),
                "tokens_seen_in_train": round(seen_tokens / max(n_tokens, 1), 4),
                "negation_share": round(
                    float(np.mean([has_negation(" ".join(t)) for t in toks])), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def distinctive_words(
    a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str, min_count: int = 50, top: int = 15
) -> pd.DataFrame:
    """Words most over-represented in `a` versus `b` and vice versa: log2 ratio of their rates
    per million tokens (add-one smoothed), alphabetic words seen at least `min_count` times."""
    ca = Counter(t for text in a["text"] for t in _tokens(text))
    cb = Counter(t for text in b["text"] for t in _tokens(text))
    na, nb = sum(ca.values()), sum(cb.values())
    words = [w for w in set(ca) | set(cb) if w.isalpha() and ca[w] + cb[w] >= min_count]
    rows = [
        {
            "word": w,
            f"count_{name_a}": ca[w],
            f"count_{name_b}": cb[w],
            f"per_million_{name_a}": round(1e6 * ca[w] / na, 1),
            f"per_million_{name_b}": round(1e6 * cb[w] / nb, 1),
            "log2_ratio": round(math.log2(((ca[w] + 1) / na) / ((cb[w] + 1) / nb)), 3),
        }
        for w in words
    ]
    d = pd.DataFrame(rows).sort_values(["log2_ratio", "word"], ascending=[False, True])
    more_a = d.head(top).assign(more_common_in=name_a)
    more_b = d.tail(top).iloc[::-1].assign(more_common_in=name_b)
    return pd.concat([more_a, more_b], ignore_index=True)


def training_history(summaries: dict[int, dict]) -> pd.DataFrame:
    """DistilBERT's held-out accuracy and loss after each epoch, per seed, from each run's
    `train_summary.json` (`log_history` of the Hugging Face Trainer). `selected` marks the epoch
    the Trainer kept (best held-out accuracy, `load_best_model_at_end`)."""
    rows = []
    for seed, summary in summaries.items():
        evals = {}
        for h in summary.get("log_history", []):
            # keep the first evaluation of each epoch: after training, the Trainer re-evaluates
            # the *kept* model and logs it under the last epoch, which must not overwrite it
            if "eval_accuracy" in h and "epoch" in h:
                evals.setdefault(float(h["epoch"]), h)
        for epoch, h in sorted(evals.items()):
            rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "heldout_accuracy": round(float(h["eval_accuracy"]), 4),
                    "heldout_loss": round(float(h["eval_loss"]), 4),
                }
            )
    out = pd.DataFrame(rows, columns=["seed", "epoch", "heldout_accuracy", "heldout_loss"])
    if out.empty:
        return out.assign(selected=pd.Series(dtype=bool))
    best = out.loc[out.groupby("seed")["heldout_accuracy"].idxmax(), ["seed", "epoch"]]
    key = set(zip(best["seed"], best["epoch"], strict=True))
    out["selected"] = [(s, e) in key for s, e in zip(out["seed"], out["epoch"], strict=True)]
    return out


def classical_structure(name: str, pipeline) -> dict:
    """Size of one fitted classical pipeline: vocabulary, learned parameters and, for trees,
    their shape. Parameters are the numbers the model stores to make a prediction."""
    vec = pipeline.named_steps["vec"]
    clf = pipeline.named_steps["clf"]
    n_features = len(vec.vocabulary_)
    row = {"model": name, "n_features": n_features}
    if name == "logreg":
        row["n_parameters"] = int(clf.coef_.size + clf.intercept_.size)
    elif name == "mnb":
        row["n_parameters"] = int(clf.feature_log_prob_.size + clf.class_log_prior_.size)
    elif name == "gnb":
        svd = pipeline.named_steps["svd"]
        row["svd_components"] = int(svd.n_components)
        # the SVD's projection matrix, plus a mean and variance per class and component, + priors
        row["n_parameters"] = int(
            svd.components_.size + clf.theta_.size + clf.var_.size + clf.class_prior_.size
        )
    elif name == "dtree":
        row.update(
            depth=int(clf.get_depth()),
            leaves=int(clf.get_n_leaves()),
            n_parameters=int(clf.tree_.node_count),  # one split rule or leaf value per node
        )
    elif name == "rf":
        depths = [t.get_depth() for t in clf.estimators_]
        row.update(
            n_trees=len(clf.estimators_),
            mean_depth=round(float(np.mean(depths)), 1),
            leaves=int(sum(t.get_n_leaves() for t in clf.estimators_)),
            n_parameters=int(sum(t.tree_.node_count for t in clf.estimators_)),
        )
    return row


def transformer_structure(model_dir, history: pd.DataFrame | None = None) -> dict:
    """Size of a saved Hugging Face classifier: parameters counted from its weights file,
    vocabulary from its config; with `history`, the epoch each seed kept and its mean score."""
    import json

    from safetensors import safe_open

    n = 0
    with safe_open(str(model_dir / "model.safetensors"), "np") as f:
        for key in f.keys():
            n += int(np.prod(f.get_slice(key).get_shape()))
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    row = {
        "model": "distilbert",
        "n_features": int(cfg["vocab_size"]),
        "n_parameters": n,
        "layers": int(cfg["n_layers"]),
        "hidden_size": int(cfg["dim"]),
        "attention_heads": int(cfg["n_heads"]),
    }
    if history is not None and len(history):
        kept = history[history["selected"]].sort_values("seed")
        row["selected_params"] = "epoch kept per seed: " + ", ".join(
            f"{int(s)}: {int(e)}" for s, e in zip(kept["seed"], kept["epoch"], strict=True)
        )
        row["heldout_acc"] = round(float(kept["heldout_accuracy"].mean()), 4)
    return row
