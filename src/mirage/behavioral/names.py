"""Which film people carry sentiment in SST-2's training labels (majordecisions.md D32).

The name-shortcut test swaps a name whose training phrases are mostly positive for one whose
phrases are mostly negative. Names are chosen from the *labels*, never from a model's weights,
so the test is equally fair to every model. The chosen pairs live in configs/behavioral.yaml;
`name_associations` regenerates the evidence behind them (results/name_associations.csv).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Surnames of directors and actors likely to appear in SST-2 (Rotten Tomatoes, ~2000-2002).
# Surnames that are also everyday words (cage, stone, lee, bay, ford, grant, cruise, lynch, hunt)
# are left out: their counts would mix the person with the word.
FILM_PEOPLE = (
    "spielberg scorsese soderbergh hitchcock kubrick eastwood allen altman tarantino coppola "
    "burton jackson sandler seagal schwarzenegger stallone murphy carrey williams hanks pitt "
    "nicholson niro streep roberts crowe freeman damon affleck diesel damme bruckheimer "
    "schumacher shyamalan lucas cameron ritchie wayans lopez madonna carey kline nolte travolta "
    "costner gibson depp kidman gere stiller myers ferrell chan wilson spacey hopkins binoche "
    "godard truffaut bergman kurosawa miyazaki herzog demme redford penn douglas willis jolie"
).split()


def name_associations(
    train: pd.DataFrame, candidates=FILM_PEOPLE, min_phrases: int = 15
) -> pd.DataFrame:
    """Per name: training phrases containing it, their positive share, and smoothed log-odds
    relative to the overall positive share (> 0 = the name comes with positive labels)."""
    toks = train["text"].str.lower().str.split().map(set)
    labels = train["label"].to_numpy()
    base = labels.mean()
    rows = []
    for name in candidates:
        mask = toks.map(lambda s, n=name: n in s).to_numpy()
        n = int(mask.sum())
        if n < min_phrases:
            continue
        pos = int(labels[mask].sum())
        log_odds = np.log((pos + 1) / (n - pos + 1)) - np.log(base / (1 - base))
        rows.append({"name": name, "phrases": n, "pos_rate": pos / n, "log_odds": log_odds})
    cols = ["name", "phrases", "pos_rate", "log_odds"]
    return pd.DataFrame(rows, columns=cols).sort_values("log_odds", ascending=False)


def pick_pairs(assoc: pd.DataFrame, k: int = 6) -> list[tuple[str, str]]:
    """The k most positive names paired with the k most negative (strongest with strongest)."""
    pos = assoc[assoc["log_odds"] > 0].head(k)["name"].tolist()
    neg = assoc[assoc["log_odds"] < 0].tail(k)["name"].tolist()[::-1]
    return [(p.capitalize(), n.capitalize()) for p, n in zip(pos, neg, strict=False)]


PROBE_PATH_NAME = "name_probe_suite.csv"


def name_probe_suite(
    assoc: pd.DataFrame, tested: set[str], n_sentences: int, seed: int
) -> pd.DataFrame:
    """Every name in `assoc` placed in the same `n_sentences` neutral sentences (D45).

    The sentences are the name-shortcut templates with a work noun and a mild adjective (which
    alone carries the sentiment), sampled once, so names differ only by the name. `tested` marks
    the names already used in the main suite's pairs; the others are the out-of-sample check.
    """
    import itertools

    from mirage.behavioral import wordlists as W
    from mirage.behavioral.templates import NAME_SHORTCUT_TEMPLATES

    combos = list(
        itertools.product(
            NAME_SHORTCUT_TEMPLATES,
            W.WORK_NOUNS,
            [(1, a) for a in W.MILD_POS] + [(0, a) for a in W.MILD_NEG],
        )
    )
    rng = np.random.default_rng(seed)
    picked = [combos[i] for i in sorted(rng.choice(len(combos), n_sentences, replace=False))]
    rows = []
    for _, r in assoc.iterrows():
        name = r["name"].capitalize()
        for c, (template, noun, (label, adj)) in enumerate(picked):
            rows.append(
                {
                    "item_id": f"name_probe-{r['name']}-{c}",
                    "text": template.format(name=name, noun=noun, adj=adj),
                    "label": label,
                    "name": r["name"],
                    "sentence": c,
                    "log_odds": float(r["log_odds"]),
                    "tested": r["name"] in tested,
                }
            )
    return pd.DataFrame(rows)
