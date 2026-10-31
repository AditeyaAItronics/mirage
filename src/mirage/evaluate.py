"""All result tables, computed from prediction files only.

Seed policy (majordecisions.md D13): aggregate tables report DistilBERT as mean +/- std over its
seeds; per-item analyses (gaps, McNemar, failures) use its median-accuracy seed. The seed is chosen
on sst2_val; the reported in-distribution numbers come from sst2_test (D22).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from mirage import paths
from mirage.behavioral.suite import case_failures, dir_cases
from mirage.metrics import summarize
from mirage.predict import load_predictions
from mirage.stats import gap_closed_ci, holm, mcnemar_test, paired_diff_ci, rate_ci

REFERENCE = "distilbert"
INDIST = "sst2_test"  # reported in-distribution test set; sst2_val is for choices only
SETS = ("sst2_val", "sst2_test", "sst2_neg", "imdb")


def median_seed(preds: pd.DataFrame, model: str = REFERENCE) -> int:
    """Seed whose sst2_val accuracy is the median across seeds (lower-middle for even counts)."""
    val = preds[(preds["model"] == model) & (preds["dataset"] == "sst2_val")]
    if val.empty:
        raise ValueError(
            "median_seed needs sst2_val predictions: pass all predictions, then filter by dataset"
        )
    acc = val.groupby("seed")["correct"].mean().sort_values()
    return int(acc.index[(len(acc) - 1) // 2])


def canonical(preds: pd.DataFrame) -> pd.DataFrame:
    """Keep one seed per model: DistilBERT's median seed, the only seed for everything else."""
    if (preds["model"] == REFERENCE).any():
        keep = (preds["model"] != REFERENCE) | (preds["seed"] == median_seed(preds))
        preds = preds[keep]
    return preds


def main_table(preds: pd.DataFrame, datasets=SETS) -> pd.DataFrame:
    rows = []
    for (m, v, s, d), g in preds[preds["dataset"].isin(datasets)].groupby(
        ["model", "variant", "seed", "dataset"]
    ):
        rows.append({"model": m, "variant": v, "seed": s, "dataset": d, **summarize(g)})
    per_seed = pd.DataFrame(rows)
    agg = per_seed.groupby(["model", "variant", "dataset"]).agg(
        n_seeds=("seed", "nunique"),
        accuracy=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1=("macro_f1", "mean"),
        ece=("ece", "mean"),
    )
    wide = agg.unstack("dataset")
    wide.columns = [f"{metric}__{ds}" for metric, ds in wide.columns]
    if {f"accuracy__{INDIST}", "accuracy__imdb"} <= set(wide.columns):
        wide["imdb_acc_drop"] = wide[f"accuracy__{INDIST}"] - wide["accuracy__imdb"]
    return wide.reset_index()


def _error_flags(preds: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-item (or per-case) failure flags for every test type of one model, indexed by id."""
    out = {}
    for ds in SETS:
        g = preds[preds["dataset"] == ds]
        if len(g):
            out[ds] = (~g["correct"].astype(bool)).set_axis(g["item_id"]).rename(ds)
    beh = preds[preds["dataset"] == "behavioral"]
    if len(beh):
        cf = case_failures(beh)
        for tt, g in cf.groupby("test_type"):
            out[f"beh_{tt}"] = g["failed"].set_axis(g["case_id"]).rename(tt)
    return out


def error_table(preds: pd.DataFrame, **ci_kw) -> pd.DataFrame:
    """H1: error rate per test type with CI, and relative to the model's SST-2 test error."""
    rows = []
    for (m, v), g in canonical(preds).groupby(["model", "variant"]):
        flags = _error_flags(g)
        base_err = flags[INDIST].mean() if INDIST in flags else np.nan
        for test, f in flags.items():
            ci = rate_ci(f.to_numpy(), **ci_kw)
            rows.append(
                {
                    "model": m,
                    "variant": v,
                    "test": test,
                    "error": ci["estimate"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "n": ci["n"],
                    "relative_to_test": ci["estimate"] / base_err if base_err else np.nan,
                }
            )
    return pd.DataFrame(rows)


def negation_gap(
    preds: pd.DataFrame, tests=("sst2_neg", "beh_negation", "beh_negation_dir"), **ci_kw
) -> pd.DataFrame:
    """H2/H3: classical negation error minus DistilBERT's, before/after negation marking."""
    preds = canonical(preds)
    flags = {(m, v): _error_flags(g) for (m, v), g in preds.groupby(["model", "variant"])}
    ref = flags.get((REFERENCE, "base"))
    if ref is None:
        raise ValueError("negation_gap needs DistilBERT predictions as the reference")
    rows = []
    for (m, v), f in flags.items():
        if m == REFERENCE or v != "base":
            continue
        for test in tests:
            if test not in f or test not in ref:
                continue
            aligned = pd.concat(
                {
                    "base": f[test],
                    "ref": ref[test],
                    "negmark": flags.get((m, "negmark"), {}).get(test),
                },
                axis=1,
                join="inner",
            )
            row = {"model": m, "test": test, "n": len(aligned)}
            gap = paired_diff_ci(aligned["base"], aligned["ref"], **ci_kw)
            row |= {
                "gap_base": gap["estimate"],
                "gap_base_lo": gap["ci_low"],
                "gap_base_hi": gap["ci_high"],
                "gap_base_p": gap["p_value"],
            }
            if "negmark" in aligned and aligned["negmark"].notna().all():
                g1 = paired_diff_ci(aligned["negmark"], aligned["ref"], **ci_kw)
                closed = gap_closed_ci(aligned["base"], aligned["negmark"], aligned["ref"], **ci_kw)
                row |= {
                    "gap_negmark": g1["estimate"],
                    "gap_negmark_lo": g1["ci_low"],
                    "gap_negmark_hi": g1["ci_high"],
                    "gap_negmark_p": g1["p_value"],
                    "gap_closed": closed["estimate"],
                    "gap_closed_lo": closed["ci_low"],
                    "gap_closed_hi": closed["ci_high"],
                    "gap_closed_p": closed["p_value"],
                }
            rows.append(row)
    out = pd.DataFrame(rows)
    # One family per question: H2 = every base gap, H3 = every gap-closed ratio (D28).
    for col in ("gap_base_p", "gap_negmark_p", "gap_closed_p"):
        if col in out:
            out[col.replace("_p", "_p_holm")] = holm(out[col])
    return out


def negmark_effect(
    preds: pd.DataFrame,
    tests=("sst2_neg", "beh_negation_dir", "sst2_test", "imdb"),
    **ci_kw,
) -> pd.DataFrame:
    """H3 within each classical model: error(base) - error(negmark) on the same items.

    Needs no DistilBERT. sst2_test and imdb are controls: marking should help on negation without
    costing accuracy elsewhere. Holm-adjusted across every (model, test) row (D28).
    DIR cases are compared on pairs that are applicable under both variants.
    """
    preds = canonical(preds)
    rows = []
    for m, g in preds.groupby("model"):
        if m == REFERENCE or not {"base", "negmark"} <= set(g["variant"]):
            continue
        fb = _error_flags(g[g["variant"] == "base"])
        fn = _error_flags(g[g["variant"] == "negmark"])
        for test in tests:
            if test not in fb or test not in fn:
                continue
            both = pd.concat({"base": fb[test], "negmark": fn[test]}, axis=1, join="inner")
            d = paired_diff_ci(both["base"], both["negmark"], **ci_kw)
            rows.append(
                {
                    "model": m,
                    "test": test,
                    "n": len(both),
                    "error_base": float(both["base"].mean()),
                    "error_negmark": float(both["negmark"].mean()),
                    "reduction": d["estimate"],
                    "reduction_lo": d["ci_low"],
                    "reduction_hi": d["ci_high"],
                    "p_value": d["p_value"],
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_value"])
        out["significant"] = out["p_holm"] < 0.05
    return out


def mcnemar_table(preds: pd.DataFrame, datasets=SETS) -> pd.DataFrame:
    preds = canonical(preds)
    preds = preds[preds["variant"] == "base"]
    rows = []
    for ds in datasets:
        d = preds[preds["dataset"] == ds]
        wide = d.pivot_table(index="item_id", columns="model", values="correct", aggfunc="first")
        for a, b in itertools.combinations(sorted(wide.columns), 2):
            pair = wide[[a, b]].dropna()
            rows.append(
                {"dataset": ds, "model_a": a, "model_b": b, **mcnemar_test(pair[a], pair[b])}
            )
    out = pd.DataFrame(rows)
    if len(out):  # one family per dataset: all pairwise model comparisons on it (D28)
        out["p_holm"] = out.groupby("dataset")["p_value"].transform(lambda p: holm(p))
        out["significant"] = out["p_holm"] < 0.05
    return out


def _behavioral(g: pd.DataFrame) -> pd.DataFrame:
    return g[g["dataset"] == "behavioral"]


def word_errors(preds: pd.DataFrame) -> pd.DataFrame:
    """Failure rate per behavioral test and sentiment word, so one odd word can't drive a result."""
    rows = []
    for (m, v), g in canonical(preds).groupby(["model", "variant"]):
        b = _behavioral(g)
        if b.empty:
            continue
        words = b.drop_duplicates("case_id")[["case_id", "word"]]
        cf = case_failures(b).merge(words, on="case_id")
        agg = cf.groupby(["test_type", "word"])["failed"].agg(n="size", fail_rate="mean")
        rows.append(agg.reset_index().assign(model=m, variant=v))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def negation_control(preds: pd.DataFrame, **ci_kw) -> pd.DataFrame:
    """D1 control: negation failures restricted to positive words the model always gets right in
    the simple test. If a model fails "not <word>" only for words it already misreads, the
    failure says nothing about negation."""
    rows = []
    for (m, v), g in canonical(preds).groupby(["model", "variant"]):
        b = _behavioral(g)
        if b.empty:
            continue
        simple = b[(b["test_type"] == "simple") & (b["label"] == 1)]
        always_right = simple.groupby("word")["correct"].all()
        known = set(always_right[always_right].index)
        neg = b[b["test_type"] == "negation"]
        failed = ~neg["correct"].astype(bool)
        row = {"model": m, "variant": v, "n_words_known": len(known), "n_words": len(always_right)}
        for name, f in (("all", failed), ("known_words", failed[neg["word"].isin(known)])):
            if len(f):
                ci = rate_ci(f.to_numpy(), **ci_kw)
                row |= {
                    f"neg_error_{name}": ci["estimate"],
                    f"neg_error_{name}_lo": ci["ci_low"],
                    f"neg_error_{name}_hi": ci["ci_high"],
                    f"n_{name}": ci["n"],
                }
        rows.append(row)
    return pd.DataFrame(rows)


def negation_direction(preds: pd.DataFrame, **ci_kw) -> pd.DataFrame:
    """D5 / D27: does adding "not" move each model's own answer?

    flip_failure: among pairs whose affirmative sentence the model gets right, the share where the
    negated twin keeps the same label. Free of label bias: a model that always says "negative"
    has no applicable positive pairs instead of a perfect score.
    no_move: share of all pairs where p_pos did not move in the expected direction at all.
    """
    rows = []
    for (m, v), g in canonical(preds).groupby(["model", "variant"]):
        dc = dir_cases(_behavioral(g))
        for tt, d in dc.groupby("test_type"):
            row = {
                "model": m,
                "variant": v,
                "test": tt,
                "scored": bool(d["scored"].iloc[0]),
                "n_pairs": len(d),
                "applicable": float(d["applicable"].mean()),
                "mean_shift": float(d["shift"].mean()),
            }
            for name, f in (
                ("flip_failure", d.loc[d["applicable"], "failed"]),
                ("no_move", d["no_move"]),
            ):
                if len(f):
                    ci = rate_ci(f.to_numpy().astype(bool), **ci_kw)
                    row |= {
                        name: ci["estimate"],
                        f"{name}_lo": ci["ci_low"],
                        f"{name}_hi": ci["ci_high"],
                    }
            rows.append(row)
    return pd.DataFrame(rows)


def run(ci_resamples: int = 1000) -> dict[str, pd.DataFrame]:
    preds = load_predictions()
    kw = {"n_resamples": ci_resamples}
    tables = {
        "main_table": main_table(preds),
        "errors_by_type": error_table(preds, **kw),
        "mcnemar": mcnemar_table(preds),
        "word_errors": word_errors(preds),
        "negation_control": negation_control(preds, **kw),
        "negation_direction": negation_direction(preds, **kw),
        "negmark_effect": negmark_effect(preds, **kw),
    }
    if (preds["model"] == REFERENCE).any():
        tables["negation_gap"] = negation_gap(preds, **kw)
    for name, df in tables.items():
        df.to_csv(paths.RESULTS / f"{name}.csv", index=False)
    return tables
