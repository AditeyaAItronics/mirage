"""All result tables, computed from prediction files only.

Seed policy (majordecisions.md D13): aggregate tables report DistilBERT as mean +/- std over its
seeds; per-item analyses (gaps, McNemar, failures) use its median-accuracy seed. The seed is chosen
on sst2_val; the reported in-distribution numbers come from sst2_test (D22).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm

from mirage import paths
from mirage.behavioral.suite import case_failures, dir_cases
from mirage.metrics import summarize
from mirage.models.base import ABLATION_VARIANTS, external_models
from mirage.predict import load_predictions
from mirage.stats import gap_closed_ci, holm, mcnemar_test, paired_diff_ci, rate_ci

REFERENCE = "distilbert"
INDIST = "sst2_test"  # reported in-distribution test set; sst2_val is for choices only
SETS = ("sst2_val", "sst2_test", "sst2_neg", "imdb", "imdb_edges")


def holm_family(*model_cols: pd.Series, variant: pd.Series | None = None) -> np.ndarray:
    """Holm family label per row, so additions never change earlier adjusted p-values:
    "external" if any model in the row is a ready-made external model (D39), "ablation" if the
    row's variant only exists to test a hypothesis step, e.g. single words (D42), else "ours"."""
    external = set(external_models())
    hit = np.zeros(len(model_cols[0]), dtype=bool)
    for col in model_cols:
        hit |= col.isin(external).to_numpy()
    out = np.where(hit, "external", "ours")
    if variant is not None:
        out = np.where(~hit & variant.isin(ABLATION_VARIANTS).to_numpy(), "ablation", out)
    return out


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
    if "sst2_neg" in out:  # B9: negated sentences never used to pick DistilBERT's seed
        neg = out["sst2_neg"]
        out["sst2_neg_testonly"] = neg[neg.index.str.startswith("sst2_test")]
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
    preds: pd.DataFrame,
    tests=("sst2_neg", "sst2_neg_testonly", "beh_negation", "beh_negation_dir"),
    **ci_kw,
) -> pd.DataFrame:
    """H2/H3: classical negation error minus DistilBERT's along the feature ladder (D42):
    single words (unigram) -> + word pairs (base) -> + negation marking (negmark).

    gap_* = error(classical) - error(DistilBERT), paired. pairs_closed = share of the single-word
    gap removed by adding word pairs; gap_closed = share of the base gap removed by marking;
    ladder_closed = share of the single-word gap removed by both. Each comparison is aligned on
    its own items: DIR pairs count only where every model involved got the affirmative right, so
    the unigram columns never change the base / negmark ones.
    """
    preds = canonical(preds)
    flags = {(m, v): _error_flags(g) for (m, v), g in preds.groupby(["model", "variant"])}
    ref = flags.get((REFERENCE, "base"))
    if ref is None:
        raise ValueError("negation_gap needs DistilBERT predictions as the reference")
    rows = []
    external = set(external_models())  # H2 is our classical models vs our DistilBERT (D39)
    for (m, v), f in flags.items():
        if m == REFERENCE or v != "base" or m in external:
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
            uni = flags.get((m, "unigram"), {}).get(test)
            if uni is not None:
                row |= _ladder(
                    uni, f[test], flags.get((m, "negmark"), {}).get(test), ref[test], ci_kw
                )
            rows.append(row)
    out = pd.DataFrame(rows)
    # One family per question: H2 = every base gap, H3 = every gap-closed ratio (D28); the ladder
    # columns (D42) are families of their own.
    for col in (
        "gap_base_p",
        "gap_negmark_p",
        "gap_closed_p",
        "gap_unigram_p",
        "pairs_closed_p",
        "ladder_closed_p",
    ):
        if col in out:
            out[col.replace("_p", "_p_holm")] = holm(out[col])
    return out


def _ladder(uni: pd.Series, base: pd.Series, negmark, ref: pd.Series, ci_kw: dict) -> dict:
    """The single-word rung of the H2/H3 ladder (D42), each comparison on its own aligned items."""

    def ci(prefix: str, res: dict) -> dict:
        return {
            prefix: res["estimate"],
            f"{prefix}_lo": res["ci_low"],
            f"{prefix}_hi": res["ci_high"],
            f"{prefix}_p": res["p_value"],
        }

    a = pd.concat({"uni": uni, "ref": ref}, axis=1, join="inner")
    out = {"n_unigram": len(a)} | ci("gap_unigram", paired_diff_ci(a["uni"], a["ref"], **ci_kw))
    b = pd.concat({"uni": uni, "base": base, "ref": ref}, axis=1, join="inner")
    out |= ci("pairs_closed", gap_closed_ci(b["uni"], b["base"], b["ref"], **ci_kw))
    if negmark is not None:
        c = pd.concat({"uni": uni, "neg": negmark, "ref": ref}, axis=1, join="inner")
        out |= ci("ladder_closed", gap_closed_ci(c["uni"], c["neg"], c["ref"], **ci_kw))
    return out


def negmark_effect(
    preds: pd.DataFrame,
    tests=("sst2_neg", "sst2_neg_testonly", "beh_negation_dir", "sst2_test", "imdb"),
    **ci_kw,
) -> pd.DataFrame:
    """H3 within each classical model: error(base) - error(variant) on the same items, for every
    negation-marking variant (negmark, negmark_w3, ...).

    Needs no DistilBERT. sst2_test and imdb are controls: marking should help on negation without
    costing accuracy elsewhere. Holm-adjusted across every (model, variant, test) row (D28).
    DIR cases are compared on pairs that are applicable under both variants.
    """
    preds = canonical(preds)
    rows = []
    for (m, variant), g in preds.groupby(["model", "variant"]):
        # marking variants only: the single-word ablation is in negation_gap's ladder (D42)
        if m == REFERENCE or variant == "base" or variant in ABLATION_VARIANTS:
            continue
        base = preds[(preds["model"] == m) & (preds["variant"] == "base")]
        if base.empty:
            continue
        fb = _error_flags(base)
        fn = _error_flags(g)
        for test in tests:
            if test not in fb or test not in fn:
                continue
            both = pd.concat({"base": fb[test], "negmark": fn[test]}, axis=1, join="inner")
            d = paired_diff_ci(both["base"], both["negmark"], **ci_kw)
            rows.append(
                {
                    "model": m,
                    "variant": variant,
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


LENGTH_BINS = (0, 100, 200, 360, np.inf)
LENGTH_LABELS = ("<=100", "101-200", "201-360", ">360 (capped)")


def length_table(preds: pd.DataFrame, **ci_kw) -> pd.DataFrame:
    """C8: IMDb error by original review length (words), per model and variant.

    DistilBERT trains on ~9-word phrases; if its error climbs with length faster than the
    classical models' (which ignore word position), part of its IMDb drop is length, not domain.
    The last bin holds the reviews capped to head + tail words (D23).
    """
    imdb = canonical(preds)
    imdb = imdb[imdb["dataset"] == "imdb"]
    if imdb.empty or "n_words" not in imdb:
        return pd.DataFrame()
    imdb = imdb.assign(
        length=pd.cut(imdb["n_words"], LENGTH_BINS, labels=LENGTH_LABELS, right=True)
    )
    rows = []
    for (m, v, b), g in imdb.groupby(["model", "variant", "length"], observed=True):
        ci = rate_ci((~g["correct"].astype(bool)).to_numpy(), **ci_kw)
        rows.append(
            {
                "model": m,
                "variant": v,
                "length": b,
                "n": ci["n"],
                "error": ci["estimate"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def shift_decomposition(preds: pd.DataFrame) -> pd.DataFrame:
    """B4 (D31): split each model's SST-2 -> IMDb error increase into two parts.

    length_format = err(imdb) - err(imdb_edges): the same reviews, whole vs first + last sentence
                    (paired bootstrap).
    content       = err(imdb_edges) - err(sst2_test): SST-length text, IMDb vs SST-2
                    (independent samples; normal-approximation CI and p).
    Holm-adjusted within each part across models and variants.
    """
    rows = []
    for (m, v), g in canonical(preds).groupby(["model", "variant"]):
        f = _error_flags(g)
        if not {"imdb", "imdb_edges", INDIST} <= set(f):
            continue
        both = pd.concat({"full": f["imdb"], "edges": f["imdb_edges"]}, axis=1, join="inner")
        lf = paired_diff_ci(both["full"], both["edges"])
        e, t = f["imdb_edges"].astype(float), f[INDIST].astype(float)
        diff = e.mean() - t.mean()
        se = np.sqrt(e.var(ddof=1) / len(e) + t.var(ddof=1) / len(t))
        rows.append(
            {
                "model": m,
                "variant": v,
                "err_sst2_test": t.mean(),
                "err_imdb_edges": e.mean(),
                "err_imdb": f["imdb"].mean(),
                "length_format": lf["estimate"],
                "length_format_lo": lf["ci_low"],
                "length_format_hi": lf["ci_high"],
                "length_format_p": lf["p_value"],
                "content": diff,
                "content_lo": diff - 1.96 * se,
                "content_hi": diff + 1.96 * se,
                "content_p": float(2 * norm.sf(abs(diff) / se)) if se > 0 else 1.0,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["family"] = holm_family(out["model"], variant=out["variant"])
    for part in ("length_format", "content"):
        if len(out):
            out[f"{part}_p_holm"] = out.groupby("family")[f"{part}_p"].transform(lambda p: holm(p))
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
    if len(out):
        # One family per dataset: all pairwise comparisons of our models on it (D28). Pairs with
        # an external model form their own family, so adding them leaves the original p-values
        # unchanged (D39).
        out["family"] = holm_family(out["model_a"], out["model_b"])
        out["p_holm"] = out.groupby(["dataset", "family"])["p_value"].transform(lambda p: holm(p))
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


def name_effect(preds: pd.DataFrame, tests=("name_shortcut", "name_swap"), **ci_kw) -> pd.DataFrame:
    """D32: how much does swapping a name move each model's answer?

    name_shortcut: film people whose training phrases are mostly positive (original) vs mostly
    negative (perturbed). shift = mean p_pos(original) - p_pos(perturbed); > 0 means the model
    learned the names' reputations. name_swap (unseen Indian vs Western first names) is the
    contrast. flip_rate = share of pairs whose predicted label changes. Holm per test.
    """
    rows = []
    for (m, v), g in canonical(preds).groupby(["model", "variant"]):
        b = _behavioral(g)
        for tt in tests:
            d = b[b["test_type"] == tt]
            if d.empty:
                continue
            w = d.pivot_table(index="case_id", columns="role", values=["p_pos", "pred"])
            shift = paired_diff_ci(w["p_pos"]["original"], w["p_pos"]["perturbed"], **ci_kw)
            rows.append(
                {
                    "model": m,
                    "variant": v,
                    "test": tt,
                    "n_pairs": len(w),
                    "flip_rate": float((w["pred"]["original"] != w["pred"]["perturbed"]).mean()),
                    "shift": shift["estimate"],
                    "shift_lo": shift["ci_low"],
                    "shift_hi": shift["ci_high"],
                    "shift_p": shift["p_value"],
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["family"] = holm_family(out["model"], variant=out["variant"])
        out["shift_p_holm"] = out.groupby(["test", "family"])["shift_p"].transform(
            lambda p: holm(p)
        )
    return out


def typo_direction(preds: pd.DataFrame, tests=("typo", "typo_noun"), **ci_kw) -> pd.DataFrame:
    """C11 / D37: which way do typos push each model, split by the sentence's sentiment?

    shift = mean p_pos(typo) - p_pos(original) with a paired CI. A model that only loses the
    sentiment word drifts toward its prior from both sides; one that reads "garbled => negative"
    drops sharply for positive sentences and barely moves for negative ones.
    """
    rows = []
    for (m, v), g in canonical(preds).groupby(["model", "variant"]):
        b = _behavioral(g)
        for tt in tests:
            d = b[b["test_type"] == tt]
            if d.empty:
                continue
            w = d.pivot_table(index="case_id", columns="role", values=["p_pos", "pred", "label"])
            for polarity, lab in (("positive", 1), ("negative", 0)):
                x = w[w["label"]["original"] == lab]
                if x.empty:
                    continue
                s = paired_diff_ci(x["p_pos"]["perturbed"], x["p_pos"]["original"], **ci_kw)
                rows.append(
                    {
                        "model": m,
                        "variant": v,
                        "test": tt,
                        "polarity": polarity,
                        "n_pairs": len(x),
                        "flip_rate": float(
                            (x["pred"]["original"] != x["pred"]["perturbed"]).mean()
                        ),
                        "shift": s["estimate"],
                        "shift_lo": s["ci_low"],
                        "shift_hi": s["ci_high"],
                    }
                )
    return pd.DataFrame(rows)


# D40: the four core shortcut tests behind the robustness score, equally weighted. typo_noun is a
# control, name_swap is near-impossible to fail (D32) and the *_neg tests are unscored probes.
ROBUSTNESS_TESTS = {
    "beh_negation_dir": "negation",
    "sst2_neg": "real negation",
    "beh_typo": "typo",
    "beh_name_shortcut": "names",
}


def pair_accuracy(beh: pd.DataFrame, test_type: str) -> pd.Series:
    """Per case: True only if the model gets *every* sentence of the pair right (D40).

    Stricter than the invariance / direction checks in errors_by_type, which a model passes by
    never changing its answer: a model that can't read the sentiment word doesn't flip on a typo
    in it. Requiring both sentences right means robustness can't exceed competence.
    """
    t = beh[beh["test_type"] == test_type]
    return t.groupby("case_id")["correct"].agg(lambda c: bool(c.astype(bool).all()))


def mirage_gap(preds: pd.DataFrame, tests=ROBUSTNESS_TESTS) -> pd.DataFrame:
    """D40: how good a model looks (SST-2 test accuracy) vs how it behaves (robustness).

    robustness = mean over the core tests of the share of cases the model gets fully right
    (accuracy on real negated sentences; both sentences right for behavioral pairs).
    gap = accuracy - robustness. DistilBERT is its median seed, like every per-item table.
    """
    rows = []
    for (m, v), g in canonical(preds).groupby(["model", "variant"]):
        test, beh = g[g["dataset"] == INDIST], g[g["dataset"] == "behavioral"]
        row = {"model": m, "variant": v, "accuracy": float(test["correct"].mean())}
        for t in tests:
            if t.startswith("beh_"):
                passed = pair_accuracy(beh, t.removeprefix("beh_"))
            else:
                passed = g.loc[g["dataset"] == t, "correct"]
            row[f"pass__{t}"] = float(passed.mean()) if len(passed) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    cols = [f"pass__{t}" for t in tests]
    if out.empty:
        return pd.DataFrame()
    out = out.dropna(subset=["accuracy", *cols])
    out["robustness"] = out[cols].mean(axis=1)
    out["gap"] = out["accuracy"] - out["robustness"]
    out["external"] = out["model"].isin(external_models())
    return out.sort_values(["variant", "gap"], ascending=[True, False]).reset_index(drop=True)


def mirage_gap_sensitivity(gap: pd.DataFrame, tests=ROBUSTNESS_TESTS) -> pd.DataFrame:
    """Leave each component out in turn: does the robustness ordering of models hold? (D40)"""
    from scipy.stats import spearmanr

    rows = []
    for v, g in gap.groupby("variant"):
        if len(g) < 3:
            continue
        full = g["robustness"].rank(ascending=False)
        for test in tests:
            rest = [f"pass__{t}" for t in tests if t != test]
            loo = g[rest].mean(axis=1).rank(ascending=False)
            rows.append(
                {
                    "variant": v,
                    "left_out": test,
                    "n_models": len(g),
                    "spearman": float(spearmanr(full, loo).statistic),
                    "same_order": bool((full == loo).all()),
                    "same_top": g["model"].iloc[full.argmin()] == g["model"].iloc[loo.argmin()],
                    "same_bottom": g["model"].iloc[full.argmax()] == g["model"].iloc[loo.argmax()],
                }
            )
    return pd.DataFrame(rows)


def fingerprint_table(
    gap: pd.DataFrame, main: pd.DataFrame, warning: pd.DataFrame | None = None
) -> pd.DataFrame:
    """D40 shortcut fingerprint (base variants): the four pass rates, calibration on IMDb
    (1 - ECE) and how well low confidence flags IMDb errors (AUROC, from the warn stage)."""
    out = gap[gap["variant"] == "base"].drop(columns="variant")
    m = main[main["variant"] == "base"]
    if "ece__imdb" in m:
        out = out.merge(m[["model"]].assign(calibration=1 - m["ece__imdb"]), on="model", how="left")
    if warning is not None and len(warning):
        w = warning[(warning["variant"] == "base") & (warning["predictor"] == "low_conf")]
        out = out.merge(
            w[["model", "auroc"]].rename(columns={"auroc": "warning_auroc"}), on="model", how="left"
        )
    return out


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
        "imdb_by_length": length_table(preds, **kw),
        "shift_decomposition": shift_decomposition(preds),
        "name_effect": name_effect(preds, **kw),
        "typo_direction": typo_direction(preds, **kw),
    }
    tables["mirage_gap"] = mirage_gap(preds)
    tables["mirage_gap_sensitivity"] = mirage_gap_sensitivity(tables["mirage_gap"])
    if (preds["model"] == REFERENCE).any():
        tables["negation_gap"] = negation_gap(preds, **kw)
    for name, df in tables.items():
        df.to_csv(paths.RESULTS / f"{name}.csv", index=False)
    return tables
