"""G8 (D53): score the pre-registered claims (docs/replication/confirmatory_claims.md) on untouched data.

The fresh data are the second behavioral suite (`behavioral_v2`, frame set B, D10) and a second
IMDb sample (`imdb_fresh`, `imdb_fresh_edges`). They are renamed to the dataset names the original
analyses expect, so every claim is scored by the *same* functions as the original result, with the
same cluster intervals (G7). SST-2 sets stay as they are: they hold H1's comparison sentences and
DistilBERT's seed choice. A claim whose data aren't in yet is "pending", never passed.

Reads prediction files only (plus the warn stage's distances for imdb_fresh, passed in).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mirage.evaluate import (
    _cluster_map,
    _clusters,
    canonical,
    h1_test,
    main_table,
    negation_gap,
    shift_decomposition,
    typo_direction,
)
from mirage.stats import diff_independent_ci

FRESH = {
    "behavioral_v2": "behavioral",
    "imdb_fresh": "imdb",
    "imdb_fresh_edges": "imdb_edges",
}
TRANSFORMERS = ("distilbert", "distilbert_hf", "roberta_large")
CLASSICAL = ("logreg", "mnb", "rf", "dtree", "gnb")
ALL_MODELS = (*CLASSICAL, *TRANSFORMERS)
WARN_MODELS = (*TRANSFORMERS, "logreg", "rf", "mnb")


def fresh_view(preds: pd.DataFrame) -> pd.DataFrame:
    """The fresh data under the original dataset names, plus the SST-2 sets the analyses need."""
    keep = preds[preds["dataset"].isin([*FRESH, "sst2_val", "sst2_test", "sst2_neg"])]
    return keep.assign(dataset=keep["dataset"].replace(FRESH))


def _row(n, connects, claim, rule, passed, detail):
    return {
        "claim": n,
        "connects_to": connects,
        "claim_text": claim,
        "pass_rule": rule,
        "status": "pending" if passed is None else ("replicated" if passed else "not replicated"),
        "detail": detail,
    }


def _get(df: pd.DataFrame, **where) -> pd.DataFrame:
    """Rows matching every `column=value` (or list of values); none if a column is missing."""
    if not set(where) <= set(df.columns):
        return df.iloc[0:0]
    m = pd.Series(True, index=df.index)
    for k, v in where.items():
        m &= df[k].isin(v) if isinstance(v, list | tuple) else df[k] == v
    return df[m]


def _all_present(df: pd.DataFrame, models) -> bool:
    return set(models) <= set(df["model"])


def _fmt(df: pd.DataFrame, cols) -> str:
    return "; ".join(
        f"{r['model']}: " + ", ".join(f"{c} {r[c]:+.3f}" for c in cols if c in r)
        for _, r in df.iterrows()
    )


def flip_difference(preds: pd.DataFrame, model: str, n_resamples: int = 1000) -> dict | None:
    """Claim 5: flip rate on positive minus negative typo pairs, cluster CI by adjective."""
    b = canonical(preds)
    b = b[(b["model"] == model) & (b["variant"] == "base") & (b["dataset"] == "behavioral")]
    d = b[b["test_type"] == "typo"]
    if d.empty:
        return None
    w = d.pivot_table(index="case_id", columns="role", values=["pred", "label"])
    flip = (w["pred"]["original"] != w["pred"]["perturbed"]).astype(float)
    pos = w["label"]["original"] == 1
    cmap = _cluster_map(b)
    return diff_independent_ci(
        flip[pos].to_numpy(),
        flip[~pos].to_numpy(),
        n_resamples=n_resamples,
        clusters_a=_clusters(flip[pos].index, cmap),
        clusters_b=_clusters(flip[~pos].index, cmap),
    )


def behavioral_pass(preds: pd.DataFrame, model: str) -> float | None:
    """Claim 10: mean share of negation, typo and name pairs with both sentences right."""
    from mirage.evaluate import pair_accuracy

    b = canonical(preds)
    b = b[(b["model"] == model) & (b["variant"] == "base") & (b["dataset"] == "behavioral")]
    if b.empty:
        return None
    return float(
        np.mean([pair_accuracy(b, t).mean() for t in ("negation_dir", "typo", "name_shortcut")])
    )


def score_claims(
    preds: pd.DataFrame, warning: pd.DataFrame | None = None, n_resamples: int = 1000
) -> pd.DataFrame:
    """One row per pre-registered claim. `warning`: compare_predictors rows on imdb_fresh
    (columns model, predictor, auroc, diff_vs_conf, diff_lo, diff_hi)."""
    fv = fresh_view(preds)
    has_beh = (fv["dataset"] == "behavioral").any()
    has_imdb = (fv["dataset"] == "imdb").any()
    kw = {"n_resamples": n_resamples}
    rows = []

    h1 = h1_test(fv, **kw) if has_beh else pd.DataFrame(columns=["model", "test"])
    gap = negation_gap(fv, **kw) if has_beh else pd.DataFrame(columns=["model", "test"])
    typo = typo_direction(fv, **kw) if has_beh else pd.DataFrame(columns=["model", "test"])
    from mirage.evaluate import name_effect

    names = name_effect(fv, **kw) if has_beh else pd.DataFrame(columns=["model", "test"])
    names = names[names["variant"] == "base"] if len(names) else names
    typo = typo[typo["variant"] == "base"] if len(typo) else typo

    # 1. H1 typos: every transformer's typo error at least 10 points above its plain error
    t = _get(h1, model=list(TRANSFORMERS), test="beh_typo")
    ok = None if not _all_present(t, TRANSFORMERS) else bool((t["diff_lo"] >= 0.10).all())
    rows.append(
        _row(
            1,
            "H1",
            "Transformers' typo error is ≥ 10 points above their plain error",
            "lower bound of the difference ≥ 0.10, all three",
            ok,
            _fmt(t, ["diff", "diff_lo"]),
        )
    )
    # 2. H1 negation: LR and MNB
    t = _get(h1, model=["logreg", "mnb"], test="beh_negation_dir")
    ok = None if not _all_present(t, ("logreg", "mnb")) else bool((t["diff_lo"] > 0).all())
    rows.append(
        _row(
            2,
            "H1",
            "LR and MNB fail more directional-negation pairs than plain sentences",
            "lower bound > 0, both",
            ok,
            _fmt(t, ["diff", "diff_lo"]),
        )
    )
    # 3. H2: single-word models vs DistilBERT on directional negation
    t = _get(gap, test="beh_negation_dir")
    ok = (
        None
        if not _all_present(t, CLASSICAL) or "gap_unigram_lo" not in t
        else bool((t["gap_unigram_lo"] >= 0.10).all())
    )
    rows.append(
        _row(
            3,
            "H2",
            "Every single-word classical model is ≥ 10 points worse than DistilBERT",
            "lower bound of the gap ≥ 0.10, all five",
            ok,
            _fmt(t, ["gap_unigram", "gap_unigram_lo"]),
        )
    )
    # 4. H3 templates: LR ladder closes more than half
    t = _get(gap, model="logreg", test="beh_negation_dir")
    ok = (
        None
        if t.empty or "ladder_closed_lo" not in t
        else bool((t["ladder_closed_lo"] > 0.5).all())
    )
    rows.append(
        _row(
            4,
            "H3",
            "Pairs + marking close more than half of LR's directional-negation gap",
            "lower bound of ladder_closed > 0.5",
            ok,
            _fmt(t, ["ladder_closed", "ladder_closed_lo"]),
        )
    )
    # 5. typo default: DistilBERT flips more positive than negative pairs
    d = flip_difference(fv, "distilbert", n_resamples) if has_beh else None
    rows.append(
        _row(
            5,
            "Findings §2",
            "DistilBERT flips more positive-sentence typo pairs than negative",
            "lower bound of the flip-rate difference > 0",
            None if d is None else bool(d["ci_low"] > 0),
            "" if d is None else f"{d['estimate']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]",
        )
    )
    # 6. noun control
    t = _get(typo, model=["distilbert", "distilbert_hf"], test="typo_noun")
    rates = (
        t.assign(f=t["flip_rate"] * t["n_pairs"]).groupby("model")[["f", "n_pairs"]].sum()
        if len(t)
        else pd.DataFrame()
    )
    ok = None if len(rates) < 2 else bool((rates["f"] / rates["n_pairs"] <= 0.02).all())
    rows.append(
        _row(
            6,
            "Findings §2",
            "A noun typo almost never flips either DistilBERT",
            "noun-typo flip rate ≤ 0.02, both",
            ok,
            "; ".join(f"{m}: {r.f / r.n_pairs:.3f}" for m, r in rates.iterrows()),
        )
    )
    # 7. HF DistilBERT shares the default
    t = _get(typo, model="distilbert_hf", test="typo", polarity="positive")
    ok = None if t.empty else bool((t["flip_rate"] >= 0.30).all())
    rows.append(
        _row(
            7,
            "Findings §3",
            "HF DistilBERT flips ≥ 30% of positive-sentence typo pairs",
            "flip rate ≥ 0.30",
            ok,
            _fmt(t, ["flip_rate"]),
        )
    )
    # 8. RoBERTa-large's default points the other way
    t = _get(typo, model="roberta_large", test="typo")
    t = t.set_index("polarity") if "polarity" in t else t
    ok = (
        None
        if len(t) < 2
        else bool(abs(t.loc["negative", "shift"]) > abs(t.loc["positive", "shift"]))
    )
    rows.append(
        _row(
            8,
            "Findings §4",
            "RoBERTa-large's typo shift is larger for negative sentences",
            "|shift negative| > |shift positive|",
            ok,
            ""
            if len(t) < 2
            else f"negative {t.loc['negative', 'shift']:+.3f}, positive {t.loc['positive', 'shift']:+.3f}",
        )
    )
    # 9. names
    t = _get(names, model=["mnb", "distilbert", "distilbert_hf"], test="name_shortcut").set_index(
        "model"
    )
    ok = (
        None
        if len(t) < 3
        else bool(
            t.loc["mnb", "flip_rate"] >= 0.20
            and t.loc["distilbert", "flip_rate"] <= 0.02
            and t.loc["distilbert_hf", "flip_rate"] <= 0.02
        )
    )
    rows.append(
        _row(
            9,
            "Findings §5",
            "A name swap flips MNB often and both DistilBERTs almost never",
            "MNB ≥ 0.20; both DistilBERTs ≤ 0.02",
            ok,
            "; ".join(f"{m}: {r.flip_rate:.3f}" for m, r in t.iterrows()),
        )
    )
    # 10. leaderboard twins' behaviour
    lr, nb = (behavioral_pass(fv, m) if has_beh else None for m in ("logreg", "mnb"))
    ok = None if lr is None or nb is None else bool(lr - nb >= 0.10)
    rows.append(
        _row(
            10,
            "Findings §1",
            "MNB passes fewer negation, typo and name pairs than LR",
            "LR − MNB ≥ 0.10",
            ok,
            "" if ok is None else f"LR {lr:.3f}, MNB {nb:.3f}",
        )
    )

    # IMDb claims
    w = warning if warning is not None else pd.DataFrame(columns=["model", "predictor"])
    dist = _get(w, predictor="distance")
    ok = None if not _all_present(dist, ALL_MODELS) else bool((dist["auroc"] <= 0.56).all())
    rows.append(
        _row(
            11,
            "H4",
            "Embedding distance doesn't pick out IMDb errors",
            "distance AUROC ≤ 0.56, every model",
            ok,
            _fmt(dist, ["auroc"]),
        )
    )
    t = _get(dist, model=list(WARN_MODELS))
    ok = None if not _all_present(t, WARN_MODELS) else bool((-t["diff_hi"] >= 0.10).all())
    rows.append(
        _row(
            12,
            "H4",
            "Confidence beats distance by ≥ 0.10 AUROC (3 transformers, LR, RF, MNB)",
            "lower bound of (confidence − distance) ≥ 0.10, all six",
            ok,
            "; ".join(
                f"{r['model']}: {-r['diff_vs_conf']:+.3f} (lower {-r['diff_hi']:+.3f})"
                for _, r in t.iterrows()
            ),
        )
    )
    comb = _get(w, predictor="combined")
    ok = None if not _all_present(comb, ALL_MODELS) else bool((comb["diff_vs_conf"] < 0.02).all())
    rows.append(
        _row(
            13,
            "H4",
            "Adding distance to confidence adds < 0.02 AUROC",
            "combined − confidence < 0.02, every model",
            ok,
            _fmt(comb, ["diff_vs_conf"]),
        )
    )
    sd = shift_decomposition(fv) if has_imdb else pd.DataFrame(columns=["model", "variant"])
    sd = sd[sd["variant"] == "base"] if len(sd) else sd
    ok = None if not _all_present(sd, ALL_MODELS) else bool((sd["length_format_hi"] < 0).all())
    rows.append(
        _row(
            14,
            "Findings §8",
            "Full reviews are easier than their first and last sentences",
            "upper bound of (full − edges error) < 0, every model",
            ok,
            _fmt(sd, ["length_format", "length_format_hi"]),
        )
    )
    ok = None if not _all_present(sd, ALL_MODELS) else bool((sd["content_lo"] > 0).all())
    rows.append(
        _row(
            15,
            "Findings §8",
            "At SST-2 length, IMDb text is harder than SST-2",
            "lower bound of (edges − SST-2 test error) > 0, every model",
            ok,
            _fmt(sd, ["content", "content_lo"]),
        )
    )
    main = main_table(fv) if has_imdb else pd.DataFrame(columns=["model", "variant"])
    main = main[main["variant"] == "base"].set_index("model") if len(main) else main
    acc = main["accuracy__imdb"].get("roberta_large") if "accuracy__imdb" in main else None
    ok = None if acc is None or pd.isna(acc) else bool(acc >= 0.946)
    rows.append(
        _row(
            16,
            "Findings §12",
            "RoBERTa-large is at least as accurate on fresh IMDb as on SST-2 test",
            "fresh-IMDb accuracy ≥ 0.946",
            ok,
            "" if ok is None else f"{acc:.3f}",
        )
    )
    four = ("logreg", "rf", "mnb", "distilbert")
    ece = main["ece__imdb"].reindex(list(four)) if "ece__imdb" in main else pd.Series(dtype=float)
    ok = None if ece.isna().any() or ece.empty else bool(ece.idxmax() == "mnb")
    rows.append(
        _row(
            17,
            "Findings §11",
            "MNB is the worst-calibrated of LR, RF, MNB and DistilBERT on IMDb",
            "highest ECE of the four",
            ok,
            "; ".join(f"{m}: {v:.3f}" for m, v in ece.items() if pd.notna(v)),
        )
    )
    return pd.DataFrame(rows)


SCORED_TESTS = ("simple", "negation", "negation_dir", "typo", "typo_noun", "name_shortcut")


def by_frame(preds: pd.DataFrame, n_resamples: int = 1000) -> pd.DataFrame:
    """D10 (D53): failure rate per model, test and sentence frame, in both suites (frame set A =
    the original suite, B = behavioral_v2), with cluster CIs; plus a chi-square test of whether
    the frames of one set differ, Holm-adjusted across (model, test, set)."""
    from scipy.stats import chi2_contingency

    from mirage.behavioral.suite import case_failures
    from mirage.behavioral.templates import frame_of
    from mirage.stats import holm, rate_ci

    rows, tests = [], []
    # pick seeds on everything: DistilBERT's median seed needs its sst2_val rows (D53)
    preds = canonical(preds[preds["variant"] == "base"])
    for ds, frame_set in (("behavioral", "A"), ("behavioral_v2", "B")):
        p = preds[preds["dataset"] == ds]
        if p.empty:
            continue
        for m, g in p.groupby("model"):
            first = g[g["role"].isin(["single", "original"])].drop_duplicates("case_id")
            frame = first.set_index("case_id").apply(
                lambda r: frame_of(r["text"], r["test_type"]), axis=1
            )
            word = ("word:" + first["word"].astype(str)).set_axis(first["case_id"])
            cf = case_failures(g)
            cf = cf[cf["test_type"].isin(SCORED_TESTS)]
            cf = cf.assign(frame=cf["case_id"].map(frame))
            for tt, d in cf.groupby("test_type"):
                table = []
                for fr, f in d.groupby("frame"):
                    ci = rate_ci(
                        f["failed"].to_numpy(),
                        clusters=word.reindex(f["case_id"]).to_numpy(),
                        n_resamples=n_resamples,
                    )
                    rows.append(
                        {
                            "model": m,
                            "test": tt,
                            "frame_set": frame_set,
                            "frame": fr,
                            "n": ci["n"],
                            "fail_rate": ci["estimate"],
                            "ci_low": ci["ci_low"],
                            "ci_high": ci["ci_high"],
                        }
                    )
                    table.append([f["failed"].sum(), (~f["failed"]).sum()])
                tab = np.array(table)
                p_val = (
                    float(chi2_contingency(tab)[1])
                    if len(tab) > 1 and (tab.sum(axis=0) > 0).all()
                    else np.nan
                )
                tests.append({"model": m, "test": tt, "frame_set": frame_set, "p_frames": p_val})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    t = pd.DataFrame(tests)
    t["p_frames_holm"] = holm(t["p_frames"])
    return out.merge(t, on=["model", "test", "frame_set"], how="left")
