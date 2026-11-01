"""`python -m mirage <stage>` - one entry point for every pipeline stage.

Stages only talk through files under artifacts/ and results/ (majordecisions.md D7), so any stage
can be re-run on its own once its inputs exist.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from mirage import paths
from mirage.config import load_config, save_resolved, set_seed
from mirage.models.base import CLASSICAL_MODELS

VARIANTS = ("base", "negmark", "negmark_w3", "unigram")  # see models.classical.VARIANT_NEGMARK


# ---------------------------------------------------------------- stages


def stage_prepare(args) -> None:
    from mirage.data import prepare

    print(prepare().to_string(index=False))


def stage_train_classical(args) -> None:
    from mirage.data import load
    from mirage.models.classical import train_all

    cfg = load_config("classical")
    set_seed(cfg["seed"])
    train, heldout = load("sst2_train"), load("sst2_heldout")
    models = tuple(args.models) if args.models else CLASSICAL_MODELS
    log_path = paths.RESULTS / "tuning_classical.csv"
    for variant in args.variants:
        fitted, log = train_all(train, heldout, cfg, variant=variant, models=models)
        for name, clf in fitted.items():
            out = paths.model_dir(name, variant, cfg["seed"])
            clf.save(out)
            save_resolved(cfg, out)
        old = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
        if len(old):  # replace rows for the models just re-tuned, keep the rest
            old = old[~(old["model"].isin(models) & (old["variant"] == variant))]
        pd.concat([old, log], ignore_index=True).to_csv(log_path, index=False)
        print(log.to_string(index=False))


def stage_train_distilbert(args) -> None:
    from mirage.data import load
    from mirage.models.distilbert import train

    cfg = load_config("distilbert")
    seeds = args.seeds or cfg["seeds"]
    train_df, heldout = load("sst2_train"), load("sst2_heldout")
    if args.limit:  # smoke-test mode: tiny run to check the pipeline end to end
        train_df, heldout = train_df.sample(args.limit, random_state=0), heldout.head(200)
    for seed in seeds:
        set_seed(seed)
        out = paths.model_dir("distilbert", "base", seed)
        summary = train(train_df, heldout, cfg, seed, out)
        save_resolved(cfg, out / "run")
        print(f"seed {seed}: held-out accuracy {summary['heldout'].get('eval_accuracy'):.4f}")


def stage_build_suite(args) -> None:
    from mirage.behavioral.suite import build_suite, save_suite

    df = build_suite()
    save_suite(df)
    print(
        df.groupby(["test_type", "kind"]).agg(rows=("text", "size"), cases=("case_id", "nunique"))
    )
    # Evidence behind configs/behavioral.yaml name_pairs (D32); needs the prepared data.
    from mirage.behavioral.names import name_associations
    from mirage.data import load

    try:
        train = pd.concat([load("sst2_train"), load("sst2_heldout")])
    except FileNotFoundError:
        return
    name_associations(train).to_csv(paths.RESULTS / "name_associations.csv", index=False)


def _saved_models(model_filter=None, variant_filter=None):
    """Yield (name, variant, seed, dir) for every saved model."""
    for d in sorted(paths.MODELS.glob("*__*")):
        parts = d.name.split("__")
        name, variant = parts[0], parts[1]
        seed = int(parts[2].removeprefix("seed")) if len(parts) > 2 else 0
        if model_filter and name not in model_filter:
            continue
        if variant_filter and variant not in variant_filter:
            continue
        yield name, variant, seed, d


def stage_predict(args) -> None:
    from mirage.models.base import load_classifier
    from mirage.predict import EVAL_SETS, predictions_path, run_predictions

    datasets = tuple(args.datasets) if args.datasets else EVAL_SETS
    for name, variant, seed, d in _saved_models(args.models, args.variants):
        files = [predictions_path(name, variant, seed, ds) for ds in datasets]
        model_time = max(f.stat().st_mtime for f in d.rglob("*") if f.is_file())
        fresh = all(f.exists() and f.stat().st_mtime >= model_time for f in files)
        if fresh and not getattr(args, "overwrite", False):
            # e.g. DistilBERT predictions made on the Kaggle GPU: don't redo them on a CPU.
            # Retraining a model makes its files newer, so stale predictions are never kept.
            print(f"skip {d.name}: predictions are newer than the model (--overwrite to redo)")
            continue
        clf = load_classifier(d)
        for p in run_predictions(clf, variant, seed, datasets):
            print("wrote", p.name)


def stage_predict_external(args) -> None:
    """Ready-made Hugging Face models (D39): no training, seed 0, raw text."""
    from mirage.models.external import EXTERNAL_SEED, ExternalClassifier
    from mirage.predict import predictions_path, run_predictions

    cfg = load_config("external")
    datasets = tuple(args.datasets) if args.datasets else tuple(cfg["datasets"])
    for name, spec in cfg["models"].items():
        if args.models and name not in args.models:
            continue
        todo = [
            ds
            for ds in datasets
            if getattr(args, "overwrite", False)
            or not predictions_path(name, "base", EXTERNAL_SEED, ds).exists()
        ]
        if not todo:
            print(f"skip {name}: predictions exist (--overwrite to redo)")
            continue
        clf = ExternalClassifier.from_hub(name, spec, cfg)
        for p in run_predictions(clf, "base", EXTERNAL_SEED, todo):
            print("wrote", p.name)


def stage_evaluate(args) -> None:
    from mirage import evaluate

    for name, df in evaluate.run(ci_resamples=args.resamples).items():
        print(f"== {name} ({len(df)} rows)")


def stage_explain(args) -> None:
    from mirage.data import load
    from mirage.explain import linear, overlap, trees
    from mirage.models.base import load_classifier

    cfg = load_config("classical")
    out = paths.EXPLAIN
    out.mkdir(parents=True, exist_ok=True)
    rankings: dict[str, list[str]] = {}
    tables = []
    for variant in VARIANTS:
        loaded = {
            n: load_classifier(d) for n, v, _, d in _saved_models(CLASSICAL_MODELS, [variant])
        }
        if "logreg" in loaded:
            t = linear.top_lr_weights(loaded["logreg"], k=args.k)
            tables.append(t.assign(variant=variant))
        if "mnb" in loaded:
            tables.append(linear.top_mnb_ratios(loaded["mnb"], k=args.k).assign(variant=variant))
        for n in ("rf", "dtree"):
            if n in loaded:
                tables.append(trees.top_importances(loaded[n], k=args.k).assign(variant=variant))
        if variant == "base":
            for n, clf in loaded.items():
                if n == "gnb":
                    continue  # SVD components have no word identity
                if n in ("logreg", "mnb"):
                    feats = (linear.top_lr_weights if n == "logreg" else linear.top_mnb_ratios)(
                        clf, k=2000
                    )
                    # rank by |score| so positive and negative shortcuts both count
                    feats = feats.reindex(feats["score"].abs().sort_values(ascending=False).index)
                else:
                    feats = trees.top_importances(clf, k=2000)
                rankings[n] = overlap.unigram_ranking(feats["feature"].tolist())
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(out / "top_features.csv", index=False)

    tree, text = trees.fit_explain_tree(load("sst2_train"), cfg)
    (out / "tree_depth4.txt").write_text(text, encoding="utf-8")
    from mirage import plots

    plots.tree_figure(tree.pipeline, cfg["explain_tree_depth"])

    if not args.skip_ig:
        ig_rows = _run_ig(args.max_ig)
        if ig_rows is not None:
            from mirage.explain.ig import mean_abs_attribution

            rankings["distilbert"] = mean_abs_attribution(ig_rows).index.tolist()
    if len(rankings) >= 2:
        overlap.overlap_matrix(rankings, k=args.overlap_k).to_csv(out / "overlap.csv", index=False)
    print("explain outputs in", out)


def _run_ig(max_items: int):
    """IG + deletion check on DistilBERT failures (median seed). Returns the IG table or None."""
    from mirage.evaluate import median_seed
    from mirage.explain.deletion import deletion_check
    from mirage.explain.ig import attribute
    from mirage.models.base import load_classifier
    from mirage.predict import load_predictions

    try:
        preds = load_predictions(model="distilbert")
    except FileNotFoundError:
        print("no DistilBERT predictions - skipping IG")
        return None
    seed = median_seed(preds)
    fails = preds[(preds["seed"] == seed) & ~preds["correct"].astype(bool)]
    # Half the budget from real SST-2 sentences, half from behavioral cases. Held-out (tuning)
    # and IMDb (long reviews, slow IG) are excluded.
    real = fails[fails["dataset"].isin(["sst2_neg", "sst2_val", "sst2_test"])].drop_duplicates(
        "text"
    )
    beh = fails[fails["dataset"] == "behavioral"].drop_duplicates("text")
    n_real = min(len(real), max_items - min(len(beh), max_items // 2))
    texts = real["text"].head(n_real).tolist() + beh["text"].head(max_items - n_real).tolist()
    clf = load_classifier(paths.model_dir("distilbert", "base", seed))
    ig = attribute(clf, texts)
    ig.assign(
        words=ig["words"].map(json.dumps), attributions=ig["attributions"].map(json.dumps)
    ).to_csv(paths.EXPLAIN / "ig_failures.csv", index=False)
    deletion_check(clf.predict_proba, ig["words"].tolist(), ig["attributions"].tolist()).to_csv(
        paths.EXPLAIN / "deletion_check.csv", index=False
    )
    print(f"IG on {len(ig)} failures; max convergence delta {ig['delta'].max():.3f}")
    return ig


def stage_occlusion(args) -> None:
    """H5 with one ranking rule for every model: leave-one-word-out importance (D43)."""
    from mirage.data import load
    from mirage.evaluate import median_seed
    from mirage.explain.occlusion import (
        chance_jaccard,
        effect_agreement,
        occlusion_table,
        split_half_overlap,
        split_half_reliability,
        word_effects,
        word_importance,
    )
    from mirage.explain.overlap import overlap_matrix
    from mirage.models.base import load_classifier
    from mirage.predict import load_predictions

    cfg = load_config("explain")["occlusion"]
    texts = load(cfg["dataset"])["text"].tolist()
    out = paths.EXPLAIN
    out.mkdir(parents=True, exist_ok=True)
    importance, rankings, ceiling, tables = [], {}, [], {}
    for name in cfg["models"]:
        if name == "distilbert":
            try:
                seed = median_seed(load_predictions(model="distilbert", dataset="sst2_val"))
            except FileNotFoundError:
                print("no DistilBERT predictions - skipping it")
                continue
        else:
            seed = load_config("classical")["seed"]
        d = paths.model_dir(name, "base", seed)
        if not d.exists():
            print(f"no saved model {d.name} - skipping it")
            continue
        # Cache each model's word-removal table (DistilBERT takes ~10 min on CPU). Named like a
        # prediction file, so a different seed (e.g. a new median seed) never reuses another
        # model's table; a retrained model or regenerated data makes it stale, as in `predict`.
        cache = (
            paths.ARTIFACTS / "occlusion" / f"{name}__base__seed{seed}__{cfg['dataset']}.parquet"
        )
        inputs = [f for f in d.rglob("*") if f.is_file()] + [
            paths.DATA / f"{cfg['dataset']}.parquet"
        ]
        newest_input = max(f.stat().st_mtime for f in inputs if f.exists())
        if cache.exists() and cache.stat().st_mtime >= newest_input:
            occ = pd.read_parquet(cache)
        else:
            occ = occlusion_table(load_classifier(d).predict_proba, texts)
            cache.parent.mkdir(parents=True, exist_ok=True)
            occ.to_parquet(cache, index=False)
        tables[name] = occ
        imp = word_importance(occ, cfg["min_count"])
        importance.append(imp.assign(model=name))
        rankings[name] = imp["word"].tolist()
        half = split_half_overlap(occ, cfg["min_count"], cfg["top_k"], cfg["split_seed"])
        chance = chance_jaccard(len(imp), cfg["top_k"])
        ceiling.append({"model": name, "n_ranked": len(imp), **half, "chance_jaccard": chance})
        print(f"{name}: {len(occ)} word removals, {len(imp)} words ranked, split-half {half}")
    if not importance:
        return
    pd.concat(importance, ignore_index=True).to_csv(out / "occlusion_importance.csv", index=False)
    pd.DataFrame(ceiling).to_csv(out / "occlusion_split_half.csv", index=False)
    if len(rankings) < 2:
        return
    overlap_matrix(rankings, k=cfg["top_k"]).to_csv(out / "overlap_occlusion.csv", index=False)
    # Main H5 measure: agreement of word effects, corrected for each model's reliability (D43).
    agreement, reliability = [], []
    for mc in (cfg["min_count"], cfg["sensitivity_min_count"]):
        effects = word_effects(tables, mc)
        rel = {
            m: split_half_reliability(
                o, effects.index, cfg["reliability_splits"], cfg["split_seed"]
            )
            for m, o in tables.items()
        }
        res = effect_agreement(effects, rel, cfg["bootstrap_resamples"], cfg["split_seed"])
        agreement.append(res.assign(min_count=mc))
        reliability += [{"model": m, "min_count": mc, "reliability": r} for m, r in rel.items()]
    pd.concat(agreement, ignore_index=True).to_csv(out / "occlusion_agreement.csv", index=False)
    pd.DataFrame(reliability).to_csv(out / "occlusion_reliability.csv", index=False)


def stage_warn(args) -> None:
    from mirage.data import load
    from mirage.evaluate import canonical
    from mirage.predict import load_predictions
    from mirage.stats import holm
    from mirage.warning.distance import (
        fit_distance,
        length_adjusted,
        one_sentence_each,
        sentence_distances,
        text_distance,
    )
    from mirage.warning.error_detection import compare_predictors, error_by_decile, ood_sanity

    cfg = load_config("warning")
    density = fit_distance(load("sst2_train"), cfg)
    imdb = load("imdb")
    dist = text_distance(density, imdb, "imdb", cfg)
    n_words = pd.Series(imdb["text"].str.split().str.len().to_numpy(), index=imdb["item_id"])
    in_dist = text_distance(density, load("sst2_test"), "sst2_test", cfg)
    sanity = ood_sanity(in_dist.to_numpy(), dist.to_numpy())
    print(f"sanity ({cfg['distance']}, {cfg['unit']}): AUROC SST-2 test vs IMDb = {sanity:.3f}")

    # F4 / D30: how much of the distance is just length?
    sd = sentence_distances(density, imdb, "imdb", cfg)
    n_sent = sd.groupby("owner").size().reindex(imdb["item_id"]).to_numpy()
    matched = ood_sanity(in_dist.to_numpy(), one_sentence_each(sd, cfg["seed"]).to_numpy())
    adjusted = pd.Series(length_adjusted(dist.to_numpy(), n_sent), index=imdb["item_id"])
    sanity_rows = {
        "full reviews (as used)": sanity,
        "one random sentence per review (length-matched)": matched,
        "corr(distance, log #sentences)": float(np.corrcoef(np.log1p(n_sent), dist)[0, 1]),
    }
    pd.Series(sanity_rows, name="value").rename_axis("check").reset_index().to_csv(
        paths.RESULTS / "early_warning_sanity.csv", index=False
    )
    for k, v in sanity_rows.items():
        print(f"  {k}: {v:.3f}")

    preds = canonical(load_predictions())  # seed choice needs sst2_val rows
    preds = preds[preds["dataset"] == "imdb"]
    rows, deciles = [], []
    for (m, v), g in preds.groupby(["model", "variant"]):
        d = dist.loc[g["item_id"]].to_numpy()
        res = compare_predictors(
            g,
            d,
            cfg["cv_folds"],
            cfg["seed"],
            cfg["bootstrap_resamples"],
            length=n_words.loc[g["item_id"]].to_numpy(),
            extra={"distance_adj": adjusted.loc[g["item_id"]].to_numpy()},
        )
        rows.append(res.assign(model=m, variant=v))
        if v == "base":
            conf = g["p_pos"].map(lambda p: max(p, 1 - p)).to_numpy()
            deciles += [
                error_by_decile(g, d, "distance").assign(model=m),
                error_by_decile(g, 1 - conf, "low confidence").assign(model=m),
            ]
    out = pd.concat(rows, ignore_index=True).assign(ood_sanity_auroc=sanity)
    # H4 family per predictor: "beats confidence" across every model and variant (D28). External
    # models are a family of their own, so they don't change our models' p-values (D39).
    from mirage.evaluate import holm_family

    out["family"] = holm_family(out["model"], variant=out["variant"])
    out["diff_p_holm"] = out.groupby(["predictor", "family"])["diff_p"].transform(lambda p: holm(p))
    out["beats_conf"] = (out["diff_p_holm"] < 0.05) & (out["diff_vs_conf"] > 0)
    out.to_csv(paths.RESULTS / "early_warning.csv", index=False)
    pd.concat(deciles, ignore_index=True).to_csv(
        paths.RESULTS / "early_warning_deciles.csv", index=False
    )
    print(out.round(3).to_string(index=False))


def stage_figures(args) -> None:
    from mirage import plots
    from mirage.evaluate import canonical
    from mirage.metrics import reliability_bins
    from mirage.predict import load_predictions

    R = paths.RESULTS
    if (R / "errors_by_type.csv").exists():
        errors = pd.read_csv(R / "errors_by_type.csv")
        for v in errors["variant"].unique():
            plots.error_heatmap(errors, v)
    if (R / "negation_gap.csv").exists():
        gaps = pd.read_csv(R / "negation_gap.csv")
        for t in gaps["test"].unique():
            plots.negation_gap_bars(gaps, t)
    if (R / "mirage_gap.csv").exists():  # D40: headline figure + fingerprint, report and slide
        from mirage.evaluate import fingerprint_table

        gap = pd.read_csv(R / "mirage_gap.csv")
        if len(gap):
            warn = R / "early_warning.csv"
            fp = fingerprint_table(
                gap,
                pd.read_csv(R / "main_table.csv"),
                pd.read_csv(warn) if warn.exists() else None,
            )
            fp.to_csv(R / "shortcut_fingerprint.csv", index=False)
            for slide in (False, True):
                plots.mirage_gap_chart(gap, "base", slide=slide)
                plots.shortcut_fingerprint(fp, slide=slide)
    if (R / "imdb_by_length.csv").exists():
        lengths = pd.read_csv(R / "imdb_by_length.csv")
        if len(lengths):
            plots.error_by_length(lengths, "base")
    if (R / "early_warning_deciles.csv").exists():
        dec = pd.read_csv(R / "early_warning_deciles.csv")
        for m, g in dec.groupby("model"):
            plots.error_vs_score(g, m)
    if (paths.EXPLAIN / "overlap.csv").exists():
        ov = pd.read_csv(paths.EXPLAIN / "overlap.csv")
        plots.overlap_heatmap(ov, "jaccard")
        plots.overlap_heatmap(ov, "rbo")
    if (paths.EXPLAIN / "occlusion_agreement.csv").exists():  # D43: H5's main measure
        agr = pd.read_csv(paths.EXPLAIN / "occlusion_agreement.csv")
        rel = pd.read_csv(paths.EXPLAIN / "occlusion_reliability.csv")
        mc = agr["min_count"].min()  # the primary threshold
        plots.overlap_heatmap(
            agr[agr["min_count"] == mc],
            "corrected",
            name="h5_word_effect_agreement",
            diagonal=rel[rel["min_count"] == mc].set_index("model")["reliability"].to_dict(),
            title="Do models agree on what each word does?\n"
            "(correlation of word effects, corrected for reliability)",
            diagonal_label="each model's own reliability",
            value_label="agreement (corrected Spearman)",
        )
    if (paths.EXPLAIN / "overlap_occlusion.csv").exists():  # D43: one ranking rule for all
        ov = pd.read_csv(paths.EXPLAIN / "overlap_occlusion.csv")
        half = paths.EXPLAIN / "occlusion_split_half.csv"
        diag = pd.read_csv(half).set_index("model") if half.exists() else None
        for metric in ("jaccard", "rbo"):
            plots.overlap_heatmap(
                ov,
                metric,
                name=f"overlap_occlusion_{metric}",
                diagonal=None if diag is None else diag[metric].to_dict(),
                title=f"Shared top-50 words, one ranking rule (leave one word out, {metric})",
            )
    if (paths.EXPLAIN / "ig_failures.csv").exists():
        ig = pd.read_csv(paths.EXPLAIN / "ig_failures.csv")
        for i, r in ig.head(4).iterrows():
            attrs = json.loads(r["attributions"])
            if r["pred"] == 0:  # point every strip toward the positive class
                attrs = [-a for a in attrs]
            plots.attribution_strip(
                json.loads(r["words"]), attrs, f"pred={r['pred']}: {r['text'][:60]}", name=f"ig_{i}"
            )
    try:
        preds = canonical(load_predictions(variant="base"))
    except FileNotFoundError:
        preds = None
    if preds is not None:
        val = preds[preds["dataset"] == "sst2_val"]
        classical = val[val["model"].isin(CLASSICAL_MODELS)]  # not the external models (D39)
        if len(classical):
            best = classical.groupby("model")["correct"].mean().idxmax()
            panels = {}
            for m in [best, "distilbert"]:
                for ds in ("sst2_test", "imdb"):
                    g = preds[(preds["model"] == m) & (preds["dataset"] == ds)]
                    if len(g):
                        panels[f"{m} - {ds}"] = reliability_bins(g["p_pos"], g["label"])
            if panels:
                plots.reliability_diagram(panels, "reliability")
    print("figures in", paths.FIGURES)


def stage_all(args) -> None:
    """Everything after training: predictions -> evaluation -> explanations -> warning -> figures.

    `predict-external` is not included: it downloads 1.9 GB and is meant for the Kaggle GPU
    (notebooks/kaggle_external.ipynb, D39). Its prediction files are picked up automatically.
    """
    ns = argparse.Namespace(
        models=None,
        variants=None,
        datasets=None,
        resamples=1000,
        k=25,
        skip_ig=False,
        max_ig=100,
        overlap_k=50,
    )
    for fn in (
        stage_build_suite,
        stage_predict,
        stage_evaluate,
        stage_explain,
        stage_occlusion,
        stage_warn,
        stage_figures,
    ):
        print(f"\n######## {fn.__name__.removeprefix('stage_')}")
        fn(ns)


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mirage", description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)

    sub.add_parser(
        "prepare", help="download SST-2/IMDb, build splits + negation subset"
    ).set_defaults(fn=stage_prepare)

    s = sub.add_parser("train-classical", help="grid-search the 5 classical models")
    s.add_argument("--variants", nargs="+", choices=VARIANTS, default=["base"])
    s.add_argument("--models", nargs="+", choices=CLASSICAL_MODELS)
    s.set_defaults(fn=stage_train_classical)

    s = sub.add_parser("train-distilbert", help="fine-tune DistilBERT (GPU; run on Kaggle/Colab)")
    s.add_argument("--seeds", nargs="+", type=int)
    s.add_argument("--limit", type=int, help="train on N examples only (smoke test)")
    s.set_defaults(fn=stage_train_distilbert)

    sub.add_parser("build-suite", help="generate the behavioral test suite").set_defaults(
        fn=stage_build_suite
    )

    s = sub.add_parser("predict", help="write prediction files for every saved model")
    s.add_argument("--models", nargs="+")
    s.add_argument("--variants", nargs="+", choices=VARIANTS)
    s.add_argument("--datasets", nargs="+")
    s.add_argument("--overwrite", action="store_true", help="redo existing prediction files")
    s.set_defaults(fn=stage_predict)

    s = sub.add_parser(
        "predict-external", help="predictions from ready-made HF models (configs/external.yaml)"
    )
    s.add_argument("--models", nargs="+")
    s.add_argument("--datasets", nargs="+")
    s.add_argument("--overwrite", action="store_true", help="redo existing prediction files")
    s.set_defaults(fn=stage_predict_external)

    s = sub.add_parser("evaluate", help="main table, errors by type, gaps, McNemar")
    s.add_argument("--resamples", type=int, default=1000)
    s.set_defaults(fn=stage_evaluate)

    s = sub.add_parser("explain", help="weights, splits, importances, IG, deletion, overlap")
    s.add_argument("--k", type=int, default=25)
    s.add_argument("--skip-ig", action="store_true")
    s.add_argument("--max-ig", type=int, default=100)
    s.add_argument("--overlap-k", type=int, default=50)
    s.set_defaults(fn=stage_explain)

    sub.add_parser(
        "occlusion", help="H5: leave-one-word-out word rankings and their overlap (D43)"
    ).set_defaults(fn=stage_occlusion)
    sub.add_parser(
        "warn", help="embeddings, Gaussian distance, error-detection AUROC"
    ).set_defaults(fn=stage_warn)
    sub.add_parser("figures", help="render every figure from the result CSVs").set_defaults(
        fn=stage_figures
    )
    sub.add_parser("all", help="every stage after training").set_defaults(fn=stage_all)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths.ensure_dirs()
    args.fn(args)


if __name__ == "__main__":
    main()
