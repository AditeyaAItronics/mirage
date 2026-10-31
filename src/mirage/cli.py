"""`python -m mirage <stage>` - one entry point for every pipeline stage.

Stages only talk through files under artifacts/ and results/ (majordecisions.md D7), so any stage
can be re-run on its own once its inputs exist.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from mirage import paths
from mirage.config import load_config, save_resolved, set_seed
from mirage.models.base import CLASSICAL_MODELS

VARIANTS = ("base", "negmark")


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
        fitted, log = train_all(train, heldout, cfg, negmark=variant == "negmark", models=models)
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
    from mirage.predict import EVAL_SETS, run_predictions

    datasets = tuple(args.datasets) if args.datasets else EVAL_SETS
    for _name, variant, seed, d in _saved_models(args.models, args.variants):
        clf = load_classifier(d)
        for p in run_predictions(clf, variant, seed, datasets):
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
    real = fails[fails["dataset"].isin(["sst2_neg", "sst2_val"])].drop_duplicates("text")
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


def stage_warn(args) -> None:
    from mirage.data import load
    from mirage.evaluate import canonical
    from mirage.predict import load_predictions
    from mirage.warning.embed import cached_embeddings
    from mirage.warning.error_detection import compare_predictors, error_by_decile, ood_sanity
    from mirage.warning.gaussian import GaussianDensity

    cfg = load_config("warning")
    emb = {
        name: cached_embeddings(
            load(name), name, cfg["embedding_model"], cfg["embedding_batch_size"]
        )
        for name in ("sst2_train", "sst2_val", "imdb")
    }
    density = GaussianDensity().fit(emb["sst2_train"])
    imdb = load("imdb")
    dist = pd.Series(density.score(emb["imdb"]), index=imdb["item_id"])
    sanity = ood_sanity(density.score(emb["sst2_val"]), dist.to_numpy())
    print(f"sanity: distance AUROC SST-2 val vs IMDb = {sanity:.3f}")

    preds = canonical(load_predictions())  # seed choice needs sst2_val rows
    preds = preds[preds["dataset"] == "imdb"]
    rows, deciles = [], []
    for (m, v), g in preds.groupby(["model", "variant"]):
        d = dist.loc[g["item_id"]].to_numpy()
        res = compare_predictors(g, d, cfg["cv_folds"], cfg["seed"], cfg["bootstrap_resamples"])
        rows.append(res.assign(model=m, variant=v))
        if v == "base":
            conf = g["p_pos"].map(lambda p: max(p, 1 - p)).to_numpy()
            deciles += [
                error_by_decile(g, d, "distance").assign(model=m),
                error_by_decile(g, 1 - conf, "low confidence").assign(model=m),
            ]
    out = pd.concat(rows, ignore_index=True).assign(ood_sanity_auroc=sanity)
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
    if (R / "early_warning_deciles.csv").exists():
        dec = pd.read_csv(R / "early_warning_deciles.csv")
        for m, g in dec.groupby("model"):
            plots.error_vs_score(g, m)
    if (paths.EXPLAIN / "overlap.csv").exists():
        ov = pd.read_csv(paths.EXPLAIN / "overlap.csv")
        plots.overlap_heatmap(ov, "jaccard")
        plots.overlap_heatmap(ov, "rbo")
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
        classical = val[val["model"] != "distilbert"]
        if len(classical):
            best = classical.groupby("model")["correct"].mean().idxmax()
            panels = {}
            for m in [best, "distilbert"]:
                for ds in ("sst2_val", "imdb"):
                    g = preds[(preds["model"] == m) & (preds["dataset"] == ds)]
                    if len(g):
                        panels[f"{m} - {ds}"] = reliability_bins(g["p_pos"], g["label"])
            if panels:
                plots.reliability_diagram(panels, "reliability")
    print("figures in", paths.FIGURES)


def stage_all(args) -> None:
    """Everything after training: predictions -> evaluation -> explanations -> warning -> figures."""
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
    s.set_defaults(fn=stage_predict)

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
