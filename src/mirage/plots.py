"""All report figures. Each function takes a result table and writes one PNG + PDF.

Colors follow a validated colorblind-safe palette (majordecisions.md D15): models keep a fixed
categorical slot in every figure (color follows the entity, never its rank); magnitudes use one
blue ramp light -> dark. Static report figures, so light mode only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: stages run from the CLI / Kaggle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from mirage import paths

# Fixed categorical order (validated adjacent-pair CVD-safe). Model -> slot never changes.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Slots 7-8: ready-made external models (D39); the 8-slot palette was re-validated for them.
MODEL_ORDER = [
    "logreg",
    "dtree",
    "rf",
    "mnb",
    "gnb",
    "distilbert",
    "distilbert_hf",
    "roberta_large",
]
TEST_ORDER = [
    "sst2_val",
    "sst2_test",
    "sst2_neg",
    "sst2_neg_testonly",
    "imdb",
    "imdb_edges",
    "beh_simple",
    "beh_negation",
    "beh_negation_neg",
    "beh_negation_dir",
    "beh_negation_dir_neg",
    "beh_typo",
    "beh_typo_noun",
    "beh_name_swap",
    "beh_name_shortcut",
]
MODEL_COLOR = dict(zip(MODEL_ORDER, SERIES, strict=False))
VARIANT_COLOR = {
    "base": SERIES[0],
    "negmark": SERIES[1],
    "negmark_w3": SERIES[2],
    "unigram": SERIES[3],
}
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "mirage_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def _save(fig, name: str, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or paths.FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    return out_dir / f"{name}.png"


def _order(models) -> list[str]:
    return [m for m in MODEL_ORDER if m in set(models)]


def error_heatmap(errors: pd.DataFrame, variant: str = "base", out_dir=None) -> Path:
    """Error rate by model x test type, annotated with the rate and n."""
    d = errors[errors["variant"] == variant]
    tests = [t for t in TEST_ORDER if t in set(d["test"])] + sorted(
        set(d["test"]) - set(TEST_ORDER)
    )
    grid = d.pivot(index="model", columns="test", values="error").reindex(
        index=_order(d["model"]), columns=tests
    )
    n = d.pivot(index="model", columns="test", values="n").reindex(index=grid.index, columns=tests)
    vmax = max(0.5, float(np.nanmax(grid.to_numpy(dtype=float))))
    fig, ax = plt.subplots(figsize=(1.1 * grid.shape[1] + 2.5, 0.65 * grid.shape[0] + 1.8))
    im = ax.imshow(grid.to_numpy(dtype=float), cmap=BLUE_RAMP, vmin=0, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(grid.shape[1]), grid.columns, rotation=30, ha="right")
    ax.set_yticks(range(grid.shape[0]), grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            if np.isnan(v):
                continue
            ink = "white" if v / vmax > 0.45 else INK
            ax.text(
                j,
                i,
                f"{v:.0%}\nn={int(n.iat[i, j])}",
                ha="center",
                va="center",
                fontsize=8,
                color=ink,
            )
    fig.colorbar(im, ax=ax, label="error / failure rate", shrink=0.8)
    ax.set_title(f"Error rate by model and test type ({variant})")
    return _save(fig, f"error_heatmap_{variant}", out_dir)


def negation_gap_bars(gaps: pd.DataFrame, test: str = "sst2_neg", out_dir=None) -> Path:
    """Gap to DistilBERT along the H2/H3 ladder (D42), with paired-bootstrap CIs:
    single words -> + word pairs (base) -> + negation marking. Rungs without data are skipped."""
    d = (
        gaps[gaps["test"] == test]
        .set_index("model")
        .reindex(_order(gaps["model"]))
        .dropna(how="all")
    )
    x = np.arange(len(d))
    rungs = [
        (col, variant, label)
        for col, variant, label in (
            ("gap_unigram", "unigram", "single words"),
            ("gap_base", "base", "+ word pairs (base)"),
            ("gap_negmark", "negmark", "+ negation marking"),
        )
        if col in d
    ]
    width = 0.8 / len(rungs)
    fig, ax = plt.subplots(figsize=(1.5 * len(d) + 2, 4))
    for i, (col, variant, label) in enumerate(rungs):
        off = (i - (len(rungs) - 1) / 2) * width
        err = np.vstack([d[col] - d[f"{col}_lo"], d[f"{col}_hi"] - d[col]])
        ax.bar(
            x + off,
            d[col],
            width=width * 0.9,
            color=VARIANT_COLOR[variant],
            label=label,
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.errorbar(x + off, d[col], yerr=err, fmt="none", ecolor=INK_2, capsize=3, linewidth=1)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xticks(x, d.index)
    ax.set_ylabel("negation error − DistilBERT negation error")
    ax.set_title(f"Negation gap to DistilBERT ({test})")
    ax.legend()
    return _save(fig, f"negation_gap_{test}", out_dir)


def error_vs_score(deciles: pd.DataFrame, model: str, out_dir=None) -> Path:
    """IMDb error rate by decile of Gaussian distance vs decile of low confidence."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for (label, g), color in zip(deciles.groupby("score"), SERIES, strict=False):
        ax.plot(g["decile"] + 1, g["error"], marker="o", markersize=6, color=color, label=label)
    ax.set_xlabel("decile of score (1 = most familiar / most confident)")
    ax.set_ylabel("error rate on IMDb")
    ax.set_xticks(range(1, 11))
    ax.set_title(f"Which score flags {model}'s IMDb errors?")
    ax.legend()
    return _save(fig, f"error_vs_score_{model}", out_dir)


def error_by_length(table: pd.DataFrame, variant: str = "base", out_dir=None) -> Path:
    """IMDb error per review-length bin, one line per model with its 95% CI band (C8)."""
    d = table[table["variant"] == variant]
    labels = list(dict.fromkeys(d["length"]))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in _order(d["model"].unique()):
        g = d[d["model"] == m].set_index("length").reindex(labels)
        color = SERIES[MODEL_ORDER.index(m) % len(SERIES)] if m in MODEL_ORDER else "gray"
        x = range(len(labels))
        ax.plot(x, g["error"], marker="o", markersize=6, linewidth=2, color=color, label=m)
        ax.fill_between(x, g["ci_low"], g["ci_high"], color=color, alpha=0.12, linewidth=0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("IMDb review length (words, before capping)")
    ax.set_ylabel("error rate")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Does error grow with review length? ({variant})")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    return _save(fig, f"imdb_error_by_length_{variant}", out_dir)


def reliability_diagram(bins: dict[str, pd.DataFrame], name: str, out_dir=None) -> Path:
    """bins: panel title -> output of metrics.reliability_bins. One panel per entry."""
    fig, axes = plt.subplots(1, len(bins), figsize=(3.6 * len(bins), 3.6), squeeze=False)
    for ax, (title, b) in zip(axes[0], bins.items(), strict=True):
        b = b[b["count"] > 0]
        ax.plot([0.5, 1], [0.5, 1], color=INK_2, linewidth=1, linestyle="--", label="perfect")
        ax.bar(
            b["confidence"],
            b["accuracy"],
            width=1 / 15 * 0.9,
            color=SERIES[0],
            alpha=0.85,
            label="accuracy",
        )
        ax.set_xlim(0.5, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence")
        ax.set_title(title, fontsize=9)
    axes[0][0].set_ylabel("accuracy")
    axes[0][0].legend(loc="upper left")
    return _save(fig, name, out_dir)


def overlap_heatmap(
    overlap: pd.DataFrame,
    metric: str = "jaccard",
    out_dir=None,
    name: str | None = None,
    diagonal: dict[str, float] | None = None,
    title: str | None = None,
    diagonal_label: str = "split-half agreement",
    value_label: str | None = None,
) -> Path:
    """Pairwise model agreement (top-50 overlap by default). `diagonal` (e.g. each model's own
    split-half agreement or reliability, D43) replaces the 1.0s on the diagonal, so agreement
    between models can be read against each model's agreement with itself."""
    models = _order(set(overlap["model_a"]) | set(overlap["model_b"])) or sorted(
        set(overlap["model_a"]) | set(overlap["model_b"])
    )
    m = pd.DataFrame(np.eye(len(models)), index=models, columns=models)
    for mod, v in (diagonal or {}).items():
        if mod in m.index:
            m.loc[mod, mod] = v
    for _, r in overlap.iterrows():
        if r["model_a"] in m.index and r["model_b"] in m.index:
            m.loc[r["model_a"], r["model_b"]] = m.loc[r["model_b"], r["model_a"]] = r[metric]
    fig, ax = plt.subplots(figsize=(0.8 * len(models) + 2, 0.8 * len(models) + 1.5))
    im = ax.imshow(m.to_numpy(), cmap=BLUE_RAMP, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(models)), models, rotation=30, ha="right")
    ax.set_yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(len(models)):
            v = m.iat[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 0.5 else INK,
            )
    fig.colorbar(im, ax=ax, label=value_label or metric, shrink=0.8)
    title = title or f"Shared shortcut words (top-50 {metric})"
    ax.set_title(title + (f"\ndiagonal: {diagonal_label}" if diagonal else ""), fontsize=10)
    return _save(fig, name or f"overlap_{metric}", out_dir)


def tree_figure(pipeline, depth: int, out_dir=None) -> Path:
    """The depth-limited explanation tree, drawn with its word-feature splits."""
    from sklearn.tree import plot_tree

    names = list(pipeline.named_steps["vec"].get_feature_names_out())
    fig, ax = plt.subplots(figsize=(2.2 * 2**depth / 2 + 4, 1.6 * depth + 2))
    ax.grid(False)
    plot_tree(
        pipeline.named_steps["clf"],
        feature_names=names,
        class_names=["neg", "pos"],
        filled=False,
        impurity=False,
        proportion=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title(f"Decision tree, depth {depth}: splits are 'tf-idf of word <= threshold'")
    return _save(fig, f"tree_depth{depth}", out_dir)


def attribution_strip(
    words: list[str], attrs: list[float], title: str, out_dir=None, name: str = "ig"
) -> Path:
    """One sentence, each word shaded by attribution (diverging: red = toward negative).

    `attrs` must point toward the POSITIVE class. IG returns attributions toward the predicted
    class, so the caller flips their sign when the prediction is negative.
    """
    a = np.asarray(attrs, dtype=float)
    lim = np.abs(a).max() or 1.0
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(words)), 1.2))
    ax.axis("off")
    for i, (w, v) in enumerate(zip(words, a, strict=True)):
        ax.text(
            i,
            0,
            w,
            ha="center",
            va="center",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": cmap((v / lim + 1) / 2),
                "edgecolor": "none",
            },
        )
    ax.set_xlim(-0.7, len(words) - 0.3)
    ax.set_title(title, fontsize=9, loc="left")
    return _save(fig, name, out_dir)


SHORT_NAME = {
    "logreg": "Logistic regression",
    "dtree": "Decision tree",
    "rf": "Random forest",
    "mnb": "Multinomial NB",
    "gnb": "Gaussian NB",
    "distilbert": "DistilBERT (ours)",
    "distilbert_hf": "DistilBERT (HF)",
    "roberta_large": "RoBERTa-large",
}
# C12 (D52): models whose accuracy may be inflated by training on our test data; marked wherever
# their accuracy is plotted, with a footnote.
ACCURACY_CAVEAT = {"roberta_large": "†"}
ACCURACY_NOTE = "† accuracy possibly inflated: likely trained on SST-2 and IMDb"


def _color(model: str) -> str:
    return MODEL_COLOR.get(model, INK_2)


def _spread(y: np.ndarray, min_gap: float) -> np.ndarray:
    """Nudge label heights apart (keeping their order) so direct labels don't collide."""
    order = np.argsort(y)
    out = np.asarray(y, dtype=float).copy()
    for a, b in zip(order[:-1], order[1:], strict=True):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def mirage_gap_chart(gap: pd.DataFrame, variant: str = "base", slide: bool = False, out_dir=None):
    """D40: accuracy (what a leaderboard shows) vs robustness (what the model handles).

    Dots below the diagonal look better than they behave; the thin drop line is the mirage.
    Filled dots = our models, hollow = ready-made external models (different training data).
    """
    d = gap[gap["variant"] == variant].set_index("model").reindex(_order(gap["model"]))
    d = d.dropna(subset=["accuracy", "robustness"])
    fs = 15 if slide else 10
    fig, ax = plt.subplots(figsize=(13.33, 7.5) if slide else (7.5, 5.5))
    lo = min(0.4, float(d[["accuracy", "robustness"]].min().min()) - 0.05)
    ax.plot([lo, 1], [lo, 1], color=INK_2, linewidth=1, linestyle="--")
    ax.text(  # along the diagonal, in its upper-left corner
        lo + 0.03,
        lo + 0.04,
        "robustness = accuracy",
        fontsize=fs * 0.8,
        color=INK_2,
        rotation=45,
        rotation_mode="anchor",
        transform_rotates_text=True,
    )
    for m, r in d.iterrows():
        ax.plot(
            [r["accuracy"]] * 2,
            [r["robustness"], r["accuracy"]],
            color=_color(m),
            linewidth=1.2,
            alpha=0.6,
        )
        hollow = bool(r.get("external", False))
        ax.scatter(
            r["accuracy"],
            r["robustness"],
            s=(170 if slide else 70),
            zorder=3,
            facecolor=SURFACE if hollow else _color(m),
            edgecolor=_color(m),
            linewidth=2.5 if hollow else 2,
        )
    worst = d["gap"].idxmax()  # the biggest mirage gets a second, bold line under its label
    # Models on (almost) the same spot share one label, e.g. our DistilBERT and HF's (D46).
    groups: list[list[str]] = []
    for m, r in d.iterrows():
        for grp in groups:
            g0 = d.loc[grp[0]]
            if (
                abs(g0["accuracy"] - r["accuracy"]) < 0.01
                and abs(g0["robustness"] - r["robustness"]) < 0.01
            ):
                grp.append(m)
                break
        else:
            groups.append([m])
    anchors = [d.loc[g[0]] for g in groups]
    label_y = _spread(np.array([a["robustness"] for a in anchors]), 0.035 if slide else 0.03)
    for grp, r, yl in zip(groups, anchors, label_y, strict=True):
        text = " = ".join(SHORT_NAME.get(m, m) + ACCURACY_CAVEAT.get(m, "") for m in grp)
        text += f"  {r['robustness']:.0%}"
        bold = worst in grp
        if bold:
            text += f"\nlooks {r['accuracy']:.0%}, behaves {r['robustness']:.0%}"
        ax.annotate(
            text,
            (r["accuracy"], r["robustness"]),
            xytext=(r["accuracy"] + 0.012, yl),
            fontsize=fs * 0.85,
            color=INK,
            va="top" if bold else "center",
            fontweight="bold" if bold else "normal",
        )
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(lo, 1.02)
    ax.set_xlabel("SST-2 test accuracy (what a leaderboard shows)", fontsize=fs)
    ax.set_ylabel("robustness (share of shortcut tests fully right)", fontsize=fs)
    ax.tick_params(labelsize=fs * 0.85)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_title("Accuracy hides how models fail", fontsize=fs * 1.3, loc="left")
    if d.get("external", pd.Series(False)).any():
        ax.text(
            1.0,
            lo + 0.01,
            "hollow = ready-made external model"
            + (f"\n{ACCURACY_NOTE}" if set(ACCURACY_CAVEAT) & set(d.index) else ""),
            ha="right",
            fontsize=fs * 0.75,
            color=INK_2,
        )
    return _save(fig, f"mirage_gap{'_slide' if slide else ''}", out_dir)


FINGERPRINT_COLUMNS = {
    "pass__beh_negation_dir": "negation",
    "pass__sst2_neg": "real negation",
    "pass__beh_typo": "typos",
    "pass__beh_name_shortcut": "names",
    "calibration": "calibration (1−ECE, IMDb)",
    "warning_auroc": "error warning (AUROC)",
}


def shortcut_fingerprint(table: pd.DataFrame, slide: bool = False, out_dir=None) -> Path:
    """D40: small multiples, one panel per model, one bar per check (higher = better).

    Bars, not a radar chart: radar areas distort and are hard to read from a distance. Each
    panel wears its model's colour; values are printed in ink. AUROC 0.5 is chance (marked).
    """
    cols = [c for c in FINGERPRINT_COLUMNS if c in table]
    d = table.set_index("model").reindex(_order(table["model"])).dropna(subset=cols, how="all")
    n = len(d)
    ncol = 4 if n > 4 else n
    nrow = int(np.ceil(n / ncol))
    fs = 13 if slide else 9
    size = (13.33, 7.5) if slide else (3.2 * ncol, 2.6 * nrow + 0.6)
    fig, axes = plt.subplots(nrow, ncol, figsize=size, squeeze=False, sharex=True, sharey=True)
    y = np.arange(len(cols))
    for ax, (m, r) in zip(axes.flat, d.iterrows(), strict=False):
        vals = r[cols].astype(float).to_numpy()
        ax.barh(y, vals, color=_color(m), height=0.7, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals, strict=True):
            if not np.isnan(v):
                ax.text(v + 0.03, yi, f"{v:.0%}", va="center", fontsize=fs * 0.85, color=INK)
        if "warning_auroc" in cols:
            yi = cols.index("warning_auroc")
            ax.plot([0.5, 0.5], [yi - 0.45, yi + 0.45], color=INK_2, linewidth=1, linestyle=":")
        ax.set_title(SHORT_NAME.get(m, m), fontsize=fs * 1.1, loc="left", color=INK)
        ax.set_xlim(0, 1.25)
        ax.set_axisbelow(True)
        ax.grid(axis="y", visible=False)
        ax.set_yticks(y, [FINGERPRINT_COLUMNS[c] for c in cols], fontsize=fs * 0.9)
        ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"], fontsize=fs * 0.8)
    axes.flat[0].invert_yaxis()  # shared y: first check on top in every panel
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(
        "Each model fails in its own way (higher = better; dotted line = chance AUROC)",
        fontsize=fs * 1.25,
        fontweight="bold",
        x=0.01,
        ha="left",
        color=INK,
    )
    return _save(fig, f"shortcut_fingerprint{'_slide' if slide else ''}", out_dir)
