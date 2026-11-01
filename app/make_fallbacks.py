"""Render a fallback image for every curated example into app/screenshots/ (D41).

For when the laptop or projector misbehaves: each image shows every model's positive
probability for sentence A (and B), with flips marked. Run from the repo root:
    python app/make_fallbacks.py
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np

from mirage import paths
from mirage.demo import load_examples, load_models, score
from mirage.plots import INK, INK_2, SHORT_NAME, SURFACE, _color

OUT = paths.ROOT / "app" / "screenshots"


def render(example: dict, models: dict) -> None:
    texts = [example["a"]] + ([example["b"]] if "b" in example else [])
    sc = score(models, texts)
    w = sc.pivot(index="model", columns="text_idx", values="p_pos").reindex(list(models))
    y = np.arange(len(w))[::-1]
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.axvline(0.5, color=INK_2, linewidth=1, linestyle="--")
    for yi, (m, r) in zip(y, w.iterrows(), strict=True):
        c = _color(m)
        if len(texts) == 2:
            a, b = r[0], r[1]
            if abs(b - a) >= 0.02:  # a zero-length arrow would leave a stray arrowhead
                ax.annotate(
                    "", (b, yi), (a, yi), arrowprops={"arrowstyle": "->", "color": c, "lw": 2.5}
                )
            ax.scatter([a], [yi], s=90, facecolor=SURFACE, edgecolor=c, linewidth=2.5, zorder=3)
            ax.scatter([b], [yi], s=90, color=c, zorder=3)
            if (a >= 0.5) != (b >= 0.5):
                ax.text(1.03, yi, "FLIPS", va="center", fontsize=13, fontweight="bold", color=INK)
        else:
            ax.barh(yi, r[0], color=c, height=0.6)
            ax.text(min(r[0], 0.93) + 0.02, yi, f"{r[0]:.0%}", va="center", fontsize=12, color=INK)
    ax.set_yticks(y, [SHORT_NAME.get(m, m) for m in w.index], fontsize=13)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=11)
    ax.set_xlabel("probability the review is positive", fontsize=12)
    ax.grid(axis="y", visible=False)
    lines = [f"A: {example['a']}"] + (
        [f"B: {example['b']}  (hollow = A, filled = B)"] if len(texts) == 2 else []
    )
    ax.set_title(example["title"] + "\n" + "\n".join(lines), loc="left", fontsize=14)
    fig.text(
        0.01,
        0.01,
        "\n".join(textwrap.wrap(example["talking_point"].strip(), 170)),
        fontsize=10,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{example['id']}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    models = load_models()
    for ex in load_examples():
        render(ex, models)
        print("wrote", OUT / f"{ex['id']}.png")
