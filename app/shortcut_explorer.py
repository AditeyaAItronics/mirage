"""Mirage: Shortcut Explorer, the live demo (D41).

Run from the repo root:  streamlit run app/shortcut_explorer.py
UI only: every number comes from mirage.demo, which is unit-tested.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from mirage.config import load_config
from mirage.demo import (
    UnusualInput,
    load_examples,
    load_models,
    occlusion,
    score,
    trim,
    unknown_words,
)
from mirage.plots import MODEL_COLOR, SHORT_NAME

CFG = load_config("demo")
EXTERNAL = set(load_config("external")["models"])

st.set_page_config(page_title="Mirage: Shortcut Explorer", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Loading models (once, about 20-40 seconds)...")
def get_models(light: bool) -> dict:
    return load_models(light=light, cfg=CFG)


@st.cache_resource(show_spinner="Fitting the unusual-input check...")
def get_unusual(percentile: float):
    try:
        return UnusualInput(percentile)
    except Exception as e:  # missing embeddings or sentence-transformers: demo still works
        print(f"unusual-input flag disabled: {e}")
        return None


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    light = st.toggle(
        "Light mode (slow laptops)",
        value=False,
        help="Leaves out the random forest and RoBERTa-large.",
    )
    low_conf = st.slider("Low-confidence flag below", 0.5, 0.95, float(CFG["low_confidence"]), 0.05)
    show_words = st.toggle("Show word importance", value=True)
    show_notes = st.toggle("Show presenter notes", value=True)
    if st.button("Reset to examples"):
        for k in ("mode", "own_a", "own_b", "compare"):
            st.session_state.pop(k, None)
        st.rerun()
    st.caption(
        "Word importance = how much the positive probability drops when that word is removed. "
        "Greyed-out words were never seen in training by that model."
    )

models = get_models(light)
unusual = get_unusual(float(CFG["unusual_percentile"]))

# ------------------------------------------------------------------ header + input
st.title("Mirage: Shortcut Explorer")
st.markdown(
    "Type a sentence and see how **every model** reads it, and **why**. "
    "Accuracy says these models are 59-92% right; this shows *how* they get there."
)

mode = st.radio("Input", ["Curated examples", "Your own sentence"], horizontal=True, key="mode")
note = None
if mode == "Curated examples":
    examples = load_examples()
    ex = st.selectbox("Example", examples, format_func=lambda e: e["title"])
    text_a, text_b, note = ex["a"], ex.get("b"), ex["talking_point"]
else:
    compare = st.toggle("Compare two sentences", key="compare")
    text_a = st.text_input("Sentence", "The movie was not bad at all.", key="own_a")
    text_b = (
        st.text_input("Second sentence", "The movie was bad.", key="own_b") if compare else None
    )

texts = []
for t in (text_a, text_b):
    if t is None:
        continue
    t, cut = trim(t, int(CFG["max_words"]))
    if cut:
        st.info(f"Long input: only the first {CFG['max_words']} words are used.")
    if t:
        texts.append(t)
if not texts:
    st.warning("Type a sentence to start.")
    st.stop()

if note and show_notes:
    st.info(f"**Presenter note:** {note}")


# ------------------------------------------------------------------ rendering helpers
def word_chips(words, importance, unknown) -> str:
    """Words shaded by importance: blue = pushed positive, red = pushed negative."""
    lim = max(0.05, float(pd.Series(importance).abs().max()))
    out = []
    for i, w in enumerate(words):
        v = float(importance[i]) / lim
        if unknown is not None and unknown[i]:
            style = "color:#9a9893;font-style:italic;text-decoration:underline dotted;"
            title = "not in this model's vocabulary"
        else:
            rgb = (42, 120, 214) if v > 0 else (227, 73, 72)
            style = f"background:rgba({rgb[0]},{rgb[1]},{rgb[2]},{abs(v) * 0.75:.2f});"
            title = f"{importance[i]:+.2f}"
        out.append(
            f'<span title="{title}" style="{style}padding:1px 4px;border-radius:4px;white-space:nowrap;'
            f'margin-right:3px;">{html.escape(w)}</span>'
        )
    return " ".join(out)


def verdict(p: float) -> str:
    label, color = ("positive", "#2a78d6") if p >= 0.5 else ("negative", "#e34948")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;'
        f'font-weight:600;">{label}</span>'
    )


def prob_bar(p: float, color: str) -> str:
    return (
        '<div style="background:#e4e3df;border-radius:4px;height:10px;width:140px;'
        'display:inline-block;vertical-align:middle;">'
        f'<div style="background:{color};width:{p * 100:.0f}%;height:10px;border-radius:4px;">'
        f"</div></div> {p:.0%} positive"
    )


def results_table(text: str, idx: int, sc: pd.DataFrame) -> None:
    flag_unusual, dist = unusual(text) if unusual else (False, None)
    if flag_unusual:
        st.warning(
            "**Unusual input:** this sentence is further from the training data than 95% of SST-2 "
            "test sentences. Unusual doesn't mean wrong (H4), but treat verdicts with care."
        )
    rows = ['<table style="width:100%;border-collapse:collapse;">']
    rows.append(
        "<tr style='text-align:left;color:#52514e;'><th>model</th><th>verdict</th>"
        "<th>probability</th><th>flags</th>"
        + ("<th>word importance</th>" if show_words else "")
        + "</tr>"
    )
    for name, clf in models.items():
        r = sc[(sc["model"] == name) & (sc["text_idx"] == idx)].iloc[0]
        color = MODEL_COLOR.get(name, "#52514e")
        flags = []
        if r["confidence"] < low_conf:
            flags.append("⚠️ low confidence")
        unknown = unknown_words(clf, text)
        if unknown and any(unknown):
            flags.append(f"🔍 {sum(unknown)} unseen word(s)")
        label = SHORT_NAME.get(name, name) + (
            " <small>(external)</small>" if name in EXTERNAL else ""
        )
        cells = [
            f'<span style="color:{color};font-weight:700;">●</span> {label}',
            verdict(r["p_pos"]),
            prob_bar(r["p_pos"], "#52514e"),  # neutral: colour belongs to the model dot
            "<br>".join(flags) or "—",
        ]
        if show_words:
            occ = occlusion(clf, text)
            cells.append(word_chips(occ["word"].tolist(), occ["importance"].to_numpy(), unknown))
        rows.append(
            "<tr style='border-top:1px solid #e4e3df;'>"
            + "".join(
                f"<td style='padding:6px 8px;vertical-align:middle;'>{c}</td>" for c in cells[:4]
            )
            + "".join(f"<td style='padding:6px 8px;line-height:1.9;'>{c}</td>" for c in cells[4:])
            + "</tr>"
        )
    rows.append("</table>")
    st.markdown("".join(rows), unsafe_allow_html=True)


# ------------------------------------------------------------------ results
sc = score(models, texts, low_conf)


def change_table() -> None:
    """The punchline of a pair: per model, A vs B and whether the verdict flips."""
    w = sc.pivot(index="model", columns="text_idx", values="p_pos").reindex(list(models))
    rows = [
        '<table style="width:100%;border-collapse:collapse;font-size:1.05rem;">'
        "<tr style='text-align:left;color:#52514e;'><th>model</th><th>A</th><th>B</th>"
        "<th>change in positive probability</th><th>verdict</th></tr>"
    ]
    for m, (a, b) in w[[0, 1]].iterrows():
        flips = (a >= 0.5) != (b >= 0.5)
        bg = "background:#fdecea;" if flips else ""
        color = MODEL_COLOR.get(m, "#52514e")
        rows.append(
            f"<tr style='border-top:1px solid #e4e3df;{bg}'>"
            f"<td style='padding:6px 8px;'><span style='color:{color};font-weight:700;'>●</span> "
            f"{SHORT_NAME.get(m, m)}{' <small>(external)</small>' if m in EXTERNAL else ''}</td>"
            f"<td>{verdict(a)} {a:.0%}</td><td>{verdict(b)} {b:.0%}</td>"
            f"<td style='font-weight:600;'>{b - a:+.0%}</td>"
            f"<td style='font-weight:700;'>{'FLIPS' if flips else 'same'}</td></tr>"
        )
    rows.append("</table>")
    st.markdown("".join(rows), unsafe_allow_html=True)


if len(texts) == 2:
    st.subheader("What changed between A and B?")
    st.markdown(f"**A:** “{html.escape(texts[0])}”  \n**B:** “{html.escape(texts[1])}”")
    change_table()
    st.caption("A flip on a change that shouldn't matter (a name, a typo in a noun) is a shortcut.")
    st.subheader("Why? Each sentence in detail")
    for i, tab in enumerate(st.tabs([f"A: {texts[0]}", f"B: {texts[1]}"])):
        with tab:
            results_table(texts[i], i, sc)
else:
    st.subheader(f"“{texts[0]}”")
    results_table(texts[0], 0, sc)

st.divider()
st.caption(
    "Models: our five classical models and DistilBERT were trained on SST-2 movie-review phrases. "
    "DistilBERT (HF) and RoBERTa-large are ready-made models from Hugging Face (external). "
    "Project: github.com/AditeyaAItronics/mirage"
)
