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


# Results are cached per (model, text), so moving a slider or toggle never re-runs a model, and
# switching light mode on and off reuses what both modes share. `_clf` is the model itself
# (not hashed); `name` identifies it in the cache key.
@st.cache_data(show_spinner=False, max_entries=512)
def get_scores(_clf, name: str, texts: tuple[str, ...]) -> pd.DataFrame:
    return score({name: _clf}, list(texts))


@st.cache_data(show_spinner=False, max_entries=512)
def get_occlusion(_clf, name: str, text: str) -> pd.DataFrame:
    return occlusion(_clf, text)


@st.cache_data(show_spinner=False, max_entries=256)
def get_unusual_flag(_check, percentile: float, text: str) -> tuple[bool, float]:
    return _check(text)


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
        "Word importance = how much the positive probability drops when that word is removed."
    )

models = get_models(light)
UNUSUAL_PCT = float(CFG["unusual_percentile"])
unusual = get_unusual(UNUSUAL_PCT)

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
# Colours that read on both the light and the dark Streamlit theme: semi-transparent greys for
# lines and tracks, inherited text colour for labels. Blue / red keep the project's meaning
# (positive / negative) everywhere.
POS, NEG = (42, 120, 214), (227, 73, 72)
LINE = "rgba(128,128,128,0.3)"
TRACK = "rgba(128,128,128,0.25)"
MUTED = "opacity:0.7;"
UNSEEN = "color:#8f8d88;font-style:italic;text-decoration:underline dotted;"
CELL = "padding:6px 8px;vertical-align:middle;"


def rgba(rgb: tuple, alpha: float) -> str:
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.2f})"


def model_label(name: str) -> str:
    color = MODEL_COLOR.get(name, "#8f8d88")
    ext = f' <small style="{MUTED}">(external)</small>' if name in EXTERNAL else ""
    return (
        f'<span style="white-space:nowrap;"><span style="color:{color};font-weight:700;">●</span> '
        f"{SHORT_NAME.get(name, name)}</span>{ext}"
    )


def chip(word: str, style: str, title: str) -> str:
    return (
        f'<span title="{html.escape(title)}" style="{style}padding:2px 6px;border-radius:4px;'
        f'white-space:nowrap;display:inline-block;margin:2px 4px 2px 0;">{html.escape(word)}</span>'
    )


def word_chips(words, importance, unknown) -> str:
    """Words shaded by importance: blue = pushed positive, red = pushed negative.

    Shading is scaled per model (its most important word is darkest), so read it as "which words
    mattered to this model", and the hover value for the size.
    """
    lim = max(0.05, float(pd.Series(importance).abs().max()))
    out = []
    for i, w in enumerate(words):
        v = float(importance[i])
        if unknown is not None and unknown[i]:
            out.append(chip(w, UNSEEN, "never seen in training by this model"))
        else:
            shade = rgba(POS if v > 0 else NEG, abs(v) / lim * 0.75)
            out.append(chip(w, f"background:{shade};", f"{v:+.2f} to the positive probability"))
    return "".join(out)


def word_key() -> str:
    """The colour key for the word chips."""
    return (
        f'<div style="font-size:0.9rem;margin:2px 0 8px 0;{MUTED}">Key: '
        + chip("pushed positive", f"background:{rgba(POS, 0.6)};", "")
        + chip("pushed negative", f"background:{rgba(NEG, 0.6)};", "")
        + chip("never seen in training", UNSEEN, "")
        + " Darker = mattered more to that model. Hover a word for its exact effect.</div>"
    )


def verdict(p: float) -> str:
    label, rgb = ("positive", POS) if p >= 0.5 else ("negative", NEG)
    return (
        f'<span style="background:{rgba(rgb, 1)};color:white;padding:2px 8px;border-radius:10px;'
        f'font-weight:600;white-space:nowrap;">{label}</span>'
    )


def prob_bar(p: float) -> str:
    # neutral fill (the text colour): colour belongs to the model dot and the verdict pill
    return (
        f'<span style="white-space:nowrap;"><span style="background:{TRACK};border-radius:4px;'
        'height:10px;width:140px;display:inline-block;vertical-align:middle;">'
        f'<span style="background:currentColor;opacity:0.55;width:{p * 100:.0f}%;height:10px;'
        f'border-radius:4px;display:block;"></span></span> {p:.0%} positive</span>'
    )


def table(header: list[str], rows: list[str], font: str = "1rem") -> str:
    head = "".join(f"<th style='{CELL}'>{h}</th>" for h in header)
    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:{font};">'
        f"<tr style='text-align:left;{MUTED}'>{head}</tr>" + "".join(rows) + "</table>"
    )


def row(cells: list[str], bg: str = "") -> str:
    tds = "".join(f"<td style='{CELL}'>{c}</td>" for c in cells)
    return f"<tr style='border-top:1px solid {LINE};{bg}'>{tds}</tr>"


def results_table(text: str, idx: int, sc: pd.DataFrame) -> None:
    flag_unusual = unusual is not None and get_unusual_flag(unusual, UNUSUAL_PCT, text)[0]
    if flag_unusual:
        st.warning(
            "**Unusual input:** this sentence is further from the training data than 95% of SST-2 "
            "test sentences. Unusual doesn't mean wrong (H4), but treat verdicts with care."
        )
    rows, why = [], []
    for name, clf in models.items():
        r = sc[(sc["model"] == name) & (sc["text_idx"] == idx)].iloc[0]
        unknown = unknown_words(clf, text)
        flags = []
        if r["confidence"] < low_conf:
            flags.append("⚠️ low confidence")
        if unknown and any(unknown):
            flags.append(f"🔍 {sum(unknown)} unseen word(s)")
        rows.append(
            row(
                [
                    model_label(name),
                    verdict(r["p_pos"]),
                    prob_bar(r["p_pos"]),
                    "<br>".join(flags) or "—",
                ]
            )
        )
        if show_words:
            occ = get_occlusion(clf, name, text)
            chips = word_chips(occ["word"].tolist(), occ["importance"].to_numpy(), unknown)
            why.append(
                f"<div style='{CELL}'>{model_label(name)}</div><div style='{CELL}'>{chips}</div>"
            )
    st.markdown(table(["model", "verdict", "probability", "flags"], rows), unsafe_allow_html=True)
    if show_words:
        # its own full-width section, so a sentence's words sit on one line instead of being
        # squeezed into a table column
        st.markdown("**Which words drove each verdict?**")
        st.markdown(
            word_key()
            + "<div style='display:grid;grid-template-columns:max-content 1fr;align-items:center;"
            f"border-top:1px solid {LINE};'>" + "".join(why) + "</div>",
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------ results
with st.spinner(f"Asking {len(models)} models..."):
    sc = pd.concat([get_scores(clf, name, tuple(texts)) for name, clf in models.items()])


def change_table() -> None:
    """The punchline of a pair: per model, A vs B and whether the verdict flips."""
    w = sc.pivot(index="model", columns="text_idx", values="p_pos").reindex(list(models))
    rows = []
    for m, (a, b) in w[[0, 1]].iterrows():
        flips = (a >= 0.5) != (b >= 0.5)
        rows.append(
            row(
                [
                    model_label(m),
                    f"{verdict(a)} {a:.0%}",
                    f"{verdict(b)} {b:.0%}",
                    f"<b>{b - a:+.0%}</b>",
                    "<b>FLIPS</b>" if flips else "same",
                ],
                bg=f"background:{rgba(NEG, 0.14)};" if flips else "",
            )
        )
    header = ["model", "A", "B", "change in positive probability", "verdict"]
    st.markdown(table(header, rows, font="1.05rem"), unsafe_allow_html=True)


if len(texts) == 2:
    st.subheader("What changed between A and B?")
    st.markdown(f"**A:** “{html.escape(texts[0])}”  \n**B:** “{html.escape(texts[1])}”")
    change_table()
    st.caption("A flip on a change that shouldn't matter (a name, a typo in a noun) is a shortcut.")
    st.subheader("Why? Each sentence in detail")
    for i, tab in enumerate(st.tabs([f"A: {texts[0]}", f"B: {texts[1]}"])):
        with tab, st.spinner("Removing one word at a time..."):
            results_table(texts[i], i, sc)
else:
    st.subheader(f"“{texts[0]}”")
    with st.spinner("Removing one word at a time..."):
        results_table(texts[0], 0, sc)

st.divider()
st.caption(
    "Models: our five classical models and DistilBERT were trained on SST-2 movie-review phrases. "
    "DistilBERT (HF) and RoBERTa-large are ready-made models from Hugging Face (external). "
    "Project: github.com/AditeyaAItronics/mirage"
)
