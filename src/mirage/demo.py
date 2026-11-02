"""Logic behind the Shortcut Explorer demo (app/shortcut_explorer.py, D41).

The app is UI only; everything it shows comes from here, so it can be tested without Streamlit:
- every model's verdict on a sentence,
- leave-one-word-out importance (works the same for every model, including the forest),
- which words a classical model has never seen (its vocabulary),
- two honest flags: low confidence (which H4 showed predicts errors) and unusual input (which
  it showed doesn't, so it's labelled "unusual", not "probably wrong").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mirage import paths
from mirage.config import load_config
from mirage.explain.occlusion import leave_one_out, occlusion_table  # noqa: F401 (re-export)
from mirage.text import normalize


def trim(text: str, max_words: int) -> tuple[str, bool]:
    """Strip and cut to max_words words; also report whether anything was cut."""
    words = text.split()
    return " ".join(words[:max_words]), len(words) > max_words


def _distilbert_seed() -> int:
    from mirage.evaluate import median_seed
    from mirage.predict import load_predictions

    try:
        return median_seed(load_predictions(model="distilbert", dataset="sst2_val"))
    except (FileNotFoundError, ValueError):
        return 42


def load_models(light: bool = False, cfg: dict | None = None) -> dict:
    """name -> classifier, in display order. Models that aren't available are skipped."""
    from mirage.models.base import load_classifier
    from mirage.models.external import ExternalClassifier

    cfg = cfg or load_config("demo")
    external = load_config("external")
    out = {}
    for name in cfg["models"]:
        if light and name in cfg["light_skip"]:
            continue
        try:
            if name in external["models"]:
                out[name] = ExternalClassifier.from_hub(name, external["models"][name], external)
            else:
                seed = _distilbert_seed() if name == "distilbert" else 42
                out[name] = load_classifier(paths.model_dir(name, "base", seed))
        except (OSError, FileNotFoundError) as e:  # not trained / not downloaded: leave it out
            print(f"demo: skipping {name}: {e}")
    return out


def score(models: dict, texts: list[str], low_confidence: float = 0.70) -> pd.DataFrame:
    """One row per (model, text): p_pos, label, confidence and the low-confidence flag."""
    rows = []
    for name, clf in models.items():
        p = clf.predict_proba(list(texts))[:, 1]
        for i, pp in enumerate(p):
            conf = max(pp, 1 - pp)
            rows.append(
                {
                    "model": name,
                    "text_idx": i,
                    "p_pos": float(pp),
                    "label": "positive" if pp >= 0.5 else "negative",
                    "confidence": float(conf),
                    "low_confidence": bool(conf < low_confidence),
                }
            )
    return pd.DataFrame(rows)


def occlusion(clf, text: str) -> pd.DataFrame:
    """Word importance: p_pos(full) - p_pos(without the word). > 0 = the word pushed positive.

    The same rule as the H5 analysis (explain.occlusion, D43); one batched call per model.
    """
    occ = occlusion_table(clf.predict_proba, [text])
    return pd.DataFrame({"word": occ["word"], "importance": occ["delta"]})


def unknown_words(clf, text: str) -> list[bool] | None:
    """Per word: True if a classical model has never seen any of its tokens (not in vocabulary).

    None for models without a word vocabulary (DistilBERT and external models split unknown words
    into pieces instead).
    """
    pipe = getattr(clf, "pipeline", None)
    if pipe is None or "vec" not in pipe.named_steps:
        return None
    vocab = pipe.named_steps["vec"].vocabulary_
    out = []
    for w in text.split():
        toks = [t for t in normalize(w).split() if any(c.isalnum() for c in t)]
        out.append(bool(toks) and not any(t in vocab for t in toks))
    return out


class UnusualInput:
    """Flags input whose most unusual sentence is further from SST-2 training sentences than
    `percentile` % of SST-2 test sentences. Same distance and unit as the warn stage (D25)."""

    def __init__(self, percentile: float = 95, cfg: dict | None = None) -> None:
        from mirage.data import load
        from mirage.warning.distance import fit_distance, text_distance

        self.cfg = cfg or load_config("warning")
        self.scorer = fit_distance(load("sst2_train"), self.cfg)
        ref = text_distance(self.scorer, load("sst2_test"), "sst2_test", self.cfg)
        self.threshold = float(np.nanpercentile(ref.to_numpy(), percentile))
        self._model = None

    def distance(self, text: str) -> float:
        from sentence_transformers import SentenceTransformer

        from mirage.text import sentences

        if self._model is None:
            self._model = SentenceTransformer(self.cfg["embedding_model"])
        sents = sentences(text) or [text]
        emb = self._model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
        return float(np.max(self.scorer.score(emb)))

    def __call__(self, text: str) -> tuple[bool, float]:
        d = self.distance(text)
        return d > self.threshold, d


# Which robustness tests (mirage_gap.csv columns, D40) back each curated example's shortcut, and
# how to name them on screen. Shortcuts without a matching test (vocabulary, sarcasm) show none.
SHORTCUT_TESTS = {
    "names": [("beh_name_shortcut", "name-swap pairs")],
    "typos": [("beh_typo", "typo pairs")],
    "negation": [
        ("beh_negation_dir", "negation template pairs"),
        ("sst2_neg", "real negated sentences"),
    ],
}


def load_findings(path=None) -> pd.DataFrame | None:
    """Each model's Mirage Gap row (base variant, D40), indexed by model: accuracy, robustness,
    gap and the pass rate of every robustness test (`pass__<test>`). None if `evaluate` hasn't
    written results/mirage_gap.csv, so the demo still runs without it."""
    path = path or paths.RESULTS / "mirage_gap.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[df["variant"] == "base"].set_index("model")


def flipped(scores: pd.DataFrame) -> list[str]:
    """Models whose predicted label differs between sentence A (text 0) and B (text 1)."""
    lab = scores.pivot(index="model", columns="text_idx", values="label")
    return [m for m in scores["model"].unique() if lab.loc[m, 0] != lab.loc[m, 1]]


def shortcut_failures(
    findings: pd.DataFrame | None, model: str, shortcut: str | None
) -> list[tuple[float, str]]:
    """(failure rate, test name) for every full test behind `shortcut`, for one model.

    A pair fails unless the model gets both sentences right; a real sentence fails if it's wrong.
    Empty if there are no findings, the model isn't in them, or no test backs the shortcut."""
    if findings is None or model not in findings.index:
        return []
    out = []
    for test, label in SHORTCUT_TESTS.get(shortcut, []):
        rate = findings.loc[model, f"pass__{test}"]
        if pd.notna(rate):
            out.append((1 - float(rate), label))
    return out


def load_examples(path=None) -> list[dict]:
    """The curated examples in app/examples.yaml, in presentation order."""
    import yaml

    path = path or paths.ROOT / "app" / "examples.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_example(models: dict, example: dict) -> list[str]:
    """Return the expectations an example no longer meets (empty = behaves as documented).

    Expectations about models that aren't loaded (e.g. light mode) are skipped.
    """
    texts = [example["a"]] + ([example["b"]] if "b" in example else [])
    sc = score({m: c for m, c in models.items() if _mentioned(m, example)}, texts)
    pred = sc.set_index(["model", "text_idx"])["label"]
    failures = []
    for kind, names in example.get("expect", {}).items():
        for m in names:
            if m not in models:
                continue
            a = pred[(m, 0)]
            if kind in ("positive", "negative"):
                ok = a == kind
            else:
                flipped = a != pred[(m, 1)]
                ok = flipped if kind == "flips" else not flipped
            if not ok:
                failures.append(f"{example['id']}: {m} expected {kind}")
    return failures


def _mentioned(model: str, example: dict) -> bool:
    return any(model in names for names in example.get("expect", {}).values())
