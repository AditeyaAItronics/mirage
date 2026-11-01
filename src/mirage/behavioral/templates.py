"""CheckList-style generators. Each returns a list of row dicts for the suite.

Row fields: text, label, test_type, kind ("MFT" or "INV"), case_id, role, scored, word.
`word` is the sentiment adjective the case is built on, so failures can be broken down per word
and negation failures checked against the same words in the simple test (shortfalls.md D1).
- MFT (minimum functionality): one row per case, failure = wrong label.
- INV (invariance): two rows per case (role original / perturbed), failure = prediction changes.
- DIR (directional): two rows per case, an affirmative sentence and its negated twin; failure =
  adding the negation does not move the model's *own* prediction (scored in suite.py, D27).
`label` is always the intended sentiment, so every row fits the predictions contract.
"""

from __future__ import annotations

import itertools

import numpy as np

from mirage.behavioral import wordlists as W

SIMPLE_TEMPLATES = [
    "The {noun} was {adj}.",
    "This {noun} is {adj}.",
    "I thought the {noun} was {adj}.",
    "Honestly, the {noun} is {adj}.",
]
NEGATION_TEMPLATES = [
    "The {noun} was not {adj}.",
    "This {noun} isn't {adj}.",
    "I don't think the {noun} was {adj}.",
    "The {noun} was never {adj}.",
]
# (affirmative, negated) twins: same frame, noun and adjective; only the negation differs.
PAIRED_TEMPLATES = [
    ("The {noun} was {adj}.", "The {noun} was not {adj}."),
    ("This {noun} is {adj}.", "This {noun} isn't {adj}."),
    ("I thought the {noun} was {adj}.", "I don't think the {noun} was {adj}."),
    ("The {noun} was always {adj}.", "The {noun} was never {adj}."),
]
# {name} is a film person; the sentiment comes only from {adj} (D32).
NAME_SHORTCUT_TEMPLATES = [
    "The new {name} film was {adj}.",
    "I watched the {name} movie and it was {adj}.",
    "{name}'s latest {noun} is {adj}.",
    "Another {name} {noun}, and it was {adj}.",  # not "a {adj} one": "a uneven" (D32)
]
NAME_TEMPLATES = ["{name} {verb} the {noun}.", "{name} said the {noun} was {adj}."]


def _sample(rng: np.random.Generator, pool: list, k: int) -> list:
    if k > len(pool):
        raise ValueError(f"asked for {k} cases but only {len(pool)} unique combinations exist")
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in sorted(idx)]


def _fill(templates, adjs, k, rng) -> list[tuple[str, str]]:
    """k distinct (text, adjective) pairs."""
    pool = list(itertools.product(templates, W.NOUNS, adjs))
    return [(t.format(noun=n, adj=a), a) for t, n, a in _sample(rng, pool, k)]


def simple(n: int, rng: np.random.Generator) -> list[dict]:
    rows = []
    for label, adjs, k in ((1, W.POS_ADJ, n // 2), (0, W.NEG_ADJ, n - n // 2)):
        rows += [
            {"text": t, "label": label, "test_type": "simple", "word": a}
            for t, a in _fill(SIMPLE_TEMPLATES, adjs, k, rng)
        ]
    return [
        {**r, "kind": "MFT", "case_id": f"simple-{i}", "role": "single", "scored": True}
        for i, r in enumerate(rows)
    ]


def negation(n_pos: int, n_neg: int, rng: np.random.Generator) -> list[dict]:
    """'not <positive>' -> negative (scored). 'not <negative>' -> labelled 1, NOT scored.

    "not bad" is at best mildly positive, so negated negatives are reported separately as an
    ambiguous probe rather than counted as failures (majordecisions.md D11).
    """
    rows = [
        {"text": t, "label": 0, "test_type": "negation", "scored": True, "word": a}
        for t, a in _fill(NEGATION_TEMPLATES, W.POS_ADJ, n_pos, rng)
    ]
    rows += [
        {"text": t, "label": 1, "test_type": "negation_neg", "scored": False, "word": a}
        for t, a in _fill(NEGATION_TEMPLATES, W.NEG_ADJ, n_neg, rng)
    ]
    return [
        {**r, "kind": "MFT", "case_id": f"{r['test_type']}-{i}", "role": "single"}
        for i, r in enumerate(rows)
    ]


def negation_dir(n_pos: int, n_neg: int, rng: np.random.Generator) -> list[dict]:
    """Directional negation pairs (shortfalls.md D5).

    The plain negation MFT expects "not <positive>" to be negative, so a model that leans negative
    passes it for free. Here each case is "X" and "not X" and the question is whether the model's
    own answer moves. Positive adjectives are scored; "not <negative>" pairs are an unscored
    probe, as in D11. `label` is the intended label of each sentence.
    """
    rows = []
    for test_type, adjs, n, scored, pos_label in (
        ("negation_dir", W.POS_ADJ, n_pos, True, 1),
        ("negation_dir_neg", W.NEG_ADJ, n_neg, False, 0),
    ):
        pool = list(itertools.product(PAIRED_TEMPLATES, W.NOUNS, adjs))
        for i, ((aff, neg), noun, adj) in enumerate(_sample(rng, pool, n)):
            for role, t, label in (("original", aff, pos_label), ("perturbed", neg, 1 - pos_label)):
                rows.append(
                    {
                        "text": t.format(noun=noun, adj=adj),
                        "label": label,
                        "role": role,
                        "case_id": f"{test_type}-{i}",
                        "word": adj,
                        "test_type": test_type,
                        "kind": "DIR",
                        "scored": scored,
                    }
                )
    return rows


def name_shortcut(n: int, pairs: list[tuple[str, str]], rng: np.random.Generator) -> list[dict]:
    """Invariance: swap a name whose training phrases are mostly positive for one whose phrases
    are mostly negative (D32). The sentence's sentiment comes only from a mild adjective, so the
    prediction should not move. `original` always holds the positive-associated name, so
    p_pos(original) - p_pos(perturbed) > 0 means the model learned the names' reputations.
    """
    texts = {}
    for t, (good, bad), noun, (label, adj) in itertools.product(
        NAME_SHORTCUT_TEMPLATES,
        pairs,
        W.WORK_NOUNS,
        [(1, a) for a in W.MILD_POS] + [(0, a) for a in W.MILD_NEG],
    ):
        a = t.format(name=good, noun=noun, adj=adj)
        texts.setdefault(a, (t.format(name=bad, noun=noun, adj=adj), label, adj))
    pool = sorted(texts.items())
    rows = []
    for i, (orig, (pert, label, adj)) in enumerate(_sample(rng, pool, n)):
        for role, text in (("original", orig), ("perturbed", pert)):
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "role": role,
                    "case_id": f"name_shortcut-{i}",
                    "word": adj,
                    "test_type": "name_shortcut",
                    "kind": "INV",
                    "scored": True,
                }
            )
    return rows


def make_typo(word: str, rng: np.random.Generator) -> str:
    """Swap two adjacent inner characters or drop one inner character; never returns `word`."""
    for _ in range(20):
        if rng.random() < 0.5 and len(word) >= 4:
            i = int(rng.integers(1, len(word) - 2))
            out = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        else:
            i = int(rng.integers(1, len(word) - 1))
            out = word[:i] + word[i + 1 :]
        if out != word:
            return out
    raise ValueError(f"could not perturb {word!r}")


def typo(n: int, rng: np.random.Generator) -> list[dict]:
    """Invariance: a typo in the sentiment word only. Short words (<4 letters) are excluded."""
    pool = [
        (t, noun, adj, label)
        for label, adjs in ((1, W.POS_ADJ), (0, W.NEG_ADJ))
        for t, noun, adj in itertools.product(SIMPLE_TEMPLATES, W.NOUNS, adjs)
        if len(adj) >= 4
    ]
    rows = []
    for i, (t, noun, adj, label) in enumerate(_sample(rng, pool, n)):
        typo_adj = make_typo(adj, rng)
        for role, a in (("original", adj), ("perturbed", typo_adj)):
            text = t.format(noun=noun, adj=a)
            rows.append(
                {"text": text, "label": label, "role": role, "case_id": f"typo-{i}", "word": adj}
            )
    return [{**r, "test_type": "typo", "kind": "INV", "scored": True} for r in rows]


def typo_noun(n: int, rng: np.random.Generator) -> list[dict]:
    """Control for `typo` (C11, D37): the typo goes in the NOUN, the sentiment word stays intact.

    "The movie was brilliant." -> "The moive was brilliant." A model that turns negative here reacts
    to any garbled word, not to losing the sentiment word.
    """
    pool = [
        (t, noun, adj, label)
        for label, adjs in ((1, W.POS_ADJ), (0, W.NEG_ADJ))
        for t, noun, adj in itertools.product(SIMPLE_TEMPLATES, W.NOUNS, adjs)
    ]
    rows = []
    for i, (t, noun, adj, label) in enumerate(_sample(rng, pool, n)):
        typo_noun_ = make_typo(noun, rng)
        for role, nn in (("original", noun), ("perturbed", typo_noun_)):
            rows.append(
                {
                    "text": t.format(noun=nn, adj=adj),
                    "label": label,
                    "role": role,
                    "case_id": f"typo_noun-{i}",
                    "word": adj,
                }
            )
    return [{**r, "test_type": "typo_noun", "kind": "INV", "scored": True} for r in rows]


def name_swap(n: int, rng: np.random.Generator) -> list[dict]:
    """Invariance: swap an Indian name for a Western one (and back) in otherwise identical text."""
    pool = []
    for label, verbs, adjs in ((1, W.POS_VERBS, W.POS_ADJ), (0, W.NEG_VERBS, W.NEG_ADJ)):
        for a, b in zip(W.NAMES_A, W.NAMES_B, strict=True):
            for noun in W.NOUNS:
                pool.append((NAME_TEMPLATES[0], a, b, noun, verbs, adjs, label))
                pool.append((NAME_TEMPLATES[1], b, a, noun, verbs, adjs, label))
    rows = []
    for i, (t, name1, name2, noun, verbs, adjs, label) in enumerate(_sample(rng, pool, n)):
        verb, adj = verbs[int(rng.integers(len(verbs)))], adjs[int(rng.integers(len(adjs)))]
        for role, name in (("original", name1), ("perturbed", name2)):
            text = t.format(name=name, verb=verb, noun=noun, adj=adj)
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "role": role,
                    "case_id": f"name_swap-{i}",
                    "word": adj,
                }
            )
    return [{**r, "test_type": "name_swap", "kind": "INV", "scored": True} for r in rows]
