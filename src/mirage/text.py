"""Text normalization and negation marking.

SST-2 is pre-tokenized and lowercased ("it 's", "does n't", "great , really"). Templates and IMDb
are not ("doesn't"). `normalize` maps any input into SST style so every model sees the same tokens
for the same word. See majordecisions.md D5/D6.
"""

from __future__ import annotations

import re

NEGATION_WORDS: frozenset[str] = frozenset({"not", "never", "no", "nothing", "n't"})
SCOPE_END: frozenset[str] = frozenset({".", ",", "!", "?", ";", ":", "...", "--"})
NEG_PREFIX = "NOT_"
# Fixed phrases where the negation word does not negate what follows (D31): "not only great but
# also funny", "no doubt the best", "nothing but praise".
NEG_EXCEPTIONS: frozenset[tuple[str, str]] = frozenset(
    {("not", "only"), ("not", "just"), ("no", "doubt"), ("nothing", "but"), ("never", "fails")}
)
SCHEMES = ("punct", "window3")

_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_CURLY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_NT = re.compile(r"(\w)n't\b")  # "doesn't" -> "does n't", "can't" -> "ca n't" (SST style)
_CLITIC = re.compile(r"(\w)'(s|re|ve|ll|d|m)\b")  # "it's" -> "it 's"
# Periods are handled per token, below; commas inside numbers ("10,000") stay, as in SST-2.
_PUNCT = re.compile(r"(\.\.\.|--|[!?;:()\"]|(?<!\d),|,(?!\d))")
_GLUED = re.compile(r"([a-z]{2,})\.([a-z]{2,})")  # "movie.the" (IMDb) -> "movie . the"
_INITIALS = re.compile(r"^(?:[a-z]\.)+$")  # "j." "l.a." "p.o.w." stay whole, as in SST-2
_ABBREV = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "st",
        "jr",
        "sr",
        "vs",
        "bros",
        "etc",
        "al",
        "inc",
        "co",
        "ltd",
        "prof",
        "mt",
    }
)
_SPACES = re.compile(r"\s+")


def _split_period(tok: str) -> str:
    if not tok.endswith(".") or set(tok) == {"."} or _INITIALS.match(tok) or tok[:-1] in _ABBREV:
        return tok
    return tok.rstrip(".") + " ."


def normalize(text: str) -> str:
    """Lowercase and tokenize into SST-2 style. Idempotent on already-normalized text.

    A sentence-final period becomes its own token, but abbreviation periods stay attached
    ("mr.", "l.a.", "j."), matching SST-2. Otherwise they would also end negation scopes.
    """
    s = _BR.sub(" ", text).translate(_CURLY).lower()
    s = _NT.sub(r"\1 n't", s)
    s = _CLITIC.sub(r"\1 '\2", s)
    s = _PUNCT.sub(r" \1 ", s)
    s = _GLUED.sub(r"\1 . \2", s)
    return " ".join(_split_period(t) for t in _SPACES.split(s.strip()) if t)


def tokens(text: str) -> list[str]:
    """Whitespace tokenizer for already-normalized text (module-level so it pickles)."""
    return text.split()


def has_negation(normalized: str) -> bool:
    return any(tok in NEGATION_WORDS for tok in normalized.split())


_SENT_END = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str, min_words: int = 3) -> list[str]:
    """Raw-text sentences (split after . ! ? + space) with at least `min_words` words."""
    return [s for s in _SENT_END.split(_BR.sub(" ", text)) if len(s.split()) >= min_words]


def negation_mark(normalized: str, scheme: str = "punct") -> str:
    """Prefix tokens in a negation's scope with NOT_.

    scheme="punct" (Pang et al., the `negmark` variant): from a negation word up to the next
    punctuation mark or the end of the text. "not good movie ." -> "not NOT_good NOT_movie ."
    scheme="window3" (the `negmark_w3` variant, D31): at most the next 3 tokens, still cut at
    punctuation, and fixed phrases in NEG_EXCEPTIONS ("not only", "no doubt") open no scope.
    The negation word itself is kept unchanged; a new negation word restarts the scope.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown negation scheme {scheme!r}; expected one of {SCHEMES}")
    window = scheme == "window3"
    toks = normalized.split()
    out: list[str] = []
    left = 0  # tokens still in scope; len(toks) means "until punctuation"
    for i, tok in enumerate(toks):
        if tok in SCOPE_END:
            left = 0
            out.append(tok)
        elif tok in NEGATION_WORDS:
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            exception = window and (tok, nxt) in NEG_EXCEPTIONS
            left = 0 if exception else (3 if window else len(toks))
            out.append(tok)
        else:
            out.append(NEG_PREFIX + tok if left > 0 else tok)
            left = max(left - 1, 0)
    return " ".join(out)


class Preprocessor:
    """Picklable raw-text -> model-text transform used inside sklearn vectorizers."""

    def __init__(self, negmark: bool = False) -> None:
        self.negmark = negmark

    def __call__(self, text: str) -> str:
        # negmark: False (base), True (Pang scope, as in models saved before D31) or a scheme.
        s = normalize(text)
        if not self.negmark:
            return s
        return negation_mark(s, "punct" if self.negmark is True else self.negmark)

    def __repr__(self) -> str:
        return f"Preprocessor(negmark={self.negmark})"
