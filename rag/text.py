"""Small text utilities shared by the offline embedder and the extractive answerer."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a about above after again all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had
    has have having he her here hers him his how i if in into is it its itself just me more
    most my no nor not now of off on once only or other our ours out over own same she should
    so some such than that the their theirs them then there these they this those through to
    too under until up very was we were what when where which while who whom why will with
    would you your yours yourself tell know please get much many
    """.split()
)


def stem(token: str) -> str:
    """Very light suffix stripping so 'policies'~'policy', 'approved'~'approve'."""
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, treating each bullet/line as its own unit."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^([-*•]|\d+[.)])\s+", "", line)
        for sentence in _SENTENCE_RE.split(line):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences
