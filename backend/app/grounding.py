"""Deterministic helpers that keep answers honest: quote verification, unsupported-entity
guard, confidence calculation."""
from __future__ import annotations

import re

STOP = set("""a an the of to in on for with and or but is are was were be been being do does did what which who whom
whose when where why how that this these those it its as at by from about into over after before between across
than then so if not no can could should would will may might must have has had i you we they he she them us our
your their me my any all each every both either neither more most less least much many some such own same other
per via vs versus""".split())
QUESTION_WORDS = {"what", "which", "who", "when", "where", "why", "how", "compare", "list", "show", "tell", "explain",
                  "describe", "summarize", "summarise", "give", "find", "is", "are", "do", "does", "can", "will",
                  "should", "please", "identify", "contrast", "outline"}
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w%$ ]+", " ", s.lower())).strip()


def content_words(text: str) -> list[str]:
    return [w for w in (m.lower().strip("'-") for m in _WORD.findall(text)) if w and w not in STOP]


def quote_in_text(quote: str, text: str) -> bool:
    q = normalize(quote)
    return len(q) >= 8 and q in normalize(text)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def best_sentence(text: str, query: str, max_len: int = 300) -> str:
    q = set(content_words(query))
    sents = split_sentences(text) or [text]
    best = max(sents, key=lambda s: (len(q & set(content_words(s))), -len(s)))
    return best[:max_len]


def unsupported_entities(question: str, evidence_texts: list[str]) -> list[str]:
    """Capitalised, non-initial tokens in the question (e.g. 'Mars') that appear nowhere in the
    retrieved evidence. A cheap, transparent guard against answering about things the corpus never mentions."""
    toks = _WORD.findall(question)
    corpus = normalize(" ".join(evidence_texts))
    missing = []
    for i, t in enumerate(toks):
        if i == 0:
            continue
        base = re.sub(r"'s$", "", t)
        if not base[0].isalpha() or not base[0].isupper() or base.lower() in QUESTION_WORDS | STOP:
            continue
        if normalize(base) not in corpus and base not in missing:
            missing.append(base)
    return missing


CONFIDENCE_TEXT = {
    "high": "Multiple signals agree: the evidence directly supports this answer.",
    "medium": "The evidence supports this answer, but it is partial or sources disagree.",
    "low": "Little or no direct support was found in the indexed documents.",
}


def compute_confidence(*, insufficient: bool, conflict: bool, cited_docs: int, top_score: float,
                       high_score: float, min_score: float, llm_confidence: str | None) -> str:
    if insufficient or cited_docs == 0 or top_score < min_score:
        return "low"
    level = "high" if top_score >= high_score else "medium"
    if conflict:
        level = "medium"  # disagreeing sources can never produce high confidence
    order = {"low": 0, "medium": 1, "high": 2}
    if llm_confidence in order:
        level = min(level, llm_confidence, key=order.get)
    return level
