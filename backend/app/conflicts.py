"""Deterministic conflict detection over quantitative claims (e.g. '12 weeks' vs '16 weeks').

Two sentences from DIFFERENT documents conflict when they state the same kind of quantity
(same unit) with different values, and the rest of the sentence is near-identical (so they
describe the same fact). This is intentionally conservative; the LLM may add semantic
conflicts which are verified against the evidence before being reported."""
from __future__ import annotations

import re
from collections import Counter

from .grounding import content_words, split_sentences

_NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
             "sixteen": 16, "eighteen": 18, "twenty": 20, "thirty": 30, "sixty": 60, "ninety": 90}
_UNIT = r"(weeks?|days?|months?|years?|hours?|percent|%)"
_QTY = re.compile(rf"(?<![\w.])(\d+(?:\.\d+)?|{'|'.join(_NUMWORDS)})[\s-]*{_UNIT}\b", re.I)
JACCARD_MIN = 0.55


def _unit(u: str) -> str:
    u = u.lower()
    return "percent" if u in ("%", "percent") else u.rstrip("s")


def _value(v: str) -> float:
    return float(_NUMWORDS.get(v.lower(), v)) if not v.replace(".", "").isdigit() else float(v)


def quantity_claims(text: str) -> list[dict]:
    out = []
    for s in split_sentences(text):
        for m in _QTY.finditer(s):
            rest = (s[: m.start()] + " " + s[m.end():])
            out.append({"sentence": s, "value": _value(m.group(1)), "unit": _unit(m.group(2)),
                        "label": f"{m.group(1)} {m.group(2)}".replace(" %", "%"), "words": set(content_words(rest))})
            break  # one primary quantity per sentence
    return out


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def detect_conflicts(items: list[dict]) -> list[dict]:
    """items: dicts with document_id, document, page, section, text, (optional) id.
    Returns conflicts: {topic, claims:[{document, document_id, page, claim, quote, evidence_id}], assessment}."""
    nodes = []
    for it in items:
        for c in quantity_claims(it["text"]):
            nodes.append({**c, "doc_id": it["document_id"], "document": it["document"], "page": it["page"],
                          "section": it.get("section") or "", "evidence_id": it.get("id")})
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    linked = set()
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if a["doc_id"] != b["doc_id"] and a["unit"] == b["unit"] and a["value"] != b["value"] \
                    and _jaccard(a["words"], b["words"]) >= JACCARD_MIN:
                parent[find(i)] = find(j)
                linked.update((i, j))

    clusters: dict[int, list[int]] = {}
    for i in linked:
        clusters.setdefault(find(i), []).append(i)

    conflicts = []
    for idxs in clusters.values():
        members = [nodes[i] for i in sorted(idxs)]
        seen, claims = set(), []
        for n in members:
            key = (n["doc_id"], n["page"], n["label"].lower())
            if key in seen:
                continue
            seen.add(key)
            claims.append({"document": n["document"], "document_id": n["doc_id"], "page": n["page"],
                           "section": n["section"], "claim": n["label"], "quote": n["sentence"][:300], "evidence_id": n["evidence_id"]})
        if len({c["document_id"] for c in claims}) < 2:
            continue
        common = set.intersection(*[m["words"] for m in members])
        freq = Counter(w for m in members for w in m["words"])
        topic_words = sorted(common, key=lambda w: (-freq[w], w))[:3] or [w for w, _ in freq.most_common(3)]
        sections = {c["section"].strip().lower() for c in claims if c["section"].strip()}
        if len(sections) == 1:  # all claims sit under the same heading: use it as the topic
            topic = next(c["section"] for c in claims if c["section"].strip()).strip()
        else:
            topic = " ".join(topic_words)
        conflicts.append({"topic": topic, "claims": claims,
                          "assessment": "The available documents disagree. No authoritative source has been established."})
    return conflicts
