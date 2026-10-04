"""Investigation pipeline: retrieve -> guard -> (LLM | extractive) -> validate -> conflicts -> confidence."""
from __future__ import annotations

import re
import time

from .conflicts import detect_conflicts
from .engine import Engine
from .grounding import (CONFIDENCE_TEXT, best_sentence, compute_confidence, normalize, quote_in_text,
                        unsupported_entities)
from .llm import LLMError
from .store import now_iso

INSUFFICIENT_ANSWER = ("I couldn't find enough support in your indexed documents to answer this reliably.")


class InvestigationError(Exception):
    pass


def _validate_citations(raw: list[dict], evidence: list[dict], question: str) -> list[dict]:
    by_id = {e["id"]: e for e in evidence}
    out, seen = [], set()
    for c in raw or []:
        e = by_id.get(c.get("evidence_id"))
        if not e or e["id"] in seen:
            continue
        quote = (c.get("quote") or "").strip()
        verified = quote_in_text(quote, e["text"])
        if not verified:  # never keep an unverifiable quote: snap to a real sentence from the cited chunk
            quote = best_sentence(e["text"], question)
        seen.add(e["id"])
        out.append({"evidence_id": e["id"], "document": e["document"], "document_id": e["document_id"], "page": e["page"],
                    "section": e["section"], "quote": quote[:300], "score": e["score"], "quote_verified": verified})
    return out


def _validate_conflicts(raw: list[dict], evidence: list[dict]) -> list[dict]:
    by_id = {e["id"]: e for e in evidence}
    out = []
    for c in raw or []:
        claims, docs = [], set()
        for cl in c.get("claims", []):
            e = by_id.get(cl.get("evidence_id"))
            claim = (cl.get("claim") or "").strip()
            if not e or not claim or not (normalize(claim) in normalize(e["text"]) or quote_in_text(cl.get("quote", ""), e["text"])):
                continue
            quote = cl.get("quote", "") if quote_in_text(cl.get("quote", ""), e["text"]) else best_sentence(e["text"], claim)
            claims.append({"document": e["document"], "document_id": e["document_id"], "page": e["page"], "claim": claim,
                           "quote": quote[:300], "evidence_id": e["id"]})
            docs.add(e["document_id"])
        if len(docs) >= 2:
            out.append({"topic": c.get("topic", "").strip() or "unspecified", "claims": claims,
                        "assessment": "The available documents disagree. No authoritative source has been established."})
    return out


def _merge_conflicts(llm_conf: list[dict], det: list[dict]) -> list[dict]:
    def sig(c):
        return {(x["document_id"], normalize(x["claim"])) for x in c["claims"]}

    merged = list(llm_conf)
    for d in det:
        if not any(len(sig(d) & sig(m)) >= 2 for m in merged):
            merged.append(d)
    return merged


def _extractive_answer(question: str, evidence: list[dict], conflicts: list[dict]) -> tuple[str, list[dict]]:
    """No-LLM mode: quote the most relevant sentence from each top source. Clearly labelled by the caller."""
    cites, lines = [], []
    for c in conflicts:
        parts = [f"{x['document']} (p.{x['page']}) states {x['claim']} [{x['evidence_id']}]" for x in c["claims"]]
        lines.append(f"The documents disagree about {c['topic']}: " + "; ".join(parts) + ".")
        for x in c["claims"]:
            cites.append({"evidence_id": x["evidence_id"], "quote": x["quote"]})
    for e in evidence[:3]:
        if any(e["id"] == c["evidence_id"] for c in cites):
            continue
        s = best_sentence(e["text"], question)
        lines.append(f"{e['document']} (p.{e['page']}): {s} [{e['id']}]")
        cites.append({"evidence_id": e["id"], "quote": s})
    return "\n\n".join(lines), cites


def investigate(engine: Engine, question: str, *, top_k: int | None = None, min_score: float | None = None,
                doc_ids: list[str] | None = None, collection: str | None = None) -> dict:
    t0 = time.perf_counter()
    question = question.strip()
    top_k = top_k or engine.settings.top_k
    min_score = engine.min_score if min_score is None else min_score
    scope = engine.scope_docs(doc_ids, collection)
    evidence, best = engine.retrieve(question, top_k, min_score, scope)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    guard = unsupported_entities(question, [e["text"] for e in evidence]) if evidence else []
    mode, answer, citations, conflicts, llm_conf = "none", "", [], [], None
    insufficient = not evidence or bool(guard)
    notes = []
    if guard:
        notes.append("Not found in retrieved evidence: " + ", ".join(guard))

    if not insufficient:
        det = detect_conflicts(evidence)
        if engine.llm is not None:
            mode = "llm"
            try:
                raw = engine.llm.investigate(question, evidence)
            except LLMError as e:
                raise InvestigationError(str(e)) from e
            insufficient = bool(raw.get("insufficient_evidence"))
            answer = (raw.get("answer") or "").strip()
            llm_conf = raw.get("confidence")
            if not insufficient:
                citations = _validate_citations(raw.get("citations", []), evidence, question)
                conflicts = _merge_conflicts(_validate_conflicts(raw.get("conflicts", []), evidence), det)
        else:
            mode = "extractive"
            conflicts = det
            answer, raw_c = _extractive_answer(question, evidence, conflicts)
            citations = _validate_citations(raw_c, evidence, question)
        if not insufficient:  # drop markers that point at nothing we kept
            valid = {c["evidence_id"] for c in citations}
            answer = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) in valid else "", answer)
            insufficient = insufficient or not citations

    if insufficient:
        answer = answer if (mode == "llm" and answer) else INSUFFICIENT_ANSWER
        evidence, citations, conflicts = [], [], []

    cited_docs = len({c["document_id"] for c in citations})
    confidence = compute_confidence(insufficient=insufficient, conflict=bool(conflicts), cited_docs=cited_docs, top_score=best,
                                    high_score=engine.embedder.default_high_score, min_score=min_score, llm_confidence=llm_conf)
    cited_ids = {c["evidence_id"] for c in citations}
    return {
        "question": question, "answer": answer, "confidence": confidence, "confidence_note": CONFIDENCE_TEXT[confidence],
        "insufficient_evidence": insufficient, "conflict_detected": bool(conflicts),
        "citations": citations, "conflicts": conflicts,
        "evidence": [{**e, "cited": e["id"] in cited_ids} for e in evidence],
        "sources_analyzed": len(evidence), "relevant_sources": cited_docs,
        "documents_searched": len(scope), "chunks_searched": len(engine.index), "notes": notes,
        "meta": {"mode": mode, "embedder": engine.embedder.name, "model": engine.settings.anthropic_model if mode == "llm" else None,
                 "top_k": top_k, "min_score": min_score, "retrieval_ms": retrieval_ms,
                 "total_ms": int((time.perf_counter() - t0) * 1000), "generated_at": now_iso()},
    }
