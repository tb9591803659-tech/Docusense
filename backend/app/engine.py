"""Engine: owns store, embedder, vector index and LLM; implements ingestion + retrieval."""
from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from pathlib import Path

from .chunking import chunk_pages
from .config import Settings
from .embeddings import Embedder, get_embedder
from .extraction import ExtractionError, extract
from .index import VectorIndex, doc_of
from .llm import AnthropicLLM
from .store import Store, now_iso

STAGES = ["uploading", "extracting", "chunking", "embedding", "indexing"]


class IngestError(Exception):
    def __init__(self, stage: str, message: str, status: int = 422):
        super().__init__(message)
        self.stage, self.message, self.status = stage, message, status


class DuplicateDocument(Exception):
    def __init__(self, doc: dict):
        super().__init__("duplicate")
        self.doc = doc


def safe_filename(name: str) -> str:
    base = Path((name or "").replace("\\", "/")).name
    base = re.sub(r"[^\w.\- ]", "_", base).strip(" .")
    return base[:120] or "document"


class Engine:
    def __init__(self, settings: Settings, embedder: Embedder | None = None, llm=None):
        self.settings = settings
        self.store = Store(settings.data_dir)
        self.embedder = embedder or get_embedder(settings.embedding_backend, settings.embedding_model)
        self.index = VectorIndex(settings.data_dir / "index", self.embedder.dim, self.embedder.name)
        self.llm = llm if llm is not None else (
            AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model) if settings.anthropic_api_key else None)
        self._sync_index()

    # -- index consistency (model change, crash recovery)
    def _sync_index(self):
        store_ids = set(self.store.chunks)
        if self.index.stale or set(self.index.ids) != store_ids:
            self.index.reset()
            chunks = sorted(self.store.chunks.values(), key=lambda c: (c["doc_id"], c["ord"]))
            if chunks:
                self.index.add([c["id"] for c in chunks], self.embedder.encode([c["text"] for c in chunks]))
            else:
                self.index.save()

    @property
    def min_score(self) -> float:
        return self.settings.min_score if self.settings.min_score is not None else self.embedder.default_min_score

    # -- ingestion
    def ingest(self, filename: str, data: bytes, collection: str | None = None) -> tuple[dict, list[dict]]:
        stages: list[dict] = []

        def run(name, fn):
            t = time.perf_counter()
            try:
                out = fn()
            except ExtractionError as e:
                stages.append({"stage": name, "status": "failed", "ms": 0})
                raise IngestError(name, str(e)) from e
            stages.append({"stage": name, "status": "done", "ms": int((time.perf_counter() - t) * 1000)})
            return out

        filename = safe_filename(filename)
        doc_id = uuid.uuid4().hex[:12]
        sha = hashlib.sha256(data).hexdigest()
        dup = self.store.find_by_hash(sha)
        if dup:
            raise DuplicateDocument(dup)
        path = self.store.dir / "uploads" / f"{doc_id}{Path(filename).suffix.lower()}"

        def save():
            if not data:
                raise ExtractionError("The file is empty.")
            path.write_bytes(data)

        try:
            run("uploading", save)
            ex = run("extracting", lambda: extract(filename, data))
            chunks = run("chunking", lambda: chunk_pages(ex.pages, doc_id))
            if not chunks:
                raise IngestError("chunking", "The document produced no usable text chunks.")
            vecs = run("embedding", lambda: self.embedder.encode([c["text"] for c in chunks]))

            def index():
                self.index.add([c["id"] for c in chunks], vecs)

            run("indexing", index)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        doc = {"id": doc_id, "name": Path(filename).stem, "filename": filename,
               "type": Path(filename).suffix.lower().lstrip("."), "pages": len(ex.pages), "page_basis": ex.page_basis,
               "chunks": len(chunks), "status": "indexed", "added_at": now_iso(), "size": len(data),
               "sha256": sha, "collection": collection or None}
        self.store.add_document(doc, chunks)
        return doc, stages

    def delete_document(self, doc_id: str) -> bool:
        doc = self.store.docs.get(doc_id)
        if not doc:
            return False
        self.index.remove_doc(doc_id)
        self.store.delete_document(doc_id)
        for p in (self.store.dir / "uploads").glob(f"{doc_id}.*"):
            p.unlink(missing_ok=True)
        return True

    # -- retrieval
    def scope_docs(self, doc_ids: list[str] | None, collection: str | None) -> set[str]:
        docs = self.store.docs.values()
        if collection:
            docs = [d for d in docs if d.get("collection") == collection]
        ids = {d["id"] for d in docs}
        return ids & set(doc_ids) if doc_ids else ids

    def retrieve(self, question: str, top_k: int, min_score: float, scope: set[str]) -> tuple[list[dict], float]:
        """Returns (evidence, best_score). Evidence is de-duplicated and diversified across documents."""
        if not scope or not len(self.index):
            return [], 0.0
        hits = self.index.search(self.embedder.encode([question])[0], top_k * 5, scope)
        best = hits[0][1] if hits else 0.0
        seen, per_doc, out = set(), {}, []
        cap = max(2, math.ceil(top_k / 2))
        for cid, score in hits:
            if score < min_score or len(out) >= top_k:
                continue
            c = self.store.chunks.get(cid)
            if not c:
                continue
            key = re.sub(r"\s+", " ", c["text"].lower())[:200]
            if key in seen or per_doc.get(c["doc_id"], 0) >= cap:
                continue
            seen.add(key)
            per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
            d = self.store.docs[c["doc_id"]]
            out.append({"id": len(out) + 1, "chunk_id": cid, "document_id": d["id"], "document": d["name"],
                        "page": c["page"], "section": c["section"], "text": c["text"], "score": round(score, 4)})
        return out, best

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Hybrid search: exact text/name matches first, then semantic hits."""
        q = query.strip().lower()
        results, seen = [], set()
        if not q:
            return results
        for c in self.store.chunks.values():
            d = self.store.docs.get(c["doc_id"])
            if not d:
                continue
            in_text = q in c["text"].lower()
            if in_text or q in d["name"].lower() or q in (c["section"] or "").lower():
                i = c["text"].lower().find(q)
                snippet = c["text"][max(0, i - 80): i + len(q) + 120] if i >= 0 else c["text"][:200]
                results.append({"document_id": d["id"], "document": d["name"], "page": c["page"], "section": c["section"],
                                "snippet": ("…" if i > 80 else "") + snippet.strip() + "…", "score": 1.0, "match": "exact"})
                seen.add(c["id"])
        scope = set(self.store.docs)
        for cid, s in self.index.search(self.embedder.encode([query])[0], limit, scope):
            c = self.store.chunks.get(cid)
            if c and cid not in seen and s >= self.min_score:
                d = self.store.docs[c["doc_id"]]
                results.append({"document_id": d["id"], "document": d["name"], "page": c["page"], "section": c["section"],
                                "snippet": c["text"][:240] + "…", "score": round(s, 4), "match": "semantic"})
        return results[:limit]
