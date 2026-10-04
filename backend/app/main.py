"""FastAPI application."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, load_settings
from .conflicts import detect_conflicts
from .engine import DuplicateDocument, Engine, IngestError
from .grounding import normalize
from .pipeline import InvestigationError, investigate
from .schemas import QuestionRequest, QuestionResponse, SaveRequest, SearchRequest
from .store import now_iso

import uuid

log = logging.getLogger("docusense")
DEMO_DIR = Path(__file__).resolve().parent.parent / "demo_data"


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="DocuSense AI", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
    holder: dict = {"engine": engine}

    def eng() -> Engine:  # created lazily so the embedding model loads once, on first use
        if holder["engine"] is None:
            holder["engine"] = Engine(settings)
        return holder["engine"]

    @app.on_event("startup")
    def _warm():
        eng()

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Something went wrong on our side. Please try again."})

    # ---------------------------------------------------------------- health / stats
    @app.get("/health")
    def health():
        e = eng()
        try:
            probe = e.store.dir / ".probe"
            probe.write_text("ok")
            probe.unlink()
            writable = True
        except OSError:
            writable = False
        return {"status": "ok" if writable else "degraded", "documents": len(e.store.docs), "chunks": len(e.store.chunks),
                "vector_index": {"backend": e.index.backend, "vectors": len(e.index), "consistent": set(e.index.ids) == set(e.store.chunks)},
                "embedding": {"name": e.embedder.name, "dim": e.embedder.dim, "ready": True,
                              "semantic": not e.embedder.name.startswith("hashing")},
                "llm": {"configured": e.llm is not None, "model": settings.anthropic_model if e.llm else None},
                "storage": {"writable": writable, "kind": "local-filesystem"}}

    def _all_conflicts() -> list[dict]:
        e = eng()
        items = []
        for c in e.store.chunks.values():
            d = e.store.docs.get(c["doc_id"])
            if d:
                items.append({"document_id": d["id"], "document": d["name"], "page": c["page"], "section": c["section"], "text": c["text"]})
        found = detect_conflicts(items)
        sig = lambda c: {(x["document_id"], normalize(x["claim"])) for x in c["claims"]}
        for inv in e.store.investigations:  # semantic conflicts surfaced by the LLM in past investigations
            for c in inv["response"].get("conflicts", []):
                if all(x["document_id"] in e.store.docs for x in c["claims"]) and not any(len(sig(c) & sig(f)) >= 2 for f in found):
                    found.append(c)
        return found

    @app.get("/conflicts")
    def conflicts():
        return {"conflicts": _all_conflicts()}

    @app.get("/stats")
    def stats():
        e = eng()
        invs = e.store.investigations
        n = len(invs)
        return {"documents": len(e.store.docs), "chunks": len(e.store.chunks), "investigations": n,
                "conflicts_detected": len(_all_conflicts()),
                "insufficient_evidence": sum(1 for i in invs if i["response"]["insufficient_evidence"]),
                "avg_evidence_sources": round(sum(i["response"]["sources_analyzed"] for i in invs) / n, 2) if n else None,
                "confidence_distribution": {k: sum(1 for i in invs if i["response"]["confidence"] == k) for k in ("high", "medium", "low")},
                "enough_data": n >= 3}

    # ---------------------------------------------------------------- documents
    def _public(d: dict) -> dict:
        return {k: v for k, v in d.items() if k != "sha256"}

    def _upload_one(f: UploadFile, collection: str | None) -> dict:
        limit = settings.max_upload_mb * 1024 * 1024
        data = f.file.read(limit + 1)
        name = f.filename or "document"
        if len(data) > limit:
            return {"filename": name, "ok": False, "error": f"File exceeds the {settings.max_upload_mb} MB limit."}
        try:
            doc, stages = eng().ingest(name, data, collection)
            return {"filename": name, "ok": True, "document": _public(doc), "stages": stages}
        except DuplicateDocument as d:
            return {"filename": name, "ok": False, "error": "This document is already indexed.", "document": _public(d.doc)}
        except IngestError as err:
            return {"filename": name, "ok": False, "error": err.message, "failed_stage": err.stage}

    @app.post("/documents/upload")
    def upload(files: list[UploadFile] = File(...), collection: str | None = None):
        if not files:
            raise HTTPException(400, "No files provided.")
        return {"results": [_upload_one(f, collection) for f in files]}

    @app.get("/documents")
    def list_documents():
        docs = sorted(eng().store.docs.values(), key=lambda d: d["added_at"], reverse=True)
        return {"documents": [_public(d) for d in docs]}

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str):
        e = eng()
        d = e.store.docs.get(doc_id)
        if not d:
            raise HTTPException(404, "Document not found.")
        return {**_public(d), "chunk_previews": [{"id": c["id"], "page": c["page"], "section": c["section"], "text": c["text"]}
                                                 for c in e.store.doc_chunks(doc_id)]}

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str):
        if not eng().delete_document(doc_id):
            raise HTTPException(404, "Document not found.")
        return {"deleted": doc_id}

    @app.post("/demo/load")
    def load_demo():
        e = eng()
        existing = {d["filename"] for d in e.store.docs.values()}
        results = []
        for f in sorted(DEMO_DIR.glob("*.txt")):
            if f.name in existing:
                results.append({"filename": f.name, "ok": True, "skipped": True})
                continue
            try:
                doc, stages = e.ingest(f.name, f.read_bytes(), "NovaTech Demo")
                results.append({"filename": f.name, "ok": True, "document": _public(doc), "stages": stages})
            except (IngestError, DuplicateDocument) as err:
                results.append({"filename": f.name, "ok": False, "error": str(err)})
        return {"results": results, "documents": len(e.store.docs)}

    # ---------------------------------------------------------------- questions / search
    @app.post("/questions", response_model=QuestionResponse)
    def ask(req: QuestionRequest):
        e = eng()
        if not e.store.docs:
            raise HTTPException(400, "Upload at least one document before asking a question.")
        try:
            result = investigate(e, req.question, top_k=req.top_k, min_score=req.min_score,
                                 doc_ids=req.document_ids, collection=req.collection)
        except InvestigationError as err:
            raise HTTPException(502, str(err))
        result["id"] = None
        if req.persist:
            result["id"] = uuid.uuid4().hex[:12]
            e.store.add_investigation({"id": result["id"], "question": req.question, "created_at": now_iso(),
                                       "saved": False, "response": result})
        return result

    @app.post("/search")
    def search(req: SearchRequest):
        return {"query": req.query, "results": eng().search(req.query, req.limit)}

    # ---------------------------------------------------------------- investigations
    @app.get("/investigations")
    def investigations(saved: bool | None = None):
        items = eng().store.investigations
        if saved is not None:
            items = [i for i in items if bool(i.get("saved")) == saved]
        return {"investigations": [{"id": i["id"], "question": i["question"], "created_at": i["created_at"], "saved": i.get("saved", False),
                                    "confidence": i["response"]["confidence"], "conflict_detected": i["response"]["conflict_detected"],
                                    "insufficient_evidence": i["response"]["insufficient_evidence"]} for i in items]}

    @app.get("/investigations/{inv_id}")
    def get_investigation(inv_id: str):
        inv = eng().store.get_investigation(inv_id)
        if not inv:
            raise HTTPException(404, "Investigation not found.")
        return inv

    @app.patch("/investigations/{inv_id}")
    def save_investigation(inv_id: str, req: SaveRequest):
        inv = eng().store.update_investigation(inv_id, saved=req.saved)
        if not inv:
            raise HTTPException(404, "Investigation not found.")
        return {"id": inv_id, "saved": inv["saved"]}

    @app.delete("/investigations/{inv_id}")
    def delete_investigation(inv_id: str):
        if not eng().store.delete_investigation(inv_id):
            raise HTTPException(404, "Investigation not found.")
        return {"deleted": inv_id}

    return app


app = create_app()
