"""JSON-file persistence behind a small interface (swap for object storage / a DB later)."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    MAX_INVESTIGATIONS = 200

    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "uploads").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.docs: dict[str, dict] = self._read("documents.json", {})
        self.chunks: dict[str, dict] = self._read("chunks.json", {})
        self.investigations: list[dict] = self._read("investigations.json", [])

    # -- io
    def _read(self, name, default):
        p = self.dir / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return default
        return default

    def _write(self, name, obj):
        tmp = self.dir / (name + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.dir / name)

    # -- documents
    def add_document(self, doc: dict, chunks: list[dict]):
        with self.lock:
            self.docs[doc["id"]] = doc
            for c in chunks:
                self.chunks[c["id"]] = c
            self._write("documents.json", self.docs)
            self._write("chunks.json", self.chunks)

    def delete_document(self, doc_id: str) -> bool:
        with self.lock:
            if doc_id not in self.docs:
                return False
            del self.docs[doc_id]
            self.chunks = {k: v for k, v in self.chunks.items() if v["doc_id"] != doc_id}
            self._write("documents.json", self.docs)
            self._write("chunks.json", self.chunks)
            return True

    def find_by_hash(self, sha: str):
        return next((d for d in self.docs.values() if d.get("sha256") == sha), None)

    def doc_chunks(self, doc_id: str) -> list[dict]:
        return sorted((c for c in self.chunks.values() if c["doc_id"] == doc_id), key=lambda c: c["ord"])

    # -- investigations
    def add_investigation(self, inv: dict):
        with self.lock:
            self.investigations.insert(0, inv)
            keep = [i for i in self.investigations if i.get("saved")]
            rest = [i for i in self.investigations if not i.get("saved")][: self.MAX_INVESTIGATIONS]
            self.investigations = sorted(keep + rest, key=lambda i: i["created_at"], reverse=True)
            self._write("investigations.json", self.investigations)

    def get_investigation(self, inv_id: str):
        return next((i for i in self.investigations if i["id"] == inv_id), None)

    def update_investigation(self, inv_id: str, **fields):
        with self.lock:
            inv = self.get_investigation(inv_id)
            if inv is None:
                return None
            inv.update(fields)
            self._write("investigations.json", self.investigations)
            return inv

    def delete_investigation(self, inv_id: str) -> bool:
        with self.lock:
            n = len(self.investigations)
            self.investigations = [i for i in self.investigations if i["id"] != inv_id]
            self._write("investigations.json", self.investigations)
            return len(self.investigations) < n
