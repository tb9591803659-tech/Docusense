"""Persistent vector index. FAISS (inner product on normalized vectors) when installed,
otherwise an exact numpy search. Vectors are always persisted to vectors.npy."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def doc_of(chunk_id: str) -> str:
    return chunk_id.rsplit(":", 1)[0]


class VectorIndex:
    def __init__(self, path: Path, dim: int, embedder_name: str):
        self.path, self.dim, self.embedder_name = Path(path), dim, embedder_name
        self.path.mkdir(parents=True, exist_ok=True)
        self.ids: list[str] = []
        self.vecs = np.zeros((0, dim), dtype="float32")
        self.stale = False
        self._faiss = None
        self._load()

    @property
    def backend(self) -> str:
        return "faiss" if faiss is not None else "numpy"

    def __len__(self):
        return len(self.ids)

    def _load(self):
        meta_p, vec_p = self.path / "meta.json", self.path / "vectors.npy"
        if not (meta_p.exists() and vec_p.exists()):
            self._rebuild()
            return
        meta = json.loads(meta_p.read_text())
        if meta.get("embedder") != self.embedder_name or meta.get("dim") != self.dim:
            self.stale = True  # embeddings from a different model; caller must re-embed
            self._rebuild()
            return
        self.ids = meta["ids"]
        self.vecs = np.load(vec_p).astype("float32")
        self._rebuild()

    def _rebuild(self):
        if faiss is not None:
            idx = faiss.IndexFlatIP(self.dim)
            if len(self.ids):
                idx.add(self.vecs)
            self._faiss = idx

    def save(self):
        tmp = self.path / "vectors.tmp.npy"
        np.save(tmp, self.vecs)
        os.replace(tmp, self.path / "vectors.npy")
        meta = {"embedder": self.embedder_name, "dim": self.dim, "ids": self.ids}
        (self.path / "meta.tmp").write_text(json.dumps(meta))
        os.replace(self.path / "meta.tmp", self.path / "meta.json")
        if faiss is not None and self._faiss is not None:
            faiss.write_index(self._faiss, str(self.path / "index.faiss"))

    def reset(self):
        self.ids, self.vecs, self.stale = [], np.zeros((0, self.dim), dtype="float32"), False
        self._rebuild()

    def add(self, ids: list[str], vecs: np.ndarray):
        if len(ids) != len(vecs):
            raise ValueError("ids/vectors length mismatch")
        self.ids += ids
        self.vecs = np.vstack([self.vecs, vecs.astype("float32")])
        if self._faiss is not None:
            self._faiss.add(vecs.astype("float32"))
        self.save()

    def remove_doc(self, doc_id: str):
        keep = [i for i, c in enumerate(self.ids) if doc_of(c) != doc_id]
        self.ids = [self.ids[i] for i in keep]
        self.vecs = self.vecs[keep] if keep else np.zeros((0, self.dim), dtype="float32")
        self._rebuild()
        self.save()

    def search(self, q: np.ndarray, k: int, allowed_docs: set[str] | None = None) -> list[tuple[str, float]]:
        n = len(self.ids)
        if n == 0:
            return []
        q = q.astype("float32").reshape(1, -1)
        kk = n if allowed_docs else min(n, k)
        if self._faiss is not None:
            scores, idxs = self._faiss.search(q, kk)
            pairs = [(self.ids[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i >= 0]
        else:
            s = self.vecs @ q[0]
            order = np.argsort(-s)[:kk]
            pairs = [(self.ids[i], float(s[i])) for i in order]
        if allowed_docs:
            pairs = [p for p in pairs if doc_of(p[0]) in allowed_docs]
        return pairs[:k]
