"""Section-aware chunking. A chunk never spans pages, so page citations are exact."""
from __future__ import annotations

import re

from .extraction import Page

TARGET = 900
HARD_MAX = 1400
_NUM_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def heading_of(block: str) -> str | None:
    """Return the heading text if the block looks like a heading, else None."""
    line = block.strip()
    if "\n" in line:
        return None
    if line.startswith("#"):
        return line.lstrip("#").strip() or None
    if len(line) > 70 or line.endswith((".", ":", ";", ",", "!", "?")):
        return None
    if not any(c.isalpha() for c in line):
        return None
    if _NUM_HEADING.match(line) or line.isupper() or (line[0].isupper() and len(line.split()) <= 8):
        return line
    return None


def _split_long(block: str) -> list[str]:
    if len(block) <= HARD_MAX:
        return [block]
    out, cur = [], ""
    for s in _SENT.split(block):
        if cur and len(cur) + len(s) + 1 > TARGET:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_pages(pages: list[Page], doc_id: str) -> list[dict]:
    chunks: list[dict] = []
    section = ""

    def flush(buf: list[str], page: int, sec: str):
        text = "\n\n".join(buf).strip()
        if text:
            chunks.append({"id": f"{doc_id}:{len(chunks)}", "doc_id": doc_id, "page": page,
                           "section": sec, "text": text, "ord": len(chunks)})

    for page in pages:
        buf: list[str] = []
        buf_sec = section
        size = 0
        for block in [b.strip() for b in re.split(r"\n\s*\n", page.text) if b.strip()]:
            h = heading_of(block)
            if h:
                flush(buf, page.number, buf_sec)
                buf, size, section, buf_sec = [], 0, h, h
                continue
            for piece in _split_long(block):
                if buf and size + len(piece) > TARGET:
                    flush(buf, page.number, buf_sec)
                    buf, size = [], 0
                buf.append(piece)
                size += len(piece)
        flush(buf, page.number, buf_sec)
    return chunks
