"""Pydantic models for the public API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    evidence_id: int
    document: str
    document_id: str
    page: int
    section: str = ""
    quote: str
    score: float
    quote_verified: bool = True


class ConflictClaim(BaseModel):
    document: str
    document_id: str
    page: int
    claim: str
    quote: str = ""
    evidence_id: Optional[int] = None
    section: str = ""


class Conflict(BaseModel):
    topic: str
    claims: list[ConflictClaim]
    assessment: str


class EvidenceItem(BaseModel):
    id: int
    chunk_id: str
    document_id: str
    document: str
    page: int
    section: str = ""
    text: str
    score: float
    cited: bool = False


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    min_score: Optional[float] = Field(default=None, ge=0, le=1)
    document_ids: Optional[list[str]] = None
    collection: Optional[str] = None
    persist: bool = True


class QuestionResponse(BaseModel):
    id: Optional[str] = None
    question: str
    answer: str
    confidence: Literal["high", "medium", "low"]
    confidence_note: str
    insufficient_evidence: bool
    conflict_detected: bool
    citations: list[Citation]
    conflicts: list[Conflict]
    evidence: list[EvidenceItem]
    sources_analyzed: int
    relevant_sources: int
    documents_searched: int
    chunks_searched: int
    notes: list[str] = []
    meta: dict


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=10, ge=1, le=50)


class SaveRequest(BaseModel):
    saved: bool = True
