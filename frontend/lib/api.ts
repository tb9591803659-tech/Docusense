const DEFAULT_API = "https://docusense-production-3ca8.up.railway.app";
// Env var wins (set NEXT_PUBLIC_API_URL=http://localhost:8000 for local backend); trailing slashes are stripped.
export const API = (process.env.NEXT_PUBLIC_API_URL || DEFAULT_API).replace(/\/+$/, "");

export type Doc = { id: string; name: string; filename: string; type: string; pages: number; chunks: number; status: string; added_at: string; collection?: string | null };
export type Evidence = { id: number; document: string; document_id: string; page: number; section: string; text: string; score: number; cited: boolean };
export type Citation = { evidence_id: number; document: string; page: number; section: string; quote: string; score: number };
export type Claim = { document: string; page: number; claim: string; quote?: string; evidence_id?: number | null };
export type Conflict = { topic: string; claims: Claim[]; assessment: string };
export type Answer = {
  id: string | null; question: string; answer: string; confidence: "high" | "medium" | "low"; confidence_note: string;
  insufficient_evidence: boolean; conflict_detected: boolean; citations: Citation[]; conflicts: Conflict[]; evidence: Evidence[];
  sources_analyzed: number; relevant_sources: number; documents_searched: number; chunks_searched: number; notes: string[];
  meta: { mode: string; embedder: string; generated_at: string; total_ms: number };
};
export type Health = { status: string; documents: number; chunks: number; embedding: { name: string; semantic: boolean }; llm: { configured: boolean; model: string | null }; vector_index: { backend: string; vectors: number; consistent: boolean } };

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try { res = await fetch(`${API}${path}`, init); } catch { throw new Error("Cannot reach the DocuSense backend."); }
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try { const j = await res.json(); if (typeof j.detail === "string") msg = j.detail; else if (Array.isArray(j.detail)) msg = "Please check your input."; } catch {}
    throw new Error(msg);
  }
  return res.json();
}

const json = (body: unknown): RequestInit => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  health: () => call<Health>("/health"),
  documents: () => call<{ documents: Doc[] }>("/documents").then((r) => r.documents),
  deleteDoc: (id: string) => call(`/documents/${id}`, { method: "DELETE" }),
  upload: (files: File[]) => { const f = new FormData(); files.forEach((x) => f.append("files", x)); return call<{ results: { filename: string; ok: boolean; error?: string; stages?: { stage: string; ms: number }[] }[] }>("/documents/upload", { method: "POST", body: f }); },
  loadDemo: () => call<{ documents: number }>("/demo/load", { method: "POST" }),
  ask: (question: string) => call<Answer>("/questions", json({ question })),
  conflicts: () => call<{ conflicts: Conflict[] }>("/conflicts").then((r) => r.conflicts),
  stats: () => call<{ documents: number; chunks: number; investigations: number; conflicts_detected: number }>("/stats"),
  recent: () => call<{ investigations: { id: string; question: string; created_at: string }[] }>("/investigations").then((r) => r.investigations),
  investigation: (id: string) => call<{ response: Answer }>(`/investigations/${id}`).then((r) => r.response),
  search: (query: string) => call<{ results: { document: string; page: number; section: string; snippet: string }[] }>("/search", json({ query })).then((r) => r.results),
};
