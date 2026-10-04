"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import Constellation from "@/components/Constellation";
import { tiltProps } from "@/components/tilt";
import { api, Answer, Conflict, Doc, Health } from "@/lib/api";

type Tab = "investigate" | "documents" | "conflicts";
const SUGGESTED = [
  "What is the parental leave policy?",
  "Which documents disagree about parental leave?",
  "Compare remote work with office attendance.",
  "What benefits are available after one year?",
  "What is NovaTech's policy for employees working on Mars?",
];
const LOADING = ["Searching indexed chunks…", "Finding relevant evidence…", "Comparing sources…", "Generating grounded response…"];

export default function Page() {
  const [tab, setTab] = useState<Tab>("investigate");
  const [docs, setDocs] = useState<Doc[] | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [recent, setRecent] = useState<{ id: string; question: string; created_at: string }[]>([]);
  const [restored, setRestored] = useState<Answer | null>(null);

  const say = (m: string) => { setToast(m); setTimeout(() => setToast(null), 3500); };
  const refresh = useCallback(async () => {
    try {
      const [d, h, r] = await Promise.all([api.documents(), api.health(), api.recent()]);
      setDocs(d); setHealth(h); setRecent(r.slice(0, 8));
    } catch (e) { setDocs([]); say((e as Error).message); }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  return (
    <div className="shell">
      <nav className="side" aria-label="Main">
        <h1>◈ DocuSense</h1>
        {(["investigate", "documents", "conflicts"] as Tab[]).map((t) => (
          <button key={t} className={`nav ${tab === t ? "active" : ""}`} aria-current={tab === t} onClick={() => setTab(t)}>
            {t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
        <div className="label" style={{ marginTop: 16 }}>Recent</div>
        {recent.length === 0 && <small>No investigations yet.</small>}
        {recent.map((r) => (
          <button key={r.id} className="nav" title={r.question} onClick={async () => { try { setRestored(await api.investigation(r.id)); setTab("investigate"); } catch (e) { say((e as Error).message); } }}>
            <small>{r.question.length > 24 ? r.question.slice(0, 24) + "…" : r.question}</small>
          </button>
        ))}
        {health && (
          <small style={{ marginTop: "auto" }}>
            Backend: {health.status}<br />Embeddings: {health.embedding.semantic ? "semantic" : "lexical fallback"}<br />LLM: {health.llm.configured ? "connected" : "not configured (extractive mode)"}
          </small>
        )}
      </nav>
      <main key={tab} className="main page">
        {toast && <div className="card" role="status">{toast}</div>}
        {docs === null ? <p className="muted">Loading…</p> :
          docs.length === 0 && tab !== "documents" ? <Welcome onDone={refresh} say={say} goDocs={() => setTab("documents")} /> :
          tab === "investigate" ? <Investigate docs={docs} restored={restored} onAsked={refresh} /> :
          tab === "documents" ? <Documents docs={docs} onChange={refresh} say={say} /> :
          <Conflicts />}
      </main>
    </div>
  );
}

function Welcome({ onDone, say, goDocs }: { onDone: () => void; say: (m: string) => void; goDocs: () => void }) {
  const [busy, setBusy] = useState(false);
  return (
    <div className="card hero">
      <Constellation mode="idle" height={300} className="hero-canvas" />
      <h2 className="glow">Welcome to DocuSense</h2>
      <p className="muted">Turn a folder of documents into an investigable knowledge base. Investigate documents. Follow the evidence.</p>
      <div className="row">
        <button className="primary" onClick={goDocs}>Upload documents</button>
        <button disabled={busy} onClick={async () => { setBusy(true); try { await api.loadDemo(); say("Demo workspace loaded."); onDone(); } catch (e) { say((e as Error).message); } setBusy(false); }}>
          {busy ? "Indexing demo documents…" : "Load NovaTech demo workspace"}
        </button>
      </div>
    </div>
  );
}

function Investigate({ docs, restored, onAsked }: { docs: Doc[]; restored: Answer | null; onAsked: () => void }) {
  const [q, setQ] = useState("");
  const [res, setRes] = useState<Answer | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState(0);
  const [sel, setSel] = useState<number | null>(null);
  const evRefs = useRef<Record<number, HTMLDivElement | null>>({});
  useEffect(() => { if (restored) { setRes(restored); setQ(restored.question); setErr(null); } }, [restored]);
  useEffect(() => { if (!busy) return; setStep(0); const t = setInterval(() => setStep((s) => Math.min(s + 1, LOADING.length - 1)), 900); return () => clearInterval(t); }, [busy]);

  const ask = async (question: string) => {
    if (question.trim().length < 3) return;
    setBusy(true); setErr(null); setSel(null);
    try { setRes(await api.ask(question)); onAsked(); } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  };
  const jump = (id: number) => { setSel(id); evRefs.current[id]?.scrollIntoView({ behavior: "smooth", block: "nearest" }); };
  const renderAnswer = (text: string) => text.split(/(\[\d+\])/g).map((p, i) => {
    const m = p.match(/^\[(\d+)\]$/);
    return m ? <button key={i} className="cite" aria-label={`Show evidence ${m[1]}`} onClick={() => jump(Number(m[1]))}>[{m[1]}]</button> : <span key={i}>{p}</span>;
  });

  return (
    <>
      <h2>Investigate</h2>
      <div className="card">
        <label htmlFor="q" className="label">What would you like to investigate?</label>
        <textarea id="q" rows={3} value={q} placeholder={`Ask anything across ${docs.length} indexed document${docs.length === 1 ? "" : "s"}…`}
          onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") ask(q); }} />
        <div className="row" style={{ marginTop: 8 }}>
          <button className="primary" disabled={busy} onClick={() => ask(q)}>Investigate →</button>
          <span className="muted">Ctrl/⌘ + Enter</span>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          {SUGGESTED.map((s) => <button key={s} className="chip" disabled={busy} onClick={() => { setQ(s); ask(s); }}>{s}</button>)}
        </div>
      </div>
      {busy && <div className="card think rise" role="status"><Constellation mode="think" height={170} /><div><strong>Analyzing your documents…</strong><ul>{LOADING.map((l, i) => <li key={l} className={i < step ? "done" : i === step ? "active" : ""}><span className="dot" />{l}</li>)}</ul></div></div>}
      {err && !busy && <div className="card err" role="alert"><strong>We couldn’t complete this investigation.</strong><p>{err}</p><button onClick={() => ask(q)}>Try again</button></div>}
      {res && !busy && !err && (
        <div className="grid2">
          <div>
            <div className="card rise">
              <div className="label">{res.insufficient_evidence ? "Insufficient evidence" : "Answer"}</div>
              <p className="answer">{renderAnswer(res.answer)}</p>
              <div className="row">
                <span className="label">Evidence confidence</span>
                <span className={`badge ${res.confidence}`}>{res.confidence === "high" ? "◉" : res.confidence === "medium" ? "◐" : "○"} {res.confidence.toUpperCase()}</span>
              </div>
              <p className="muted">{res.confidence_note}</p>
              <p className="muted">
                {res.sources_analyzed} sources analyzed · {res.relevant_sources} cited · {res.conflicts.length} conflict{res.conflicts.length === 1 ? "" : "s"} · {res.documents_searched} documents / {res.chunks_searched} chunks searched · {res.meta.mode === "llm" ? "LLM answer" : "extractive mode (no LLM configured)"} · {res.meta.total_ms} ms
              </p>
              {res.insufficient_evidence && <p className="muted">Try uploading more documents or asking a more specific question.{res.notes.length > 0 && ` (${res.notes.join("; ")})`}</p>}
            </div>
            {res.conflicts.map((c, i) => (
              <div key={i} className="card"><div className="label">⚠ Conflict detected — {c.topic}</div>
                {c.claims.map((cl, j) => <p key={j}><strong>{cl.document}</strong> (page {cl.page}): {cl.claim}{cl.evidence_id ? <button className="cite" onClick={() => jump(cl.evidence_id!)}>[{cl.evidence_id}]</button> : null}</p>)}
                <p className="muted">{c.assessment}</p></div>
            ))}
          </div>
          <div>
            <div className="label" style={{ marginBottom: 8 }}>Evidence · {res.evidence.length} source{res.evidence.length === 1 ? "" : "s"}</div>
            {res.evidence.length === 0 && <p className="muted">No relevant evidence found.</p>}
            {res.evidence.map((e, idx) => (
              <div key={e.id} ref={(el) => { evRefs.current[e.id] = el; }} className={`card ev rise ${sel === e.id ? "sel" : ""}`} style={{ "--i": idx } as React.CSSProperties} {...tiltProps} onClick={() => setSel(e.id)}>
                <strong>[{e.id}] {e.document}</strong>
                <div className="label">Page {e.page}{e.section ? ` • ${e.section}` : ""}</div>
                <q>{res.citations.find((c) => c.evidence_id === e.id)?.quote ?? e.text.slice(0, 220)}</q>
                <span className="muted">Relevance {Math.round(e.score * 100)}%{e.cited ? " · cited" : ""}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

function Documents({ docs, onChange, say }: { docs: Doc[]; onChange: () => void; say: (m: string) => void }) {
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<{ document: string; page: number; section: string; snippet: string }[] | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const upload = async (files: File[]) => {
    if (!files.length) return;
    setBusy(true);
    try {
      const r = await api.upload(files);
      r.results.forEach((x) => say(x.ok ? `✓ ${x.filename} indexed` : `✕ ${x.filename}: ${x.error}`));
      onChange();
    } catch (e) { say((e as Error).message); }
    setBusy(false);
  };
  const doSearch = async () => { if (!query.trim()) { setHits(null); return; } try { setHits(await api.search(query)); } catch (e) { say((e as Error).message); } };

  return (
    <>
      <h2>Documents</h2>
      <div className={`drop ${over ? "over" : ""}`} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); upload(Array.from(e.dataTransfer.files)); }}>
        {busy ? "Uploading, extracting, chunking, embedding and indexing…" : <>Drop PDF, DOCX or TXT files here, or <button onClick={() => input.current?.click()}>choose files</button></>}
        <input ref={input} type="file" multiple hidden accept=".pdf,.docx,.txt,.md" onChange={(e) => { upload(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
      </div>
      <div className="row" style={{ margin: "12px 0" }}>
        <input type="text" aria-label="Search documents" placeholder="Search document names and content…" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && doSearch()} style={{ maxWidth: 360 }} />
        <button onClick={doSearch}>Search</button>
        {docs.length === 0 && <button onClick={async () => { setBusy(true); try { await api.loadDemo(); onChange(); } catch (e) { say((e as Error).message); } setBusy(false); }}>Load demo workspace</button>}
      </div>
      {hits && (hits.length === 0 ? <p className="muted">No matches.</p> : hits.map((h, i) => <div key={i} className="card"><strong>{h.document}</strong> <span className="label">Page {h.page}{h.section ? ` • ${h.section}` : ""}</span><p className="muted">{h.snippet}</p></div>))}
      {docs.length === 0 ? <div className="card"><strong>No documents yet.</strong><p className="muted">Upload your first document to start investigating.</p></div> : (
        <table>
          <thead><tr><th>Document</th><th>Type</th><th>Pages</th><th>Chunks</th><th>Status</th><th>Added</th><th></th></tr></thead>
          <tbody>{docs.map((d) => (
            <tr key={d.id}><td>{d.name}</td><td>{d.type.toUpperCase()}</td><td>{d.pages}</td><td>{d.chunks}</td><td>{d.status}</td><td>{new Date(d.added_at).toLocaleString()}</td>
              <td><button aria-label={`Delete ${d.name}`} onClick={async () => { if (confirm(`Delete “${d.name}”?`)) { try { await api.deleteDoc(d.id); say("Document deleted."); onChange(); } catch (e) { say((e as Error).message); } } }}>Delete</button></td></tr>
          ))}</tbody>
        </table>
      )}
    </>
  );
}

function Conflicts() {
  const [items, setItems] = useState<Conflict[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.conflicts().then(setItems).catch((e) => setErr(e.message)); }, []);
  return (
    <>
      <h2>Conflicts</h2>
      <p className="muted">Where your sources disagree</p>
      {err && <div className="card err">{err}</div>}
      {items && items.length === 0 && <div className="card"><strong>No conflicts detected.</strong><p className="muted">We’ll surface contradictory claims when they appear.</p></div>}
      {items?.map((c, i) => (
        <div key={i} className="card"><div className="label">⚠ Conflict detected</div><h3 style={{ margin: "4px 0" }}>{c.topic}</h3>
          {c.claims.map((cl, j) => <div key={j} className="row" style={{ justifyContent: "space-between" }}><span><strong>{cl.document}</strong> <span className="muted">Page {cl.page}</span></span><strong>{cl.claim}</strong></div>)}
          {c.claims.map((cl, j) => cl.quote ? <q key={j} className="muted" style={{ display: "block" }}>{cl.document}: {cl.quote}</q> : null)}
          <p className="muted">Authority: Undetermined. {c.assessment}</p></div>
      ))}
    </>
  );
}
