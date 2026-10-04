import unittest

from app.chunking import chunk_pages, heading_of
from app.conflicts import detect_conflicts, quantity_claims
from app.engine import IngestError, DuplicateDocument, safe_filename
from app.extraction import ExtractionError, Page, extract
from app.grounding import quote_in_text, unsupported_entities
from app.llm import LLMError
from app.pipeline import InvestigationError, investigate
from tests.helpers import FakeLLM, load_demo, make_docx, make_engine, make_pdf


class ExtractionTests(unittest.TestCase):
    def test_txt_native_pages(self):
        r = extract("a.txt", b"page one\fpage two")
        self.assertEqual([p.number for p in r.pages], [1, 2])
        self.assertEqual(r.page_basis, "native")

    def test_pdf(self):
        r = extract("a.pdf", make_pdf(["Alpha policy text", "Beta policy text"]))
        self.assertEqual(len(r.pages), 2)
        self.assertIn("Beta", r.pages[1].text)

    def test_docx_headings(self):
        r = extract("a.docx", make_docx([("h", "Parental Leave"), ("p", "Employees get 12 weeks.")]))
        self.assertIn("# Parental Leave", r.pages[0].text)

    def test_errors(self):
        for name, data in [("a.txt", b""), ("a.exe", b"x"), ("a.pdf", b"not a pdf"), ("a.docx", b"not docx"), ("a.txt", b"  \n ")]:
            with self.assertRaises(ExtractionError, msg=name):
                extract(name, data)

    def test_filename_sanitised(self):
        self.assertEqual(safe_filename("../../etc/pass wd.txt"), "pass wd.txt")
        self.assertNotIn("/", safe_filename("a/b\\c.txt"))


class ChunkingTests(unittest.TestCase):
    def test_pages_sections_and_ids(self):
        pages = [Page(1, "INTRO\n\nHello world."), Page(2, "Parental Leave\n\nTwelve weeks of leave.")]
        ch = chunk_pages(pages, "d1")
        self.assertEqual([(c["page"], c["section"]) for c in ch], [(1, "INTRO"), (2, "Parental Leave")])
        self.assertEqual(ch[0]["id"], "d1:0")

    def test_long_block_split_and_heading_rules(self):
        ch = chunk_pages([Page(1, ("Sentence number one is here. " * 120).strip())], "d")
        self.assertGreater(len(ch), 1)
        self.assertTrue(all(len(c["text"]) <= 1400 for c in ch))
        self.assertIsNone(heading_of("This is a normal sentence."))
        self.assertEqual(heading_of("# Vacation"), "Vacation")


class ConflictTests(unittest.TestCase):
    def test_quantities(self):
        self.assertEqual(quantity_claims("Employees receive twelve weeks of leave.")[0]["value"], 12.0)

    def test_demo_corpus_has_exactly_the_parental_leave_conflict(self):
        e = load_demo(make_engine())
        items = [{"document_id": d["id"], "document": d["name"], "page": c["page"], "section": c["section"], "text": c["text"]}
                 for c in e.store.chunks.values() for d in [e.store.docs[c["doc_id"]]]]
        found = detect_conflicts(items)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["topic"], "Parental Leave")
        self.assertEqual({c["claim"] for c in found[0]["claims"]}, {"12 weeks", "16 weeks"})

    def test_agreement_and_same_document_are_not_conflicts(self):
        mk = lambda d, t: {"document_id": d, "document": d, "page": 1, "section": "S", "text": t}
        same = "Eligible employees receive {} weeks of paid parental leave."
        self.assertEqual(detect_conflicts([mk("A", same.format(12)), mk("B", same.format(12))]), [])
        self.assertEqual(detect_conflicts([mk("A", same.format(12) + " " + same.format(16))]), [])
        self.assertEqual(detect_conflicts([mk("A", "Vacation is 15 days per year."), mk("B", "Sick leave is 10 days per year.")]), [])


class RetrievalAndIndexTests(unittest.TestCase):
    def setUp(self):
        self.e = load_demo(make_engine())

    def test_chunks_embeddings_index(self):
        self.assertEqual(len(self.e.index), len(self.e.store.chunks))
        self.assertEqual(self.e.index.vecs.shape[1], self.e.embedder.dim)

    def test_semantic_retrieval_finds_handbook_and_hr(self):
        ev, _ = self.e.retrieve("What is the parental leave policy?", 6, self.e.min_score, set(self.e.store.docs))
        names = {x["document"] for x in ev}
        self.assertTrue({"NovaTech Employee Handbook", "NovaTech HR Policy"} <= names)
        self.assertEqual(len({x["chunk_id"] for x in ev}), len(ev))  # no duplicates

    def test_index_persists_across_restart(self):
        from app.engine import Engine
        e2 = Engine(self.e.settings)
        self.assertEqual(e2.index.ids, self.e.index.ids)
        self.assertEqual(len(e2.store.docs), 5)

    def test_delete_removes_vectors(self):
        doc = next(d for d in self.e.store.docs.values() if "Handbook" in d["name"])
        self.assertTrue(self.e.delete_document(doc["id"]))
        self.assertTrue(all(not i.startswith(doc["id"]) for i in self.e.index.ids))
        self.assertEqual(set(self.e.index.ids), set(self.e.store.chunks))

    def test_duplicate_and_bad_uploads(self):
        f = next(iter(__import__("tests.helpers", fromlist=["DEMO"]).DEMO.glob("*.txt")))
        with self.assertRaises(DuplicateDocument):
            self.e.ingest(f.name, f.read_bytes())
        with self.assertRaises(IngestError):
            self.e.ingest("x.pdf", b"garbage")
        self.assertEqual(len(self.e.store.docs), 5)  # failed uploads leave no trace
        self.assertEqual(len(list((self.e.store.dir / "uploads").iterdir())), 5)

    def test_search(self):
        r = self.e.search("parental leave")
        self.assertTrue(r and r[0]["match"] == "exact")

    def test_scope_filter(self):
        doc = next(d for d in self.e.store.docs.values() if "Remote" in d["name"])
        ev, _ = self.e.retrieve("parental leave", 6, 0.0, {doc["id"]})
        self.assertTrue(all(x["document_id"] == doc["id"] for x in ev))


class InvestigationTests(unittest.TestCase):
    def setUp(self):
        self.e = load_demo(make_engine())

    def test1_parental_leave_conflict(self):
        r = investigate(self.e, "What is the parental leave policy?")
        self.assertTrue(r["conflict_detected"])
        self.assertEqual({c["claim"] for c in r["conflicts"][0]["claims"]}, {"12 weeks", "16 weeks"})
        self.assertEqual(r["confidence"], "medium")
        self.assertEqual(r["meta"]["mode"], "extractive")

    def test4_mars_insufficient(self):
        r = investigate(self.e, "What is the company's policy for employees working on Mars?")
        self.assertTrue(r["insufficient_evidence"])
        self.assertEqual((r["confidence"], r["evidence"], r["citations"], r["conflicts"]), ("low", [], [], []))
        self.assertIn("Mars", r["notes"][0])

    def test5_which_documents_disagree(self):
        r = investigate(self.e, "Which documents disagree about parental leave?")
        self.assertTrue(r["conflict_detected"])
        self.assertIn("12 weeks", r["answer"])
        self.assertIn("16 weeks", r["answer"])

    def test_empty_scope_is_insufficient(self):
        r = investigate(self.e, "What is the parental leave policy?", doc_ids=["nope"])
        self.assertTrue(r["insufficient_evidence"])

    def test_citations_are_verified_and_snapped(self):
        def resp(ev):
            e = ev[0]
            return {"answer": f"Something [{e['id']}] and [99].", "confidence": "high", "insufficient_evidence": False,
                    "citations": [{"evidence_id": e["id"], "quote": "this quote is completely invented by the model"},
                                  {"evidence_id": 99, "quote": "ghost"}], "conflicts": []}
        e = load_demo(make_engine(FakeLLM(resp)))
        r = investigate(e, "What is the vacation policy?")
        self.assertEqual(len(r["citations"]), 1)
        c = r["citations"][0]
        self.assertFalse(c["quote_verified"])
        ev = next(x for x in r["evidence"] if x["id"] == c["evidence_id"])
        self.assertTrue(quote_in_text(c["quote"], ev["text"]))  # snapped quote is real text
        self.assertNotIn("[99]", r["answer"])

    def test_llm_conflict_validation_drops_invented_claims(self):
        def resp(ev):
            by = {x["document"]: x["id"] for x in ev}
            return {"answer": "x", "confidence": "high", "insufficient_evidence": False,
                    "citations": [{"evidence_id": ev[0]["id"], "quote": ""}],
                    "conflicts": [{"topic": "made up", "claims": [{"evidence_id": ev[0]["id"], "claim": "99 weeks"},
                                                                  {"evidence_id": ev[1]["id"], "claim": "98 weeks"}]}]}
        e = load_demo(make_engine(FakeLLM(resp)))
        r = investigate(e, "What is the parental leave policy?")
        self.assertNotIn("made up", [c["topic"] for c in r["conflicts"]])
        self.assertTrue(r["conflict_detected"])  # deterministic detector still reports the real one

    def test_llm_insufficient_and_failure(self):
        e = load_demo(make_engine(FakeLLM({"answer": "The documents do not cover this.", "confidence": "low",
                                           "insufficient_evidence": True, "citations": [], "conflicts": []})))
        r = investigate(e, "What is the vacation policy?")
        self.assertTrue(r["insufficient_evidence"] and r["confidence"] == "low" and r["evidence"] == [])

        class Boom:
            def investigate(self, *a):
                raise LLMError("The AI provider did not respond.")
        e2 = load_demo(make_engine(Boom()))
        with self.assertRaises(InvestigationError):
            investigate(e2, "What is the vacation policy?")

    def test_unsupported_entities(self):
        self.assertEqual(unsupported_entities("What is the policy on Mars?", ["no planets here"]), ["Mars"])
        self.assertEqual(unsupported_entities("What does NovaTech's handbook say?", ["NovaTech handbook"]), [])


@unittest.skipUnless(__import__("importlib").util.find_spec("fastapi") and __import__("importlib").util.find_spec("httpx"),
                     "fastapi/httpx not installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from app.main import create_app
        e = make_engine()
        self.c = TestClient(create_app(e.settings, engine=e))

    def test_flow(self):
        self.assertEqual(self.c.get("/health").json()["status"], "ok")
        r = self.c.post("/demo/load").json()
        self.assertEqual(r["documents"], 5)
        self.assertEqual(len(self.c.get("/documents").json()["documents"]), 5)
        q = self.c.post("/questions", json={"question": "What is the parental leave policy?"}).json()
        self.assertTrue(q["conflict_detected"])
        self.assertEqual(self.c.post("/questions", json={"question": "Employees working on Mars?"}).json()["insufficient_evidence"], True)
        self.assertTrue(self.c.post("/search", json={"query": "vacation"}).json()["results"])
        self.assertEqual(len(self.c.get("/conflicts").json()["conflicts"]), 1)
        self.assertEqual(self.c.post("/documents/upload", files=[("files", ("bad.exe", b"x"))]).json()["results"][0]["ok"], False)
        self.assertEqual(self.c.get("/stats").json()["investigations"], 2)


if __name__ == "__main__":
    unittest.main()
