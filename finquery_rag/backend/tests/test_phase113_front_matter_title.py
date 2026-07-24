import sys
from unittest.mock import MagicMock

mock_embed_fn = MagicMock()
mock_st_ef = MagicMock()
mock_st_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embed_fn
for _mod in [
    "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
    "camelot", "pymupdf", "langchain_core", "langchain_core.documents", "langchain_text_splitters", "jieba_fast",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["chromadb.utils.embedding_functions"] = mock_st_ef
sys.modules["langchain_core.documents"].Document = MagicMock()
sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = MagicMock()
sys.modules["langchain_text_splitters"].MarkdownHeaderTextSplitter = MagicMock()
sys.modules["jieba_fast"].cut_for_search = lambda text: [text]

from src.services.ingest import _extract_title_from_first_page
from src.services.rag_engine import RAGEngine


class _FakePage:
    def __init__(self, lines, height=1000):
        self._lines = lines
        self.rect = type("Rect", (), {"height": height})()

    def get_text(self, mode, *args, **kwargs):
        assert mode == "dict"
        blocks = []
        for text, size, y0 in self._lines:
            blocks.append({
                "type": 0,
                "lines": [{
                    "spans": [{
                        "text": text,
                        "size": size,
                        "bbox": [0, y0, 100, y0 + 10],
                    }]
                }]
            })
        return {"blocks": blocks}


class _DummyLLM:
    pass


def test_extract_title_merges_multiline_front_matter_title():
    page = _FakePage([
        ("001", 8, 20),
        ("Rethinking Crack Segmentation: A", 22, 80),
        ("Semantic-Visual Co-refinement Paradigm with", 22, 120),
        ("the MSCrack30K Benchmark", 22, 160),
        ("Anonymous ECCV 2026 Submission", 14, 260),
        ("Abstract. Crack segmentation remains challenging", 11, 380),
    ])

    title = _extract_title_from_first_page(page)

    assert title == "Rethinking Crack Segmentation: A Semantic-Visual Co-refinement Paradigm with the MSCrack30K Benchmark"


def test_extract_title_falls_back_to_adjacent_cover_lines_when_primary_font_is_partial():
    page = _FakePage([
        ("2025", 44, 70),
        ("Driving Smart Solutions", 25, 130),
        ("ANNUALREPORT", 28, 180),
        ("www.example.com", 9, 240),
    ])

    title = _extract_title_from_first_page(page)

    assert title == "2025 Driving Smart Solutions Annual Report"


def test_front_matter_title_answer_strips_title_prefix(tmp_path):
    engine = RAGEngine(_DummyLLM(), bm25_db_path=str(tmp_path / "b.db"))
    chunks = [{
        "doc_id": "paper::title",
        "content": "Title: Rethinking Crack Segmentation",
        "metadata": {"type": "front_matter", "subtype": "title", "page": 1, "doc_name": "paper.pdf"},
        "score": 0.02,
    }]

    answer = engine.answer_front_matter_query("What is the title of this paper?", chunks)

    assert answer["answer"] == 'The title of the paper is "Rethinking Crack Segmentation".'
    assert answer["diagnostic"] == "front_matter_title"


def test_front_matter_title_falls_back_to_page_one_cover_evidence(tmp_path):
    engine = RAGEngine(_DummyLLM(), bm25_db_path=str(tmp_path / "b.db"))
    chunks = [{
        "doc_id": "report::page_1::chunk_0",
        "content": "Section: 2025\n# 2025\nDriving Smart Solutions\nANNUALREPORT\nwww.example.com",
        "metadata": {"type": "text", "page": 1, "doc_name": "report.pdf"},
        "score": 0.1,
    }]

    answer = engine.answer_front_matter_query("What is the title of this report?", chunks)

    assert answer["answer"] == 'The title of the paper is "2025 Driving Smart Solutions Annual Report".'
    assert answer["diagnostic"] == "front_matter_title"


def test_front_matter_title_can_include_reporting_period_from_cover_evidence(tmp_path):
    engine = RAGEngine(_DummyLLM(), bm25_db_path=str(tmp_path / "b.db"))
    chunks = [
        {
            "doc_id": "report::front_matter_title",
            "content": "Title: Annual financial report and financial statements",
            "metadata": {"type": "front_matter", "subtype": "title", "page": 1, "doc_name": "report.pdf"},
            "score": 1.0,
        },
        {
            "doc_id": "report::page_1::chunk_0",
            "content": "Annual financial report and financial statements\nYear to December 31, 2020",
            "metadata": {"type": "text", "page": 1, "doc_name": "report.pdf"},
            "score": 0.1,
        },
    ]

    answer = engine.answer_front_matter_query("What is the title and reporting period?", chunks)

    assert "Annual financial report and financial statements" in answer["answer"]
    assert "Year to December 31, 2020" in answer["answer"]


def test_retrieve_front_matter_chunks_uses_metadata_lookup_before_vector_search(monkeypatch, tmp_path):
    def fake_front_matter(doc_name, user_id, subtype=None):
        assert doc_name == "paper.pdf"
        assert user_id == 7
        assert subtype == "title"
        return [{
            "doc_id": "paper::front_matter_title",
            "content": "Title: Correct Paper Title",
            "metadata": {"type": "front_matter", "subtype": "title", "page": 1, "doc_name": "paper.pdf"},
            "score": 1.0,
        }]

    monkeypatch.setattr("src.services.rag_engine.get_front_matter_chunks", fake_front_matter)
    engine = RAGEngine(_DummyLLM(), bm25_db_path=str(tmp_path / "b.db"))

    chunks = engine.retrieve_front_matter_chunks(["paper.pdf"], "What is the title of this paper?", user_id=7)
    answer = engine.answer_front_matter_query("What is the title of this paper?", chunks)

    assert chunks[0]["metadata"]["subtype"] == "title"
    assert answer["answer"] == 'The title of the paper is "Correct Paper Title".'


def test_query_uses_front_matter_title_without_llm_or_vector_search(monkeypatch, tmp_path):
    class FailingLLM:
        @property
        def chat(self):
            raise AssertionError("LLM should not be called for deterministic title answers")

    engine = RAGEngine(FailingLLM(), use_hybrid=False, bm25_db_path=str(tmp_path / "b.db"))
    # Mock at orchestrator dependency level (query delegates to orchestrator)
    monkeypatch.setattr(engine._orchestrator, "_retrieve_front_matter_chunks", lambda doc_names, query, user_id: [{
        "doc_id": "paper::front_matter_title",
        "content": "Title: Deterministic Title",
        "metadata": {"type": "front_matter", "subtype": "title", "page": 1, "doc_name": "paper.pdf"},
        "score": 1.0,
    }])
    monkeypatch.setattr(engine._retrieval_pipeline, "retrieve_single", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector retrieval should not be used")))

    result = __import__("asyncio").run(engine.query("What is the title of this paper?", doc_names=["paper.pdf"], user_id=1))

    assert result["answer"] == 'The title of the paper is "Deterministic Title".'
    assert result["confidence"] == 1.0
    assert result["context_sufficient"] is True
    assert result["retrieved_chunks"][0]["type"] == "front_matter"
