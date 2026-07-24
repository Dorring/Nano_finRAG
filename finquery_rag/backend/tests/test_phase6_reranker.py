import pytest
"""Phase 6A tests: optional reranker interface."""
import os
import sys
import tempfile
import time
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mock heavy optional dependencies before importing RAGEngine.
mock_embed_fn = MagicMock()
mock_st_ef = MagicMock()
mock_st_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embed_fn
for _mod in [
    "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
    "camelot", "pymupdf", "langchain", "langchain_core", "langchain_core.documents",
    "langchain_text_splitters", "jieba_fast",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["jieba_fast"].cut_for_search = lambda text: [text]
sys.modules["chromadb.utils.embedding_functions"] = mock_st_ef

from services.rag_engine import RAGEngine
from services.reranker import CrossEncoderReranker, HeuristicReranker, NoopReranker, build_reranker


class MockLLMClient:
    def __init__(self):
        self.chat = self


def make_engine(**kwargs):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    engine = RAGEngine(
        llm_client=MockLLMClient(),
        bm25_db_path=tmp.name,
        **kwargs,
    )
    return engine, tmp.name


def cleanup(path):
    import gc
    gc.collect()
    for _ in range(3):
        try:
            os.unlink(path)
            return
        except PermissionError:
            time.sleep(0.05)


def chunk(doc_id, content, score):
    return {
        "doc_id": doc_id,
        "content": content,
        "metadata": {"type": "text", "page": 1},
        "score": score,
    }


def test_build_reranker_default_disabled():
    assert build_reranker(None) is None
    assert build_reranker("none") is None
    assert build_reranker("off") is None


def test_build_reranker_known_names():
    assert isinstance(build_reranker("noop"), NoopReranker)
    assert isinstance(build_reranker("heuristic"), HeuristicReranker)
    assert isinstance(
        build_reranker("cross-encoder", model_name_or_path="/models/reranker"),
        CrossEncoderReranker,
    )


def test_build_reranker_requires_model_for_cross_encoder():
    try:
        build_reranker("cross-encoder")
    except ValueError as exc:
        assert "RAG_RERANKER_MODEL" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_reranker_rejects_unknown():
    try:
        build_reranker("unknown")
    except ValueError as exc:
        assert "Unknown reranker" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_noop_reranker_preserves_order_and_top_k():
    chunks = [chunk("a", "alpha", 0.1), chunk("b", "beta", 0.9)]
    result = NoopReranker().rerank("beta", chunks, top_k=1)
    assert [item["doc_id"] for item in result] == ["a"]


def test_heuristic_reranker_adds_metadata_and_reorders_by_query_overlap():
    chunks = [
        chunk("a", "cash flow statement", 0.1),
        chunk("b", "revenue growth and gross margin", 0.1),
    ]
    result = HeuristicReranker(original_score_weight=0.0, lexical_weight=1.0).rerank(
        "revenue margin",
        chunks,
    )
    assert result[0]["doc_id"] == "b"
    assert result[0]["reranker"] == "heuristic"
    assert result[0]["rerank_score"] > result[1]["rerank_score"]


def test_rag_engine_default_reranker_disabled_preserves_order():
    engine, path = make_engine()
    try:
        chunks = [chunk("a", "alpha", 0.1), chunk("b", "beta", 0.9)]
        result = engine._apply_reranker("beta", chunks, top_k=2)
        assert [item["doc_id"] for item in result] == ["a", "b"]
    finally:
        cleanup(path)


def test_rag_engine_heuristic_reranker_can_reorder():
    engine, path = make_engine(reranker_name="heuristic")
    try:
        engine.reranker.original_score_weight = 0.0
        engine.reranker.lexical_weight = 1.0
        chunks = [
            chunk("a", "cash flow", 0.1),
            chunk("b", "revenue margin", 0.1),
        ]
        result = engine._apply_reranker("revenue", chunks, top_k=2)
        assert [item["doc_id"] for item in result] == ["b", "a"]
    finally:
        cleanup(path)


def test_rag_engine_reranker_respects_top_k():
    engine, path = make_engine(reranker_name="heuristic")
    try:
        chunks = [
            chunk("a", "alpha", 0.1),
            chunk("b", "beta", 0.1),
            chunk("c", "gamma", 0.1),
        ]
        result = engine._apply_reranker("gamma", chunks, top_k=1)
        assert len(result) == 1
        assert result[0]["doc_id"] == "c"
    finally:
        cleanup(path)


def test_rag_engine_retrieval_debug_tracks_candidate_counts():
    engine, path = make_engine(reranker_name="heuristic", retrieval_candidate_multiplier=3)
    try:
        chunks = [
            chunk("a", "alpha", 0.1),
            chunk("b", "beta", 0.1),
            chunk("c", "gamma", 0.1),
        ]
        engine._apply_reranker("gamma", chunks, top_k=1)
        assert engine._last_retrieval_debug == {
            "reranker": "heuristic",
            "reranker_enabled": True,
            "candidate_count": 3,
            "returned_count": 1,
            "candidate_multiplier": 3,
        }
    finally:
        cleanup(path)


def test_rag_engine_summarizes_retrieved_chunks_without_content():
    chunks = [{
        "doc_id": "user_1_q3.pdf::1",
        "content": "sensitive body",
        "metadata": {"doc_name": "q3.pdf", "page": 2, "type": "text"},
        "score": 0.4,
        "rerank_score": 0.8,
        "reranker": "heuristic",
    }]

    summary = RAGEngine._summarize_retrieved_chunks(chunks)

    assert summary == [{
        "doc_id": "user_1_q3.pdf::1",
        "filename": "q3.pdf",
        "page": 2,
        "type": "text",
        "score": 0.4,
        "rerank_score": 0.8,
        "reranker": "heuristic",
    }]
    assert "content" not in summary[0]


class FakeCrossEncoder:
    def predict(self, pairs):
        return [0.1 if "cash" in text else 0.9 for _, text in pairs]


def test_cross_encoder_reranker_uses_model_scores():
    chunks = [
        chunk("a", "cash flow", 0.9),
        chunk("b", "revenue margin", 0.1),
    ]
    reranker = CrossEncoderReranker(
        model_name_or_path="/models/fake",
        model=FakeCrossEncoder(),
    )

    result = reranker.rerank("revenue", chunks, top_k=1)

    assert len(result) == 1
    assert result[0]["doc_id"] == "b"
    assert result[0]["reranker"] == "cross-encoder"
    assert result[0]["rerank_score"] == 0.9


def test_cross_encoder_reranker_requires_model_or_path():
    try:
        CrossEncoderReranker(model_name_or_path="")
    except ValueError as exc:
        assert "requires a model" in str(exc)
    else:
        raise AssertionError("expected ValueError")
