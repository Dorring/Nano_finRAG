"""Phase 12 tests: query trace_id propagation."""
import asyncio
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock

mock_embed_fn = MagicMock()
mock_st_ef = MagicMock()
mock_st_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embed_fn
for _mod in [
    "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
    "jieba_fast",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["jieba_fast"].cut_for_search = lambda text: [text]
sys.modules["chromadb.utils.embedding_functions"] = mock_st_ef

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.eval_runner import run_case
from evaluation.evaluation import EvaluationCase
from services.rag_engine import RAGEngine


class FixedTraceLogger:
    def __init__(self, trace_id="trace-123", fail=False):
        self.trace_id = trace_id
        self.fail = fail
        self.calls = []

    def log(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("trace unavailable")
        return self.trace_id


def _engine(tmp_path, trace_logger):
    client = MagicMock()
    engine = RAGEngine(
        client,
        use_hybrid=False,
        bm25_db_path=str(tmp_path / "bm25.db"),
        trace_db_path=str(tmp_path / "trace.db"),
    )
    engine.trace_logger = trace_logger
    # Mock at orchestrator dependency level (query delegates to orchestrator)
    engine._orchestrator._trace_logger = trace_logger
    engine._retrieval_pipeline.retrieve_single = MagicMock(return_value=[{
        "doc_id": "user_1_report.pdf::1",
        "content": "Revenue was $10M.",
        "metadata": {"doc_name": "report.pdf", "page": 1, "type": "text"},
        "score": 0.9,
    }])
    engine._llm_gateway.generate = AsyncMock(return_value="Revenue was $10M.")
    return engine


def test_query_returns_trace_id_from_logger(tmp_path):
    logger = FixedTraceLogger("trace-abc")
    engine = _engine(tmp_path, logger)

    result = asyncio.run(
        engine.query(
            "What was revenue?",
            doc_names=["report.pdf"],
            user_id=1,
            n_results=1,
        )
    )

    assert result["trace_id"] == "trace-abc"
    assert logger.calls[0]["tenant_id"] == 1
    # Phase 4 hotfix: trace redacts answer content (answer=None).
    assert logger.calls[0]["answer"] is None


def test_query_trace_failure_does_not_break_answer(tmp_path):
    engine = _engine(tmp_path, FixedTraceLogger(fail=True))

    result = asyncio.run(
        engine.query(
            "What was revenue?",
            doc_names=["report.pdf"],
            user_id=1,
            n_results=1,
        )
    )

    assert result["answer"] == "Revenue was $10M."
    # When trace logger fails, a fallback UUID is generated so clients
    # always have a trace_id to correlate requests.
    assert isinstance(result["trace_id"], str) and len(result["trace_id"]) > 0


def test_eval_runner_persists_trace_id():
    class FakeEngine:
        async def query(self, **kwargs):
            return {
                "answer": "A",
                "sources": [],
                "retrieved_chunks": [],
                "searched_docs": [],
                "confidence": 1.0,
                "context_sufficient": True,
                "intent": "document_qa",
                "intent_confidence": 0.8,
                "retrieval_debug": {},
                "trace_id": "trace-runner",
            }

    case = EvaluationCase.from_dict({"id": "c1", "question": "Q"})
    prediction = asyncio.run(run_case(case, FakeEngine(), user_id=1))

    assert prediction["trace_id"] == "trace-runner"
