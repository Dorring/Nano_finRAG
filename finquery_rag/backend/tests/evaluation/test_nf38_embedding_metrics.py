"""Tests for NF38 embedding A/B metrics computation."""
from __future__ import annotations

import numpy as np

from src.evaluation.evaluation import EvaluationCase, ExpectedSource
from src.evaluation.nf37_metrics import ranking_metrics
from src.evaluation.nf38_evaluator import (
    _normalize_bm25_candidates,
    _to_rrf_format,
    _from_rrf_format,
    run_dense_diagnostic,
    run_hybrid_ranking,
)
from src.retrieval.candidate_fusion import rrf
from src.retrieval.embedding_provider import l2_normalize
from src.evaluation.nf38_dense_index import DenseIndex, build_dense_index
from src.evaluation.nf38_corpus import CanonicalEvidenceRecord, hash_embedding_text


class _StubProvider:
    def __init__(self, name: str = "stub", dimension: int = 8, seed: int = 42) -> None:
        self._name = name
        self._dimension = dimension
        self._seed = seed

    @property
    def name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        return "test-rev"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_length(self) -> int:
        return 128

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        rng = np.random.RandomState(self._seed)
        return l2_normalize(rng.randn(len(texts), self._dimension).astype(np.float32))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_documents(texts)


def _make_case(case_id: str = "c1", question: str = "What is revenue?") -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question=question,
        expected_sources=(ExpectedSource(filename="a.pdf", page=1),),
    )


def _make_records(n: int = 5) -> list[CanonicalEvidenceRecord]:
    return [
        CanonicalEvidenceRecord(
            evidence_id=f"r{i}",
            document_id="a.pdf",
            page=i + 1,
            block_type="text",
            embedding_text=f"revenue {i}",
            embedding_text_hash=hash_embedding_text(f"revenue {i}"),
        )
        for i in range(n)
    ]


def _make_index(seed: int = 42) -> DenseIndex:
    records = _make_records(5)
    provider = _StubProvider(seed=seed)
    return build_dense_index(records, provider, "ch", "eh")


def test_ranking_metrics_computes_all_ks():
    cases = [_make_case()]
    rankings = {"c1": [{"candidate_id": "r1", "evidence_id": "r1", "document_id": "a.pdf", "page": 1, "block_type": "text", "score": 0.9, "rank": 0}]}
    metrics = ranking_metrics(cases, rankings, ks=(5, 8, 20, 40))
    assert "case_hit_rate_at_5" in metrics
    assert "case_hit_rate_at_40" in metrics
    assert "source_recall_at_5" in metrics
    assert "mrr" in metrics


def test_dense_diagnostic_returns_metrics():
    cases = [_make_case()]
    index = _make_index()
    provider = _StubProvider()
    metrics = run_dense_diagnostic(cases, index, provider, ks=(5, 40))
    assert "case_hit_rate_at_5" in metrics
    assert "case_hit_rate_at_40" in metrics
    assert "source_recall_at_5" in metrics
    assert "mrr" in metrics
    assert metrics["query_count"] == 1
    assert metrics["embedding_model"] == "stub"
    assert metrics["embedding_dimension"] == 8


def test_dense_diagnostic_includes_latencies():
    cases = [_make_case()]
    index = _make_index()
    provider = _StubProvider()
    metrics = run_dense_diagnostic(cases, index, provider)
    assert "query_latencies_p50" in metrics
    assert "query_latencies_p95" in metrics
    assert metrics["query_latencies_p50"] >= 0
    assert metrics["query_latencies_p95"] >= 0


def test_hybrid_ranking_returns_rrf_and_final_metrics():
    cases = [_make_case()]
    index = _make_index()
    provider = _StubProvider()
    bm25_pool = {"c1": [{"candidate_id": "r0", "evidence_id": "r0", "document_id": "a.pdf", "page": 1, "block_type": "text", "score": 1.0, "rank": 0}]}
    result = run_hybrid_ranking(cases, index, provider, bm25_pool, reranker=None, top_k=5)
    assert "rrf" in result
    assert "final" in result
    assert "case_hit_rate_at_5" in result["final"]
    assert "mrr" in result["final"]
    assert result["reranker"] == "noop"


def test_bm25_pool_is_shared_between_variants():
    """Both variants must use the exact same BM25 pool."""
    cases = [_make_case()]
    index = _make_index()
    provider = _StubProvider()
    bm25_pool = {"c1": [{"candidate_id": "bm25_1", "evidence_id": "bm25_1", "document_id": "a.pdf", "page": 1, "block_type": "text", "score": 1.0, "rank": 0}]}

    # Run twice with the same pool — both get identical BM25 candidates
    result_a = run_hybrid_ranking(cases, index, provider, bm25_pool, reranker=None)
    result_b = run_hybrid_ranking(cases, index, provider, bm25_pool, reranker=None)
    assert result_a["rrf"] == result_b["rrf"]


def test_rrf_format_conversion_roundtrip():
    candidates = [
        {"candidate_id": "r1", "evidence_id": "r1", "document_id": "a.pdf", "page": 1, "block_type": "text", "score": 0.9, "rank": 0},
        {"candidate_id": "r2", "evidence_id": "r2", "document_id": "a.pdf", "page": 2, "block_type": "text", "score": 0.8, "rank": 1},
    ]
    rrf_format = _to_rrf_format(candidates)
    assert all("doc_id" in item for item in rrf_format)
    assert rrf_format[0]["doc_id"] == "r1"

    fused = rrf([rrf_format, rrf_format])
    back = _from_rrf_format(fused)
    assert all("candidate_id" in item for item in back)
    assert all("evidence_id" in item for item in back)


def test_normalize_bm25_candidates_extracts_metadata():
    raw = [
        {"doc_id": "r1", "content": "revenue", "metadata": {"doc_name": "a.pdf", "page": 1, "type": "text"}, "score": 1.5},
    ]
    normalized = _normalize_bm25_candidates(raw)
    assert len(normalized) == 1
    assert normalized[0]["candidate_id"] == "r1"
    assert normalized[0]["document_id"] == "a.pdf"
    assert normalized[0]["page"] == 1
    assert normalized[0]["block_type"] == "text"
    assert normalized[0]["rank"] == 0


def test_no_answer_cases_excluded_from_retrieval_denominator():
    """No-answer cases should not affect retrieval metrics."""
    cases = [
        _make_case("c1"),
        EvaluationCase(case_id="c2", question="unknown", expected_no_answer=True),
    ]
    rankings = {"c1": [{"candidate_id": "r1", "evidence_id": "r1", "document_id": "a.pdf", "page": 1, "block_type": "text", "score": 0.9, "rank": 0}]}
    metrics = ranking_metrics(cases, rankings, ks=(5,))
    # c2 has no expected_sources, so it's excluded from the denominator
    assert metrics["case_hit_rate_at_5"] == 1.0


def test_reranker_provider_is_unchanged():
    """The reranker variable must be fixed across A/B arms."""
    from src.services.reranker import build_reranker

    reranker = build_reranker("noop")
    assert reranker is not None
    assert reranker.name == "noop"

    # Using the same reranker name produces the same behavior
    reranker_a = build_reranker("noop")
    reranker_b = build_reranker("noop")
    assert reranker_a.name == reranker_b.name


def test_query_expansion_is_disabled():
    """NF38 must not use NF36 dual-query expansion."""
    # The DenseIndex.search method takes a single query vector, not expanded queries.
    # There is no query expansion in the NF38 evaluator.
    import inspect

    from src.evaluation.nf38_evaluator import run_dense_diagnostic

    source = inspect.getsource(run_dense_diagnostic)
    assert "expand" not in source.lower()
    assert "dual" not in source.lower()


def test_bge_reranker_is_disabled():
    """NF38 must not use the BGE reranker (only the default heuristic/noop)."""
    import inspect

    from src.evaluation.nf38_evaluator import run_hybrid_ranking

    source = inspect.getsource(run_hybrid_ranking)
    assert "bge_v2_m3" not in source
    assert "cross_encoder" not in source.lower()


def test_no_case_specific_embedding_logic():
    """The evaluator must not contain case-specific logic."""
    import inspect

    from src.evaluation.nf38_evaluator import run_dense_diagnostic, run_hybrid_ranking

    for func in [run_dense_diagnostic, run_hybrid_ranking]:
        source = inspect.getsource(func)
        assert "case_id ==" not in source
        assert 'case_id == "' not in source


def test_no_document_specific_embedding_logic():
    """The evaluator must not contain document-specific logic."""
    import inspect

    from src.evaluation.nf38_evaluator import run_dense_diagnostic

    source = inspect.getsource(run_dense_diagnostic)
    assert 'document_id ==' not in source
    assert 'doc_name ==' not in source


def test_question_and_label_hashes_are_independent():
    """Verify the case fingerprint separation from NF37 fix still holds."""
    from dataclasses import replace

    from src.evaluation.case_fingerprints import label_fingerprint, question_fingerprint

    cases = [
        EvaluationCase(
            case_id="c1",
            question="What is revenue?",
            expected_sources=(ExpectedSource(filename="a.pdf", page=1),),
            expected_numbers=("100",),
        ),
    ]
    q_hash = question_fingerprint(cases)
    l_hash = label_fingerprint(cases)
    assert q_hash != l_hash

    # Question-only change
    modified = [replace(cases[0], question="What is net income?")]
    assert question_fingerprint(modified) != q_hash
    assert label_fingerprint(modified) == l_hash
