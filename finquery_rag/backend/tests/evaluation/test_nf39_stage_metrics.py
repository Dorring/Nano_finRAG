"""Tests for NF39 stage metrics computation."""
from __future__ import annotations

from src.evaluation.evaluation import EvaluationCase, ExpectedSource
from src.evaluation.nf39_attribution import compute_stage_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_expected(
    chunk_id: str = "ev_1",
    filename: str = "doc_a",
    page: int | None = None,
) -> ExpectedSource:
    return ExpectedSource(filename=filename, page=page, chunk_id=chunk_id)


def _make_case(
    case_id: str = "case_1",
    expected_sources: tuple[ExpectedSource, ...] = (),
    expected_no_answer: bool = False,
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question=f"Question for {case_id}",
        expected_sources=expected_sources,
        expected_answer_contains=(),
        expected_numbers=(),
        expected_no_answer=expected_no_answer,
        expected_calculations=(),
        expected_intent=None,
        document_names=("doc_a",),
        tags=(),
        metadata={},
    )


def _make_candidate(eid: str, doc: str = "doc_a", page: int = 1) -> dict:
    return {
        "evidence_id": eid,
        "candidate_id": eid,
        "document_id": doc,
        "page": page,
        "block_type": "text",
        "parent_id": None,
        "table_id": None,
        "score": 0.5,
        "rrf_score": 0.5,
        "reranker_score": 0.5,
    }


def _make_rankings(
    case_ids: list[str],
    candidates_per_case: int = 10,
) -> dict[str, list[dict]]:
    return {
        cid: [_make_candidate(f"ev_{cid}_{i}", page=i) for i in range(1, candidates_per_case + 1)]
        for cid in case_ids
    }


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_rrf_reranker_final_metrics_use_same_denominator():
    """RRF, Reranker, and Final metrics must use the same case set as denominator.

    All three stages must compute metrics over the same set of eligible
    cases (those with expected_sources and not expected_no_answer).
    The case_count field must be identical across stages.
    """
    cases = [
        _make_case("case_1", expected_sources=(_make_expected("ev_1"),)),
        _make_case("case_2", expected_sources=(_make_expected("ev_2"),)),
        _make_case("case_3", expected_sources=(_make_expected("ev_3"),)),
    ]
    rankings = _make_rankings(["case_1", "case_2", "case_3"])

    rrf_metrics = compute_stage_metrics(cases=cases, rankings=rankings, ks=(5, 8, 20, 40))
    reranker_metrics = compute_stage_metrics(cases=cases, rankings=rankings, ks=(5, 8, 20))
    final_metrics = compute_stage_metrics(cases=cases, rankings=rankings, ks=(5,))

    assert rrf_metrics["case_count"] == reranker_metrics["case_count"] == final_metrics["case_count"]
    assert rrf_metrics["case_count"] == 3, "All 3 cases should be in denominator"


def test_no_answer_excluded_from_retrieval_metrics():
    """No-answer cases must not be in the retrieval metric denominator."""
    cases = [
        _make_case("case_1", expected_sources=(_make_expected("ev_1"),)),
        _make_case("case_2", expected_sources=(_make_expected("ev_2"),)),
        _make_case("case_3", expected_no_answer=True),
    ]
    rankings = _make_rankings(["case_1", "case_2", "case_3"])

    metrics = compute_stage_metrics(cases=cases, rankings=rankings, ks=(5,))

    assert metrics["case_count"] == 2, (
        "No-answer case must be excluded from denominator"
    )


def test_case_without_expected_sources_excluded():
    """Cases without expected_sources must not be in the denominator."""
    cases = [
        _make_case("case_1", expected_sources=(_make_expected("ev_1"),)),
        _make_case("case_2", expected_sources=()),
    ]
    rankings = _make_rankings(["case_1", "case_2"])

    metrics = compute_stage_metrics(cases=cases, rankings=rankings, ks=(5,))

    assert metrics["case_count"] == 1


def test_metrics_contain_all_required_ks():
    """compute_stage_metrics must produce metrics at all requested K values."""
    cases = [_make_case("case_1", expected_sources=(_make_expected("ev_1"),))]
    rankings = _make_rankings(["case_1"])

    metrics = compute_stage_metrics(cases=cases, rankings=rankings, ks=(5, 8, 20, 40))

    for k in (5, 8, 20, 40):
        assert f"case_hit_rate_at_{k}" in metrics
        assert f"source_recall_at_{k}" in metrics
        assert f"all_source_coverage_at_{k}" in metrics
    assert "mrr" in metrics
    assert "case_count" in metrics


def test_mrr_uses_first_matching_rank():
    """MRR must be based on the first matching candidate's reciprocal rank."""
    expected = _make_expected("ev_3")
    cases = [_make_case("case_1", expected_sources=(expected,))]
    rankings = {
        "case_1": [_make_candidate(f"ev_{i}") for i in range(1, 6)],
    }

    metrics = compute_stage_metrics(cases=cases, rankings=rankings, ks=(5,))

    # ev_3 is at rank 3, so MRR = 1/3
    assert abs(metrics["mrr"] - 1.0 / 3.0) < 1e-9


def test_case_hit_rate_counts_any_match():
    """Case Hit Rate@K counts a case as hit if any expected source is in Top-K."""
    expected1 = _make_expected("ev_1")
    expected2 = _make_expected("ev_999")  # not in candidates
    cases = [
        _make_case("case_1", expected_sources=(expected1, expected2)),
    ]
    rankings = {"case_1": [_make_candidate(f"ev_{i}") for i in range(1, 6)]}

    metrics = compute_stage_metrics(cases=cases, rankings=rankings, ks=(5,))

    # ev_1 is at rank 1, so case is hit
    assert metrics["case_hit_rate_at_5"] == 1.0


def test_source_recall_counts_matched_expected():
    """Source Recall@K = matched expected sources / total expected sources."""
    expected1 = _make_expected("ev_1")
    expected2 = _make_expected("ev_999")  # not in candidates
    cases = [
        _make_case("case_1", expected_sources=(expected1, expected2)),
    ]
    rankings = {"case_1": [_make_candidate(f"ev_{i}") for i in range(1, 6)]}

    metrics = compute_stage_metrics(cases=cases, rankings=rankings, ks=(5,))

    # 1 out of 2 expected sources matched
    assert abs(metrics["source_recall_at_5"] - 0.5) < 1e-9
