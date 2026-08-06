from __future__ import annotations

import pytest

from src.pdf_retrieval_v4.query_plan_models import QueryPlan, RetrievalConstraints
from src.pdf_retrieval_v4.v4_gate08_pool import build_query, merge_raw_protected


def _plan() -> QueryPlan:
    return QueryPlan(
        plan_id="p",
        plan_version="v1",
        raw_question="What was revenue in FY2025?",
        document_scope=("aapl_fy2025",),
        task_type="table_single_fact",
        operation=None,
        issuer="aapl_fy2025",
        metric_phrases=("revenue",),
        periods=("FY2025",),
        evidence_shapes=("atomic_fact",),
        operand_slots=(),
        retrieval_routes=(),
        constraints=RetrievalConstraints(),
        raw_protection_required=True,
        answerability_check_required=True,
    )


def test_raw_candidates_unchanged_and_residual_appended() -> None:
    raw = [
        {"candidate_key": "raw-a", "stage_rank": 1, "score": 2.0},
        {"candidate_key": "raw-b", "stage_rank": 2, "score": 1.0},
    ]
    structured = [
        {"original_candidate_identity": "raw-b"},
        {"original_candidate_identity": "new-c"},
    ]
    result = merge_raw_protected(raw, structured, structured_k=40)
    assert [item["candidate_key"] for item in result["combined_pool"]] == ["raw-a", "raw-b", "new-c"]
    assert result["raw_candidate_loss"] is False
    assert result["raw_candidate_hash_before"] == result["raw_candidate_hash_after"]


def test_query_builder_keeps_raw_question_and_concept_features() -> None:
    plan = _plan()
    text = build_query(plan, slot={"raw_metric_phrase": "net revenue", "period": "FY2025", "concept_candidates": ["sales"]})
    assert "What was revenue in FY2025?" in text
    assert "net revenue" in text
    assert "sales" in text


def test_shadow_runtime_path_fails_closed() -> None:
    from src.pdf_retrieval_v4.shadow_index_reader import ShadowIndexReader

    with pytest.raises(ValueError, match="unsafe_shadow_runtime_path"):
        ShadowIndexReader(__import__("pathlib").Path("/tmp/other-index"))
