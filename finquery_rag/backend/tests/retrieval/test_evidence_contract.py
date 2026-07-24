"""Regression tests for score provenance and context-aligned validation."""

from src.domain.calculation import CalculationStatus
from src.retrieval.candidate_fusion import normalize_scores, rrf
from src.retrieval.context_builder import ContextBuilder, EvidenceSufficiencyEvaluator
from src.validation.answerability import AnswerabilityEvaluator


def _chunk(*, score=0.0, content="Revenue was 125 million."):
    return {
        "doc_id": "annual.pdf::page_5::chunk_1",
        "content": content,
        "metadata": {
            "doc_name": "annual.pdf",
            "page": 5,
            "parent_id": "annual.pdf::page_5::parent",
            "parent_excerpt": "Revenue was 125 million in the annual report.",
        },
        "score": score,
    }


def test_single_retriever_rrf_score_is_not_rejected_by_dense_threshold():
    fused = rrf([[_chunk(score=0.91)]])
    normalized = normalize_scores(fused)
    result = EvidenceSufficiencyEvaluator().evaluate(normalized)

    assert normalized[0]["score_kind"] == "rrf"
    assert normalized[0]["score"] > 0.008
    assert result.is_sufficient is True


def test_context_evidence_uses_parent_expanded_text_seen_by_generator():
    builder = ContextBuilder()
    context, _ = builder.build([_chunk()])

    assert "Revenue was 125 million in the annual report." in context
    assert builder.last_context_evidence[0]["content"] in context
    assert "Matched child evidence" in builder.last_context_evidence[0]["content"]


def test_query_matched_evidence_can_override_only_score_calibration():
    evaluator = AnswerabilityEvaluator()
    from src.domain.evidence import EvidenceItem
    from src.retrieval.context_builder import SufficiencyResult

    result = evaluator.evaluate(
        question="What was revenue?",
        intent="document_qa",
        evidence=(EvidenceItem.from_chunk(_chunk(score=0.01)),),
        sufficiency_result=SufficiencyResult(False, 0.01, 0.01),
        calculation_result=None,
        requested_documents=("annual.pdf",),
        has_query_matched_evidence=True,
    )

    assert result.status.value == "answerable"
    assert "query_matched_evidence" in result.reason_codes
