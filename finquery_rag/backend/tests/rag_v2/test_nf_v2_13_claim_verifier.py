from __future__ import annotations

from rag_v2.generation.contracts import AnswerEnvelopeV1
from rag_v2.runtime.semantic_claims import (
    SemanticClaimDecision,
    SemanticClaimVerifierV1,
)


def _packet(metric: str, value: str = "16.7") -> dict:
    return {
        "query_id": "q1",
        "route": "DIRECT",
        "validation_status": "VERIFIED",
        "allowed_citation_ids": ["EV-1"],
        "evidence_items": [{
            "citation_id": "EV-1",
            "metric": metric,
            "period": "FY2025",
            "value": value,
            "unit": None,
            "currency": None,
            "scale": None,
        }],
    }


def _answer(text: str) -> AnswerEnvelopeV1:
    return AnswerEnvelopeV1(
        query_id="q1",
        route="DIRECT",
        answer_text=text,
        citation_ids=("E1",),
        generator_provider="sealed",
        generator_model="fixture",
    )


def test_safe_numeric_metric_period_claim_is_supported():
    result = SemanticClaimVerifierV1().verify(
        _packet("Data Center", "115186"),
        _answer("Data Center revenue in FY2025 was 115186 [E1]."),
    )
    assert result.decision is SemanticClaimDecision.SUPPORTED
    assert result.unsupported_claims == ()


def test_unsupported_unit_claim_is_blocked_without_benchmark_rule():
    result = SemanticClaimVerifierV1().verify(
        _packet("Total volume1"),
        _answer("Total volume reported by Visa in FY2025 was 16.7 cubic feet [E1]."),
    )
    assert result.decision is SemanticClaimDecision.UNSUPPORTED
    assert "SCV_UNIT_UNSUPPORTED" in result.reason_codes


def test_network_transaction_claim_remains_supported():
    result = SemanticClaimVerifierV1().verify(
        _packet("Transactions processed on Visa's networks2", "257.5"),
        _answer("Visa's networks processed 257.5 transactions in FY2025. [E1]."),
    )
    assert result.decision is SemanticClaimDecision.SUPPORTED
