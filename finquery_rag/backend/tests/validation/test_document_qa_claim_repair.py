"""Regression coverage for generic partial-claim repair in document QA."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.evidence import EvidenceItem  # noqa: E402
from src.domain.validation import ValidationStatus  # noqa: E402
from src.validation.response_validator import ResponseValidator  # noqa: E402
from src.validation.validation_policy import get_policy_for_intent  # noqa: E402


def test_document_qa_ungrounded_numeric_claim_is_repairable_not_blocked():
    """Keep an evidence-backed answer recoverable when one claim is unsafe."""
    policy = get_policy_for_intent("document_qa")
    assert policy.strict_numeric_grounding is False

    result = ResponseValidator().validate(
        answer="Revenue was $10 million. Margin was 99%.",
        intent="document_qa",
        evidence=(
            EvidenceItem(
                chunk_id="chunk-1",
                content="Revenue was $10 million for the reporting period.",
                document_name="report.pdf",
                page=1,
                content_type="text",
                score=0.9,
                rerank_score=None,
                metadata={},
            ),
        ),
        sources=(),
        calculation_result=None,
    )

    assert result.status is ValidationStatus.REPAIRABLE
