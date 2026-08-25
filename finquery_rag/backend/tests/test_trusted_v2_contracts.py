"""TV2-01 focused contract tests."""

from __future__ import annotations

import pytest

from src.runtime import (
    ReleaseStatus,
    V2ExecutionOutcome,
    V2ExecutionStatus,
)


@pytest.mark.parametrize(
    ("status", "release_status"),
    [
        (V2ExecutionStatus.FAIL_CLOSED, ReleaseStatus.NOT_APPLICABLE),
        (V2ExecutionStatus.EXECUTION_ERROR, ReleaseStatus.RELEASED),
    ],
)
def test_non_release_outcomes_cannot_cross_release_boundary(
    status: V2ExecutionStatus,
    release_status: ReleaseStatus,
) -> None:
    with pytest.raises(ValueError):
        V2ExecutionOutcome(
            status=status,
            release_status=release_status,
        )


def test_ready_outcome_requires_answer_and_explicit_release() -> None:
    with pytest.raises(ValueError, match="non-empty answer"):
        V2ExecutionOutcome(
            status=V2ExecutionStatus.READY_FOR_RELEASE,
            release_status=ReleaseStatus.RELEASED,
        )


def test_structured_ids_use_first_seen_stable_deduplication() -> None:
    outcome = V2ExecutionOutcome(
        status=V2ExecutionStatus.FAIL_CLOSED,
        evidence_ids=["fact-2", "fact-1", "fact-2"],
        citation_ids=["cite-2", "cite-2", "cite-1"],
        calculation_ids=["calc-1", "calc-1"],
        reason_codes=["MISSING_SLOT", "MISSING_SLOT"],
        release_status=ReleaseStatus.NOT_RELEASED,
    )

    assert outcome.evidence_ids == ["fact-2", "fact-1"]
    assert outcome.citation_ids == ["cite-2", "cite-1"]
    assert outcome.calculation_ids == ["calc-1"]
    assert outcome.reason_codes == ["MISSING_SLOT"]
