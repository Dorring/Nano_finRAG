"""P1.5-S2-D: a relational task that cannot be verified must not release.

`compare-002` is the case this exists for.  Both operands bound correctly -- Apple
`(8,077)` and Visa `1,926` -- and the released answer was:

    "The verified evidence reports General and administrative for FY2025 at
     8,077 ... while the v_fy2025 calculation indicates that General and
     administrative was the metric for FY2025."

It never compares anything, and `expected_higher` is Visa.  It released because
`comparison` was not an operation the plan could express, so no structured result
existed, so the validator had nothing to check and passed.

The invariant asserted here is permanent rather than a stopgap for that one
case: **a task requiring a relational operation and holding no admissible
structured result cannot release.**  It holds whether the result is missing
because the operands never bound, because the calculator was blocked, or because
a planner forgot to name the operation -- every one of which is a reason to
refuse and none of which is a reason to let prose answer.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from rag_v2.adaptive import AdaptiveRAGStateV1
from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan
from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
    ComparisonRelation,
)
from src.runtime.trusted_v2_validation import TrustedReleaseValidationCapability

GAP_REASON = "RELATIONAL_RESULT_REQUIRED"


def _state(operation: str | None, intent: Intent = Intent.MULTI_EVIDENCE) -> AdaptiveRAGStateV1:
    slots = (
        RequiredSlot(
            slot_id="s1",
            metric="General and administrative",
            period="FY2025",
            role="value",
            value_type="numeric",
            unit=None,
            entity="Apple",
        ),
        RequiredSlot(
            slot_id="s2",
            metric="General and administrative",
            period="FY2025",
            role="value",
            value_type="numeric",
            unit=None,
            entity="Visa",
        ),
    )
    plan = SupervisorPlan(
        intent=intent,
        required_slots=slots,
        operation=operation,
        next_action=Action.RETRIEVE,
    )
    return AdaptiveRAGStateV1.new(
        request_id="relational-invariant",
        query="Compare Apple and Visa",
        intent=intent.value,
        task_type=intent.value,
        required_slots=[slot.to_dict() for slot in slots],
        plan={"supervisor_plan": plan.to_dict()},
    )


def _comparison_result(*, well_formed: bool = True) -> CalculationResult:
    return CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation.COMPARISON,
        operands=(
            CalculationOperand(name="lhs", value=Decimal("-8077"), evidence_chunk_id="E1", slot_id="s1"),
            CalculationOperand(name="rhs", value=Decimal("1926"), evidence_chunk_id="E2", slot_id="s2"),
        ),
        relation=ComparisonRelation.RHS_GT_LHS if well_formed else None,
    )


# --- the invariant -----------------------------------------------------------------


def test_a_comparison_with_no_structured_result_must_not_release() -> None:
    """`compare-002`'s shape: the plan asks for a comparison and nothing computed one."""

    assert TrustedReleaseValidationCapability._relational_result_gap(
        _state("comparison")
    ) == GAP_REASON


def test_a_comparison_with_a_structured_result_may_release() -> None:
    state = _state("comparison")
    state._calculation_result_obj = _comparison_result()

    assert TrustedReleaseValidationCapability._relational_result_gap(state) is None


def test_a_blocked_relational_result_is_not_admissible() -> None:
    """A BLOCKED calculation is an honest refusal, not a result to release on."""

    state = _state("comparison")
    state._calculation_result_obj = CalculationResult(
        status=CalculationStatus.BLOCKED,
        operation=CalculationOperation.COMPARISON,
        error_code="INSUFFICIENT_OPERANDS",
    )

    assert TrustedReleaseValidationCapability._relational_result_gap(state) == GAP_REASON


def test_a_malformed_relational_result_is_not_admissible() -> None:
    """Executed, but with no stated relation -- prose with a status field."""

    state = _state("comparison")
    state._calculation_result_obj = _comparison_result(well_formed=False)

    assert TrustedReleaseValidationCapability._relational_result_gap(state) == GAP_REASON


def test_a_ranking_with_no_ordering_must_not_release() -> None:
    state = _state("ranking", intent=Intent.MULTI_EVIDENCE)
    state._calculation_result_obj = CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation.RANKING,
        operands=(
            CalculationOperand(name="a", value=Decimal("1"), evidence_chunk_id="E1", slot_id="s1"),
            CalculationOperand(name="b", value=Decimal("2"), evidence_chunk_id="E2", slot_id="s2"),
        ),
        ordering_groups=None,
    )

    assert TrustedReleaseValidationCapability._relational_result_gap(state) == GAP_REASON


# --- and nothing else changes ------------------------------------------------------


@pytest.mark.parametrize(
    "operation",
    ["difference", "average", "percentage_share", "scale_conversion"],
)
def test_an_arithmetic_plan_is_unaffected(operation: str) -> None:
    """No other stratum's release depends on this invariant."""

    assert TrustedReleaseValidationCapability._relational_result_gap(
        _state(operation, intent=Intent.CALCULATION)
    ) is None


def test_a_plan_with_no_operation_is_unaffected() -> None:
    """Which is every cross-entity fixture as it stands today."""

    assert TrustedReleaseValidationCapability._relational_result_gap(
        _state(None, intent=Intent.DIRECT_FACT)
    ) is None


def test_a_state_with_no_plan_is_unaffected() -> None:
    """The guard is a release condition, not a precondition on having a plan."""

    state = _state(None, intent=Intent.DIRECT_FACT)
    state.plan = {}

    assert TrustedReleaseValidationCapability._relational_result_gap(state) is None
