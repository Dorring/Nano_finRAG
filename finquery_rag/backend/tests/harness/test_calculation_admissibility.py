"""H2A-1B: a calculation result is usable only when the runtime admits it.

Evidence taught the rule once (H2A-1E): the existence of a value is not the
trustworthiness of that value.  Calculation has the same shape and had the same
defect:

    calculation_result_id is not None  ->  "there is a usable result"

which is false whenever the calculator ran and honestly reported ``BLOCKED``.
The coordinator guarded the release path, so no bad answer escaped -- but that is
path safety, and H1 already showed path safety is not safety.  A projection
helper that can be handed a blocked result and an id must be unable to produce a
trusted-looking payload, whatever calls it.

The fix puts the identity on the result and makes it unobtainable unless the
result is admissible, so there is one truth and it cannot disagree with status.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1
from src.domain.calculation import (
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
)
from src.runtime.trusted_v2_coordinator import _structured_calculations

ALL_OPERATIONS = list(CalculationOperation)


def _result(status: CalculationStatus, **kw: object) -> CalculationResult:
    return CalculationResult(
        status=status,
        operation=CalculationOperation.GROWTH_RATE,
        formula_version="growth_rate.v1",
        **kw,  # type: ignore[arg-type]
    )


def _state_with(result: CalculationResult) -> AdaptiveRAGStateV1:
    state = AdaptiveRAGStateV1.new("r", "q")
    state._calculation_result_obj = result
    return state


# --- direct projection safety ------------------------------------------------


@pytest.mark.parametrize(
    "status, extras",
    [
        (CalculationStatus.BLOCKED, {"error_code": "INSUFFICIENT_OPERANDS"}),
        (CalculationStatus.FAILED, {"error_code": "PRIMITIVE_EXCEPTION"}),
        (CalculationStatus.NOT_APPLICABLE, {}),
        (CalculationStatus.READY, {}),
    ],
)
def test_an_inadmissible_result_cannot_be_projected(
    status: CalculationStatus, extras: dict[str, object]
) -> None:
    """The caller supplies a plausible-looking id; the result refuses to have one."""

    result = _result(status, **extras)

    assert result.calculation_id is None
    assert _structured_calculations(_state_with(result), ["C1-forged"]) == []


def test_a_forged_id_cannot_manufacture_admissibility() -> None:
    """The defect stated directly: a non-None id must not be sufficient.

    Before the fix this returned a payload carrying ``status: blocked`` and the
    caller's invented ``calculation_id``.
    """

    blocked = _result(CalculationStatus.BLOCKED, error_code="INSUFFICIENT_OPERANDS")

    payload = _structured_calculations(_state_with(blocked), ["C1-fake"])

    assert payload == []


def test_an_admissible_result_still_projects_exactly_as_before() -> None:
    """Governed, not narrowed.  A real result must be unaffected."""

    executed = _result(
        CalculationStatus.EXECUTED, value=Decimal("0.0209"), unit="ratio"
    )

    payload = _structured_calculations(_state_with(executed), [executed.calculation_id])

    assert len(payload) == 1
    assert payload[0]["status"] == "executed"
    assert payload[0]["calculation_id"] == executed.calculation_id
    assert payload[0]["calculation_id"].startswith("C1-")


def test_the_identity_covers_arithmetic_and_operands_only() -> None:
    """Two results that compute the same thing are the same calculation."""

    left = _result(CalculationStatus.EXECUTED, value=Decimal("0.0209"), unit="ratio")
    right = _result(CalculationStatus.EXECUTED, value=Decimal("0.0209"), unit="ratio")
    different = _result(
        CalculationStatus.EXECUTED, value=Decimal("0.0300"), unit="ratio"
    )

    assert left.calculation_id == right.calculation_id
    assert left.calculation_id != different.calculation_id


# --- identity ownership ------------------------------------------------------


def test_the_capability_and_the_result_cannot_report_different_ids() -> None:
    """One truth.  The capability reads the id off the result; it does not mint one.

    When the id was computed beside the result, a producer could pair a BLOCKED
    result with a non-None id and the two would disagree with nothing to
    reconcile them.
    """

    from tests.harness.harness_support import BlockedCalculation

    blocked = BlockedCalculation()
    assert blocked.last_calculation_id is None

    state = AdaptiveRAGStateV1.new("r", "q")
    result = blocked.calculate(state)
    assert result.calculation_id is None
    assert result.is_admissible is False


# --- required calculation must not vanish ------------------------------------


def test_a_required_calculation_that_cannot_be_admitted_is_not_erased() -> None:
    """Silently dropping it would read downstream as "no calculation requested".

    The projection yields nothing, which is correct for a *projection*.  What
    must not happen is the runtime concluding the calculation was optional: the
    refusal has to survive somewhere explicit.  This pins that the two are
    different, by showing the projection is empty while the result still
    carries its reason.
    """

    blocked = _result(CalculationStatus.BLOCKED, error_code="INSUFFICIENT_OPERANDS")

    assert _structured_calculations(_state_with(blocked), ["C1"]) == []
    # The refusal is not erased: it remains on the authoritative result.
    assert blocked.status is CalculationStatus.BLOCKED
    assert blocked.error_code == "INSUFFICIENT_OPERANDS"
    assert blocked.is_admissible is False


# --- the nine operations are untouched ---------------------------------------


@pytest.mark.parametrize("operation", ALL_OPERATIONS)
def test_every_operation_can_produce_an_admissible_identity(
    operation: CalculationOperation,
) -> None:
    """Identity is a property of admissibility, not of which operation ran."""

    result = CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=operation,
        value=Decimal("1"),
        formula_version=f"{operation.value}.v1",
    )

    assert result.is_admissible
    assert result.calculation_id is not None


def test_the_nine_operations_are_unchanged() -> None:
    """B hardens admission; it does not touch arithmetic."""

    assert [op.value for op in CalculationOperation] == [
        "difference",
        "growth_rate",
        "percentage_share",
        "sum",
        "average",
        "gross_margin",
        "net_margin",
        "debt_ratio",
        "scale_conversion",
    ]


# --- the real production path ------------------------------------------------


def test_a_real_calculation_query_reaches_release() -> None:
    """retrieve -> bind -> calculate -> admissible -> generate -> release."""

    from src.runtime import V2ExecutionStatus
    from src.runtime.harness_runtime_mode import AgentRuntimeMode
    from tests.harness.h1_integration import FIXTURES, run_fixture

    fixture = next(f for f in FIXTURES if f.fixture_id == "calculation_growth_rate")
    outcome = run_fixture(fixture, AgentRuntimeMode.HARNESS_V3)

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert outcome.calculations
    assert outcome.calculations[0]["status"] == "executed"


def test_a_real_blocked_calculation_produces_no_trusted_calculation() -> None:
    """calculate -> inadmissible -> nothing published, nothing released."""

    from src.runtime import V2ExecutionStatus
    from src.runtime.harness_runtime_mode import AgentRuntimeMode
    from tests.harness.h1_integration import FIXTURES, run_fixture

    fixture = next(f for f in FIXTURES if f.fixture_id == "calculation_blocked")
    outcome = run_fixture(fixture, AgentRuntimeMode.HARNESS_V3)

    assert outcome.status is not V2ExecutionStatus.READY_FOR_RELEASE
    assert outcome.calculations == []
    assert "CALCULATION_INVALID" in outcome.reason_codes
