"""H2A-2A.1 — conflict coverage in the legacy/harness_v3 differential.

The H1 differential corpus has ten fixtures and **none of them has two
disagreeing admissible candidates for one slot**, so its `reason_codes` and
`release_status` are identical across modes by construction and it contributes
no coverage to the conflict gate. E was verified provider-independently and
end-to-end by the readiness benchmark; the differential contract, which is the
thing that will police the H2A-2 SlotBinding refactor, saw nothing.

This adds that coverage.

**It is added here rather than to `FIXTURES` on purpose.** `FIXTURES` is sealed:
`sealed_digest()` is recorded in the H1 seal document and in the committed
integration report. Adding a fixture there would change the digest and put the
seal's own record out of step with every future run. A conflict fixture
constructed in this file gets the coverage without editing a frozen record.

The expectations below are **authored**, not produced by the conflict
implementation: "these two candidates disagree, so nothing may be released" is a
domain claim, and the test asks whether the runtime agrees. A guard whose
expectation came from the code under test would only confirm that the code
agrees with itself.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_v2.adaptive.adaptive_contracts import ReasonCode
from src.runtime import V2ExecutionStatus
from src.runtime.harness_runtime_mode import AgentRuntimeMode
from tests.harness.equivalence import assert_decision_equivalent
from tests.harness.h1_integration import H1Fixture, run_fixture
from tests.test_trusted_v2_r4_binder import _fact

#: One slot, two admissible candidates, contradictory values. No majority, no
#: corroboration, no representation equivalence -- an unresolved conflict.
CONFLICT_FIXTURE = H1Fixture(
    fixture_id="differential-conflict",
    description=(
        "Two admissible candidates for one slot disagree and no safe consensus "
        "exists. Nothing may be released, in either runtime mode."
    ),
    query="What was Apple's FY2024 revenue?",
    slots=({"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"},),
    facts={
        "CF1": _fact("CF1", slots=("revenue",), value="100"),
        "CF2": _fact("CF2", slots=("revenue",), value="999"),
    },
    routes=(("", ("CF1", "CF2")),),
    # Authored, not derived: a conflict is a fail-closed terminal.
    expect_released=False,
    expect_status="FAIL_CLOSED",
    expect_reasons=(ReasonCode.EVIDENCE_CONFLICT.value,),
)


def _outcome(mode: AgentRuntimeMode) -> Any:
    return run_fixture(CONFLICT_FIXTURE, mode)


def test_the_fixture_really_presents_a_conflict() -> None:
    """Without this the comparison below would be vacuous.

    Two candidates must actually be admissible for the slot; a fixture that
    retrieved one of them would be a fact query wearing a conflict's name.
    """

    assert CONFLICT_FIXTURE.routes == (("", ("CF1", "CF2")),)
    assert len(CONFLICT_FIXTURE.facts) == 2
    assert CONFLICT_FIXTURE.facts["CF1"]["value"] != CONFLICT_FIXTURE.facts["CF2"]["value"]


@pytest.mark.parametrize("mode", [AgentRuntimeMode.LEGACY, AgentRuntimeMode.HARNESS_V3])
def test_a_conflict_never_releases_in_either_mode(mode: AgentRuntimeMode) -> None:
    """The safety property the H2A-2 SlotBinding refactor must not break."""

    outcome = _outcome(mode)

    assert outcome.release_status.value != "RELEASED"
    assert outcome.status is not V2ExecutionStatus.READY_FOR_RELEASE
    assert outcome.answer is None


def test_the_two_modes_agree_on_the_conflict() -> None:
    """The differential claim, over a shape the sealed corpus does not cover."""

    assert_decision_equivalent(_outcome(AgentRuntimeMode.LEGACY), _outcome(AgentRuntimeMode.HARNESS_V3))


def test_the_conflict_is_reported_as_a_conflict_not_as_missing_evidence() -> None:
    """The distinction E exists to draw.

    "No candidates" and "candidates that disagree" are different states, and a
    runtime that reported the second as the first would be erasing evidence it
    actually had -- which is how a slot looks empty when it is contested.
    """

    codes = set(_outcome(AgentRuntimeMode.HARNESS_V3).reason_codes)

    assert ReasonCode.EVIDENCE_CONFLICT.value in codes
    assert ReasonCode.MISSING_SLOT.value not in codes
