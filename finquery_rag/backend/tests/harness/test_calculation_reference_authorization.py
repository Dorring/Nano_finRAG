"""H2A-2D-3A: the calculation-reference invariant, exercised for the first time.

`CALCULATION_PROVENANCE_MISMATCH` existed in the validator and had **no test
anywhere**. The 2D-1 audit classified the sibling citation guard as vacuous; this
one is not vacuous, but an invariant nobody exercises is an invariant nobody can
tell has stopped working.

**Producer classification, by producer rather than by candidate type.** The
distinction matters because the same `CandidateExecutionResult` type carries
different trust semantics depending on who filled it:

| producer | `calculation_ids` origin | what the guard is |
| --- | --- | --- |
| `trusted_v2_generation.py:396` (live) | `state.calculation_result_id`, itself `CalculationResult.calculation_id` | **consistency / wiring** |
| `trusted_v2_validation.py:274` (repairer) | copied from the candidate it repairs | **consistency / wiring** |
| `trusted_v2_validation.py:345` (string) | declares nothing | **not applicable** |
| a substituted/foreign generator | declares its own ids | **authorization** |

So on every live production path this guard checks that the candidate's
declaration agrees with the authority; it is a real authorization boundary only
where a candidate can declare ids independently of that authority. The tests
below author the candidate side literally and let the admitted side come from
the real admissible result, which is the direction the anti-circularity rule
requires: the candidate is never populated by reading the authority it is
checked against.
"""

from __future__ import annotations

from typing import Any

from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1
from src.runtime.trusted_v2_generation import CandidateExecutionResult
from src.domain.calculation import CalculationResult, CalculationStatus
from src.runtime.trusted_v2_validation import TrustedReleaseValidationCapability
from tests.harness.test_calculation_identity_oracle import GOLDEN_ID, _result
from tests.test_trusted_v2_r4_binder import _fact

#: Authored literal: an id no admissible result in this file can produce.
FABRICATED_ID = "C1-deadbeefdeadbeef"

CITATION = "citation-E1"


def _state(result: CalculationResult | None) -> AdaptiveRAGStateV1:
    """A state whose evidence is admitted and whose calculation authority is set.

    H2A-2D-3B: the authority is attached as a *result*, not assigned as an id.
    The state's `calculation_result_id` is now derived from this object, so a
    test can no longer declare an id the result does not have -- which is the
    drift this phase removes.
    """

    state = AdaptiveRAGStateV1.new("r", "q")
    state.add_evidence(
        [EvidencePacketV1.from_mapping({**_fact("E1"), "citation_id": CITATION})]
    )
    state.bound_evidence_ids = ["E1"]
    state._calculation_result_obj = result
    return state


def _candidate(declared: tuple[str, ...]) -> CandidateExecutionResult:
    """The candidate declares its own references, authored here."""

    return CandidateExecutionResult(
        candidate_answer="Growth Rate: 2.09%",
        route="CALCULATION_SIMPLE",
        route_reason="fixture",
        bound_evidence_ids=("E1",),
        citation_ids=(CITATION,),
        calculation_ids=declared,
    )


#: The authored admissible vector from the frozen oracle; its id is GOLDEN_ID.
ADMISSIBLE = _result()

#: An inadmissible result, which authorizes nothing.
BLOCKED = CalculationResult(status=CalculationStatus.BLOCKED)


def _validate(result: CalculationResult | None, declared: tuple[str, ...]) -> Any:
    return TrustedReleaseValidationCapability().validate(
        _state(result), _candidate(declared)
    )


# --- authorized ---------------------------------------------------------------


def test_a_candidate_declaring_the_admitted_calculation_is_authorized() -> None:
    """The candidate's literal matches the authority. Not a mismatch.

    Asserted as the *absence* of this reason code rather than as an overall
    pass: other contracts in the same validator may still have things to say,
    and conflating them would hide which invariant fired.
    """

    result = _validate(ADMISSIBLE, (GOLDEN_ID,))

    assert "CALCULATION_PROVENANCE_MISMATCH" not in result.reason_codes


# --- unauthorized -------------------------------------------------------------


def test_a_fabricated_calculation_reference_is_rejected() -> None:
    """Rejected by this invariant, not by a later generic failure."""

    result = _validate(ADMISSIBLE, (FABRICATED_ID,))

    assert result.passed is False
    assert "CALCULATION_PROVENANCE_MISMATCH" in result.reason_codes


def test_declaring_extra_calculation_references_is_rejected() -> None:
    """A superset is not authorized either -- the declaration must match."""

    result = _validate(ADMISSIBLE, (GOLDEN_ID, FABRICATED_ID))

    assert "CALCULATION_PROVENANCE_MISMATCH" in result.reason_codes


# --- inadmissible results authorize nothing -----------------------------------


def test_a_blocked_calculation_cannot_authorize_an_id() -> None:
    """H2A-1B: a BLOCKED result has no identity, so it cannot vouch for one.

    The state's authority is unset, exactly as an inadmissible result leaves it,
    and a candidate that nevertheless names an id must be refused -- otherwise
    the identifier for a calculation that never ran could be released.
    """

    result = _validate(BLOCKED, (GOLDEN_ID,))

    assert result.passed is False
    assert "CALCULATION_PROVENANCE_MISMATCH" in result.reason_codes


def test_a_blocked_calculation_and_a_silent_candidate_do_not_mismatch() -> None:
    """Nothing to compare: no authority and no declaration is not a conflict.

    This is the boundary between "declared and unauthorized" and "nothing was
    claimed". The former is refused above; the latter is not this guard's
    business, and treating the two as one is what made the sibling citation
    guard vacuous.
    """

    result = _validate(BLOCKED, ())

    assert "CALCULATION_PROVENANCE_MISMATCH" not in result.reason_codes
