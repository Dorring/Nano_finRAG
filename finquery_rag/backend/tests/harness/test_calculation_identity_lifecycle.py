"""H2A-2D-3B: a calculation id cannot survive the run that produced it.

The state mirror was migrated to a derived property first; this covers the
capability's own `last_calculation_id`, which was the more dangerous of the two.
It was a stored field kept in step with `last_result` by hand -- assigned the id
on success, reset to ``None`` on failure. A paired assignment is correct only
while both halves are remembered every time, and its failure mode is not a wrong
value but a *stale* one: an id from a previous execution surviving into a run
that produced no id at all.

That is why the assertions below run two calculations on **one capability
instance**. A single-run test cannot see the defect, and the existing equality
tests could not either -- they would pass against a mirror that never resets.

Every assertion reads the field under test only *against* something independent:
the result the caller was handed, or the absence of any id at all. The identity
oracle's frozen literal is deliberately not reused -- that vector was authored as
a `CalculationResult` directly, while the calculator derives its own operands and
formula version, so matching it would prove only that two different vectors
coincide.
"""

from __future__ import annotations

from typing import Any

from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1
from rag_v2.contracts import Intent
from src.runtime.trusted_v2_calculation import DeterministicCalculationCapability
from tests.harness.test_calculation_identity_oracle import OPERANDS
from tests.test_trusted_v2_r4_binder import _fact, _plan, _slot


def _state(*, with_operands: bool) -> AdaptiveRAGStateV1:
    """A calculation state, with or without resolvable operands."""

    state = AdaptiveRAGStateV1.new("r", "q")
    current, previous = OPERANDS[0], OPERANDS[1]
    packets = [
        EvidencePacketV1.from_mapping(
            {**_fact("CUR", slots=("current",), period="FY2024", value=str(current[1])),
             "citation_id": f"citation-{current[2]}"}
        ),
        EvidencePacketV1.from_mapping(
            {**_fact("PRI", slots=("prior",), period="FY2023", value=str(previous[1])),
             "evidence_id": "PRI", "fact_id": "PRI",
             "citation_id": f"citation-{previous[2]}"}
        ),
    ]
    state.add_evidence(packets)
    state.plan = {
        "supervisor_plan": _plan(
            _slot("current", period="FY2024", role="current"),
            _slot("prior", period="FY2023", role="prior"),
            intent=Intent.CALCULATION,
            operation="growth_rate",
        ).to_dict()
    }
    if with_operands:
        state.bound_evidence_ids = ["CUR", "PRI"]
        state.bound_slot_bindings = {"current": ["CUR"], "prior": ["PRI"]}
    return state


def _capability() -> DeterministicCalculationCapability:
    return DeterministicCalculationCapability()


def _run(capability: Any, *, with_operands: bool) -> Any:
    return capability.calculate(_state(with_operands=with_operands))


# --- the identity follows the result ------------------------------------------


def test_a_successful_calculation_exposes_an_identity() -> None:
    """The identity oracle's frozen literal is not reused here.

    That vector was authored directly as a `CalculationResult`; the calculator
    builds its own operands and `formula_version`, so it produces a different --
    equally valid -- identity. Reusing the literal would only be testing that two
    different vectors coincide.
    """

    capability = _capability()
    result = _run(capability, with_operands=True)

    identifier = result.calculation_id
    assert identifier is not None and identifier.startswith("C1-")
    # The mirror agrees with the authority the caller was handed -- the result
    # is independent of the field under test.
    assert capability.last_calculation_id == identifier


def test_a_failed_calculation_exposes_no_identity() -> None:
    capability = _capability()
    result = _run(capability, with_operands=False)

    assert result.calculation_id is None
    assert capability.last_calculation_id is None


# --- the lifecycle: no residue across runs ------------------------------------


def test_an_id_does_not_survive_into_a_run_that_produced_none() -> None:
    """The defect a single-run test cannot see.

    Run A succeeds and Run B blocks.  A stored mirror that forgets to reset would
    still be advertising Run A's identity, and a coordinator reading it would
    attach a calculation id to a run whose calculation never happened.
    """

    capability = _capability()

    from_run_a = _run(capability, with_operands=True).calculation_id
    assert from_run_a is not None
    assert capability.last_calculation_id == from_run_a

    blocked = _run(capability, with_operands=False)
    assert blocked.calculation_id is None
    assert capability.last_calculation_id is None


def test_a_new_identity_replaces_the_previous_one() -> None:
    """The other direction: a stale id must not outlive a newer success either."""

    capability = _capability()

    assert _run(capability, with_operands=False).calculation_id is None
    assert _run(capability, with_operands=True).calculation_id is not None

    # Read through the result, not through the field under test.
    assert capability.last_result is not None
    assert capability.last_calculation_id == capability.last_result.calculation_id
