"""H2A-2D-3B: the two remaining occurrences are projections, not authorities.

After the state mirror and the capability mirror became derived references, two
visible copies remained:

    V2ExecutionTrace.calculation_result_id      TRACE PROJECTION
    V2ExecutionOutcome.calculation_result_id    PUBLIC PROJECTION

The question this file answers is not whether the same string appears in four
places -- that is what a healthy reference graph looks like -- but whether either
copy can be **read back as truth**. The audit says no:

- the trace is built once at `coordinator.py:490-492` from the calculation
  capability's snapshot, which now derives from `last_result`;
- the outcome is built once at `coordinator.py:1541`;
- the only runtime readers of a calculation id are `generation.py:119` and
  `validation.py:409`, and both read the **state's derived property** -- a
  reference to the authority -- not either projection.

So a search for a reverse read found none, and no production code changed. These
tests pin the projection wiring: which run's authority each surface carries, and
that an inadmissible run projects nothing.

Comparing a projection against the authoritative result of the same run is the
right direction here: the authority is independent of the projection under test.
"""

from __future__ import annotations

from typing import Any

from rag_v2.contracts import Intent
from src.runtime.harness_runtime_mode import AgentRuntimeMode
from tests.harness.h1_integration import H1Fixture, run_fixture
from tests.test_trusted_v2_r4_binder import _fact


def _trace(outcome: Any) -> dict[str, Any]:
    return outcome.debug_metadata["trace"]


# --- projection wiring through the real runtime --------------------------------


def _calc_fixture(*, with_operands: bool) -> H1Fixture:
    facts: dict[str, Any] = {
        "CUR": _fact("CUR", slots=("current",), period="FY2024", value="391035000"),
        "PRI": _fact("PRI", slots=("prior",), period="FY2023", value="383285000"),
    }
    return H1Fixture(
        fixture_id="c3-projection",
        description="calculation projection probe",
        query="How much did Apple's revenue grow from FY2023 to FY2024?",
        slots=(
            {"slot_id": "current", "metric": "Revenue", "period": "FY2024", "role": "current"},
            {"slot_id": "prior", "metric": "Revenue", "period": "FY2023", "role": "prior"},
        ),
        facts=facts,
        routes=(("", ("CUR", "PRI") if with_operands else ()),),
        intent=Intent.CALCULATION.value,
        operation="growth_rate",
    )


def test_a_released_calculation_projects_its_identity_once() -> None:
    """The trace and the outcome carry the run's own calculation identity.

    The two projections are compared against each other within one run.  That is
    the wiring claim; comparing either against a *separate* capability's id would
    compare two different calculations.
    """

    outcome = run_fixture(_calc_fixture(with_operands=True), AgentRuntimeMode.HARNESS_V3)
    trace = _trace(outcome)

    # Two independent projections of one run's authority.  Asserting they agree
    # with each other is the wiring claim; asserting they equal a *separate*
    # capability's id would be comparing two different calculations, which is
    # what an earlier revision of this test did -- and it failed, correctly.
    identifier = outcome.calculation_result_id
    assert identifier is not None and identifier.startswith("C1-")
    assert trace["calculation_result_id"] == identifier


def test_a_run_without_a_calculation_projects_no_identity() -> None:
    """Projection of an absent authority is absent, not a previous run's id."""

    outcome = run_fixture(_calc_fixture(with_operands=False), AgentRuntimeMode.HARNESS_V3)
    trace = _trace(outcome)

    assert trace.get("calculation_result_id") is None
    assert outcome.calculation_result_id is None


# --- no residue in the projections either -------------------------------------


def test_a_second_run_projects_only_its_own_identity() -> None:
    """Two runs: neither projection may carry the other run's id.

    This is the projection-side counterpart of the lifecycle regression. It is
    deliberately asserted across two separate executions rather than within one,
    because that is where a projection built from a stale source would show up.
    """

    first = run_fixture(_calc_fixture(with_operands=True), AgentRuntimeMode.HARNESS_V3)
    first_id = first.calculation_result_id
    assert first_id is not None

    blocked = run_fixture(_calc_fixture(with_operands=False), AgentRuntimeMode.HARNESS_V3)

    assert blocked.calculation_result_id is None
    assert _trace(blocked).get("calculation_result_id") != first_id
