"""H2A-2C-3: multi-support binding survives every hop to the public surface.

H2A-2C-2 made a slot keep every independent support of its canonical value.  The
question this file answers is whether anything downstream quietly puts it back
to one -- the failure mode that would make the binding change cosmetic.

The answer, established by tracing rather than by reading:

    binding            slot -> (X1, X2)          both independent sources
    claim provenance   slot:revenue -> (X1, X2)  both, with both citations
    public citations   one entry per support     both, with per-support page
    calculation        one operand per slot      run once, on one value
    answer text        one value, one inline citation

The last line is the only place a single id appears, and it is not a loss: the
renderer states one quantity, and `citations` and `claim_provenance` are the
authoritative, structured surfaces.  Two sources supporting one figure is one
claim with two witnesses, not two claims -- rendering the value twice would be
the defect this file exists to rule out.

Expected support sets come from the sealed gold and from the authored fixtures.
Nothing here is read back out of the production binding or generation helpers.
"""

from __future__ import annotations

from typing import Any

from rag_v2.contracts import Intent
from src.runtime.harness_runtime_mode import AgentRuntimeMode
from tests.benchmark.tv2_readiness_cases import cases
from tests.harness.h1_integration import H1Fixture, run_fixture
from tests.test_trusted_v2_r4_binder import _fact


def _run(case_id: str) -> Any:
    for case in cases():
        if case.case_id == case_id:
            return case, run_fixture(case.fixture, AgentRuntimeMode.HARNESS_V3)
    raise AssertionError(f"no sealed case {case_id!r}")


def _trace(outcome: Any) -> dict[str, Any]:
    return outcome.debug_metadata["trace"]


# --- the two readiness targets ------------------------------------------------


def test_cross_source_keeps_both_independent_sources() -> None:
    """Two documents, one figure.  The gold names both; so must every surface."""

    case, outcome = _run("cross-source-001")
    expected = list(case.expected_evidence_ids)
    assert expected == ["X1", "X2"]

    assert _trace(outcome)["bound_evidence_ids"] == expected
    assert outcome.evidence_ids == expected
    assert outcome.citation_ids == list(case.expected_citation_ids)
    assert outcome.route == "MULTI"

    slots = {claim.claim_id: claim for claim in outcome.claim_provenance}
    assert slots["slot:revenue"].bound_evidence_ids == tuple(expected)
    assert slots["slot:revenue"].citation_ids == tuple(case.expected_citation_ids)


def test_multi_evidence_keeps_both_representation_equivalent_sources() -> None:
    """`1 million` and `1000 thousand` are one figure, and both are kept.

    This case is also the 2B integration proof: if the two supports were not
    canonicalised to one quantity the binding would see a conflict and abstain,
    so a release here means the scale canonicalisation is doing the work.
    """

    case, outcome = _run("multi-evidence-001")
    expected = list(case.expected_evidence_ids)
    assert expected == ["Q1", "Q2"]

    assert _trace(outcome)["bound_evidence_ids"] == expected
    assert outcome.evidence_ids == expected
    assert outcome.citation_ids == list(case.expected_citation_ids)
    assert outcome.route == "MULTI"

    slots = {claim.claim_id: claim for claim in outcome.claim_provenance}
    assert slots["slot:revenue"].bound_evidence_ids == tuple(expected)


def test_a_corroborated_value_is_rendered_once() -> None:
    """Two witnesses, one number.

    The assertion is deliberately about how many times the figure appears, not
    about the text: rendering `100` twice would be a second claim, and a
    consumer reading the answer would have no way to tell corroboration from
    two disagreeing figures that happen to match.
    """

    case, outcome = _run("cross-source-001")
    assert outcome.answer is not None
    assert outcome.answer.count("100") == 1, outcome.answer


# --- ordering and identity ----------------------------------------------------


def test_each_support_keeps_its_own_source_location() -> None:
    """Corroboration must not blur into one provenance record.

    Two supports are two places the figure was found.  Collapsing them into one
    citation would leave the released answer citing a source it may not have
    read -- the provenance defect H2A-1C fixed for a single support.
    """

    _, outcome = _run("cross-source-001")
    by_evidence = {item["evidence_id"]: item for item in outcome.citations}

    assert set(by_evidence) == {"X1", "X2"}
    assert by_evidence["X1"]["citation_id"] == "citation-X1"
    assert by_evidence["X2"]["citation_id"] == "citation-X2"


def test_a_losing_candidate_never_reaches_provenance() -> None:
    """The exclusion side of the same contract.

    `multi-evidence-002` supplies two candidates for one slot that disagree, so
    nothing is bound -- and an empty provenance set is what "nothing was
    admitted" has to look like.  If a losing candidate could appear here, the
    released citations would name evidence the binding refused.
    """

    case, outcome = _run("multi-evidence-002")
    assert case.expected_release is False

    assert outcome.evidence_ids == []
    assert outcome.citation_ids == []
    assert _trace(outcome)["bound_evidence_ids"] == []
    assert all(not claim.bound_evidence_ids for claim in outcome.claim_provenance)


# --- calculation --------------------------------------------------------------

CALC_FIXTURE = H1Fixture(
    fixture_id="c3-calc-multi-support",
    description="One operand carried by two independent sources.",
    query="How much did Apple's revenue grow from FY2023 to FY2024?",
    slots=(
        {"slot_id": "current", "metric": "Revenue", "period": "FY2024", "role": "current"},
        {"slot_id": "prior", "metric": "Revenue", "period": "FY2023", "role": "prior"},
    ),
    # CUR-1 and CUR-2 state the same quantity at different scales from distinct
    # physical sources; PRI-1 is the other operand's single witness.
    facts={
        "CUR-1": _fact("CUR-1", slots=("current",), period="FY2024", value="1"),
        "CUR-2": {
            **_fact("CUR-2", slots=("current",), period="FY2024", value="1000"),
            "scale": "thousand",
        },
        "PRI-1": _fact("PRI-1", slots=("prior",), period="FY2023", value="900"),
    },
    routes=(("", ("CUR-1", "CUR-2", "PRI-1")),),
    intent=Intent.CALCULATION.value,
    operation="growth_rate",
)


def test_a_multi_support_operand_is_computed_once_on_one_canonical_value() -> None:
    """Supports are provenance, not operands.

    The calculator must see one operand for the `current` slot, not two, and the
    value it sees must be the canonical 1,000,000 that both sources state -- so
    the arithmetic cannot change with how many sources agreed.
    """

    outcome = run_fixture(CALC_FIXTURE, AgentRuntimeMode.HARNESS_V3)

    assert outcome.route == "CALCULATION_SIMPLE"
    assert _trace(outcome)["bound_evidence_ids"] == ["CUR-1", "CUR-2", "PRI-1"]

    assert len(outcome.calculations) == 1, "the calculator ran more than once"
    operands = outcome.calculations[0]["operands"]
    assert len(operands) == 2, "one operand per required slot, not one per support"
    by_role = {str(op["name"]): op for op in operands}
    # The operation names its operands, not the plan's slot roles.
    assert set(by_role) == {"current", "previous"}

    # 1 million and 1000 thousand are the same figure; either support yields it.
    assert str(by_role["current"]["value"]) in {"1000000", "1000000.0", "1E+6"}


def test_a_multi_support_operand_keeps_all_of_its_supports_in_provenance() -> None:
    """The canonical value is singular; its support is not.

    `_operand_for_slot` picks a representative -- it has to, to reuse the
    existing value extraction -- and that choice cannot change the result
    because the validator proved the group agrees.  What must not happen is the
    representative silently becoming the *only* record the run keeps.
    """

    outcome = run_fixture(CALC_FIXTURE, AgentRuntimeMode.HARNESS_V3)
    slots = {claim.claim_id: claim for claim in outcome.claim_provenance}

    assert slots["slot:current"].bound_evidence_ids == ("CUR-1", "CUR-2")
    assert slots["slot:prior"].bound_evidence_ids == ("PRI-1",)

    calculation_claims = [
        claim for claim in outcome.claim_provenance if claim.claim_id.startswith("calculation:")
    ]
    assert calculation_claims, "the calculation published no claim provenance"
    assert set(calculation_claims[0].bound_evidence_ids) == {"CUR-1", "CUR-2", "PRI-1"}


def test_a_multi_support_operand_is_not_rendered_once_per_support() -> None:
    """The value appears once in the answer, with one representative locator."""

    outcome = run_fixture(CALC_FIXTURE, AgentRuntimeMode.HARNESS_V3)
    assert outcome.answer is not None
    assert outcome.answer.count("1,000,000.00") == 1, outcome.answer
