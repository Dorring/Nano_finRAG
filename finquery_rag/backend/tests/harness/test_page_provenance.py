"""H2A-1C: page provenance, and the validator boundary it revealed.

Two coupled defects, the second hidden by the first:

1. ``EvidencePacketV1`` had no ``page`` field, so the extractor's page value was
   demoted into ``metadata``.  The citation projection reads the top level, so
   every public citation lost its page.
2. Fixing that re-enabled ``CalculationRenderer``'s long-standing
   ``doc-X, p.N`` suffix -- which had never once rendered, because the page was
   never there to render.  ``GV3_NUMERIC_FIDELITY`` then read the ``1`` in
   ``p.1`` as a financial claim the evidence did not support, and failed.

The second is a scope bug, not a formatting bug: numeric fidelity must validate
claim-bearing numbers, and a source locator is not a claim.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1
from rag_v2.generation.validator import (
    _provenance_citations,
    _without_provenance_citations,
)
from src.runtime import trusted_v2_coordinator as coord
from tests.test_trusted_v2_r4_binder import _fact


def _packet(**overrides: Any) -> dict[str, Any]:
    return EvidencePacketV1.from_mapping({**_fact("E1"), **overrides}).to_dict()


def _citation(**overrides: Any) -> dict[str, Any]:
    state = AdaptiveRAGStateV1.new("r", "q")
    state.add_evidence([EvidencePacketV1.from_mapping({**_fact("E1"), **overrides})])
    return coord._structured_citations(state, ["E1"], [])[0]


def _calculation_packet(*operand_pages: Any) -> dict[str, Any]:
    return {
        "calculation_result": {
            "value": "0.0209",
            "operands": [
                {"name": f"op{index}", "value": "391", "page": page,
                 "document_name": "report.pdf", "evidence_chunk_id": f"c{index}"}
                for index, page in enumerate(operand_pages)
            ],
        }
    }


# --- defect 1: authoritative page transport ----------------------------------


def test_the_extractor_page_is_a_first_class_packet_field() -> None:
    """It used to be demoted into ``metadata``, which the projection cannot read."""

    packet = _packet(pdf_page=7)

    assert packet["page"] == 7
    assert packet["page"] == packet["metadata"]["pdf_page"]


def test_the_public_citation_carries_the_evidence_page() -> None:
    """The original defect: pdf_page=7 produced citation page=None."""

    assert _citation(pdf_page=7)["page"] == 7


def test_page_zero_is_a_page_not_an_absence() -> None:
    """``if page:`` would drop it.  Zero is a legitimate first page."""

    assert _packet(page=0)["page"] == 0
    assert _citation(page=0)["page"] == 0


def test_a_missing_page_stays_missing() -> None:
    """Provenance that is absent must read as absent, not as page 1 or 0."""

    packet = _packet(pdf_page=None)

    assert packet["page"] is None
    assert "page" not in _citation(pdf_page=None)


def test_two_evidence_items_keep_their_own_pages() -> None:
    """No last-page or shared-metadata leakage."""

    state = AdaptiveRAGStateV1.new("r", "q")
    state.add_evidence([
        EvidencePacketV1.from_mapping({**_fact("A"), "page": 7}),
        EvidencePacketV1.from_mapping({**_fact("B"), "page": 42}),
    ])

    citations = coord._structured_citations(state, ["A", "B"], [])

    assert {c["evidence_id"]: c.get("page") for c in citations} == {"A": 7, "B": 42}


def test_a_table_derived_fact_keeps_its_page() -> None:
    """Structured facts carry page provenance; turning one into a packet must not drop it."""

    table_fact = {
        **_fact("T1"),
        "page": 12,
        "metadata": {"type": "table", "table_num": "T3", "page": 12},
    }

    packet = EvidencePacketV1.from_mapping(table_fact).to_dict()

    assert packet["page"] == 12


# --- defect 2: provenance numbers are not claims -----------------------------


def test_the_renderer_emits_page_provenance() -> None:
    """The suffix the promotion re-enabled, pinned at its source."""

    from src.finance.calculation_renderer import render_calculation_result
    from src.domain.calculation import (
        CalculationOperation, CalculationResult, CalculationStatus, CalculationOperand,
    )

    result = CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation.DIFFERENCE,
        value=Decimal("8"),
        operands=(
            CalculationOperand(name="current", value=Decimal("391"), page=12,
                               document_name="report.pdf", evidence_chunk_id="c1"),
        ),
    )

    assert "p.12" in render_calculation_result(result)


def test_provenance_locators_are_reconstructed_from_structure() -> None:
    """Not from a pattern: from the same operand fields the renderer read."""

    packet = _calculation_packet(12)

    assert _provenance_citations(packet) == ["report.pdf, p.12"]


def test_a_page_number_is_not_treated_as_a_claim() -> None:
    """The failure this defect produced: 1 appears only as provenance."""

    packet = _calculation_packet(1)
    answer = "Growth Rate: 2.09%\nInputs:\n  - current = 391 … report.pdf, p.1"

    stripped = _without_provenance_citations(answer, packet)

    assert "p.1" not in stripped
    # The claim value is untouched.
    assert "391" in stripped


def test_a_genuine_claim_of_the_same_value_is_still_checked() -> None:
    """The over-suppression risk.  Do not ignore the number 1.

    A page locator and a claimed value can be the same digits; only the locator
    is provenance, and only its exact rendered string is removed.
    """

    packet = _calculation_packet(1)
    answer = "Revenue grew by 1 … report.pdf, p.1"

    stripped = _without_provenance_citations(answer, packet)

    assert "p.1" not in stripped
    assert "grew by 1" in stripped


def test_page_zero_does_not_look_like_a_claim() -> None:
    packet = _calculation_packet(0)

    assert _provenance_citations(packet) == ["report.pdf, p.0"]
    assert "p.0" not in _without_provenance_citations(
        "current = 391 … report.pdf, p.0", packet
    )


def test_the_same_value_in_two_roles_is_distinguished_by_role() -> None:
    """A claimed 391 and a page 391 must not be confused.

    The exclusion is positional and derived from the operand's page field, so it
    removes the locator and leaves the claim -- whatever the numbers are.
    """

    packet = _calculation_packet(391)
    answer = "current = 391 … report.pdf, p.391"

    stripped = _without_provenance_citations(answer, packet)

    assert "p.391" not in stripped
    assert "current = 391" in stripped


def test_no_page_means_no_provenance_string_to_remove() -> None:
    """Missing provenance must not invent a locator."""

    packet = _calculation_packet(None)

    # The document is still cited -- only the page is absent, and absent must
    # not become a locator.  No "p." fragment is invented for the renderer to
    # have written, so there is nothing of that kind to remove.
    assert _provenance_citations(packet) == ["report.pdf"]
    assert not any("p." in locator for locator in _provenance_citations(packet))
    assert _without_provenance_citations("current = 391", packet) == "current = 391"


# --- the real production path ------------------------------------------------


def test_a_real_calculation_still_releases_with_page_provenance() -> None:
    """The end-to-end regression this defect broke, and whose fix restores it."""

    from src.runtime import V2ExecutionStatus
    from src.runtime.harness_runtime_mode import AgentRuntimeMode
    from tests.harness.h1_integration import FIXTURES, run_fixture

    fixture = next(f for f in FIXTURES if f.fixture_id == "calculation_growth_rate")
    outcome = run_fixture(fixture, AgentRuntimeMode.HARNESS_V3)

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert outcome.validator_status == "PASS"


def test_the_real_citation_carries_the_trusted_evidence_page() -> None:
    """Public citation page must equal the page on the evidence it came from."""

    from src.runtime.harness_runtime_mode import AgentRuntimeMode
    from tests.harness.h1_integration import FIXTURES, run_fixture

    fixture = next(f for f in FIXTURES if f.fixture_id == "calculation_growth_rate")
    outcome = run_fixture(fixture, AgentRuntimeMode.HARNESS_V3)

    assert outcome.citations, "the fixture releases with citations"
    assert all(citation.get("page") is not None for citation in outcome.citations)
