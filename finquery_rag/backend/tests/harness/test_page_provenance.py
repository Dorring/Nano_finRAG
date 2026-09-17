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
from rag_v2.generation.validator import _claim_numbers
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
#
# The invariant: changing a document name or a page locator must not change
# which financial claim numbers are validated.
#
# The first fix for this subtracted the renderer's locator string from the
# answer -- which made the validator a consumer of the renderer's *format*, gave
# it a duplicate of the formatter's logic, and let a data-controlled document
# name delete claim text.  The check now reads the calculation's structured
# values instead, so there is no string to reconstruct and nothing to remove.


def test_the_renderer_still_emits_page_provenance() -> None:
    """The suffix C re-enabled.  Unchanged by C.1: this is about the validator."""

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


def test_a_page_number_never_enters_the_claim_set() -> None:
    """The page is provenance; it is not read at all now, rather than removed."""

    packet = _calculation_packet(12)

    claims = {str(item) for item in _claim_numbers("", packet)}

    assert "12" not in claims
    assert "0.0209" in claims


def test_changing_a_document_name_cannot_change_the_validated_claims() -> None:
    """The invariant, stated directly.

    The old implementation reconstructed a locator from ``document_name`` and
    deleted that exact string from the answer, so a name shaped like claim text
    could remove a claim.  Nothing about the name is read now.
    """

    answers = [
        _calculation_packet(7),
        {**_calculation_packet(7),
         "calculation_result": {**_calculation_packet(7)["calculation_result"],
                                "operands": [{"name": "op0", "value": "391", "page": 7,
                                              "document_name": "revenue 999"}]}},
    ]

    first = [str(item) for item in _claim_numbers("", answers[0])]
    second = [str(item) for item in _claim_numbers("", answers[1])]

    assert first == second


def test_changing_a_page_cannot_change_the_validated_claims() -> None:
    low = [str(item) for item in _claim_numbers("", _calculation_packet(1))]
    high = [str(item) for item in _claim_numbers("", _calculation_packet(99999))]

    assert low == high


def test_page_zero_is_not_a_claim() -> None:
    assert "0" not in {str(item) for item in _claim_numbers("", _calculation_packet(0))}


def test_a_prose_answer_is_still_fully_claim_bearing() -> None:
    """No calculation means no provenance, so the text is unchanged.

    C.1 must not create a blind spot: outside a calculation route every number
    the answer states is still checked.
    """

    assert _claim_numbers("Revenue was 999", {}) == [Decimal("999")]


def test_the_old_string_removal_helpers_are_gone() -> None:
    """They carried the renderer's format into the validator.  No consumer remains."""

    import rag_v2.generation.validator as validator

    assert not hasattr(validator, "_provenance_citations")
    assert not hasattr(validator, "_without_provenance_citations")


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


