"""H2A-2D-2A: page lineage, proved from the source rather than from agreement.

The page test that existed before this file asserted:

    packet["page"] == packet["metadata"]["pdf_page"]

That proves two stored copies agree. It cannot prove either one is *right*, and
it cannot survive the removal of the duplicate it depends on -- so it was pinning
the duplication rather than guarding it. It remains as a compatibility check; it
is no longer the correctness guard.

Everything here derives its expected page from an authored literal that came in
through the source mapping. The decisive test is the adversarial one: the
metadata is given a **wrong** page on purpose, and the runtime must still report
the canonical one. That is the property that has to hold before the metadata is
demoted, because it is the property that makes the demotion safe.
"""

from __future__ import annotations

from typing import Any

from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1
from rag_v2.evidence.binder_fact_view import build_runtime_binder_fact_view
from src.runtime import trusted_v2_coordinator as coord
from tests.test_trusted_v2_r4_binder import _fact

#: The wrong page.  Deliberately not a plausible neighbour of the real one, so a
#: test that accidentally passes cannot be passing by coincidence.
SENTINEL = 42


def _mapping(**overrides: Any) -> dict[str, Any]:
    """An authored source mapping, as an extractor would emit it."""

    return {**_fact("E1"), **overrides}


def _packet(**overrides: Any) -> dict[str, Any]:
    return EvidencePacketV1.from_mapping(_mapping(**overrides)).to_dict()


def _citation(**overrides: Any) -> dict[str, Any]:
    state = AdaptiveRAGStateV1.new("r", "q")
    state.add_evidence([EvidencePacketV1.from_mapping(_mapping(**overrides))])
    return coord._structured_citations(state, ["E1"], [])[0]


def _poisoned(**overrides: Any) -> dict[str, Any]:
    """A packet whose metadata bag carries a page that is not the real one."""

    packet = _packet(**overrides)
    packet["metadata"]["page"] = SENTINEL
    packet["metadata"]["pdf_page"] = SENTINEL
    return packet


# --- lineage from the source --------------------------------------------------


def test_the_authored_page_arrives_unchanged() -> None:
    """`pdf_page = 7` in, `7` out -- through the canonical field, not metadata."""

    assert _packet(pdf_page=7)["page"] == 7


def test_the_public_citation_reports_the_authored_page() -> None:
    assert _citation(pdf_page=7)["page"] == 7


def test_the_page_is_not_renumbered() -> None:
    """Physical, as the extractor recorded it -- no logical-page invention."""

    for authored in (0, 1, 7, 113):
        assert _packet(pdf_page=authored)["page"] == authored, authored


# --- the decisive property: the canonical page wins ---------------------------


def test_a_wrong_page_in_metadata_cannot_override_the_canonical_page() -> None:
    """The test this phase exists for.

    The canonical field says 7 and the metadata says 42. Every runtime reader
    must see 7. If this can fail, demoting metadata is not safe yet -- and if it
    passes, it stays passing only while the readers take page from the canonical
    field.
    """

    assert _poisoned(pdf_page=7)["page"] == 7


def test_the_public_citation_ignores_a_wrong_page_in_metadata() -> None:
    """End to end: the released citation cannot be poisoned."""

    packet = EvidencePacketV1.from_mapping(_mapping(pdf_page=7)).to_dict()
    packet["metadata"]["page"] = SENTINEL
    packet["metadata"]["pdf_page"] = SENTINEL

    state = AdaptiveRAGStateV1.new("r", "q")
    state.add_evidence([EvidencePacketV1.from_mapping(packet)])
    citation = coord._structured_citations(state, ["E1"], [])[0]

    assert citation["page"] == 7


def test_the_binder_fact_view_ignores_a_wrong_page_in_metadata() -> None:
    """The one live reader that could still reach metadata does not.

    `build_runtime_binder_fact_view` resolves fields across `[fact,
    structural_context, metadata]` and continues past ``None``, so an absent
    canonical page would let the metadata win. With the canonical page present
    it must not, and the adversarial value below is what proves it.
    """

    assert build_runtime_binder_fact_view(_poisoned(pdf_page=7)).get("page") == 7


def test_a_disagreeing_legacy_pair_resolves_to_the_canonical_name() -> None:
    """`page` and `pdf_page` in one mapping, disagreeing.

    `page` is the fact store's canonical name and wins; `pdf_page` is the
    extractor's name and is the fallback. The chosen value is the authored
    canonical one, not the other.
    """

    packet = _packet(page=7, pdf_page=SENTINEL)
    assert packet["page"] == 7


def test_an_absent_canonical_page_is_not_resurrected_from_metadata() -> None:
    """Missing stays missing -- the demotion's actual content.

    The binder ladder resolves across `[fact, structural_context, metadata]` and
    continues past ``None``, so with no canonical page it will take whatever the
    metadata bag offers. That makes the metadata a second authority on the live
    binder path.

    No current producer creates this shape: the canonical page is derived from
    the same two raw keys the bag retains, so whenever the bag has a page the
    canonical has it too. But the ladder does not know that, and a provenance
    guard that holds only because no producer has yet emitted a different shape
    is not a guard -- it is a coincidence. This is the case 2D-2B closes.
    """

    raw = _mapping()
    raw.pop("pdf_page", None)
    raw.pop("page", None)
    packet = EvidencePacketV1.from_mapping(raw).to_dict()
    packet["metadata"]["page"] = SENTINEL
    packet["metadata"]["pdf_page"] = SENTINEL

    assert build_runtime_binder_fact_view(packet).get("page") is None


# --- the ingest boundary ------------------------------------------------------


def test_a_legacy_mapping_with_only_pdf_page_is_normalised_once() -> None:
    """The one place the two names are reconciled.

    A legacy producer that only knows `pdf_page` is converted at the ingestion
    boundary, and every runtime reader afterwards consumes `.page`. That is
    what makes the boundary explicit rather than a second authority.
    """

    raw = _mapping()
    raw.pop("pdf_page", None)
    raw["pdf_page"] = 9

    packet = EvidencePacketV1.from_mapping(raw).to_dict()
    assert packet["page"] == 9


def test_an_absent_page_absent_from_the_mapping_stays_absent() -> None:
    """Provenance that is missing reads as missing, not as page 1."""

    raw = _mapping()
    raw.pop("pdf_page", None)
    raw.pop("page", None)

    assert EvidencePacketV1.from_mapping(raw).to_dict()["page"] is None


def test_an_explicit_none_page_does_not_discard_a_present_pdf_page() -> None:
    """The `dict.get(k, default)` trap, pinned from the ingest side.

    A producer emitting both keys with `page=None` would lose a present
    `pdf_page` if the fallback used `get` with a default instead of testing for
    `None`.
    """

    packet = _packet(page=None, pdf_page=7)
    assert packet["page"] == 7


# --- edges the earlier phase established --------------------------------------


def test_page_zero_survives() -> None:
    assert _packet(pdf_page=0)["page"] == 0
    assert _citation(pdf_page=0)["page"] == 0


def test_two_sources_do_not_cross_contaminate() -> None:
    """Corroboration must not blur two provenance records into one.

    H2A-2C made a slot keep several supports; each still carries its own page,
    and the citation projection must report each one against its own source.
    """

    state = AdaptiveRAGStateV1.new("r", "q")
    state.add_evidence(
        [
            EvidencePacketV1.from_mapping(_mapping(pdf_page=7)),
            EvidencePacketV1.from_mapping({**_mapping(), "evidence_id": "E2", "fact_id": "E2", "pdf_page": 113}),
        ]
    )
    citations = {item["evidence_id"]: item["page"] for item in coord._structured_citations(state, ["E1", "E2"], [])}

    assert citations == {"E1": 7, "E2": 113}
