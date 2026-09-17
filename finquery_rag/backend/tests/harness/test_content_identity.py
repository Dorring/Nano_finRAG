"""H2A-1D: evidence content identity, apart from evidence instance identity.

The consumer audit (`docs/architecture/nf-v3-h2a-1d-content-identity-audit.md`)
established that ``EvidencePacketV1.content_hash`` has exactly one functional
reader -- ``ProgressDetector.signature`` -- and has not escaped into any
persisted, sealed, cached or published contract.  So its semantics are corrected
in place rather than shadowed by a parallel field.

What was wrong: the hash covered ``to_dict()``, which includes ``evidence_id``.
Two rounds returning the same content under a new id therefore hashed
differently, and the loop read that as new information.

Three identities stay separate, which is the point of the whole phase:

    evidence_id          instance    which runtime object is this
    content_fingerprint  semantic    is this the same evidence
    page / document_id   provenance  where did it come from

Provenance must not enter the fingerprint.  Two documents reporting the same
figure are the same content and different provenance -- and that distinction is
what the H2A-1E conflict gate counts as corroboration.
"""

from __future__ import annotations

from typing import Any

from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1
from rag_v2.adaptive.adaptive_progress import ProgressDetectorV1
from tests.test_trusted_v2_r4_binder import _fact


def _packet(fact_id: str, **overrides: Any) -> EvidencePacketV1:
    return EvidencePacketV1.from_mapping(
        {**_fact(fact_id, slots=("revenue",), value="100"), **overrides}
    )


def _signature(detector: ProgressDetectorV1, packets: list[EvidencePacketV1]) -> str:
    return detector.signature(
        query="What was revenue?",
        capability="SEMANTIC_RETRIEVAL",
        packets=packets,
        filled_slots=(),
        missing_slots=(),
        conflicts=(),
        calculation_ready=False,
    )


# --- the defect --------------------------------------------------------------


def test_the_same_content_under_a_new_id_is_not_progress() -> None:
    """Three rounds, one fact, three evidence ids: the loop is not learning.

    Before this, every round reported progress and the run spent budget
    re-reading evidence it already had.
    """

    detector = ProgressDetectorV1()

    signatures = [
        _signature(detector, [_packet(fact_id)])
        for fact_id in ("E-1", "E-2", "E-3")
    ]

    assert len(set(signatures)) == 1, "content identity must not follow the instance id"


def test_the_detector_then_reports_no_progress() -> None:
    """The consequence, stated as the loop sees it."""

    detector = ProgressDetectorV1()
    state = AdaptiveRAGStateV1.new("r", "What was revenue?")

    first = _signature(detector, [_packet("E-1")])
    second = _signature(detector, [_packet("E-2")])

    assert detector.observe(state, first) is True
    assert detector.observe(state, second) is False


def test_genuinely_different_content_is_still_progress() -> None:
    """Do not fix this by collapsing everything to one signature."""

    detector = ProgressDetectorV1()
    state = AdaptiveRAGStateV1.new("r", "What was revenue?")

    first = _signature(detector, [_packet("E-1")])
    other = _signature(detector, [_packet("E-2", value="999")])

    assert detector.observe(state, first) is True
    assert detector.observe(state, other) is True


def test_a_changed_metric_or_period_is_different_content() -> None:
    for field, value in (("metric", "net_income"), ("period", "FY2023"), ("unit", "EUR")):
        left = _packet("A")
        right = EvidencePacketV1.from_mapping(
            {**_fact("A", slots=("revenue",), value="100"), field: value}
        )
        assert left.content_fingerprint != right.content_fingerprint, field


# --- provenance is not content -----------------------------------------------


def test_page_and_document_do_not_change_content_identity() -> None:
    """The distinction the conflict gate depends on.

    Two sources reporting the same figure are corroboration, not one source.
    A fingerprint that folded in the page or the document would make them look
    like the same evidence, and consensus counting works by keeping them apart.
    """

    base = _packet("A")
    moved = EvidencePacketV1.from_mapping(
        {**_fact("A", slots=("revenue",), value="100"), "page": 42, "document_id": "other.pdf"}
    )

    assert base.content_fingerprint == moved.content_fingerprint


def test_the_instance_id_is_not_content_identity() -> None:
    a = _packet("A", evidence_id="A")
    b = _packet("B", evidence_id="B")

    assert a.content_fingerprint == b.content_fingerprint
    assert a.evidence_id != b.evidence_id


def test_the_fingerprint_is_not_the_serialized_record() -> None:
    """The old hash covered ``to_dict()``, so it was a record hash.

    Pinned by showing a field that is *not* semantic content -- the citation id
    and the metadata bag -- cannot move the fingerprint.
    """

    base = _packet("A")
    decorated = EvidencePacketV1.from_mapping(
        {
            **_fact("A", slots=("revenue",), value="100"),
            "citation_id": "citation-OTHER",
            "metadata": {"anything": "at all"},
        }
    )

    assert base.content_fingerprint == decorated.content_fingerprint
