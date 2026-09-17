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


# --- H2A-2B: quantity identity, and text identity, kept apart -----------------


def test_the_same_quantity_written_two_ways_is_not_progress() -> None:
    """``1 billion`` and ``1000 million`` are one figure, not two reports.

    This is the round-two case the progress detector gets wrong when the
    magnitude is compared as a label: the run answers the retriever again with
    the same quantity in different units, and reads it as new information.
    """

    detector = ProgressDetectorV1()
    state = AdaptiveRAGStateV1.new("r", "What was revenue?")

    billion = _packet("A", value="1", scale="billion")
    million = _packet("B", value="1000", scale="million")

    assert billion.content_fingerprint == million.content_fingerprint
    assert detector.observe(state, _signature(detector, [billion])) is True
    assert detector.observe(state, _signature(detector, [million])) is False


def test_a_changed_magnitude_is_still_progress() -> None:
    """Do not fix the above by canonicalising everything to one constant."""

    detector = ProgressDetectorV1()
    state = AdaptiveRAGStateV1.new("r", "What was revenue?")

    first = _packet("A", value="1", scale="billion")
    other = _packet("B", value="1.2", scale="billion")

    assert first.content_fingerprint != other.content_fingerprint
    assert detector.observe(state, _signature(detector, [first])) is True
    assert detector.observe(state, _signature(detector, [other])) is True


def test_a_ratio_is_not_its_percentage_form_here_either() -> None:
    """The representation distinction survives into the fingerprint."""

    ratio = _packet("A", value="0.12", unit="ratio")
    percent = _packet("B", value="12", unit="%")

    assert ratio.content_fingerprint != percent.content_fingerprint


def test_a_comma_in_a_textual_field_is_part_of_the_claim() -> None:
    """``Revenue, net`` and ``Revenue net`` are different metrics.

    The helper this replaced removed every comma, which is right for a digit
    grouping separator and wrong for every other comma -- and it could not tell
    which it had.
    """

    left = _packet("A", metric="Revenue, net")
    right = _packet("B", metric="Revenue net")

    assert left.content_fingerprint != right.content_fingerprint


def test_a_malformed_number_is_not_guessed_into_a_different_one() -> None:
    """``1,5`` is not ``15``.  Neither is chosen; they stay different."""

    ambiguous = _packet("A", value="1,5", scale=None)
    comma_stripped = _packet("B", value="15", scale=None)

    assert ambiguous.content_fingerprint != comma_stripped.content_fingerprint


def test_a_grouping_separator_is_not_a_difference() -> None:
    assert _packet("A", value="1,500", scale=None).content_fingerprint == _packet(
        "B", value="1500", scale=None
    ).content_fingerprint


def test_every_content_identity_field_is_still_in_the_fingerprint() -> None:
    """The two halves must partition the fields the fingerprint claims to cover.

    A field missing from both would be ignored silently, which is how a
    fingerprint stops noticing a dimension of the claim it is supposed to
    identify.
    """

    from rag_v2.adaptive.adaptive_contracts import (
        CONTENT_IDENTITY_FIELDS,
        QUANTITY_IDENTITY_FIELDS,
        TEXT_IDENTITY_FIELDS,
    )

    assert set(TEXT_IDENTITY_FIELDS) | set(QUANTITY_IDENTITY_FIELDS) == set(
        CONTENT_IDENTITY_FIELDS
    )
    assert set(TEXT_IDENTITY_FIELDS) & set(QUANTITY_IDENTITY_FIELDS) == set()
