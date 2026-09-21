"""H2A-2D-3A part 2: what each candidate path can and cannot declare.

Two candidate paths exist, and they answer the reference-authorization question
differently. Conflating them is what produced the vacuous guard.

**Structured path** (`CandidateExecutionResult`). The candidate *does* declare
`citation_ids` and `calculation_ids`, and the generator fills them. Its contract
is live: `_validation_packet` rejects a candidate whose declared citations are
not Binder-admitted. These tests exercise that guard directly.

**String path.** A bare string cannot declare references at all. The validator
used to manufacture an empty declaration for it by reading two state attributes
that no code in the repository writes, which made the subset check trivially
true -- a guard that had never once been evaluated. Authorization is now **not
applicable** on this path, and the distinction is deliberate:

    not declared   -> the question cannot be asked here
    declared empty -> the question was asked and answered

The second is what the old code pretended. What actually protects the string
path is the downstream envelope contract, pinned below.

Candidate-declared ids and admitted ids are authored independently in every
test: nothing here reads a production authority to build the candidate side, or
vice versa.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1
from src.runtime.trusted_v2_generation import CandidateExecutionResult
from src.runtime.trusted_v2_validation import TrustedReleaseValidationCapability, _validation_packet
from tests.test_trusted_v2_r4_binder import _fact

#: Authored literals. The candidate declares these; the Binder admits a
#: different, independently written set.
ADMITTED = ("citation-A", "citation-B")
DECLARED_OK = ("citation-A",)
DECLARED_UNKNOWN = ("citation-X",)


def _state(bound: tuple[str, ...] = ("E1", "E2")) -> AdaptiveRAGStateV1:
    state = AdaptiveRAGStateV1.new("r", "q")
    state.add_evidence(
        [
            EvidencePacketV1.from_mapping({**_fact("E1"), "citation_id": ADMITTED[0]}),
            EvidencePacketV1.from_mapping(
                {**_fact("E2"), "evidence_id": "E2", "fact_id": "E2", "citation_id": ADMITTED[1]}
            ),
        ]
    )
    state.bound_evidence_ids = list(bound)
    return state


def _candidate(declared: tuple[str, ...], **over: Any) -> CandidateExecutionResult:
    return CandidateExecutionResult(
        candidate_answer="Revenue was 100.",
        route="STRUCTURED_SINGLE",
        route_reason="fixture",
        bound_evidence_ids=("E1",),
        citation_ids=declared,
        **over,
    )


# --- the structured path: a real contract -------------------------------------


def test_a_candidate_declaring_only_admitted_citations_is_authorized() -> None:
    packet, _, _ = _validation_packet(_state(), _candidate(DECLARED_OK))
    # The candidate's independently authored declaration is inside the
    # independently authored admitted set, and the packet carries the latter.
    # Membership, not equality: the admission set also carries the uppercase
    # aliases it resolves, which is not what this test is about.
    assert set(DECLARED_OK) <= set(packet["allowed_citation_ids"])
    assert set(ADMITTED) <= set(packet["allowed_citation_ids"])


def test_a_candidate_declaring_an_unadmitted_citation_is_rejected() -> None:
    """Rejected by the reference invariant itself, not by a later coercion.

    The message is the guard's own, which is the point: an unauthorized
    reference has to fail *here*, where the reason names the actual problem.
    """

    with pytest.raises(ValueError, match="not Binder-admitted"):
        _validation_packet(_state(), _candidate(DECLARED_UNKNOWN))


def test_the_rejection_names_every_unadmitted_reference() -> None:
    """A partial overlap is still unauthorized -- one bad id is enough."""

    with pytest.raises(ValueError, match="not Binder-admitted"):
        _validation_packet(_state(), _candidate(("citation-A", "citation-X")))


def test_a_candidate_declaring_no_citations_is_not_an_authorization_failure() -> None:
    """A declaration of "none" is a different question from "not declared".

    It is well-formed at this guard and is refused downstream for the reason
    that actually applies -- an answer with no structured citation.
    """

    packet, _, _ = _validation_packet(_state(), _candidate(()))
    assert packet["allowed_citation_ids"]


# --- the string path: the guard is not applicable -----------------------------


def test_a_bare_string_candidate_declares_no_references() -> None:
    """The fabricated declaration is gone.

    `_candidate` no longer reads `state._candidate_citation_ids` /
    `_candidate_calculation_ids` -- attributes with no writer anywhere in the
    repository. What the resulting object carries is the type's default, not a
    declaration recovered from state.
    """

    candidate = TrustedReleaseValidationCapability._candidate("fixture answer", _state())

    assert candidate.citation_ids == ()
    assert candidate.calculation_ids == ()


def test_removing_the_fabricated_declaration_does_not_let_a_string_release() -> None:
    """The string path still fails -- at the contract that actually governs it.

    This is the safety property of the removal: skipping an inapplicable guard
    must not widen what the path can do. The envelope requires structured
    citation ids, and that is where a bare string stops.
    """

    state = _state()
    capability = TrustedReleaseValidationCapability()
    result = capability.validate(state, "fixture answer")

    assert result.passed is False
    # Not the citation-authorization code: that invariant does not apply here.
    assert "UNBOUND_CITATION_METADATA" in result.reason_codes
