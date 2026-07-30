import pytest

from src.evaluation.frozen_candidate_integrity import (
    FrozenArtifactIntegrityError,
    RankedEvidenceCandidate,
    candidate_content_hash,
    final_context_hash,
)
from src.retrieval.candidate_identity import CandidateIdentity, CandidateKind


def _candidate(source_id="e-1", content="Revenue was 42.2 million."):
    return RankedEvidenceCandidate(
        identity=CandidateIdentity("candidate-identity/v1", 1, "report.pdf", CandidateKind.BLOCK, source_id),
        page=1,
        block_type="text",
        content=content,
    )


def test_content_hash_normalizes_line_endings():
    assert candidate_content_hash(_candidate(content="a\r\nb")) == candidate_content_hash(_candidate(content="a\nb"))


def test_final_context_hash_depends_on_order():
    first, second = _candidate("e-1"), _candidate("e-2")
    tail = [_candidate(f"e-{index}") for index in range(3, 6)]
    assert final_context_hash([first, second, *tail]) != final_context_hash([second, first, *tail])


def test_final_context_requires_exactly_five_candidates():
    with pytest.raises(FrozenArtifactIntegrityError):
        final_context_hash([_candidate()])

