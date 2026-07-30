import pytest

from src.retrieval.candidate_identity import (
    CandidateIdentityError,
    candidate_key,
    identity_from_candidate,
)


def _candidate(**values):
    base = {"tenant_id": 1, "document_id": "report.pdf", "evidence_id": "e-1", "block_type": "text"}
    base.update(values)
    return base


def test_empty_document_id_is_rejected():
    with pytest.raises(CandidateIdentityError):
        identity_from_candidate(_candidate(document_id=""))


def test_table_cell_uses_parent_row_identity():
    identity = identity_from_candidate(_candidate(block_type="table_cell", parent_row_id="row-7"))
    assert identity.kind.value == "table_row"
    assert identity.source_id == "row-7"


def test_candidate_key_is_stable_and_never_uses_empty_block_key():
    identity = identity_from_candidate(_candidate())
    assert candidate_key(identity) == candidate_key(identity)
    assert candidate_key(identity).startswith("candidate:v1:")
    assert "block::" not in candidate_key(identity)

