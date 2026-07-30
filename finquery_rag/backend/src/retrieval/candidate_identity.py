"""Stable identity contract for ranked retrieval candidates."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any


class CandidateIdentityError(ValueError):
    """Raised when a candidate cannot be safely identified."""


class CandidateKind(str, Enum):
    BLOCK = "block"
    TABLE_ROW = "table_row"


@dataclass(frozen=True)
class CandidateIdentity:
    schema_version: str
    tenant_id: int
    document_id: str
    kind: CandidateKind
    source_id: str
    collection_id: str | None = None


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateIdentityError(f"{field} is required")
    return value.strip()


def _tenant(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        raise CandidateIdentityError("tenant_id is required")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CandidateIdentityError("tenant_id is required") from exc


def identity_from_candidate(candidate: dict[str, Any]) -> CandidateIdentity:
    metadata = candidate.get("metadata") or {}
    tenant = _tenant(candidate.get("tenant_id", metadata.get("user_id")))
    document = _text(
        candidate.get("document_id", metadata.get("doc_name")), "document_id"
    )
    block_type = candidate.get("block_type", metadata.get("type", "text"))
    evidence_id = candidate.get("evidence_id", candidate.get("candidate_id", candidate.get("doc_id")))
    if block_type == "table_cell":
        source_id = _text(
            candidate.get("parent_row_id", metadata.get("parent_row_id", candidate.get("parent_id", metadata.get("parent_id")))),
            "parent_row_id",
        )
        kind = CandidateKind.TABLE_ROW
    elif block_type == "table_row":
        source_id = _text(
            candidate.get("row_id", metadata.get("row_id", evidence_id)), "row_id"
        )
        kind = CandidateKind.TABLE_ROW
    else:
        source_id = _text(evidence_id, "evidence_id")
        kind = CandidateKind.BLOCK
    return CandidateIdentity(
        schema_version="candidate-identity/v1",
        tenant_id=tenant,
        document_id=document,
        kind=kind,
        source_id=source_id,
        collection_id=candidate.get("collection_id"),
    )


def candidate_key(identity: CandidateIdentity) -> str:
    payload = {
        "schema_version": identity.schema_version,
        "tenant_id": identity.tenant_id,
        "document_id": identity.document_id,
        "kind": identity.kind.value,
        "source_id": identity.source_id,
        "collection_id": identity.collection_id,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "candidate:v1:" + hashlib.sha256(raw).hexdigest()

