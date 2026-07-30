"""Content-addressed candidate integrity helpers for NF39 R2."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Any

from src.retrieval.candidate_identity import CandidateIdentity, candidate_key


class FrozenArtifactIntegrityError(ValueError):
    """Raised when a frozen evaluation artifact is incomplete or inconsistent."""


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_rendered_bytes(rendered: str) -> bytes:
    return unicodedata.normalize("NFC", rendered).replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


@dataclass(frozen=True)
class RankedEvidenceCandidate:
    identity: CandidateIdentity
    page: int | None
    block_type: str
    content: str
    content_format: str = "text/plain"
    parent_id: str | None = None
    table_id: str | None = None
    section_path: tuple[str, ...] = ()
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None

    @property
    def candidate_key(self) -> str:
        return candidate_key(self.identity)


def render_candidate_for_context(candidate: RankedEvidenceCandidate) -> str:
    """Render one evidence item using the production context reference format."""
    name = candidate.identity.document_id
    source = f"{name}, p{candidate.page}" if candidate.page is not None else name
    rendered = f"[{source}]\n{candidate.content}"
    if not candidate.content.strip():
        raise FrozenArtifactIntegrityError("Rendered candidate content is empty")
    return rendered


def candidate_content_hash(candidate: RankedEvidenceCandidate) -> str:
    return hashlib.sha256(canonical_rendered_bytes(render_candidate_for_context(candidate))).hexdigest()


def candidate_manifest_row(candidate: RankedEvidenceCandidate) -> dict[str, Any]:
    rendered = render_candidate_for_context(candidate)
    return {
        "candidate_key": candidate.candidate_key,
        "identity": {
            "tenant_id": candidate.identity.tenant_id,
            "document_id": candidate.identity.document_id,
            "kind": candidate.identity.kind.value,
            "source_id": candidate.identity.source_id,
            "collection_id": candidate.identity.collection_id,
        },
        "page": candidate.page,
        "block_type": candidate.block_type,
        "parent_id": candidate.parent_id,
        "table_id": candidate.table_id,
        "content_hash": candidate_content_hash(candidate),
        "content_char_count": len(rendered),
        "content_byte_count": len(canonical_rendered_bytes(rendered)),
        "renderer_schema": "context-renderer/v1",
    }


def final_context_hash(candidates: list[RankedEvidenceCandidate]) -> str:
    if len(candidates) != 5:
        raise FrozenArtifactIntegrityError(f"Expected Final Top-5, got {len(candidates)}")
    entries = [
        {"rank": rank, "candidate_key": candidate.candidate_key, "content_hash": candidate_content_hash(candidate)}
        for rank, candidate in enumerate(candidates, 1)
    ]
    return hashlib.sha256(stable_json_bytes(entries)).hexdigest()


def validate_candidates(candidates: list[RankedEvidenceCandidate], *, expected_count: int, label: str) -> None:
    if len(candidates) != expected_count:
        raise FrozenArtifactIntegrityError(f"{label}: expected {expected_count} candidates, got {len(candidates)}")
    keys = [candidate.candidate_key for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise FrozenArtifactIntegrityError(f"{label}: duplicate candidate keys")
    for candidate in candidates:
        if not candidate_content_hash(candidate):
            raise FrozenArtifactIntegrityError(f"{label}: missing content hash")


def validate_rankings(rankings: dict[str, list[RankedEvidenceCandidate]], *, expected_cases: int, expected_count: int, label: str) -> None:
    if len(rankings) != expected_cases:
        raise FrozenArtifactIntegrityError(f"{label}: expected {expected_cases} cases, got {len(rankings)}")
    for case_id, candidates in rankings.items():
        validate_candidates(candidates, expected_count=expected_count, label=case_id)

