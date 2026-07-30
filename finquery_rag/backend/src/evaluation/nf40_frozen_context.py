"""Fail-closed loading of the NF39 R2 local frozen-context snapshot."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.frozen_candidate_integrity import (
    canonical_rendered_bytes,
    stable_json_bytes,
)
from src.application.frozen_evaluation import FrozenEvaluationContext


class FrozenContextError(ValueError):
    """Raised when a frozen NF39 R2 payload cannot be rehydrated exactly."""


@dataclass(frozen=True)
class FrozenContextCandidate:
    case_id: str
    rank: int
    candidate_key: str
    content_hash: str
    rendered_content: str
    document_id: str
    source_id: str
    page: int | None
    block_type: str


@dataclass(frozen=True)
class FrozenCaseContext:
    case_id: str
    candidates: tuple[FrozenContextCandidate, ...]
    final_context_hash: str


def _content_hash(rendered: str) -> str:
    return hashlib.sha256(canonical_rendered_bytes(rendered)).hexdigest()


def _context_hash(candidates: list[FrozenContextCandidate]) -> str:
    entries = [
        {"rank": candidate.rank, "candidate_key": candidate.candidate_key, "content_hash": candidate.content_hash}
        for candidate in candidates
    ]
    return hashlib.sha256(stable_json_bytes(entries)).hexdigest()


def load_frozen_contexts(
    payload_path: Path,
    final_manifest_path: Path,
) -> dict[str, FrozenCaseContext]:
    manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_schema") != "nf39-r2/v1":
        raise FrozenContextError("NF40 requires an NF39 R2 final-context manifest")
    records = [json.loads(line) for line in payload_path.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 135:
        raise FrozenContextError(f"Expected 135 records, got {len(records)}")
    grouped: dict[str, list[FrozenContextCandidate]] = {}
    manifest_cases = manifest.get("cases") or {}
    for row in records:
        manifest_case = manifest_cases.get(row["case_id"])
        if not isinstance(manifest_case, dict):
            raise FrozenContextError(f"{row['case_id']}: missing manifest case")
        candidate_rows = manifest_case.get("candidates") or []
        candidate_manifest = next(
            (
                item for item in candidate_rows
                if item.get("candidate_key") == row["candidate_key"]
                and item.get("rank") == row["rank"]
            ),
            None,
        )
        if not isinstance(candidate_manifest, dict):
            raise FrozenContextError(f"{row['case_id']}: candidate not in manifest")
        identity = candidate_manifest.get("identity") or {}
        document_id = identity.get("document_id")
        source_id = identity.get("source_id")
        if not isinstance(document_id, str) or not document_id:
            raise FrozenContextError(f"{row['case_id']}: candidate missing document identity")
        if not isinstance(source_id, str) or not source_id:
            raise FrozenContextError(f"{row['case_id']}: candidate missing source identity")
        candidate = FrozenContextCandidate(
            case_id=row["case_id"], rank=row["rank"], candidate_key=row["candidate_key"],
            content_hash=row["content_hash"], rendered_content=row["rendered_content"],
            document_id=document_id,
            source_id=source_id,
            page=candidate_manifest.get("page"),
            block_type=str(candidate_manifest.get("block_type") or "text"),
        )
        grouped.setdefault(candidate.case_id, []).append(candidate)
    if len(grouped) != 27:
        raise FrozenContextError(f"Expected 27 cases, got {len(grouped)}")
    contexts = {}
    for case_id, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda candidate: candidate.rank)
        if [candidate.rank for candidate in ordered] != [1, 2, 3, 4, 5]:
            raise FrozenContextError(f"{case_id}: invalid candidate ranks")
        if len({candidate.candidate_key for candidate in ordered}) != 5:
            raise FrozenContextError(f"{case_id}: duplicate candidate key")
        if any(_content_hash(candidate.rendered_content) != candidate.content_hash for candidate in ordered):
            raise FrozenContextError(f"{case_id}: content hash mismatch")
        expected = manifest["cases"].get(case_id, {}).get("final_context_hash")
        actual = _context_hash(ordered)
        if actual != expected:
            raise FrozenContextError(f"{case_id}: final context hash mismatch")
        contexts[case_id] = FrozenCaseContext(case_id, tuple(ordered), actual)
    return contexts


def as_evaluation_context(context: FrozenCaseContext) -> FrozenEvaluationContext:
    """Convert a verified local snapshot into production-compatible chunks.

    The frozen rendered strings are kept verbatim as the model context.  The
    chunk form exists only for the unchanged calculator and validator APIs.
    """
    chunks: list[dict] = []
    sources: list[dict] = []
    for candidate in context.candidates:
        marker, separator, content = candidate.rendered_content.partition("\n")
        if not separator or not marker.startswith("[") or not marker.endswith("]"):
            raise FrozenContextError(f"{candidate.case_id}: malformed rendered candidate")
        chunks.append(
            {
                "doc_id": candidate.source_id,
                "chunk_id": candidate.source_id,
                "content": content,
                "score": 0.0,
                "metadata": {
                    "filename": candidate.document_id,
                    "document_name": candidate.document_id,
                    "page": candidate.page,
                    "type": candidate.block_type,
                },
            }
        )
        sources.append(
            {
                "filename": candidate.document_id,
                "page": candidate.page,
                "type": candidate.block_type,
                "score": 0.0,
                "chunk_id": candidate.source_id,
            }
        )
    return FrozenEvaluationContext(
        context="\n\n---\n\n".join(item.rendered_content for item in context.candidates),
        chunks=tuple(chunks),
        sources=tuple(sources),
        document_names=tuple(dict.fromkeys(item.document_id for item in context.candidates)),
        final_context_hash=context.final_context_hash,
    )
