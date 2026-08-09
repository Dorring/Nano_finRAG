from __future__ import annotations

import hashlib
from typing import Any

from .evidence_set_models import EvidenceRef

TYPE_MAP = {
    "atomic": "atomic_fact",
    "comparison": "comparison_fact",
    "bucket": "bucket_fact",
    "row_matrix": "row_matrix",
    "narrative": "narrative_evidence",
}


def attach_candidate(
    candidate_key: str,
    candidate_rank: int,
    structured_view: dict[str, Any] | None,
    document_scope: tuple[str, ...],
) -> list[EvidenceRef]:
    if structured_view is None:
        evidence_id = "raw:" + hashlib.sha256(candidate_key.encode()).hexdigest()
        return [
            EvidenceRef(
                evidence_id,
                "raw_candidate",
                candidate_key,
                candidate_rank,
                document_scope[0] if len(document_scope) == 1 else "",
                {},
                (),
            )
        ]
    result = []
    for fact in structured_view.get("facts") or []:
        kind = TYPE_MAP.get(str(fact.get("type") or ""), str(fact.get("type") or ""))
        evidence_id = str(fact.get("evidence_id") or "")
        if not evidence_id or kind not in set(TYPE_MAP.values()):
            continue
        result.append(
            EvidenceRef(
                evidence_id,
                kind,
                candidate_key,
                candidate_rank,
                str(structured_view.get("document_id") or ""),
                dict(fact),
                tuple(structured_view.get("source_traceback") or ()),
            )
        )
    if not result:
        evidence_id = "raw:" + hashlib.sha256(candidate_key.encode()).hexdigest()
        result.append(
            EvidenceRef(
                evidence_id,
                "raw_candidate",
                candidate_key,
                candidate_rank,
                str(structured_view.get("document_id") or ""),
                {},
                tuple(structured_view.get("source_traceback") or ()),
            )
        )
    return result


def canonicalize(refs: list[EvidenceRef]) -> list[dict[str, Any]]:
    grouped: dict[str, list[EvidenceRef]] = {}
    for ref in refs:
        grouped.setdefault(ref.evidence_id, []).append(ref)
    output = []
    for evidence_id in sorted(grouped):
        members = sorted(
            grouped[evidence_id],
            key=lambda item: (item.candidate_rank, item.candidate_key),
        )
        best = members[0]
        output.append(
            {
                "evidence_id": evidence_id,
                "evidence_type": best.evidence_type,
                "candidate_key": best.candidate_key,
                "candidate_rank": best.candidate_rank,
                "supporting_candidate_keys": sorted(
                    {item.candidate_key for item in members}
                ),
                "document_id": best.document_id,
                "payload": best.payload,
                "source_traceback": list(best.source_traceback),
            }
        )
    return output
