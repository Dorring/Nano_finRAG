"""Pure contracts for the PDF Retrieval V3 Gate 3 structured lane.

This module deliberately contains no benchmark or label access.  Gate 3 uses
it to make the raw-pool protection rule independently testable: structured
results may be appended, but never re-score, reorder, or delete raw results.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable


def payload_hash(payload: Any) -> str:
    """Return a stable digest for an audit payload."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_safe_structured_view(view: dict[str, Any]) -> bool:
    """Accept only row views with a traceable, non-inferred metric.

    Table-level periods are retrieval hints.  This predicate intentionally
    never inspects or emits a cell-to-period/value relationship.
    """
    return (
        view.get("evidence_type") == "table_row"
        and bool((view.get("metric_field") or {}).get("normalized_metric"))
        and bool(view.get("candidate_key"))
        and bool(view.get("evidence_id"))
    )


def enriched_retrieval_text(view: dict[str, Any], raw_text: str) -> str:
    """Build the V2-Lite retrieval-only representation from traced fields."""
    document = view.get("document_field") or {}
    section = view.get("section_field") or {}
    metric = view.get("metric_field") or {}
    period = view.get("period_field") or {}
    unit = view.get("unit_field") or {}
    parts = [
        "document " + " ".join(str(item) for item in (document.get("company"), document.get("fiscal_year")) if item),
        "metric " + str(metric.get("normalized_metric") or ""),
    ]
    statement = section.get("statement_title") or section.get("table_title")
    if statement:
        parts.append("statement " + str(statement))
    periods = [str(item) for item in period.get("periods") or () if item]
    if periods:
        parts.append("table periods " + " ".join(periods))
    unit_text = " ".join(str(item) for item in (unit.get("currency"), unit.get("scale")) if item)
    if unit_text:
        parts.append("unit " + unit_text)
    parts.append("row " + raw_text)
    return "\n".join(part for part in parts if part.strip())


def fixed_rrf(
    bm25: Iterable[str], dense: Iterable[str], *, k: int = 60, limit: int = 20
) -> list[tuple[str, float]]:
    """Identity-deduplicated, equal-weight reciprocal-rank fusion."""
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in (list(bm25), list(dense)):
        for rank, identity in enumerate(ranking, 1):
            if not identity:
                continue
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(identity, len(first_seen))
    ordered = sorted(scores, key=lambda item: (-scores[item], first_seen[item], item))
    return [(identity, scores[identity]) for identity in ordered[:limit]]


@dataclass(frozen=True)
class MergeResult:
    combined: list[dict[str, Any]]
    raw_unchanged: bool
    duplicate_count: int


def append_structured_residual(
    raw_pool: list[dict[str, Any]], structured: list[dict[str, Any]]
) -> MergeResult:
    """Append only structured identities missing from the frozen raw pool.

    The returned raw prefix is byte-for-byte equivalent at the object level
    (the values are copied, not re-ranked).  A structured duplicate merely
    annotates lane provenance on the appended record representation and never
    creates another candidate.
    """
    raw_copy = [dict(item) for item in raw_pool]
    raw_ids = [str(item["candidate_key"]) for item in raw_copy]
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("raw full RRF pool contains duplicate identities")
    combined = list(raw_copy)
    seen = set(raw_ids)
    duplicates = 0
    for append_rank, candidate in enumerate(structured, 1):
        identity = str(candidate["candidate_key"])
        if identity in seen:
            duplicates += 1
            continue
        record = dict(candidate)
        record["combined_append_rank"] = append_rank
        record["present_in_raw_pool"] = False
        record["present_in_structured_pool"] = True
        combined.append(record)
        seen.add(identity)
    raw_unchanged = combined[: len(raw_copy)] == raw_copy
    return MergeResult(combined=combined, raw_unchanged=raw_unchanged, duplicate_count=duplicates)
