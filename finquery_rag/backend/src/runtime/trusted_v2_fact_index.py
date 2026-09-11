"""Build candidate-aligned R4 views from the same trusted V2 fact store.

The production V2 retrieval port materializes every R4 ``candidate_key``
through :class:`StructuredFactStore`.  An index and a fact store created by
different historical pipelines can therefore be individually valid while
being unusable together.  This module creates source-only raw/structured
views directly from the fact-store records so the two assets share one stable
candidate namespace.

It is an asset-construction helper, not a retrieval policy or a model runner:
it never reads questions, Gold labels, answers, conversation history, or
model-generated summaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.pdf_retrieval_v4.candidate_aligned_view import (
    CandidateAlignedView,
    CandidateViewPair,
    make_raw_view_id,
    make_structured_view_id,
)


FACT_CANDIDATE_INDEX_SCHEMA_VERSION = "trusted-v2-fact-candidate-index-v1"


class TrustedV2FactIndexError(ValueError):
    """Raised when a fact record cannot safely become an R4 candidate view."""


@dataclass(frozen=True)
class FactCandidateViewBuildSummary:
    """Small, serializable summary of a source-only fact-index build input."""

    source_fact_count: int
    candidate_pair_count: int
    structured_view_count: int
    document_count: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": FACT_CANDIDATE_INDEX_SCHEMA_VERSION,
            "source_fact_count": self.source_fact_count,
            "candidate_pair_count": self.candidate_pair_count,
            "structured_view_count": self.structured_view_count,
            "document_count": self.document_count,
        }


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stable_unique(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return tuple(result)


def _values(record: Mapping[str, Any], *keys: str) -> tuple[str, ...]:
    values: list[Any] = []
    for key in keys:
        value = record.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(value)
        else:
            values.append(value)
    return _stable_unique(values)


def _context(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("structural_context")
    return value if isinstance(value, Mapping) else {}


def _first(record: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _text(record.get(key))
        if value:
            return value
    return None


def _candidate_key(record: Mapping[str, Any]) -> str | None:
    """Match ``StructuredFactStore`` candidate-key precedence exactly."""

    for key in ("candidate_key", "candidate_id", "candidate_keys", "candidate_ids"):
        values = _values(record, key)
        if values:
            return values[0]
    return None


def _source_text(record: Mapping[str, Any]) -> str:
    """Return physically extracted source text, never dialogue/model fields."""

    for key in ("content", "evidence_text", "source_text", "raw_content", "row_label"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _page(record: Mapping[str, Any], context: Mapping[str, Any]) -> int | None:
    for value in (record.get("pdf_page"), record.get("page"), context.get("pdf_page")):
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.strip().isdigit() and int(value.strip()) > 0:
            return int(value.strip())
    return None


def _line(label: str, value: Any) -> str | None:
    text = _text(value)
    return f"{label}: {text}" if text else None


def _joined(label: str, values: Iterable[Any]) -> str | None:
    normalized = _stable_unique(values)
    return f"{label}: {', '.join(normalized)}" if normalized else None


def _fact_views(record: Mapping[str, Any], position: int) -> CandidateViewPair:
    if not isinstance(record, Mapping):
        raise TrustedV2FactIndexError(f"fact_record_not_mapping:{position}")
    context = _context(record)
    candidate_key = _candidate_key(record)
    evidence_id = _first(record, "evidence_id", "fact_id")
    document_id = _first(
        record,
        "document_id",
        "document_name",
        "source_id",
        "physical_source_id",
    ) or _first(context, "document_id")
    if candidate_key is None:
        raise TrustedV2FactIndexError(f"missing_candidate_key:{position}")
    if evidence_id is None:
        raise TrustedV2FactIndexError(f"missing_evidence_id:{candidate_key}")
    if document_id is None:
        raise TrustedV2FactIndexError(f"missing_document_identity:{candidate_key}")
    if record.get("provenance_complete") is not True:
        raise TrustedV2FactIndexError(f"incomplete_provenance:{candidate_key}")

    metric_paths = _stable_unique(
        (
            record.get("metric"),
            record.get("normalized_metric"),
            *_values(context, "metric_paths", "row_path"),
        )
    )
    periods = _stable_unique(
        (
            record.get("period"),
            record.get("normalized_period"),
            *_values(context, "periods"),
        )
    )
    table_ids = _stable_unique(
        (
            *_values(record, "logical_table_ids", "logical_table_id", "table_fragment_id", "table_id"),
            *_values(context, "logical_table_ids", "logical_table_id", "table_fragment_id", "table_id"),
        )
    )
    row_ids = _stable_unique(
        (
            *_values(record, "row_ids", "row_id"),
            *_values(context, "row_ids"),
        )
    )
    temporal_types = _stable_unique(
        (
            *_values(record, "temporal_types", "temporal_type", "period_type", "period_kind"),
            *_values(context, "temporal_types", "temporal_type"),
        )
    )

    descriptor_values = {
        "Entity": _first(record, "entity", "issuer", "company"),
        "Metric": _first(record, "metric", "normalized_metric", "raw_metric"),
        "Period": _first(record, "period", "normalized_period", "raw_period"),
        "Value": _first(record, "value", "parsed_numeric_value", "raw_value"),
        "Currency": _first(record, "currency"),
        "Unit": _first(record, "unit"),
        "Scale": _first(record, "scale", "normalized_scale"),
        "Scope": _first(record, "scope", "scope_label"),
        "Statement Type": _first(record, "statement_type"),
        "Table": _first(record, "table_title") or _first(context, "table_title"),
    }
    common_lines = [f"Document: {document_id}"]
    page = _page(record, context)
    if page is not None:
        common_lines.append(f"Page: {page}")
    common_lines.extend(
        line for label, value in descriptor_values.items() if (line := _line(label, value))
    )
    common_lines.extend(
        line
        for line in (
            _joined("Metric Path", metric_paths),
            _joined("Periods", periods),
            _joined("Table IDs", table_ids),
            _joined("Row IDs", row_ids),
        )
        if line
    )

    source_text = _source_text(record)
    raw_lines = [*common_lines]
    if source_text:
        raw_lines.extend(("Source:", source_text))
    raw_text = "\n".join(raw_lines)
    structured_text = "\n".join((f"Evidence ID: {evidence_id}", *common_lines))
    bridge_grade = "A1_fact_store"
    raw_view = CandidateAlignedView(
        candidate_key=candidate_key,
        view_type="raw",
        view_id=make_raw_view_id(candidate_key),
        retrieval_text=raw_text,
        document_id=document_id,
        pdf_page=page,
        logical_table_ids=table_ids,
        row_ids=row_ids,
        fact_ids=(evidence_id,),
        metric_paths=metric_paths,
        periods=periods,
        temporal_types=temporal_types,
        bridge_grade=bridge_grade,
    )
    structured_view = CandidateAlignedView(
        candidate_key=candidate_key,
        view_type="structured",
        view_id=make_structured_view_id(candidate_key),
        retrieval_text=structured_text,
        document_id=document_id,
        pdf_page=page,
        logical_table_ids=table_ids,
        row_ids=row_ids,
        fact_ids=(evidence_id,),
        metric_paths=metric_paths,
        periods=periods,
        temporal_types=temporal_types,
        bridge_grade=bridge_grade,
    )
    return CandidateViewPair(candidate_key, raw_view, structured_view)


def build_fact_store_candidate_views(
    facts: Iterable[Mapping[str, Any]],
) -> tuple[list[CandidateViewPair], FactCandidateViewBuildSummary]:
    """Build one raw/structured R4 pair per provenance-complete fact.

    Every produced pair uses the fact's existing ``candidate_key``.  This is
    the critical deployment invariant: every R4 result can be materialized
    through the same fact-store artifact without an ID translation layer.
    """

    pairs: list[CandidateViewPair] = []
    seen: set[str] = set()
    documents: set[str] = set()
    count = 0
    for count, fact in enumerate(facts, 1):
        pair = _fact_views(fact, count)
        if pair.candidate_key in seen:
            raise TrustedV2FactIndexError(
                f"duplicate_candidate_key:{pair.candidate_key}"
            )
        seen.add(pair.candidate_key)
        pairs.append(pair)
        documents.add(pair.document_id)
    if not pairs:
        raise TrustedV2FactIndexError("fact_store_contains_no_candidate_views")
    return pairs, FactCandidateViewBuildSummary(
        source_fact_count=count,
        candidate_pair_count=len(pairs),
        structured_view_count=sum(pair.structured_view is not None for pair in pairs),
        document_count=len(documents),
    )


__all__ = [
    "FACT_CANDIDATE_INDEX_SCHEMA_VERSION",
    "FactCandidateViewBuildSummary",
    "TrustedV2FactIndexError",
    "build_fact_store_candidate_views",
]
