"""Query-independent provider view over frozen FinancialFactV1 metadata.

The view deliberately does not invent a canonical metric.  It preserves
source-derived row/header context so the Binder can interpret composition while
the internal FinancialFactV1 object and its identity remain unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping


_FACT_FIELDS = (
    "raw_metric",
    "normalized_metric",
    "raw_period",
    "normalized_period",
    "raw_value",
    "parsed_numeric_value",
    "raw_scale",
    "normalized_scale",
    "currency",
    "unit",
)

_SOURCE_FIELDS = (
    "row_label",
    "row_hierarchy",
    "column_header",
    "column_header_path",
    "table_title",
    "statement_title",
    "section_heading",
    "table_id",
    "row_id",
    "column_id",
    "cell_id",
    "physical_source_id",
    "document_id",
    "pdf_page",
)


def build_binder_fact_view(
    fact: Mapping[str, Any],
    fact_handle: str,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deterministic view without query or Gold access."""

    source = source_metadata or {}
    view: dict[str, Any] = {"fact_handle": str(fact_handle)}
    for key in _FACT_FIELDS:
        value = fact.get(key)
        if value is not None:
            view[key] = value
    for key in _SOURCE_FIELDS:
        value = fact.get(key)
        if value is None:
            value = source.get(key)
        if value is not None:
            view[key] = value
    # Candidate metadata is source context, never a selectable identity.
    if "candidate_rank" in fact:
        view["candidate_rank"] = fact["candidate_rank"]
    return view


def build_binder_fact_views(
    facts: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    source_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Map frozen packet order to F01..Fn and preserve every fact."""

    source_by_candidate = source_by_candidate or {}
    views: list[dict[str, Any]] = []
    for index, fact in enumerate(facts, 1):
        candidate_ids = [str(item) for item in fact.get("candidate_ids", [])]
        candidate_id = str(fact.get("candidate_id")) if fact.get("candidate_id") is not None else ""
        source = source_by_candidate.get(candidate_id)
        if source is None:
            source = next((source_by_candidate.get(item) for item in candidate_ids if item in source_by_candidate), None)
        views.append(build_binder_fact_view(fact, f"F{index:02d}", source))
    return views

