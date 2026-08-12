"""Query-independent provider view over frozen FinancialFactV1 metadata.

The view deliberately does not invent a canonical metric.  It preserves
source-derived row/header context so the Binder can interpret composition while
the internal FinancialFactV1 object and its identity remain unchanged.
"""

from __future__ import annotations

import re
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

# BinderFactViewV2 is deliberately additive.  These fields are copied from
# the FinancialFactV1 relation or from the already-linked candidate metadata;
# they are never inferred from a question or a RequiredSlot.
_V2_SOURCE_FIELDS = (
    "row_label",
    "row_path",
    "row_hierarchy",
    "column_label",
    "column_header_path",
    "multi_level_column_headers",
    "table_title",
    "statement_title",
    "statement_type",
    "section_title",
    "section_path",
    "page",
    "table_id",
    "row_id",
    "column_id",
    "cell_id",
    "physical_source_id",
    "document_id",
    "pdf_page",
    "period_value_bindings",
)


def _clean_sequence(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _parse_source_structure(source: Mapping[str, Any]) -> dict[str, Any]:
    """Read exact labelled structure lines already present in source_text.

    This is a serialization parser, not a semantic normalizer.  It only
    copies values following labels emitted by the frozen candidate serializer.
    """

    text = source.get("source_text")
    if not isinstance(text, str):
        return {}
    parsed: dict[str, Any] = {}
    labels = {
        "Statement": "statement_title",
        "Statement title": "statement_title",
        "Statement type": "statement_type",
        "Metric Path": "row_path",
        "Row Path": "row_path",
        "Row": "row_label",
        "Column Header": "column_header_path",
        "Column Headers": "column_header_path",
        "Section": "section_title",
        "Section Path": "section_path",
        "Table": "table_title",
        "Table title": "table_title",
    }
    for line in text.splitlines():
        line = line.strip()
        for label, key in labels.items():
            prefix = f"{label}:"
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                if value and key not in parsed:
                    parsed[key] = value
                break
    if "column_header_path" in parsed:
        parsed["column_header_path"] = [
            item.strip() for item in re.split(r"\s*(?:>|/|→|\|)\s*", str(parsed["column_header_path"])) if item.strip()
        ]
    for key in ("row_path", "section_path"):
        if key in parsed:
            parsed[key] = [
                item.strip() for item in re.split(r"\s*(?:>|/|→|\|)\s*", str(parsed[key])) if item.strip()
            ]
    return parsed


def _source_value(fact: Mapping[str, Any], source: Mapping[str, Any], parsed: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = fact.get(key)
        if value is None:
            value = source.get(key)
        if value is None:
            value = parsed.get(key)
        if value is not None:
            return value
    return None


def binder_fact_view_v2_field_provenance(
    fact: Mapping[str, Any],
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return provenance for every V2 field without reading query/Gold data."""

    source = source_metadata or {}
    parsed = _parse_source_structure(source)
    result: dict[str, dict[str, Any]] = {}
    for key in _FACT_FIELDS:
        if fact.get(key) is not None:
            result[key] = {"source_field": key, "source_candidate_id": source.get("candidate_id"), "origin": "financial_fact"}
    source_keys: dict[str, tuple[str, ...]] = {
        "row_label": ("row_label",),
        "row_path": ("row_path", "row_hierarchy", "metric_path"),
        "row_hierarchy": ("row_hierarchy",),
        "column_label": ("column_label",),
        "column_header_path": ("column_header_path", "column_header"),
        "multi_level_column_headers": ("multi_level_column_headers", "column_header"),
        "table_title": ("table_title",),
        "statement_title": ("statement_title", "statement_id"),
        "statement_type": ("statement_type", "statement_id"),
        "section_title": ("section_title", "section_heading"),
        "section_path": ("section_path",),
        "page": ("page", "pdf_page"),
        "table_id": ("table_id",),
        "row_id": ("row_id",),
        "column_id": ("column_id",),
        "cell_id": ("cell_id",),
        "physical_source_id": ("physical_source_id",),
        "document_id": ("document_id",),
        "pdf_page": ("pdf_page",),
        "period_value_bindings": ("period_value_bindings",),
    }
    for output_key, keys in source_keys.items():
        value = _source_value(fact, source, parsed, *keys)
        if value is not None:
            origin = "financial_fact" if any(fact.get(key) is not None for key in keys) else "source_candidate"
            source_field = next((key for key in keys if source.get(key) is not None), None)
            if source_field is None:
                source_field = next((key for key in keys if parsed.get(key) is not None), keys[0])
            result[output_key] = {
                "source_field": source_field,
                "source_candidate_id": source.get("candidate_id"),
                "origin": origin,
            }
    return result


def build_binder_fact_view_v2(
    fact: Mapping[str, Any],
    source_metadata: Mapping[str, Any] | None = None,
    fact_handle: str | None = None,
) -> dict[str, Any]:
    """Build the deterministic, query-independent BinderFactViewV2.

    ``fact`` and ``source_metadata`` must already be linked by the frozen
    FinancialFact relation.  The function accepts no question, slot, Gold,
    or reference-answer argument by design.
    """

    source = source_metadata or {}
    parsed = _parse_source_structure(source)
    handle = fact_handle or str(fact.get("fact_handle") or "")
    view: dict[str, Any] = {"fact_handle": handle}
    for key in _FACT_FIELDS:
        value = fact.get(key)
        if value is not None:
            view[key] = value

    # Keep source values structured.  In particular, never flatten a row or
    # header path into a newly generated metric label.
    values: dict[str, Any] = {
        "row_label": _source_value(fact, source, parsed, "row_label"),
        "row_path": _source_value(fact, source, parsed, "row_path", "row_hierarchy", "metric_path"),
        "row_hierarchy": _source_value(fact, source, parsed, "row_hierarchy"),
        "column_label": _source_value(fact, source, parsed, "column_label"),
        "column_header_path": _source_value(fact, source, parsed, "column_header_path", "column_header"),
        "multi_level_column_headers": _source_value(fact, source, parsed, "multi_level_column_headers", "column_header"),
        "table_title": _source_value(fact, source, parsed, "table_title"),
        "statement_title": _source_value(fact, source, parsed, "statement_title", "statement_id"),
        "statement_type": _source_value(fact, source, parsed, "statement_type", "statement_id"),
        "section_title": _source_value(fact, source, parsed, "section_title", "section_heading"),
        "section_path": _source_value(fact, source, parsed, "section_path"),
        "page": _source_value(fact, source, parsed, "page", "pdf_page"),
        "table_id": _source_value(fact, source, parsed, "table_id"),
        "row_id": _source_value(fact, source, parsed, "row_id"),
        "column_id": _source_value(fact, source, parsed, "column_id"),
        "cell_id": _source_value(fact, source, parsed, "cell_id"),
        "physical_source_id": _source_value(fact, source, parsed, "physical_source_id"),
        "document_id": _source_value(fact, source, parsed, "document_id"),
        "pdf_page": _source_value(fact, source, parsed, "pdf_page"),
        "period_value_bindings": _source_value(fact, source, parsed, "period_value_bindings"),
    }
    for key, value in values.items():
        if value is not None:
            if key in {"row_path", "row_hierarchy", "column_header_path", "multi_level_column_headers", "section_path"}:
                sequence = _clean_sequence(value)
                if sequence:
                    view[key] = sequence
            else:
                view[key] = value
    if "candidate_rank" in fact:
        view["candidate_rank"] = fact["candidate_rank"]
    return view


def build_binder_fact_views_v2(
    facts: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    source_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Project every frozen packet fact to a local F-handle, preserving order."""

    source_by_candidate = source_by_candidate or {}
    views: list[dict[str, Any]] = []
    for index, fact in enumerate(facts, 1):
        candidate_ids = [str(item) for item in fact.get("candidate_ids", [])]
        candidate_id = str(fact.get("candidate_id")) if fact.get("candidate_id") is not None else ""
        source = source_by_candidate.get(candidate_id)
        if source is None:
            source = next((source_by_candidate.get(item) for item in candidate_ids if item in source_by_candidate), None)
        views.append(build_binder_fact_view_v2(fact, source, f"F{index:02d}"))
    return views


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
