"""Oracle-blind table schema and temporal binding helpers for V4 Gate 05 R4."""

from __future__ import annotations

import re
from typing import Any


SCHEMA_TYPES = (
    "period_on_columns",
    "period_on_rows",
    "single_period_snapshot",
    "metric_by_segment_matrix",
    "period_by_segment_matrix",
    "roll_forward",
    "comparison_change_table",
    "maturity_or_bucket_table",
    "ratio_percentage_table",
    "mixed_or_unsupported",
)

_YEAR = re.compile(r"(?<![A-Za-z0-9])(?:FY\s*)?((?:19|20)\d{2})(?![A-Za-z0-9])", re.I)
_DATE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2}(?:,|\s)\s*((?:19|20)\d{2})\b",
    re.I,
)
_CHANGE = re.compile(
    r"\b(change|changes|variance|delta|growth|growth rate|%\s*change|percentage change|vs\.?|versus|prior)\b",
    re.I,
)
_BUCKET = re.compile(
    r"\b(?:less than|more than|within|after|over|thereafter|maturit|bucket|past due|days? past|years? remaining)\b|\b\d+\s*[-–]\s*\d+\s*(?:years?|months?)\b",
    re.I,
)
_ROLL = re.compile(
    r"\b(?:opening|beginning|balance at|balance as of|balance in|closing|ending|additions|disposals|transfers|activity|roll[- ]?forward)\b",
    re.I,
)
_RATIO = re.compile(r"%|percent|percentage|ratio|margin|rate|basis point", re.I)
_YEAR_LABEL = re.compile(r"^\s*(?:FY\s*)?(?:19|20)\d{2}\s*$", re.I)
_NON_FACT_LABEL = re.compile(r"\b(?:page|note|age|director since|committees?)\b", re.I)


def text(value: Any) -> str:
    return str(value or "").strip()


def period_from_text(value: Any) -> tuple[str | None, str | None]:
    """Parse only an explicit year/date expression; never infer from a query."""
    raw = text(value)
    years = sorted(set(_YEAR.findall(raw)))
    if len(years) != 1:
        return None, None
    year = years[0]
    lowered = raw.lower()
    if _DATE.search(raw) or re.search(r"\b(as of|balance at|balance as of|december 31)\b", lowered):
        return f"FY{year}", "instant"
    if re.search(r"\b(three|six|nine|quarter|months?)\b", lowered) and "ended" in lowered:
        return f"FY{year}", "quarter_duration"
    if re.search(r"\b(year ended|years ended|annual|fiscal)\b", lowered):
        return f"FY{year}", "annual_duration"
    return f"FY{year}", "annual_duration"


def _header_paths(table: dict[str, Any]) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for key, values in (table.get("column_header_paths") or {}).items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        result[index] = [text(value) for value in values if text(value)]
    return result


def _row_labels(table: dict[str, Any]) -> list[str]:
    labels = []
    for row in table.get("rows", []):
        labels.extend([text(row.get("raw_label")), text(row.get("raw_text"))])
        labels.extend(text(value) for value in row.get("metric_path", []))
    return [value for value in labels if value]


def _column_periods(headers: dict[int, list[str]]) -> dict[int, list[tuple[str, str]]]:
    result: dict[int, list[tuple[str, str]]] = {}
    for index, values in headers.items():
        parsed = []
        for value in values:
            period, kind = period_from_text(value)
            if period:
                parsed.append((period, kind or "unknown_duration"))
        if parsed:
            result[index] = parsed
    return result


def _unique_periods(table: dict[str, Any], headers: dict[int, list[str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    context = table.get("table_context", {})
    values: list[str] = []
    for key in ("caption", "title", "statement"):
        values.append(text(context.get(key)))
    values.extend(value for paths in headers.values() for value in paths)
    for value in values:
        period, kind = period_from_text(value)
        if period and kind:
            result[period] = kind
    return result


def _numeric_cells(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [cell for cell in table.get("cells", []) if cell.get("parsed_value") is not None]


def classify_schema(table: dict[str, Any]) -> dict[str, Any]:
    """Classify one table using only its structural/header content."""
    headers = _header_paths(table)
    periods_by_column = _column_periods(headers)
    periods = _unique_periods(table, headers)
    header_text = " | ".join(value for paths in headers.values() for value in paths)
    row_text = " | ".join(_row_labels(table))
    context = table.get("table_context", {})
    context_text = " | ".join(text(context.get(key)) for key in ("title", "caption", "statement"))
    all_text = f"{header_text} | {row_text} | {context_text}"
    change_columns = sorted(index for index, paths in headers.items() if _CHANGE.search(" ".join(paths)))
    bucket_columns = sorted(index for index, paths in headers.items() if _BUCKET.search(" ".join(paths)))
    roll_forward = bool(_ROLL.search(row_text) or _ROLL.search(header_text))
    ratio_table = bool(_RATIO.search(header_text)) or any(
        text(cell.get("value_kind")).lower() in {"percentage", "ratio", "basis_points"}
        for cell in table.get("cells", [])
    )
    numeric = _numeric_cells(table)
    unique_header_labels = {" ".join(paths).strip().lower() for paths in headers.values() if paths}
    row_period_count = sum(1 for label in _row_labels(table) if period_from_text(label)[0])
    reasons: list[str] = []
    schema = "mixed_or_unsupported"
    if change_columns and len(periods_by_column) >= 2:
        schema = "comparison_change_table"
        reasons.append("explicit_change_axis_with_period_columns")
    elif roll_forward:
        schema = "roll_forward"
        reasons.append("opening_closing_activity_row_labels")
    elif bucket_columns or _BUCKET.search(all_text):
        schema = "maturity_or_bucket_table"
        reasons.append("bucket_or_maturity_axis")
    elif row_period_count >= 2 and len(periods_by_column) < 2:
        schema = "period_on_rows"
        reasons.append("explicit_periods_on_row_axis")
    elif len(periods_by_column) >= 2:
        schema = "period_on_columns"
        reasons.append("multiple_period_headers_on_column_axis")
    elif len(periods) == 1 and re.search(r"\b(as of|balance at|december|january|march|june|september)\b", all_text, re.I):
        schema = "single_period_snapshot"
        reasons.append("unique_explicit_snapshot_period")
    elif ratio_table:
        schema = "ratio_percentage_table"
        reasons.append("ratio_or_percentage_semantics")
    elif len(unique_header_labels) >= 2 and numeric:
        schema = "metric_by_segment_matrix"
        reasons.append("multiple_non_temporal_dimension_headers")
    if schema == "period_on_columns" and any("segment" in value.lower() for value in headers.get(0, [])):
        schema = "period_by_segment_matrix"
        reasons.append("segment_dimension_on_row_or_stub_axis")
    period_axis_indexes = sorted(periods_by_column)
    dimension_axis_indexes = sorted(index for index in headers if index not in period_axis_indexes and index not in change_columns)
    derived_measure_axes = sorted(set(change_columns + bucket_columns))
    return {
        "table_fragment_id": table.get("table_fragment_id"),
        "document_id": table.get("document_id"),
        "pdf_page": table.get("pdf_page"),
        "schema_type": schema,
        "row_axis": "period" if schema == "period_on_rows" else "metric",
        "column_axis": "period" if schema in {"period_on_columns", "comparison_change_table"} else "dimension",
        "period_axis_indexes": period_axis_indexes,
        "metric_axis_indexes": [index for index in headers if index not in period_axis_indexes],
        "dimension_axes": [{"column_index": index, "labels": headers.get(index, [])} for index in dimension_axis_indexes],
        "derived_measure_axes": [{"column_index": index, "labels": headers.get(index, [])} for index in derived_measure_axes],
        "periods": periods,
        "schema_confidence": 1.0 if schema != "mixed_or_unsupported" else 0.5,
        "schema_reasons": reasons,
    }


def _column_axis(table: dict[str, Any], schema: dict[str, Any]) -> dict[int, dict[str, Any]]:
    headers = _header_paths(table)
    periods = _column_periods(headers)
    columns: dict[int, dict[str, Any]] = {}
    period_indexes = sorted(periods)
    for index, paths in headers.items():
        parsed = periods.get(index, [])
        entry: dict[str, Any] = {"header_path": paths, "period": parsed[0][0] if len(parsed) == 1 else None, "period_type": parsed[0][1] if len(parsed) == 1 else None, "ambiguous_periods": sorted({item[0] for item in parsed})}
        header = " ".join(paths)
        if _CHANGE.search(header):
            before = [value for value in period_indexes if value < index]
            after = [value for value in period_indexes if value > index]
            # Financial comparison tables conventionally list current/newest
            # period first, followed by the base/prior period.  Preserve that
            # structural order; do not infer from a question or expected value.
            current_index = before[0] if len(before) >= 2 else (after[0] if after else None)
            base_index = before[1] if len(before) >= 2 else (after[1] if len(after) >= 2 else None)
            if current_index is not None and base_index is not None:
                entry["comparison"] = {"current_period": periods[current_index][0][0], "base_period": periods[base_index][0][0], "measure": "percentage_change" if "%" in header.lower() or "percent" in header.lower() or "growth" in header.lower() else "absolute_change"}
        if _BUCKET.search(header):
            entry["bucket_label"] = header
        columns[index] = entry
    return columns


def _metric_path(row: dict[str, Any], fact: dict[str, Any], cell: dict[str, Any]) -> list[str]:
    for value in (cell.get("metric_path"), row.get("metric_path"), fact.get("metric_path")):
        if isinstance(value, list) and any(text(item) for item in value):
            return [text(item) for item in value if text(item)]
    return []


def _value_semantics(fact: dict[str, Any], cell: dict[str, Any]) -> tuple[str | None, str | None, str | None, str | None]:
    value_kind = text(cell.get("value_kind") or fact.get("value_kind")).lower() or None
    scale = text(fact.get("scale") or cell.get("scale")) or None
    currency = text(fact.get("currency") or cell.get("currency")) or None
    if value_kind in {"percentage", "basis_points"}:
        return value_kind, "percent" if value_kind == "percentage" else "basis_points", None, None
    if value_kind == "per_share":
        return value_kind, "unit", currency, None
    if value_kind == "ratio":
        return value_kind, "unitless", None, None
    if value_kind == "currency":
        if not scale:
            return value_kind, scale, currency, "missing_scale_for_currency"
        return value_kind, scale, currency, None
    if value_kind in {"count", "date"}:
        return value_kind, "unit", currency, None
    return value_kind, scale, currency, "invalid_value_kind"


def _is_non_fact_numeric(fact: dict[str, Any], cell: dict[str, Any], row: dict[str, Any], classification: dict[str, Any]) -> bool:
    if classification.get("eligibility_class") == "non_fact_numeric":
        return True
    if row.get("row_role") in {"header", "separator"}:
        return True
    raw = text(fact.get("raw_value") or cell.get("raw_text"))
    label = " ".join([text(row.get("raw_label")), " ".join(text(x) for x in row.get("metric_path", []))])
    header = " ".join(text(x) for x in cell.get("header_path", []))
    if _YEAR_LABEL.fullmatch(raw) and (row.get("row_index", 0) == 0 or "change" in label.lower() or "year" in header.lower()):
        return True
    return bool(_NON_FACT_LABEL.search(label) and not row.get("metric_path"))


def bind_fact(table: dict[str, Any], schema: dict[str, Any], fact: dict[str, Any], cell: dict[str, Any], row: dict[str, Any], classification: dict[str, Any]) -> dict[str, Any]:
    """Produce one typed, fail-closed temporal binding for a Fact."""
    axes = _column_axis(table, schema)
    metric_path = _metric_path(row, fact, cell)
    value_kind, scale, currency, semantic_error = _value_semantics(fact, cell)
    record: dict[str, Any] = {
        "fact_id": fact.get("fact_id"),
        "cell_id": fact.get("cell_id"),
        "row_id": fact.get("row_id"),
        "table_fragment_id": table.get("table_fragment_id"),
        "document_id": table.get("document_id"),
        "pdf_page": table.get("pdf_page"),
        "raw_value": fact.get("raw_value"),
        "parsed_value": fact.get("parsed_value"),
        "metric_path": metric_path,
        "normalized_metric": " / ".join(value.lower() for value in metric_path) if metric_path else None,
        "value_kind": value_kind,
        "scale": scale,
        "currency": currency,
        "schema_type": schema.get("schema_type"),
        "temporal_binding": None,
        "temporal_source": None,
        "fact_semantic_type": None,
        "admission_status": "blocked",
        "failure_reasons": [],
    }
    if _is_non_fact_numeric(fact, cell, row, classification):
        record.update({"fact_semantic_type": "non_fact_numeric", "admission_status": "excluded", "failure_reasons": ["non_fact_numeric"]})
        return record
    if classification.get("eligibility_class") == "structurally_blocked":
        record["failure_reasons"] = ["structural_conflict"]
        return record
    if not metric_path:
        record["failure_reasons"].append("missing_metric_path")
    if fact.get("parsed_value") is None and cell.get("parsed_value") is None:
        record["failure_reasons"].append("numeric_parse_status_conflict")
    if semantic_error:
        record["failure_reasons"].append(semantic_error)
    column_index = int(cell.get("column_index", -1))
    axis = axes.get(column_index, {})
    schema_type = schema.get("schema_type")
    binding: dict[str, Any] | None = None
    source = None
    if axis.get("comparison") and schema_type == "comparison_change_table":
        binding = {"kind": "comparison", **axis["comparison"]}
        source = "comparison_column_schema"
    elif axis.get("bucket_label") or schema_type == "maturity_or_bucket_table":
        periods = schema.get("periods", {})
        if len(periods) == 1:
            binding = {"kind": "bucket", "reporting_period": next(iter(periods)), "bucket_label": axis.get("bucket_label") or "row_bucket"}
            source = "bucket_axis_with_unique_reporting_period"
        else:
            record["failure_reasons"].append("bucket_reporting_period_unresolved")
    elif axis.get("period") and schema_type in {"period_on_columns", "comparison_change_table", "period_by_segment_matrix", "ratio_percentage_table"}:
        binding = {"kind": "duration", "period": axis["period"], "period_type": axis.get("period_type") or "annual_duration"}
        source = "column_header_path"
    elif schema_type == "period_on_rows":
        for value in [row.get("raw_label"), row.get("raw_text"), *row.get("metric_path", [])]:
            period, period_type = period_from_text(value)
            if period:
                binding = {"kind": "duration", "period": period, "period_type": period_type or "annual_duration"}
                source = "row_header_path"
                break
    elif schema_type == "single_period_snapshot" and len(schema.get("periods", {})) == 1:
        period, period_type = next(iter(schema["periods"].items()))
        binding = {"kind": "point", "period": period, "period_type": period_type or "instant"}
        source = "unique_snapshot_caption"
    elif schema_type == "roll_forward":
        for value in [row.get("raw_label"), row.get("raw_text")]:
            period, period_type = period_from_text(value)
            if period:
                role = "opening_instant" if re.search(r"opening|beginning|january 1|balance in", text(value), re.I) else "closing_instant" if re.search(r"closing|ending|december 31", text(value), re.I) else "period_activity"
                binding = {"kind": "point" if role != "period_activity" else "duration", "period": period, "period_type": period_type or "instant", "role": role}
                source = "roll_forward_row_header"
                break
    if binding is None and fact.get("period") and schema_type not in {"maturity_or_bucket_table", "mixed_or_unsupported"}:
        binding = {"kind": "duration", "period": fact.get("period"), "period_type": "annual_duration"}
        source = "existing_graph_binding"
    if binding is None:
        periods = schema.get("periods", {})
        dimensions = schema.get("dimension_axes", [])
        if len(periods) >= 2 or dimensions:
            binding = {"kind": "period_set", "periods": sorted(periods)} if periods else {"kind": "not_applicable", "dimensions": dimensions}
            source = "row_or_matrix_schema"
    record["temporal_binding"] = binding
    record["temporal_source"] = source
    soft_field_warnings = {
        "missing_scale_for_currency",
        "invalid_value_kind",
        "currency_percentage_conflict",
        "per_share_scale_conflict",
        "ratio_scale_conflict",
        "bucket_reporting_period_unresolved",
    }
    hard_failures = [reason for reason in record["failure_reasons"] if reason not in soft_field_warnings]
    if record["failure_reasons"] and not hard_failures and binding:
        # A field-level uncertainty cannot create an Atomic Fact.  If the
        # table still has a stable period/dimension axis, retain a legal Row
        # or Matrix Evidence representation without inventing a Cell fact.
        if binding.get("kind") in {"point", "duration", "comparison", "bucket"}:
            if schema.get("periods") or schema.get("dimension_axes"):
                if schema.get("periods") and len(schema.get("periods", {})) >= 2:
                    binding = {"kind": "period_set", "periods": sorted(schema.get("periods", {}))}
                else:
                    binding = {"kind": "not_applicable", "dimensions": schema.get("dimension_axes", [])}
                record["temporal_binding"] = binding
                record["temporal_source"] = "row_matrix_schema_after_field_warning"
    if not record["failure_reasons"] and binding:
        if binding["kind"] in {"point", "duration"}:
            record.update({"fact_semantic_type": "atomic_fact", "admission_status": "admitted_a1"})
        elif binding["kind"] == "comparison":
            record.update({"fact_semantic_type": "comparison_fact", "admission_status": "admitted_a2"})
        elif binding["kind"] == "bucket":
            record.update({"fact_semantic_type": "bucket_fact", "admission_status": "admitted_a3"})
        elif binding["kind"] in {"period_set", "not_applicable"}:
            record.update({"fact_semantic_type": "row_matrix_evidence", "admission_status": "admitted_b"})
    elif not hard_failures and binding and binding["kind"] in {"period_set", "not_applicable"}:
        record.update({"fact_semantic_type": "row_matrix_evidence", "admission_status": "admitted_b"})
    if record["fact_semantic_type"] is None:
        record["fact_semantic_type"] = "blocked"
    return record


def classify_complete_predicate_gap(fact: dict[str, Any], cell: dict[str, Any], row: dict[str, Any]) -> str:
    value_kind, scale, _, error = _value_semantics(fact, cell)
    if error == "missing_scale_for_currency":
        return "missing_scale_for_currency"
    if error == "invalid_value_kind":
        return "invalid_value_kind"
    if value_kind == "percentage" and scale not in {None, "percent"}:
        return "currency_percentage_conflict"
    if value_kind == "per_share" and scale not in {None, "unit"}:
        return "per_share_scale_conflict"
    if value_kind == "ratio" and scale not in {None, "unitless"}:
        return "ratio_scale_conflict"
    if fact.get("parsed_value") is None and cell.get("parsed_value") is None:
        return "numeric_parse_status_conflict"
    return "other"
