"""Oracle-blind, fail-closed Fact Metric/Period recovery for Gate 05 R2/R3."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
from typing import Any


VALID_VALUE_KINDS = {"currency", "percentage", "per_share", "basis_points", "count", "ratio"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _period_from_text(value: Any) -> tuple[str | None, str | None]:
    text = _text(value)
    years = sorted(set(re.findall(r"\b((?:19|20)\d{2})\b", text)))
    if len(years) != 1:
        return None, None
    year = years[0]
    lowered = text.lower()
    if re.search(r"\b(as of|balance at|december|january|march|june|september)\b", lowered) and "ended" not in lowered and "year" not in lowered:
        return f"FY{year}", "instant"
    if re.search(r"\b(three|six|nine|quarter|months?)\b", lowered) and "ended" in lowered:
        return f"FY{year}", "quarter_duration"
    return f"FY{year}", "annual_duration"


def _load_classification(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if index:
                result[str(value.get("fact_id"))] = value
    return result


def _metric_path(row: dict[str, Any], fact: dict[str, Any], cell: dict[str, Any]) -> tuple[list[str], str | None]:
    for source, values in (("cell_metric_path", cell.get("metric_path")), ("row_metric_path", row.get("metric_path")), ("fact_metric_path", fact.get("metric_path"))):
        if isinstance(values, list) and any(_text(value) for value in values):
            cleaned = [_text(value) for value in values if _text(value)]
            return cleaned, "parent_metric_hierarchy" if source == "row_metric_path" else source
    return [], None


def _numeric_valid(fact: dict[str, Any], cell: dict[str, Any]) -> bool:
    if fact.get("parsed_value") is None and cell.get("parsed_value") is None:
        return False
    kind = _text(cell.get("value_kind") or fact.get("value_kind")).lower()
    return kind in VALID_VALUE_KINDS


def _header_periods(table: dict[str, Any], cell: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    cell_periods: set[str] = set()
    column_periods: set[str] = set()
    for value in cell.get("header_path", []):
        period, _ = _period_from_text(value)
        if period:
            cell_periods.add(period)
    for value in table.get("column_header_paths", {}).get(str(cell.get("column_index")), []):
        period, _ = _period_from_text(value)
        if period:
            column_periods.add(period)
    caption_periods: set[str] = set()
    context = table.get("table_context", {})
    for value in (context.get("caption"), context.get("title")):
        period, _ = _period_from_text(value)
        if period:
            caption_periods.add(period)
    return cell_periods, column_periods, caption_periods


def _candidate_period(table: dict[str, Any], cell: dict[str, Any], stable_by_column: dict[int, set[str]], table_periods: set[str]) -> tuple[str | None, str | None, float]:
    cell_periods, header_periods, _ = _header_periods(table, cell)
    if len(cell_periods) == 1:
        return next(iter(cell_periods)), "cell_header_path", 1.0
    if len(header_periods) == 1:
        return next(iter(header_periods)), "stable_column_schema", 0.95
    stable = stable_by_column.get(int(cell.get("column_index", -1)), set())
    if len(stable) == 1:
        return next(iter(stable)), "stable_column_schema", 0.95
    if len(table_periods) == 1:
        return next(iter(table_periods)), "unique_table_period_candidate", 0.90
    return None, None, 0.0


def _set_metric(target: dict[str, Any], path: list[str], source: str | None) -> None:
    if path:
        target["metric_path"] = path
        target["normalized_metric_path"] = " / ".join(value.lower() for value in path)
        target["normalized_metric"] = target["normalized_metric_path"]
        if source:
            target["metric_source"] = source


def recover_graph(source: dict[str, Any], classification_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a recovered graph copy and an auditable recovery summary."""
    class_map = _load_classification(classification_path)
    recovered = json.loads(json.dumps(source, ensure_ascii=False))
    summary: dict[str, Any] = {"fact_total_count": 0, "fact_eligible_count": 0, "fact_recovered_count": 0, "fact_recovery_source_counts": {}, "binding_status_counts": {}, "eligibility_class_counts": {}, "period_candidate_counts": {"cell_header_path": 0, "stable_column_schema": 0, "unique_table_period_candidate": 0, "unresolved": 0}, "metric_recovery_count": 0, "false_binding_guard_count": 0, "classification_missing_count": 0, "soft_continuation_period_inheritance_count": 0}
    for page in recovered.get("pages", []):
        for table in page.get("tables", []):
            rows = {str(row.get("row_id")): row for row in table.get("rows", [])}
            cells = {str(cell.get("cell_id")): cell for cell in table.get("cells", [])}
            facts = {str(fact.get("fact_id")): fact for fact in table.get("facts", [])}
            stable_by_column: dict[int, set[str]] = {}
            table_periods: set[str] = set()
            for fact in facts.values():
                classification = class_map.get(str(fact.get("fact_id")), {})
                if classification.get("eligibility_class") == "non_fact_numeric":
                    continue
                if fact.get("period"):
                    cell = cells.get(str(fact.get("cell_id")), {})
                    stable_by_column.setdefault(int(cell.get("column_index", -1)), set()).add(str(fact.get("period")))
                    table_periods.add(str(fact.get("period")))
            table_context = table.get("table_context", {})
            table_scale = _text(table_context.get("table_scale"))
            table_currency = _text(table_context.get("table_currency"))
            # Header periods are used only when one complete schema is available.
            for value in table.get("column_header_paths", {}).values():
                for segment in value:
                    period, _ = _period_from_text(segment)
                    if period:
                        table_periods.add(period)
            for fact in facts.values():
                fact_id = str(fact.get("fact_id"))
                summary["fact_total_count"] += 1
                classification = class_map.get(fact_id)
                if classification is None:
                    summary["classification_missing_count"] += 1
                    classification = {"eligibility_class": "unresolved"}
                category = classification.get("eligibility_class") or "unresolved"
                summary["eligibility_class_counts"][category] = summary["eligibility_class_counts"].get(category, 0) + 1
                cell = cells.get(str(fact.get("cell_id")), {})
                row = rows.get(str(fact.get("row_id")), {})
                metric_path, metric_source = _metric_path(row, fact, cell)
                if not fact.get("metric_path") and metric_path:
                    _set_metric(fact, metric_path, metric_source)
                    _set_metric(cell, metric_path, metric_source)
                    summary["metric_recovery_count"] += 1
                period_source = None
                period_confidence = 0.0
                if not fact.get("period"):
                    candidate, period_source, period_confidence = _candidate_period(table, cell, stable_by_column, table_periods)
                    if candidate and period_source in {"cell_header_path", "stable_column_schema"}:
                        fact["period"] = candidate
                        cell["normalized_period"] = candidate
                        cell["period_type"] = "annual_duration" if candidate.startswith("FY") else None
                        cell["period_source"] = period_source
                        cell["period_confidence"] = period_confidence
                        summary["fact_recovered_count"] += 1
                        summary["fact_recovery_source_counts"][period_source] = summary["fact_recovery_source_counts"].get(period_source, 0) + 1
                    elif period_source:
                        summary["period_candidate_counts"][period_source] = summary["period_candidate_counts"].get(period_source, 0) + 1
                    else:
                        summary["period_candidate_counts"]["unresolved"] += 1
                else:
                    cell["normalized_period"] = fact.get("period")
                    cell.setdefault("period_source", "existing_graph_binding")
                    cell.setdefault("period_confidence", 1.0)
                if not _text(fact.get("scale")) and table_scale:
                    fact["scale"] = table_scale
                    cell["scale"] = table_scale
                    fact["scale_source"] = "table_local_scope"
                    cell["scale_source"] = "table_local_scope"
                if not _text(fact.get("currency")) and table_currency:
                    fact["currency"] = table_currency
                    cell["currency"] = table_currency
                    fact["currency_source"] = "table_local_scope"
                    cell["currency_source"] = "table_local_scope"
                if category == "non_fact_numeric":
                    fact["fact_eligible"] = False
                    fact["fact_eligibility_class"] = category
                    fact["binding_status"] = "non_fact_numeric"
                    cell["binding_status"] = "non_fact_numeric"
                    cell["fact_eligible"] = False
                elif category == "structurally_blocked":
                    fact["fact_eligible"] = False
                    fact["fact_eligibility_class"] = category
                    fact["binding_status"] = "blocked_conflict"
                    cell["binding_status"] = "blocked_conflict"
                    cell["fact_eligible"] = False
                else:
                    value_kind = _text(cell.get("value_kind") or fact.get("value_kind")).lower()
                    scale_ok = value_kind != "currency" or bool(_text(fact.get("scale") or cell.get("scale")))
                    complete = bool(metric_path or fact.get("metric_path")) and bool(fact.get("period")) and _numeric_valid(fact, cell) and scale_ok
                    if complete:
                        fact["binding_status"] = "complete"
                        fact["fact_eligible"] = True
                        fact["fact_eligibility_class"] = "eligible_recoverable"
                        cell["binding_status"] = "complete"
                        cell["fact_eligible"] = True
                        cell["evidence_level"] = "A"
                        summary["fact_eligible_count"] += 1
                    elif metric_path or fact.get("metric_path"):
                        fact["binding_status"] = "row_only"
                        fact["fact_eligible"] = False
                        fact["fact_eligibility_class"] = "eligible_row_only"
                        cell["binding_status"] = "row_only"
                        cell["fact_eligible"] = False
                    else:
                        fact["binding_status"] = "unresolved"
                        fact["fact_eligible"] = False
                        fact["fact_eligibility_class"] = "unresolved"
                        cell["binding_status"] = "unresolved"
                        cell["fact_eligible"] = False
                fact["period_source"] = fact.get("period_source") or period_source or ("existing_graph_binding" if fact.get("period") else None)
                fact["period_confidence"] = fact.get("period_confidence", period_confidence if fact.get("period") else 0.0)
                summary["binding_status_counts"][fact.get("binding_status")] = summary["binding_status_counts"].get(fact.get("binding_status"), 0) + 1
    summary["fact_admission_rate_over_all_facts"] = summary["fact_eligible_count"] / max(1, summary["fact_total_count"])
    summary["fact_classification_map_missing"] = summary["classification_missing_count"]
    summary["false_binding_guard_count"] = summary["soft_continuation_period_inheritance_count"]
    return recovered, summary
