"""Classify Gate 05 excluded Facts without reading Question or Gold data."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE05 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05"
DEFAULT_GRAPH = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-03"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r1"
CLASSES = ("eligible_recoverable", "eligible_row_only", "non_fact_numeric", "structurally_blocked", "unresolved")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_units(path: Path) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if index:
                units.append(value)
    return units


def _write_jsonl_gz(path: Path, header: dict[str, Any], records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        compressed.write((json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        for record in records:
            compressed.write((json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _period_from_text(value: Any) -> tuple[str | None, str | None]:
    text = _text(value)
    if not text:
        return None, None
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


def _period_candidates(table: dict[str, Any], cell: dict[str, Any], facts_by_cell: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    candidates: dict[str, set[str]] = {"cell_header_path": set(), "column_header_schema": set(), "existing_fact": set()}
    for value in cell.get("header_path", []):
        period, _ = _period_from_text(value)
        if period:
            candidates["cell_header_path"].add(period)
    column_paths = table.get("column_header_paths", {})
    for value in column_paths.get(str(cell.get("column_index")), []):
        period, _ = _period_from_text(value)
        if period:
            candidates["column_header_schema"].add(period)
    for fact in facts_by_cell.values():
        if fact.get("period") and fact.get("cell_id") == cell.get("cell_id"):
            candidates["existing_fact"].add(str(fact["period"]))
    return candidates


def _numeric_equals(value: Any, other: Any) -> bool:
    try:
        return Decimal(str(value)) == Decimal(str(other))
    except (InvalidOperation, TypeError, ValueError):
        return False


def _is_non_fact_numeric(fact: dict[str, Any], cell: dict[str, Any], row: dict[str, Any], table: dict[str, Any]) -> tuple[bool, str | None]:
    row_role = _text(row.get("row_role")).lower()
    label = _text(row.get("raw_label") or row.get("raw_text")).lower()
    metric = _text(fact.get("normalized_metric") or fact.get("normalized_metric_path")).lower()
    value_kind = _text(cell.get("value_kind") or fact.get("value_kind")).lower()
    header = " / ".join(str(value) for value in cell.get("header_path", []))
    if row_role in {"header", "separator"}:
        return True, "header_or_separator_row"
    if _text(cell.get("header_path", [])[0] if cell.get("header_path") else "").lower() == "page" or metric.startswith("page /"):
        return True, "page_number_or_navigation_cell"
    if re.search(r"\b(page|contents|table of contents)\b", label) and value_kind in {"unknown_numeric", "", "count"}:
        return True, "page_number_or_navigation_cell"
    if re.match(r"^\s*\(\d+\)", label) and int(cell.get("column_index", 1)) == 0:
        return True, "footnote_marker_numeric"
    period, _ = _period_from_text(header)
    if period and _numeric_equals(fact.get("raw_value"), period[2:]):
        return True, "year_header_numeric"
    if re.search(r"\b(year ended|as of or for|in millions|except .*data)\b", label) and int(cell.get("column_index", 1)) == 0:
        return True, "header_caption_numeric"
    if value_kind == "unknown_numeric" and row_role in {"subtotal", "header"}:
        return True, "non_fact_unknown_numeric"
    return False, None


def _blocked_reason(fact: dict[str, Any], cell: dict[str, Any], row: dict[str, Any]) -> str:
    if not _text(fact.get("normalized_metric") or fact.get("normalized_metric_path")):
        return "metric_conflict"
    if _text(cell.get("period_status")).lower() == "conflicting":
        return "period_conflict"
    if _text(cell.get("scale_status")).lower() == "conflicting":
        return "scale_conflict"
    if _text(cell.get("value_kind")).lower() == "conflicting":
        return "value_kind_conflict"
    if _text(row.get("parent_row_id")) == _text(row.get("row_id")):
        return "invalid_row_parent"
    return "unknown"


def _record(fact: dict[str, Any], cell: dict[str, Any], row: dict[str, Any], table: dict[str, Any], primary: str, recoverability: str | None, reasons: list[str]) -> dict[str, Any]:
    trace = {"document_id": table.get("document_id"), "pdf_page": table.get("pdf_page"), "table_fragment_id": table.get("table_fragment_id"), "logical_table_id": table.get("logical_table_id"), "row_id": row.get("row_id"), "cell_id": cell.get("cell_id"), "fact_id": fact.get("fact_id")}
    paths = {str(key): sorted(value) for key, value in _period_candidates(table, cell, {fact.get("cell_id"): fact}).items() if value}
    return {"fact_id": fact.get("fact_id"), "cell_id": cell.get("cell_id"), "row_id": row.get("row_id"), "table_fragment_id": table.get("table_fragment_id"), "logical_table_id": table.get("logical_table_id"), "document_id": table.get("document_id"), "pdf_page": table.get("pdf_page"), "raw_value": fact.get("raw_value"), "parsed_value": fact.get("parsed_value"), "metric_path": fact.get("metric_path") or row.get("metric_path", []), "normalized_metric": fact.get("normalized_metric"), "header_path": cell.get("header_path", []), "period": fact.get("period"), "period_type": cell.get("period_type"), "scale": fact.get("scale") or cell.get("scale"), "currency": fact.get("currency") or cell.get("currency"), "value_kind": cell.get("value_kind"), "binding_status": fact.get("binding_status"), "evidence_level": "A" if fact.get("binding_status") == "complete" else "B", "row_role": row.get("row_role"), "column_index": cell.get("column_index"), "table_period_set": sorted({str(item.get("period")) for item in table.get("facts", []) if item.get("period")}), "row_period_set": sorted({str(item.get("period")) for item in table.get("facts", []) if item.get("row_id") == row.get("row_id") and item.get("period")}), "column_period_candidates": paths, "failure_reasons": sorted(set(reasons)), "eligibility_class": primary, "recoverability_class": recoverability, "source_traceback": trace}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate05", type=Path, default=DEFAULT_GATE05)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()
    stream = args.gate05 / "evidence-units.jsonl.gz"
    graph_path = args.graph / "header-graph-predictions.json"
    for path in (stream, graph_path, args.gate05 / "evidence-unit-metrics.json"):
        if not path.is_file():
            raise RuntimeError(f"missing_gate_05_r1_input:{path.name}")
    units = _load_units(stream)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    tables_by_fragment: dict[str, dict[str, Any]] = {}
    for page in graph.get("pages", []):
        for table in page.get("tables", []):
            tables_by_fragment[table.get("table_fragment_id")] = table
    facts = [unit for unit in units if unit.get("unit_type") == "fact"]
    excluded = [unit for unit in facts if unit.get("evidence_level") != "A" or unit.get("binding_status") != "complete"]
    records: list[dict[str, Any]] = []
    all_classes: dict[str, int] = {name: 0 for name in CLASSES}
    table_slices: dict[str, dict[str, Any]] = {}
    for fact in facts:
        trace = fact.get("source_traceback") or {}
        table = tables_by_fragment.get(trace.get("table_fragment_id"), {})
        rows = {row.get("row_id"): row for row in table.get("rows", [])}
        cells = {cell.get("cell_id"): cell for cell in table.get("cells", [])}
        row = rows.get(trace.get("row_id"), {})
        cell = cells.get(trace.get("cell_id"), {})
        reasons: list[str] = []
        if not fact.get("period"):
            reasons.append("missing_period")
        if not fact.get("normalized_metric") and not fact.get("normalized_metric_path"):
            reasons.append("missing_metric_path")
        if fact.get("binding_status") != "complete":
            reasons.append(f"binding_status_{fact.get('binding_status') or 'unknown'}")
        non_fact, non_fact_reason = _is_non_fact_numeric(fact, cell, row, table)
        if non_fact:
            primary, recoverability = "non_fact_numeric", non_fact_reason
        elif fact.get("binding_status") == "blocked":
            primary, recoverability = "structurally_blocked", _blocked_reason(fact, cell, row)
        elif fact.get("normalized_metric") and fact.get("period"):
            primary, recoverability = "eligible_recoverable", "already_complete"
        elif fact.get("normalized_metric"):
            candidates = _period_candidates(table, cell, {fact.get("cell_id"): fact})
            unique = set().union(*candidates.values()) if candidates else set()
            if len(unique) == 1:
                primary, recoverability = "eligible_recoverable", "period_candidate_available"
            else:
                primary, recoverability = "eligible_row_only", "period_not_unique"
        else:
            primary, recoverability = "unresolved", "metric_unresolved"
        all_classes[primary] += 1
        if fact in excluded:
            item = _record(fact, cell, row, table, primary, recoverability, reasons)
            records.append(item)
            key = f"{table.get('document_id')}:{table.get('table_fragment_id')}"
            slice_item = table_slices.setdefault(key, {"document_id": table.get("document_id"), "pdf_page": table.get("pdf_page"), "table_fragment_id": table.get("table_fragment_id"), "fact_count": 0, "class_counts": {}, "missing_period_count": 0, "missing_metric_count": 0})
            slice_item["fact_count"] += 1
            slice_item["class_counts"][primary] = slice_item["class_counts"].get(primary, 0) + 1
            slice_item["missing_period_count"] += int("missing_period" in reasons)
            slice_item["missing_metric_count"] += int("missing_metric_path" in reasons)
    records.sort(key=lambda item: str(item.get("fact_id")))
    record_ids = [str(item.get("fact_id")) for item in records]
    classification_map = []
    for fact in facts:
        trace = fact.get("source_traceback") or {}
        table = tables_by_fragment.get(trace.get("table_fragment_id"), {})
        rows = {row.get("row_id"): row for row in table.get("rows", [])}
        cells = {cell.get("cell_id"): cell for cell in table.get("cells", [])}
        row = rows.get(trace.get("row_id"), {})
        cell = cells.get(trace.get("cell_id"), {})
        non_fact, non_fact_reason = _is_non_fact_numeric(fact, cell, row, table)
        if non_fact:
            category = "non_fact_numeric"
            recoverability = non_fact_reason
        elif fact.get("binding_status") == "blocked":
            category = "structurally_blocked"
            recoverability = _blocked_reason(fact, cell, row)
        elif fact.get("normalized_metric") and fact.get("period"):
            category = "eligible_recoverable"
            recoverability = "already_complete"
        elif fact.get("normalized_metric"):
            category = "eligible_row_only"
            recoverability = "period_not_unique"
        else:
            category = "unresolved"
            recoverability = "metric_unresolved"
        classification_map.append({"fact_id": fact.get("fact_id"), "eligibility_class": category, "recoverability_class": recoverability})
    counts = {name: sum(item.get("eligibility_class") == name for item in records) for name in CLASSES}
    missing_period = [item for item in records if "missing_period" in item.get("failure_reasons", [])]
    missing_metric = [item for item in records if "missing_metric_path" in item.get("failure_reasons", [])]
    binding_counts: dict[str, int] = {}
    for item in records:
        status = str(item.get("binding_status") or "unknown")
        binding_counts[status] = binding_counts.get(status, 0) + 1
    args.out.mkdir(parents=True, exist_ok=True)
    _write(args.out / "fact-integrity-protocol.json", {"gate": "pdf_retrieval_v4_gate_05_r1", "evaluation_type": "post_benchmark_iterative_evaluation", "code_commit": args.code_commit, "input_gate": "pdf_retrieval_v4_gate_05", "classification_only": True, "fact_mutations": 0, "question_reads": 0, "gold_reads": 0, "expected_value_reads": 0, "index_builds": 0, "retrieval_runs": 0, "parameter_scan": False})
    _write(args.out / "fact-eligibility-audit.json", {"fact_record_count": len(facts), "excluded_fact_count": len(excluded), "excluded_fact_id_hash": _hash(sorted(record_ids)), "all_fact_class_counts": all_classes, "excluded_class_counts": counts, "eligible_financial_fact_count": len(facts) - all_classes["non_fact_numeric"], "eligible_complete_fact_count": all_classes["eligible_recoverable"], "records_included": "excluded-facts.jsonl.gz"})
    _write_jsonl_gz(args.out / "excluded-facts.jsonl.gz", {"format": "excluded_fact_jsonl_v1", "record_count": len(records), "source_gate": "pdf_retrieval_v4_gate_05"}, records)
    _write_jsonl_gz(args.out / "fact-classification-map.jsonl.gz", {"format": "fact_classification_map_v1", "record_count": len(classification_map), "source_gate": "pdf_retrieval_v4_gate_05"}, sorted(classification_map, key=lambda item: str(item.get("fact_id"))))
    _write(args.out / "period-failure-audit.json", {"record_count": len(missing_period), "class_counts": {key: sum(key in item.get("recoverability_class", "") or key in item.get("failure_reasons", []) for item in missing_period) for key in ("period_candidate_available", "period_not_unique", "missing_period")}, "records": missing_period})
    _write(args.out / "metric-failure-audit.json", {"record_count": len(missing_metric), "class_counts": {"row_metric_available": sum(bool(item.get("metric_path")) for item in missing_metric), "row_label_empty": sum(not item.get("metric_path") for item in missing_metric)}, "records": missing_metric})
    _write(args.out / "binding-status-audit.json", {"record_count": len(records), "binding_status_counts": dict(sorted(binding_counts.items())), "records": [{"fact_id": item.get("fact_id"), "binding_status": item.get("binding_status"), "eligibility_class": item.get("eligibility_class"), "failure_reasons": item.get("failure_reasons", [])} for item in records]})
    _write(args.out / "table-slice-audit.json", {"table_count": len(table_slices), "tables": sorted(table_slices.values(), key=lambda item: (str(item.get("document_id")), int(item.get("pdf_page") or -1), str(item.get("table_fragment_id"))))})
    unknown = sum(item.get("eligibility_class") not in CLASSES for item in records)
    missing_trace = sum(not item.get("source_traceback", {}).get("document_id") or item.get("source_traceback", {}).get("pdf_page") is None for item in records)
    integrity = {"excluded_fact_count": len(records), "expected_excluded_count": len(excluded), "classification_count_sum": sum(counts.values()), "classification_counts": counts, "classification_mutually_exclusive": all(sum(item.get("eligibility_class") == name for name in CLASSES) == 1 for item in records), "unknown_class_count": unknown, "duplicate_fact_id_count": len(record_ids) - len(set(record_ids)), "source_traceback_missing_count": missing_trace, "question_reads": 0, "gold_reads": 0, "fact_mutations": 0}
    _write(args.out / "classification-integrity.json", integrity)
    passed = len(records) == len(excluded) == 1028 and sum(counts.values()) == len(records) and unknown == 0 and integrity["classification_mutually_exclusive"] and integrity["duplicate_fact_id_count"] == 0 and missing_trace == 0
    decision = "fact_failure_taxonomy_complete" if passed else "fact_failure_taxonomy_blocked"
    next_gate = "fact_integrity_period_metric_recovery" if passed else "stop_and_fix_fact_classification"
    _write(args.out / "acceptance.json", {"gate": "pdf_retrieval_v4_gate_05_r1", "gate_passed": passed, "decision": decision, "next_gate": next_gate, "question_reads": 0, "gold_reads": 0, "expected_value_reads": 0, "index_builds": 0, "retrieval_runs": 0, "production_index_writes": 0, "parameter_scan": False, "per_query_oracle": False})
    _write(args.out / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    print(json.dumps({"decision": decision, "excluded_count": len(records), "class_counts": counts, "all_fact_class_counts": all_classes}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
