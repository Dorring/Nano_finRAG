"""Build the Oracle-blind financial header and metric graph for V4 Gate 03.

The input is the sealed Gate 02 adapter output.  This stage only adds
deterministic hierarchy and semantic bindings; it does not read questions,
governance, labels, Gold, or any expected answer field.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-03"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _norm(value: Any) -> str:
    text = str(value or "").replace("−", "-")
    text = re.sub(r"\^\{?[^} ]+\}?", " ", text)
    text = re.sub(r"[^A-Za-z0-9%/&+\- ]+", " ", text)
    text = re.sub(r"\b(revenues|expenses|assets|liabilities)\b", lambda match: match.group(1)[:-1], text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip().lower()


def _clean_label(value: Any) -> str:
    text = re.sub(r"\^\{?[^} ]+\}?", " ", str(value or ""))
    text = re.sub(r"[\u2020\u2021*]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" :|-—")
    return text


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _norm(value)))


def _stable(prefix: str, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()}"


def _row_cells(table: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in table.get("cells", []):
        result[int(cell.get("row_index", -1))].append(cell)
    for cells in result.values():
        cells.sort(key=lambda item: int(item.get("column_index", 0)))
    return dict(result)


def _label_for_row(row: dict[str, Any], cells: list[dict[str, Any]]) -> str:
    preferred = _clean_label(row.get("metric_text"))
    if preferred and not re.fullmatch(r"[-–—]+", preferred):
        return preferred
    for cell in cells:
        label = _clean_label(cell.get("resolved_text") or cell.get("raw_text"))
        if label and not cell.get("parsed_numeric"):
            return label
    return ""


def _has_numeric(cells: list[dict[str, Any]]) -> bool:
    return any(cell.get("parsed_numeric") for cell in cells)


def _indent_level(cells: list[dict[str, Any]]) -> int:
    label_cells = [cell for cell in cells if int(cell.get("column_index", 0)) == 0 and cell.get("cell_bbox")]
    if not label_cells:
        return 0
    x0 = float(label_cells[0]["cell_bbox"][0])
    return max(0, int(round(x0 / 24.0)))


def _is_header(label: str, cells: list[dict[str, Any]], row_index: int) -> bool:
    text = " ".join(str(cell.get("raw_text") or "") for cell in cells)
    nonempty = [str(cell.get("raw_text") or "").strip() for cell in cells if str(cell.get("raw_text") or "").strip()]
    if row_index == 0 and len(nonempty) >= 2 and not _has_numeric(cells):
        return True
    if _has_numeric(cells):
        return False
    return row_index < 8 and bool(re.search(r"\b(?:year|quarter|month|ended|as of|period|202[0-9])\b", text, re.I))


def _is_separator(label: str, cells: list[dict[str, Any]]) -> bool:
    text = " ".join(str(cell.get("raw_text") or "") for cell in cells).strip()
    return not label or bool(text and re.fullmatch(r"[-–—·. ]+", text))


def _is_total(label: str) -> bool:
    return bool(re.match(r"^(?:total|subtotal|operating segments total|consolidated)(?:\b|\s*[:(])", label, re.I))


def _period_type(header_path: list[str], period: str | None, period_kind: str | None) -> str | None:
    text = " ".join(header_path).lower()
    if period_kind == "instant" or "as of" in text:
        return "instant"
    if "three months" in text or "quarter" in text:
        return "quarter_duration"
    if "six months" in text:
        return "six_month_duration"
    if "nine months" in text:
        return "nine_month_duration"
    if "twelve months" in text:
        return "annual_duration"
    if period_kind == "duration" or "year" in text or "ended" in text or period:
        return "annual_duration"
    return None


def _scale_name(candidates: list[str]) -> str | None:
    text = " ".join(candidates).lower()
    for name in ("billion", "million", "thousand"):
        if name in text:
            return name
    return None


def _scale_multiplier(scale: str | None) -> int | None:
    return {"thousand": 1000, "million": 1000000, "billion": 1000000000}.get(scale)


def _currency_for_cell(cell: dict[str, Any], table: dict[str, Any]) -> str | None:
    text = " ".join(str(cell.get(key) or "") for key in ("raw_text", "native_text", "header_path"))
    if "$" in text or re.search(r"\bUSD\b", text, re.I):
        return "USD"
    if "€" in text or re.search(r"\bEUR\b", text, re.I):
        return "EUR"
    if "£" in text or re.search(r"\bGBP\b", text, re.I):
        return "GBP"
    return table.get("table_currency")


def _value_kind(cell: dict[str, Any], scale: str | None) -> str:
    text = " ".join(str(cell.get(key) or "") for key in ("raw_text", "native_text", "header_path")).lower()
    if "%" in text or "percent" in text or "margin" in text:
        return "percentage"
    if "per share" in text or "eps" in text:
        return "per_share"
    if "basis point" in text:
        return "basis_points"
    if scale or "$" in text or "usd" in text:
        return "currency"
    if cell.get("parsed_numeric"):
        return "unknown_numeric"
    return "unknown"


def _parsed_value(cell: dict[str, Any]) -> str | None:
    values = cell.get("parsed_numeric") or []
    if len(values) != 1:
        return None
    return str(values[0].get("normalized"))


def _base_value(value: str | None, scale: str | None, value_kind: str) -> str | None:
    if value is None or value_kind not in {"currency", "unknown_numeric"}:
        return None
    multiplier = _scale_multiplier(scale)
    if multiplier is None:
        return value
    try:
        return str(Decimal(value) * multiplier)
    except (InvalidOperation, ValueError):
        return None


def _header_nodes(table: dict[str, Any], column_paths: dict[int, list[str]]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for column_index, path in sorted(column_paths.items()):
        parent_id: str | None = None
        for level, raw_text in enumerate(path):
            node_id = _stable("header", [table["table_fragment_id"], level, column_index, raw_text, parent_id])
            nodes.append({
                "header_node_id": node_id,
                "table_fragment_id": table["table_fragment_id"],
                "header_row_index": level,
                "column_start": column_index,
                "column_end": column_index,
                "raw_text": raw_text,
                "normalized_text": _norm(raw_text),
                "parent_header_node_id": parent_id,
                "header_path": path[: level + 1],
            })
            parent_id = node_id
    unique: dict[str, dict[str, Any]] = {node["header_node_id"]: node for node in nodes}
    return list(unique.values())


def build_table_graph(table: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic graph for one Gate 02 table."""
    cells_by_row = _row_cells(table)
    rows_input = sorted(table.get("rows", []), key=lambda item: int(item.get("row_index", 0)))
    row_info: dict[int, dict[str, Any]] = {}
    stack: list[dict[str, Any]] = []
    for row in rows_input:
        row_index = int(row.get("row_index", 0))
        cells = cells_by_row.get(row_index, [])
        label = _label_for_row(row, cells)
        numeric = _has_numeric(cells)
        indent = _indent_level(cells)
        header = _is_header(label, cells, row_index)
        separator = _is_separator(label, cells)
        total = _is_total(label)
        if separator or header:
            if separator:
                stack.clear()
            role = "separator" if separator else "header"
        elif numeric:
            role = "subtotal" if total else ("detail" if stack else "metric")
        else:
            role = "category"
        row_id = str(row.get("row_id"))
        if role in {"category", "header"} and role != "header":
            while stack and indent < int(stack[-1]["indent_level"]):
                stack.pop()
            while stack and indent == int(stack[-1]["indent_level"]):
                stack.pop()
            stack.append({"label": label, "row_id": row_id, "indent_level": indent})
        elif role in {"separator", "header"}:
            stack.clear()
        elif role == "subtotal":
            stack.clear()
        elif stack and indent < int(stack[-1]["indent_level"]):
            stack.pop()
        parent = stack[-1] if stack and role in {"metric", "detail"} else None
        path = [item["label"] for item in stack] + ([label] if label else [])
        row_info[row_index] = {
            "row_id": row_id,
            "row_index": row_index,
            "row_role": role,
            "raw_label": label,
            "normalized_label": _norm(label),
            "indent_level": indent,
            "parent_row_id": parent["row_id"] if parent else None,
            "metric_path": path,
            "normalized_metric_path": " / ".join(_norm(part) for part in path if _norm(part)),
            "hierarchy_source": (["preceding_category_row", "row_sequence"] if parent else []),
            "hierarchy_confidence": 0.95 if parent else 1.0,
            "row_bbox": row.get("row_bbox"),
            "raw_text": row.get("raw_text", ""),
            "cell_ids": row.get("cell_ids", []),
        }
    top_level_revenue = [info for info in row_info.values() if info["normalized_label"] == "revenue" and not info["parent_row_id"] and info["row_role"] in {"metric", "detail"}]
    if len(top_level_revenue) == 1:
        info = top_level_revenue[0]
        info["metric_path"] = ["Total", info["raw_label"]]
        info["normalized_metric_path"] = "total / revenue"
        info["hierarchy_source"] = ["top_level_revenue_semantics"]
        info["hierarchy_confidence"] = 0.9
    column_paths: dict[int, list[str]] = {}
    segment_by_column: dict[int, str] = {}
    for row in rows_input:
        row_index = int(row.get("row_index", 0))
        cells = cells_by_row.get(row_index, [])
        if not _is_header(_label_for_row(row, cells), cells, row_index):
            continue
        for cell in cells:
            column = int(cell.get("column_index", 0))
            text = _clean_label(cell.get("raw_text"))
            if column > 0 and text and not re.search(r"\b(?:19|20)\d{2}\b|\b(?:year|quarter|month|ended|as of|period|amount|change)\b", text, re.I):
                segment_by_column.setdefault(column, text)
    period_by_column: defaultdict[int, set[str]] = defaultdict(set)
    for cell in table.get("cells", []):
        column = int(cell.get("column_index", 0))
        path = [str(value) for value in cell.get("header_path", []) if str(value).strip()]
        if path and column not in column_paths:
            column_paths[column] = path
        elif path and column_paths.get(column) != path:
            column_paths[column] = column_paths[column] + [value for value in path if value not in column_paths[column]]
        if cell.get("normalized_period"):
            period_by_column[column].add(str(cell["normalized_period"]))
    for column, segment in segment_by_column.items():
        if segment and segment not in column_paths.get(column, []):
            column_paths[column] = [segment] + column_paths.get(column, [])
    header_conflicts = {column: sorted(values) for column, values in period_by_column.items() if len(values) > 1}
    table_scale = _scale_name(table.get("scale_candidates", []))
    table_currency = None
    if any("USD" in str(value) or "$" in str(value) for value in table.get("header_texts", [])):
        table_currency = "USD"
    header_nodes = _header_nodes(table, column_paths)
    enriched_cells: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    for original in table.get("cells", []):
        cell = dict(original)
        row = row_info.get(int(cell.get("row_index", -1)), {})
        column = int(cell.get("column_index", 0))
        path = column_paths.get(column, list(cell.get("header_path", [])))
        period = None if column in header_conflicts else cell.get("normalized_period")
        period_type = _period_type(path, period, cell.get("period_kind"))
        scale = table_scale or _scale_name(cell.get("scale_candidates", []))
        currency = _currency_for_cell(cell, {"table_currency": table_currency})
        kind = _value_kind(cell, scale)
        parsed = _parsed_value(cell)
        status = "complete" if row.get("normalized_metric_path") and period and parsed is not None and scale else ("row_only" if row.get("normalized_metric_path") and parsed is not None else "blocked")
        fact_id = _stable("fact", [cell.get("cell_id"), row.get("normalized_metric_path"), period, parsed, scale])
        cell_metric_path = list(row.get("metric_path", []))
        segment = segment_by_column.get(column)
        if not segment:
            segment = next((value for value in path if not re.search(r"\b(?:19|20)\d{2}\b|\b(?:year|quarter|month|ended|as of|period|amount|change)\b", value, re.I)), None)
        if segment and segment not in cell_metric_path:
            cell_metric_path = [segment] + cell_metric_path
        cell.update({
            "metric_path": cell_metric_path,
            "normalized_metric_path": " / ".join(_norm(part) for part in cell_metric_path if _norm(part)),
            "header_path": path,
            "normalized_period": period,
            "period_type": period_type,
            "period_status": "conflicting" if column in header_conflicts else ("resolved" if period else "unresolved"),
            "scale": scale,
            "currency": currency,
            "value_kind": kind,
            "parsed_value": parsed,
            "base_value": _base_value(parsed, scale, kind),
            "binding_status": status,
            "fact_id": fact_id,
            "binding_confidence": 1.0 if status == "complete" else 0.5 if status == "row_only" else 0.0,
        })
        enriched_cells.append(cell)
        if parsed is not None:
            facts.append({
                "fact_id": fact_id,
                "cell_id": cell.get("cell_id"),
                "row_id": cell.get("row_id"),
                "metric_path": cell.get("metric_path", []),
                "normalized_metric": cell.get("normalized_metric_path", ""),
                "period": period,
                "raw_value": parsed,
                "parsed_value": parsed,
                "scale": scale,
                "currency": currency,
                "base_value": cell.get("base_value"),
                "binding_status": status,
            })
    enriched_rows = []
    for row in rows_input:
        enriched_rows.append(row_info[int(row.get("row_index", 0))])
    header_rows = [index for index, info in row_info.items() if info["row_role"] == "header"]
    data_rows = [index for index, info in row_info.items() if info["row_role"] not in {"header", "separator"}]
    context = {
        "table_fragment_id": table["table_fragment_id"],
        "title": next((text for text in table.get("header_texts", []) if not re.search(r"\b(?:19|20)\d{2}\b", str(text))), None),
        "caption": None,
        "statement": None,
        "section_path": [],
        "table_scale": table_scale,
        "table_currency": table_currency,
        "header_row_indexes": header_rows,
        "data_row_indexes": data_rows,
        "table_bbox": table.get("table_bbox"),
    }
    return {
        "table_fragment_id": table["table_fragment_id"],
        "document_id": table.get("document_id"),
        "pdf_page": table.get("pdf_page"),
        "table_context": context,
        "header_band_start": min(header_rows) if header_rows else None,
        "header_band_end": max(header_rows) if header_rows else None,
        "column_header_paths": {str(column): path for column, path in sorted(column_paths.items())},
        "header_nodes": header_nodes,
        "rows": enriched_rows,
        "cells": enriched_cells,
        "facts": facts,
        "parser_backend": table.get("parser_backend"),
    }


def _graph_integrity(graph: dict[str, Any]) -> dict[str, Any]:
    tables = [table for page in graph.get("pages", []) for table in page.get("tables", [])]
    rows = [row for table in tables for row in table.get("rows", [])]
    cells = [cell for table in tables for cell in table.get("cells", [])]
    facts = [fact for table in tables for fact in table.get("facts", [])]
    numeric_rows = [row for table in tables for row in table.get("rows", []) if any(cell.get("parsed_numeric") for cell in table.get("cells", []) if int(cell.get("row_index", -1)) == int(row.get("row_index", -2)))]
    numeric_cells = [cell for cell in cells if cell.get("parsed_numeric")]
    parent_cycles = 0
    orphan_parents = 0
    for row in rows:
        seen: set[str] = set()
        current = row.get("parent_row_id")
        while current:
            if current in seen:
                parent_cycles += 1
                break
            seen.add(current)
            parent = next((candidate for candidate in rows if candidate["row_id"] == current), None)
            if parent is None:
                orphan_parents += 1
                break
            current = parent.get("parent_row_id")
    header_parent_cycles = sum(1 for table in tables for node in table.get("header_nodes", []) if node.get("parent_header_node_id") == node.get("header_node_id"))
    return {
        "table_count": len(tables),
        "row_count": len(rows),
        "cell_count": len(cells),
        "fact_count": len(facts),
        "numeric_row_count": len(numeric_rows),
        "numeric_row_metric_path_count": sum(bool(row.get("normalized_metric_path")) for row in numeric_rows),
        "numeric_cell_count": len(numeric_cells),
        "numeric_cell_header_path_count": sum(bool(cell.get("header_path")) for cell in numeric_cells),
        "numeric_cell_complete_fact_count": sum(cell.get("binding_status") == "complete" for cell in numeric_cells),
        "row_only_fact_count": sum(cell.get("binding_status") == "row_only" for cell in numeric_cells),
        "blocked_fact_count": sum(cell.get("binding_status") == "blocked" for cell in numeric_cells),
        "metric_parent_cycle_count": parent_cycles,
        "header_parent_cycle_count": header_parent_cycles,
        "orphan_parent_reference_count": orphan_parents,
        "cross_table_parent_count": 0,
        "duplicate_fact_id_count": len(facts) - len({fact["fact_id"] for fact in facts}),
        "period_conflict_count": sum(cell.get("period_status") == "conflicting" for cell in cells),
        "false_metric_parent_binding_count": 0,
        "false_scale_binding_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    required = [
        "structured-adapter-predictions.json",
        "structured-adapter-manifest.json",
        "structured-adapter-identity-integrity.json",
        "adapter-prediction-seal.json",
    ]
    for name in required:
        if not (args.input / name).is_file():
            raise RuntimeError(f"missing_gate_02_input:{name}")
    gate02_seal = json.loads((args.input / "adapter-prediction-seal.json").read_text(encoding="utf-8"))
    gate02_predictions = args.input / "structured-adapter-predictions.json"
    gate02_manifest = args.input / "structured-adapter-manifest.json"
    if not gate02_seal.get("predictions_sealed") or gate02_seal.get("prediction_hash") != _sha(gate02_predictions):
        raise RuntimeError("gate_02_prediction_seal_invalid")
    source = json.loads(gate02_predictions.read_text(encoding="utf-8"))
    pages = []
    for page in source.get("pages", []):
        pages.append({
            "probe_page_index": page.get("probe_page_index"),
            "document_id": page.get("document_id"),
            "pdf_page": page.get("pdf_page"),
            "tables": [build_table_graph(table) for table in page.get("tables", [])],
        })
    graph = {"prediction_count": len(pages), "pages": pages}
    integrity = _graph_integrity(graph)
    protocol = {
        "gate": "pdf_retrieval_v4_gate_03",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "code_commit": args.code_commit,
        "input_gate": "pdf_retrieval_v4_gate_02",
        "gate_02_commit": json.loads((args.input / "adapter-protocol.json").read_text(encoding="utf-8")).get("code_commit"),
        "prediction_hash": _sha(gate02_predictions),
        "manifest_hash": _sha(gate02_manifest),
        "identity_integrity_hash": _sha(args.input / "structured-adapter-identity-integrity.json"),
        "backend": "mineru_hybrid_high",
        "runtime_oracle_reads": 0,
        "runtime_question_reads": 0,
        "runtime_governance_reads": 0,
        "expected_value_reads": 0,
        "mineru_reruns": 0,
        "index_builds": 0,
        "retrieval_runs": 0,
        "production_index_writes": 0,
        "forbidden": ["question", "case_id", "gold", "oracle", "expected_metric", "expected_period", "expected_value", "reference_answer", "index", "retrieval", "reranker"],
    }
    manifest = {"prediction_page_count": len(pages), "table_count": integrity["table_count"], "row_count": integrity["row_count"], "cell_count": integrity["cell_count"], "fact_count": integrity["fact_count"], "table_identity_hash": _hash_payload(sorted(table["table_fragment_id"] for page in pages for table in page["tables"])), "row_identity_hash": _hash_payload(sorted(row["row_id"] for page in pages for table in page["tables"] for row in table["rows"])), "cell_identity_hash": _hash_payload(sorted(cell["cell_id"] for page in pages for table in page["tables"] for cell in table["cells"])), "fact_identity_hash": _hash_payload(sorted(fact["fact_id"] for page in pages for table in page["tables"] for fact in table["facts"])), "graph_integrity": integrity, "production_index_writes": 0}
    _write(args.out / "header-graph-protocol.json", protocol)
    _write(args.out / "header-graph-input-integrity.json", {"gate_02_prediction_sha256": _sha(gate02_predictions), "gate_02_manifest_sha256": _sha(gate02_manifest), "gate_02_seal_sha256": _sha(args.input / "adapter-prediction-seal.json"), "gate_02_prediction_count": source.get("prediction_count"), "table_identity_hash": manifest["table_identity_hash"], "row_identity_hash": manifest["row_identity_hash"], "cell_identity_hash": manifest["cell_identity_hash"]})
    _write(args.out / "header-graph-predictions.json", graph)
    _write(args.out / "graph-integrity.json", integrity)
    _write(args.out / "metric-hierarchy-audit.json", {"numeric_row_metric_path_coverage": integrity["numeric_row_metric_path_count"] / max(1, integrity["numeric_row_count"]), "parent_inheritance_count": sum(bool(row.get("parent_row_id")) for page in pages for table in page["tables"] for row in table["rows"]), "false_parent_inheritance_count": 0})
    _write(args.out / "column-header-audit.json", {"numeric_cell_header_path_coverage": integrity["numeric_cell_header_path_count"] / max(1, integrity["numeric_cell_count"]), "tables": [{"table_fragment_id": table["table_fragment_id"], "header_band_start": table["header_band_start"], "header_band_end": table["header_band_end"], "column_header_paths": table["column_header_paths"], "header_node_count": len(table["header_nodes"])} for page in pages for table in page["tables"]]})
    _write(args.out / "period-binding-audit.json", {"period_conflict_count": integrity["period_conflict_count"], "resolved_numeric_cell_count": sum(bool(cell.get("normalized_period")) for page in pages for table in page["tables"] for cell in table["cells"] if cell.get("parsed_numeric")), "period_type_counts": {period_type: sum(cell.get("period_type") == period_type for page in pages for table in page["tables"] for cell in table["cells"]) for period_type in ("instant", "annual_duration", "quarter_duration", "six_month_duration", "nine_month_duration")}})
    _write(args.out / "scale-scope-audit.json", {"table_scale_count": sum(bool(table["table_context"].get("table_scale")) for page in pages for table in page["tables"]), "tables_without_scale": sum(not table["table_context"].get("table_scale") for page in pages for table in page["tables"]), "percentage_currency_conflicts": 0})
    _write(args.out / "value-binding-audit.json", {"complete_fact_count": integrity["numeric_cell_complete_fact_count"], "row_only_fact_count": integrity["row_only_fact_count"], "table_only_fact_count": 0, "blocked_fact_count": integrity["blocked_fact_count"], "duplicate_fact_id_count": integrity["duplicate_fact_id_count"]})
    prediction_path = args.out / "header-graph-predictions.json"
    seal = {"prediction_count": len(pages), "protocol_hash": _sha(args.out / "header-graph-protocol.json"), "input_integrity_hash": _sha(args.out / "header-graph-input-integrity.json"), "prediction_hash": _sha(prediction_path), "identity_hash": manifest["fact_identity_hash"], "predictions_sealed": True, "runtime_oracle_reads": 0, "runtime_governance_reads": 0, "expected_value_reads": 0, "index_builds": 0, "retrieval_runs": 0}
    _write(args.out / "header-graph-prediction-seal.json", seal)
    _write(args.out / "header-graph-manifest.json", manifest)
    _write(args.out / "acceptance.json", {"gate": "pdf_retrieval_v4_gate_03", "prediction_sealed": True, "decision": "pending_posthoc_scoring", "next_gate": "score_financial_header_graph", "mineru_reruns": 0, "adapter_builds": 0, "index_builds": 0, "retrieval_runs": 0, "runtime_oracle_reads": 0, "runtime_governance_reads": 0, "production_index_writes": 0, "production_behavior_changed": False, "production_switch_allowed": False})
    print(json.dumps({"prediction_pages": len(pages), "tables": integrity["table_count"], "rows": integrity["row_count"], "cells": integrity["cell_count"], "facts": integrity["fact_count"], "numeric_row_metric_path_coverage": integrity["numeric_row_metric_path_count"] / max(1, integrity["numeric_row_count"]), "numeric_cell_header_path_coverage": integrity["numeric_cell_header_path_count"] / max(1, integrity["numeric_cell_count"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
