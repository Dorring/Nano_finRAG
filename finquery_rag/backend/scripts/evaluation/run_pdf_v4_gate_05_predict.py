"""Generate Oracle-blind Section/Table/Row/Cell/Fact Evidence Units for V4 Gate 05."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-03"
DEFAULT_LOGICAL = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-04"
DEFAULT_SHADOW = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-04c"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl_gz(path: Path, header: dict[str, Any], units: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        compressed.write((json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        for unit in units:
            compressed.write((json.dumps(unit, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _periods(table: dict[str, Any]) -> list[str]:
    return sorted({str(cell.get("normalized_period")) for cell in table.get("cells", []) if cell.get("normalized_period")})


def _metrics(table: dict[str, Any]) -> list[str]:
    return sorted({str(row.get("normalized_metric_path")) for row in table.get("rows", []) if row.get("normalized_metric_path")})


def _source(table: dict[str, Any], row: dict[str, Any] | None = None, cell: dict[str, Any] | None = None, fact: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"document_id": table.get("document_id"), "pdf_page": table.get("pdf_page"), "table_fragment_id": table.get("table_fragment_id"), "logical_table_id": table.get("logical_table_id"), "row_id": row.get("row_id") if row else None, "cell_id": cell.get("cell_id") if cell else None, "fact_id": fact.get("fact_id") if fact else None, "table_bbox": table.get("table_context", {}).get("table_bbox"), "row_bbox": row.get("row_bbox") if row else None, "cell_bbox": cell.get("cell_bbox") if cell else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--logical", type=Path, default=DEFAULT_LOGICAL)
    parser.add_argument("--shadow", type=Path, default=DEFAULT_SHADOW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    required = [args.graph / "header-graph-predictions.json", args.graph / "header-graph-prediction-seal.json", args.logical / "logical-tables.json", args.logical / "gate-04-prediction-seal.json", args.logical / "logical-table-integrity.json"]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"missing_gate_05_input:{path.name}")
    graph_pred = args.graph / "header-graph-predictions.json"
    graph_seal = json.loads((args.graph / "header-graph-prediction-seal.json").read_text(encoding="utf-8"))
    logical_pred = args.logical / "logical-tables.json"
    logical_seal = json.loads((args.logical / "gate-04-prediction-seal.json").read_text(encoding="utf-8"))
    gate04_pred = args.logical / "gate-04-predictions.json"
    if not graph_seal.get("predictions_sealed") or graph_seal.get("prediction_hash") != _sha(graph_pred):
        raise RuntimeError("gate_03_prediction_seal_invalid")
    if not logical_seal.get("predictions_sealed") or logical_seal.get("prediction_hash") != _sha(gate04_pred):
        raise RuntimeError("gate_04_prediction_seal_invalid")
    source = json.loads(graph_pred.read_text(encoding="utf-8"))
    logical_tables = json.loads(logical_pred.read_text(encoding="utf-8")).get("logical_tables", [])
    fragment_to_logical = {fragment_id: logical["logical_table_id"] for logical in logical_tables for fragment_id in logical.get("fragment_ids", [])}
    soft_groups: dict[str, str | None] = {}
    shadow_pred = args.shadow / "continuation-shadow-predictions.json"
    shadow_seal = args.shadow / "gate-04c-prediction-seal.json"
    if shadow_pred.is_file() and shadow_seal.is_file():
        shadow = json.loads(shadow_pred.read_text(encoding="utf-8"))
        soft_groups = {fragment_id: item.get("continuation_group_id") for item in shadow.get("links", []) if item.get("continuation_candidate") for fragment_id in (item.get("left_fragment_id"), item.get("right_fragment_id"))}
    tables = []
    for page in source.get("pages", []):
        for original in page.get("tables", []):
            table = dict(original)
            table["logical_table_id"] = fragment_to_logical.get(table["table_fragment_id"])
            tables.append(table)
    units: list[dict[str, Any]] = []
    section_keys: set[str] = set()
    for table in tables:
        logical_id = table["logical_table_id"]
        context = table.get("table_context", {})
        fragment_id = table["table_fragment_id"]
        continuation_group_id = soft_groups.get(fragment_id)
        source_pages = [table.get("pdf_page")]
        section_key = _hash([table.get("document_id"), context.get("statement"), context.get("section_path", []), context.get("title"), logical_id])
        if section_key not in section_keys:
            section_keys.add(section_key)
            units.append({"evidence_unit_id": "section:" + section_key, "unit_type": "section", "evidence_level": "D", "document_id": table.get("document_id"), "logical_table_id": logical_id, "fragment_id": fragment_id, "continuation_group_id": continuation_group_id, "cross_page_merged": False, "source_pages": source_pages, "statement": context.get("statement"), "section_path": context.get("section_path", []), "title": context.get("title"), "retrieval_text": " | ".join(filter(None, [str(table.get("document_id")), str(context.get("statement") or ""), " / ".join(context.get("section_path", [])), str(context.get("title") or "")])), "source_traceback": _source(table)})
        table_unit_id = "table:" + _hash([logical_id, fragment_id])
        units.append({"evidence_unit_id": table_unit_id, "unit_type": "table", "evidence_level": "C", "document_id": table.get("document_id"), "logical_table_id": logical_id, "fragment_id": fragment_id, "continuation_group_id": continuation_group_id, "cross_page_merged": False, "source_pages": source_pages, "title": context.get("title"), "statement": context.get("statement"), "metric_set": _metrics(table), "period_set": _periods(table), "scale": context.get("table_scale"), "currency": context.get("table_currency"), "retrieval_text": " | ".join(filter(None, [str(table.get("document_id")), str(context.get("statement") or ""), str(context.get("title") or ""), "metrics: " + ", ".join(_metrics(table)), "periods: " + ", ".join(_periods(table)), "scale: " + str(context.get("table_scale") or "")])), "source_traceback": _source(table)})
        cells_by_row: dict[int, list[dict[str, Any]]] = {}
        for cell in table.get("cells", []):
            cells_by_row.setdefault(int(cell.get("row_index", -1)), []).append(cell)
        facts_by_cell = {fact.get("cell_id"): fact for fact in table.get("facts", [])}
        for row in table.get("rows", []):
            row_index = int(row.get("row_index", -1))
            row_cells = cells_by_row.get(row_index, [])
            row_unit_id = "row:" + _hash([logical_id, fragment_id, row.get("row_id")])
            row_periods = sorted({str(cell.get("normalized_period")) for cell in row_cells if cell.get("normalized_period")})
            row_values = [str(cell.get("parsed_value")) for cell in row_cells if cell.get("parsed_value") is not None]
            row_level = "B" if row.get("normalized_metric_path") and row_values else "D"
            units.append({"evidence_unit_id": row_unit_id, "unit_type": "row", "evidence_level": row_level, "document_id": table.get("document_id"), "logical_table_id": logical_id, "fragment_id": fragment_id, "continuation_group_id": continuation_group_id, "cross_page_merged": False, "source_pages": source_pages, "row_id": row.get("row_id"), "row_index": row_index, "metric_path": row.get("metric_path", []), "normalized_metric_path": row.get("normalized_metric_path"), "period_set": row_periods, "values": row_values, "scale": context.get("table_scale"), "raw_text": row.get("raw_text"), "retrieval_text": " | ".join(filter(None, [str(table.get("document_id")), str(context.get("title") or ""), str(row.get("normalized_metric_path") or ""), "periods: " + ", ".join(row_periods), "values: " + ", ".join(row_values), str(row.get("raw_text") or "")])), "source_traceback": _source(table, row=row)})
            for cell in row_cells:
                fact = facts_by_cell.get(cell.get("cell_id"))
                level = "A" if cell.get("binding_status") == "complete" and cell.get("normalized_metric_path") and cell.get("normalized_period") and cell.get("parsed_value") is not None else ("B" if cell.get("normalized_metric_path") and cell.get("parsed_value") is not None else "D")
                cell_unit_id = "cell:" + _hash([logical_id, fragment_id, cell.get("cell_id")])
                units.append({"evidence_unit_id": cell_unit_id, "unit_type": "cell", "evidence_level": level, "document_id": table.get("document_id"), "logical_table_id": logical_id, "fragment_id": fragment_id, "continuation_group_id": continuation_group_id, "cross_page_merged": False, "source_pages": source_pages, "row_id": cell.get("row_id"), "cell_id": cell.get("cell_id"), "metric_path": cell.get("metric_path", []), "normalized_metric_path": cell.get("normalized_metric_path"), "header_path": cell.get("header_path", []), "normalized_period": cell.get("normalized_period"), "period_type": cell.get("period_type"), "raw_value": cell.get("raw_text"), "parsed_value": cell.get("parsed_value"), "scale": cell.get("scale"), "currency": cell.get("currency"), "value_kind": cell.get("value_kind"), "binding_status": cell.get("binding_status"), "retrieval_text": " | ".join(filter(None, [str(table.get("document_id")), str(cell.get("normalized_metric_path") or ""), str(cell.get("normalized_period") or ""), str(cell.get("raw_text") or ""), str(cell.get("scale") or "")])), "source_traceback": _source(table, row=row, cell=cell, fact=fact)})
                if fact:
                    fact_unit_id = "fact:" + _hash([logical_id, fragment_id, fact.get("fact_id")])
                    units.append({"evidence_unit_id": fact_unit_id, "unit_type": "fact", "evidence_level": "A" if fact.get("binding_status") == "complete" else "B", "document_id": table.get("document_id"), "logical_table_id": logical_id, "fragment_id": fragment_id, "continuation_group_id": continuation_group_id, "cross_page_merged": False, "source_pages": source_pages, "fact_id": fact.get("fact_id"), "cell_id": fact.get("cell_id"), "row_id": fact.get("row_id"), "metric_path": fact.get("metric_path", []), "normalized_metric": fact.get("normalized_metric"), "period": fact.get("period"), "raw_value": fact.get("raw_value"), "parsed_value": fact.get("parsed_value"), "scale": fact.get("scale"), "currency": fact.get("currency"), "base_value": fact.get("base_value"), "binding_status": fact.get("binding_status"), "retrieval_text": " | ".join(filter(None, [str(table.get("document_id")), str(fact.get("normalized_metric") or ""), str(fact.get("period") or ""), str(fact.get("raw_value") or ""), str(fact.get("scale") or "")])), "source_traceback": _source(table, row=row, cell=cell, fact=fact)})
    ids = [unit["evidence_unit_id"] for unit in units]
    integrity = {"unit_count": len(units), "section_count": sum(unit["unit_type"] == "section" for unit in units), "table_count": sum(unit["unit_type"] == "table" for unit in units), "row_count": sum(unit["unit_type"] == "row" for unit in units), "cell_count": sum(unit["unit_type"] == "cell" for unit in units), "fact_count": sum(unit["unit_type"] == "fact" for unit in units), "level_a_count": sum(unit["evidence_level"] == "A" for unit in units), "level_b_count": sum(unit["evidence_level"] == "B" for unit in units), "level_c_count": sum(unit["evidence_level"] == "C" for unit in units), "level_d_count": sum(unit["evidence_level"] == "D" for unit in units), "duplicate_unit_id_count": len(ids) - len(set(ids)), "source_traceback_missing_count": sum(not unit.get("source_traceback", {}).get("document_id") for unit in units), "cross_page_merged_count": sum(unit.get("cross_page_merged") for unit in units), "soft_continuation_unit_count": sum(bool(unit.get("continuation_group_id")) for unit in units)}
    protocol = {"gate": "pdf_retrieval_v4_gate_05", "evaluation_type": "post_benchmark_iterative_evaluation", "code_commit": args.code_commit, "input_gate": "pdf_retrieval_v4_gate_04", "gate_03_prediction_hash": _sha(graph_pred), "gate_04_prediction_hash": _sha(gate04_pred), "gate_04c_shadow_optional": shadow_pred.is_file(), "cross_page_merged": False, "soft_continuation_links_allowed": True, "oracle_blind": True, "question_reads": 0, "governance_reads": 0, "oracle_reads": 0, "expected_value_reads": 0, "index_builds": 0, "retrieval_runs": 0, "reranker_calls": 0, "production_index_writes": 0, "production_switch_allowed": False}
    input_integrity = {"gate_03_prediction_sha256": _sha(graph_pred), "gate_03_seal_sha256": _sha(args.graph / "header-graph-prediction-seal.json"), "gate_04_prediction_sha256": _sha(gate04_pred), "gate_04_seal_sha256": _sha(args.logical / "gate-04-prediction-seal.json"), "gate_04_logical_integrity_sha256": _sha(args.logical / "logical-table-integrity.json"), "gate_04c_prediction_sha256": _sha(shadow_pred) if shadow_pred.is_file() else None, "source_table_count": len(tables)}
    _write(args.out / "gate-05-protocol.json", protocol)
    _write(args.out / "gate-05-input-integrity.json", input_integrity)
    prediction_payload = {"prediction_count": len(tables), "unit_count": len(units), "cross_page_merged": False, "evidence_units": units}
    prediction_path = args.out / "evidence-unit-predictions.jsonl.gz"
    evidence_path = args.out / "evidence-units.jsonl.gz"
    for stale in (args.out / "evidence-unit-predictions.json", args.out / "evidence-units.json"):
        if stale.exists():
            stale.unlink()
    stream_header = {"format": "evidence_unit_jsonl_v1", "prediction_count": len(tables), "unit_count": len(units), "cross_page_merged": False}
    _write_jsonl_gz(prediction_path, stream_header, units)
    _write_jsonl_gz(evidence_path, stream_header, units)
    uncompressed_hash = _hash(prediction_payload)
    _write(args.out / "evidence-units-manifest.json", {"storage": evidence_path.name, "record_count": len(units) + 1, "compressed_sha256": _sha(evidence_path), "uncompressed_sha256": uncompressed_hash, "compression": "gzip", "deterministic": True})
    _write(args.out / "evidence-unit-integrity.json", integrity)
    _write(args.out / "evidence-unit-metrics.json", integrity)
    _write(args.out / "evidence-unit-prediction-seal.json", {"prediction_count": len(tables), "unit_count": len(units), "oracle_reads_before_seal": 0, "question_reads": 0, "governance_reads": 0, "input_hash": _sha(args.out / "gate-05-input-integrity.json"), "protocol_hash": _sha(args.out / "gate-05-protocol.json"), "prediction_hash": _sha(prediction_path), "prediction_storage": prediction_path.name, "prediction_uncompressed_sha256": uncompressed_hash, "predictions_sealed": True, "cross_page_merged": False})
    _write(args.out / "acceptance.json", {"gate": "pdf_retrieval_v4_gate_05", "prediction_sealed": True, "decision": "pending_posthoc_scoring", "next_gate": "score_evidence_units", "cross_page_merged": False, "oracle_reads_runtime": 0, "question_reads": 0, "governance_reads": 0, "index_builds": 0, "retrieval_runs": 0, "production_index_writes": 0, "production_switch_allowed": False})
    _write(args.out / "next-gate.json", {"decision": "pending_posthoc_scoring", "next_gate": "score_evidence_units", "production_switch_allowed": False})
    print(json.dumps({"table_fragments": len(tables), "unit_count": len(units), "counts": integrity, "cross_page_merged": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
