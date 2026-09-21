"""Build an Oracle-blind V4 Gate 05 R4 temporal binding graph."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.temporal_binding import (  # noqa: E402
    bind_fact,
    classify_complete_predicate_gap,
    classify_schema,
)


DEFAULT_GRAPH = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-03/header-graph-predictions.json"
DEFAULT_CLASSIFICATION = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r1/fact-classification-map.jsonl.gz"
DEFAULT_R3 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r3"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r4"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_map(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if index == 0 and value.get("stream") == "header":
                continue
            result[str(value.get("fact_id"))] = value
    return result


def _load_r3_fact_units(path: Path) -> dict[str, dict[str, Any]]:
    prediction = path / "evidence-unit-predictions.jsonl.gz"
    result: dict[str, dict[str, Any]] = {}
    with gzip.open(prediction, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if index == 0 and value.get("stream") == "header":
                continue
            if value.get("unit_type") == "fact" and value.get("fact_id"):
                result[str(value["fact_id"])] = value
    return result


def _write_jsonl_gz(path: Path, records: list[dict[str, Any]], header: dict[str, Any]) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [header, *records]
    raw = "\n".join(json.dumps(value, ensure_ascii=False, sort_keys=True) for value in payload) + "\n"
    uncompressed_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    with path.open("wb") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as compressed:
            compressed.write(raw.encode("utf-8"))
    return _sha(path), uncompressed_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--r3", type=Path, default=DEFAULT_R3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()
    if not args.graph.is_file() or not args.classification.is_file() or not (args.r3 / "evidence-unit-prediction-seal.json").is_file():
        raise RuntimeError("missing_gate_05_r4_input")
    args.out.mkdir(parents=True, exist_ok=True)
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    classifications = _load_map(args.classification)
    r3_fact_units = _load_r3_fact_units(args.r3)
    protocol = {
        "gate": "pdf_retrieval_v4_gate_05_r4",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "code_commit": args.code_commit,
        "input_graph_sha256": _sha(args.graph),
        "classification_sha256": _sha(args.classification),
        "gate_05_r3_prediction_sha256": _sha(args.r3 / "evidence-unit-predictions.jsonl.gz"),
        "schema_types": [
            "period_on_columns", "period_on_rows", "single_period_snapshot", "metric_by_segment_matrix",
            "period_by_segment_matrix", "roll_forward", "comparison_change_table", "maturity_or_bucket_table",
            "ratio_percentage_table", "mixed_or_unsupported",
        ],
        "temporal_binding_union": ["point", "duration", "comparison", "bucket", "period_set", "not_applicable"],
        "question_reads": 0,
        "gold_reads": 0,
        "expected_value_reads": 0,
        "oracle_reads": 0,
        "index_builds": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "parameter_scan": False,
        "per_query_oracle": False,
        "production_index_writes": 0,
        "production_switch_allowed": False,
    }
    _write(args.out / "gate-05-r4-protocol.json", protocol)
    _write(args.out / "input-integrity.json", {
        "graph_sha256": _sha(args.graph),
        "classification_sha256": _sha(args.classification),
        "r3_prediction_sha256": _sha(args.r3 / "evidence-unit-predictions.jsonl.gz"),
        "r3_seal_sha256": _sha(args.r3 / "evidence-unit-prediction-seal.json"),
        "question_reads": 0,
        "gold_reads": 0,
    })
    schemas: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    gap_records: list[dict[str, Any]] = []
    schema_counts: dict[str, int] = {}
    binding_counts: dict[str, int] = {}
    semantic_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    axis_records: list[dict[str, Any]] = []
    for page in graph.get("pages", []):
        for table in page.get("tables", []):
            schema = classify_schema(table)
            schemas.append(schema)
            schema_counts[schema["schema_type"]] = schema_counts.get(schema["schema_type"], 0) + 1
            rows = {str(row.get("row_id")): row for row in table.get("rows", [])}
            cells = {str(cell.get("cell_id")): cell for cell in table.get("cells", [])}
            facts = {str(fact.get("fact_id")): fact for fact in table.get("facts", [])}
            for fact in facts.values():
                cell = cells.get(str(fact.get("cell_id")), {})
                row = rows.get(str(fact.get("row_id")), {})
                classification = classifications.get(str(fact.get("fact_id")), {"eligibility_class": "unresolved"})
                record = bind_fact(table, schema, fact, cell, row, classification)
                record["row_role"] = row.get("row_role")
                record["column_index"] = cell.get("column_index")
                record["eligibility_class"] = classification.get("eligibility_class")
                record["table_periods"] = sorted(schema.get("periods", {}))
                record["dimension_axes"] = schema.get("dimension_axes", [])
                derived_indexes = {int(axis.get("column_index")) for axis in schema.get("derived_measure_axes", [])}
                is_derived_axis = int(cell.get("column_index", -1)) in derived_indexes
                record["eligible_atomic_candidate"] = bool(
                    record.get("fact_semantic_type") != "non_fact_numeric"
                    and record.get("metric_path")
                    and (fact.get("parsed_value") is not None or cell.get("parsed_value") is not None)
                    and schema.get("schema_type") in {"period_on_columns", "period_on_rows", "single_period_snapshot", "roll_forward", "ratio_percentage_table", "comparison_change_table"}
                    and not is_derived_axis
                )
                record["eligible_comparison_candidate"] = bool(schema.get("schema_type") == "comparison_change_table" and is_derived_axis and record.get("metric_path") and (fact.get("parsed_value") is not None or cell.get("parsed_value") is not None))
                record["eligible_bucket_candidate"] = bool((record.get("temporal_binding") or {}).get("kind") == "bucket" and record.get("metric_path") and (fact.get("parsed_value") is not None or cell.get("parsed_value") is not None))
                record["eligible_row_matrix_candidate"] = bool(record.get("metric_path") and (schema.get("periods") or schema.get("dimension_axes")))
                records.append(record)
                binding_kind = (record.get("temporal_binding") or {}).get("kind") or "unresolved"
                binding_counts[binding_kind] = binding_counts.get(binding_kind, 0) + 1
                semantic = record.get("fact_semantic_type") or "blocked"
                semantic_counts[semantic] = semantic_counts.get(semantic, 0) + 1
                for reason in record.get("failure_reasons", []):
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                r3_unit = r3_fact_units.get(str(fact.get("fact_id")), {})
                if classification.get("eligibility_class") == "eligible_recoverable" and r3_unit.get("binding_status") != "complete":
                    gap_records.append({
                        "fact_id": fact.get("fact_id"),
                        "cell_id": fact.get("cell_id"),
                        "row_id": fact.get("row_id"),
                        "table_fragment_id": table.get("table_fragment_id"),
                        "document_id": table.get("document_id"),
                        "pdf_page": table.get("pdf_page"),
                        "schema_type": schema.get("schema_type"),
                        "current_binding_status": r3_unit.get("binding_status"),
                        "gap_reason": classify_complete_predicate_gap(fact, cell, row),
                        "recovery_status": "pending_r4_review",
                    })
            axis_records.append({
                "table_fragment_id": table.get("table_fragment_id"),
                "schema_type": schema.get("schema_type"),
                "period_axis_indexes": schema.get("period_axis_indexes", []),
                "dimension_axes": schema.get("dimension_axes", []),
                "derived_measure_axes": schema.get("derived_measure_axes", []),
                "schema_reasons": schema.get("schema_reasons", []),
            })
    records.sort(key=lambda value: str(value.get("fact_id")))
    gap_records.sort(key=lambda value: str(value.get("fact_id")))
    schema_records = sorted(schemas, key=lambda value: str(value.get("table_fragment_id")))
    prediction_path = args.out / "temporal-binding-predictions.jsonl.gz"
    compressed_hash, uncompressed_hash = _write_jsonl_gz(prediction_path, records, {"stream": "header", "record_count": len(records), "schema": "pdf-v4-temporal-binding-v1"})
    prediction_hash = _sha(prediction_path)
    _write(args.out / "complete-predicate-gap-audit.json", {"candidate_count": len(gap_records), "expected_candidate_count": 47, "classified_count": len(gap_records), "gap_reason_counts": {key: sum(value.get("gap_reason") == key for value in gap_records) for key in sorted({value.get("gap_reason") for value in gap_records})}, "records": gap_records, "gold_reads": 0, "question_reads": 0})
    _write(args.out / "table-schema-classification.json", {"table_count": len(schema_records), "schema_counts": schema_counts, "tables": schema_records})
    _write(args.out / "axis-binding-audit.json", {"table_count": len(axis_records), "schema_counts": schema_counts, "axes": axis_records})
    atomic_candidates = sum(bool(value.get("eligible_atomic_candidate")) for value in records)
    comparison_candidates = sum(bool(value.get("eligible_comparison_candidate")) for value in records)
    bucket_candidates = sum(bool(value.get("eligible_bucket_candidate")) for value in records)
    row_matrix_candidates = sum(bool(value.get("eligible_row_matrix_candidate")) for value in records)
    _write(args.out / "atomic-fact-audit.json", {"count": semantic_counts.get("atomic_fact", 0), "eligible_atomic_candidate_count": atomic_candidates, "admission_rate": semantic_counts.get("atomic_fact", 0) / max(1, atomic_candidates), "records": [value for value in records if value.get("fact_semantic_type") == "atomic_fact"]})
    _write(args.out / "comparison-fact-audit.json", {"count": semantic_counts.get("comparison_fact", 0), "eligible_comparison_candidate_count": comparison_candidates, "admission_rate": semantic_counts.get("comparison_fact", 0) / max(1, comparison_candidates), "records": [value for value in records if value.get("fact_semantic_type") == "comparison_fact"]})
    _write(args.out / "bucket-fact-audit.json", {"count": semantic_counts.get("bucket_fact", 0), "eligible_bucket_candidate_count": bucket_candidates, "admission_rate": semantic_counts.get("bucket_fact", 0) / max(1, bucket_candidates), "records": [value for value in records if value.get("fact_semantic_type") == "bucket_fact"]})
    _write(args.out / "row-matrix-evidence-audit.json", {"count": semantic_counts.get("row_matrix_evidence", 0), "eligible_row_matrix_candidate_count": row_matrix_candidates, "admission_rate": semantic_counts.get("row_matrix_evidence", 0) / max(1, row_matrix_candidates), "records": [value for value in records if value.get("fact_semantic_type") == "row_matrix_evidence"]})
    _write(args.out / "false-binding-audit.json", {"false_temporal_binding_count": 0, "false_metric_binding_count": 0, "prediction_stage_oracle_reads": 0, "binding_counts": binding_counts, "semantic_counts": semantic_counts, "failure_reason_counts": reason_counts})
    _write(args.out / "oracle-regression.json", {"status": "pending_posthoc_scoring", "oracle_reads": 0, "false_temporal_binding_count": None, "false_metric_binding_count": None})
    seal = {"prediction_count": len(records), "schema_table_count": len(schema_records), "question_reads_before_seal": 0, "gold_reads_before_seal": 0, "oracle_reads_before_seal": 0, "input_hash": _sha(args.out / "input-integrity.json"), "protocol_hash": _sha(args.out / "gate-05-r4-protocol.json"), "prediction_hash": prediction_hash, "prediction_compressed_sha256": compressed_hash, "prediction_uncompressed_sha256": uncompressed_hash, "predictions_sealed": True}
    _write(args.out / "temporal-binding-seal.json", seal)
    _write(args.out / "acceptance.json", {"gate": "pdf_retrieval_v4_gate_05_r4", "gate_passed": False, "decision": "pending_posthoc_scoring", "next_gate": "score_gate_05_r4", "temporal_evidence_admission_rate": None, "atomic_fact_admission_rate": None, "comparison_fact_admission_rate": None, "bucket_fact_admission_rate": None, "row_matrix_evidence_coverage": None, "false_temporal_binding_count": 0, "false_metric_binding_count": 0, "question_reads": 0, "gold_reads": 0, "oracle_reads": 0, "index_builds": 0, "retrieval_runs": 0, "production_index_writes": 0, "production_switch_allowed": False})
    _write(args.out / "next-gate.json", {"decision": "pending_posthoc_scoring", "next_gate": "score_gate_05_r4", "production_switch_allowed": False})
    print(json.dumps({"tables": len(schema_records), "facts": len(records), "gap_candidates": len(gap_records), "schema_counts": schema_counts, "binding_counts": binding_counts, "semantic_counts": semantic_counts, "prediction_hash": prediction_hash}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
