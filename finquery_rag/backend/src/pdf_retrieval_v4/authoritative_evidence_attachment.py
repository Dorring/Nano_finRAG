from __future__ import annotations

from collections import Counter
from typing import Any


TYPE_BY_PREFIX = {
    "atomic": "atomic_fact",
    "comparison": "comparison_fact",
    "bucket": "bucket_fact",
    "matrix": "row_matrix",
    "narrative": "narrative_evidence",
    "raw": "raw_candidate",
}


def rehydrate_evidence(
    frozen: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    tables: dict[str, dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence_id = frozen["evidence_id"]
    if frozen["evidence_type"] == "raw_candidate":
        payload = frozen.get("payload") or {}
    else:
        payload = catalog[evidence_id]
    trace = payload.get("source_traceback") or frozen.get("source_traceback") or {}
    if isinstance(trace, list):
        trace_item = trace[0] if trace else {}
    else:
        trace_item = trace
    row_id = payload.get("row_id") or trace_item.get("row_id")
    table_id = payload.get("table_fragment_id") or trace_item.get("table_fragment_id")
    row = rows.get(str(row_id), {})
    table = tables.get(str(table_id), {})
    metric = metrics.get(str(row_id), {})
    context = {
        "table_fragment_id": table_id,
        "table_index": table.get("table_index"),
        "statement_type": table.get("statement_type"),
        "table_title": table.get("table_title"),
        "row_id": row_id,
        "row_index": row.get("row_index"),
        "row_type": row.get("row_type"),
        "raw_row_label": row.get("raw_label") or metric.get("raw_row_label"),
        "parent_row_id": row.get("parent_row_id"),
        "metric_path": payload.get("metric_path") or metric.get("metric_path"),
        "metric_path_segments": metric.get("metric_path_segments") or [],
        "section_path": payload.get("section_path"),
        "pdf_page": payload.get("pdf_page") or trace_item.get("pdf_page"),
        "bbox": payload.get("bbox")
        or trace_item.get("bbox")
        or trace_item.get("row_bbox"),
    }
    return {
        "attachment_schema": "pdf-retrieval-v4/evidence-attachment/v2",
        "evidence_id": evidence_id,
        "evidence_type": frozen["evidence_type"],
        "candidate_key": frozen["candidate_key"],
        "candidate_rank": frozen["candidate_rank"],
        "supporting_candidate_keys": frozen.get("supporting_candidate_keys")
        or [frozen["candidate_key"]],
        "document_id": payload.get("document_id") or frozen.get("document_id"),
        "semantic_payload": payload,
        "context": context,
        "source_traceback": payload.get("source_traceback")
        or frozen.get("source_traceback")
        or [],
    }


def context_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "table_fragment_id",
        "statement_type",
        "table_title",
        "row_id",
        "row_index",
        "row_type",
        "raw_row_label",
        "metric_path",
        "section_path",
    ]
    typed = [item for item in records if item["evidence_type"] != "raw_candidate"]
    counts = Counter()
    for item in typed:
        for field in fields:
            if item["context"].get(field) not in (None, "", []):
                counts[field] += 1
        if item["semantic_payload"].get("equivalent_group_id"):
            counts["equivalent_group_id"] += 1
    return {
        "typed_evidence_count": len(typed),
        **{
            field: {
                "count": counts[field],
                "coverage": counts[field] / len(typed) if typed else 0.0,
            }
            for field in [*fields, "equivalent_group_id"]
        },
    }
