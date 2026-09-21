#!/usr/bin/env python3
"""Build candidate-global authoritative context and V2 rerank views without Gold."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.structure_aware_rerank_view import (  # noqa: E402
    RERANK_INSTRUCTION,
    build_rerank_document_view,
    sha256_text,
)

BASE = ROOT / "artifacts/evaluation"
TOP100 = BASE / "pdf-retrieval-v4-gate-08-r8-r2a-2/bounded-top100-predictions.jsonl.gz"
P0_VIEWS = BASE / "pdf-retrieval-v4-gate-08-r8-r3-p0/rerank-input-views.jsonl.gz"
GATE03 = BASE / "pdf-retrieval-v4-gate-03-r2"
GATE05 = BASE / "pdf-retrieval-v4-gate-05-r5"
META = BASE / "pdf-retrieval-v4-gate-06-r4/candidate-indexes/candidate-metadata.sqlite"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r3-1a"
CONTEXT = OUT / "top100-authoritative-context-v2.jsonl.gz"
VIEWS = OUT / "rerank-input-views-v2.jsonl.gz"
EXPECTED_TOP100_SHA = "ced014c357a3c8862a5ae3723a4c618556156542fd63aa2a8fd4bf766b60f01a"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if sha(TOP100) != EXPECTED_TOP100_SHA:
        raise RuntimeError("top100_input_sha_mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    structured = {item["candidate_key"]: item for item in load_jsonl(GATE05 / "structured-views.jsonl")}
    bridge = {item["candidate_key"]: item for item in load_jsonl(GATE05 / "bridge-results.jsonl")}
    evidence: dict[str, tuple[str, dict[str, Any]]] = {}
    evidence_files = {
        "atomic_fact": ("atomic-facts.jsonl", "semantic_fact_id"),
        "comparison_fact": ("comparison-facts.jsonl", "semantic_fact_id"),
        "bucket_fact": ("bucket-facts.jsonl", "semantic_fact_id"),
        "row_matrix": ("row-matrices.jsonl", "semantic_fact_id"),
        "narrative_evidence": ("narrative-evidence.jsonl", "semantic_evidence_id"),
    }
    for evidence_type, (filename, id_field) in evidence_files.items():
        for item in load_jsonl(GATE03 / filename):
            evidence[item[id_field]] = (evidence_type, item)
    tables = {item["table_fragment_id"]: item for item in load_jsonl(GATE03 / "logical-tables.jsonl")}
    rows = {item["row_id"]: item for item in load_jsonl(GATE03 / "semantic-rows.jsonl")}
    paths = {item["row_id"]: item for item in load_jsonl(GATE03 / "metric-paths.jsonl")}
    evidence.update({key: ("logical_table", item) for key, item in tables.items()})
    evidence.update({key: ("semantic_row", item) for key, item in rows.items()})
    candidate_meta: dict[str, dict[str, Any]] = {}
    connection = sqlite3.connect(f"file:{META}?mode=ro", uri=True)
    for lane, key, document_id, text, metadata_json in connection.execute(
        "SELECT lane,candidate_key,document_id,retrieval_text,metadata_json FROM view_metadata "
        "ORDER BY CASE lane WHEN 'candidate_raw_bm25' THEN 0 ELSE 1 END, view_id"
    ):
        current = candidate_meta.setdefault(key, {"document_id": document_id, "raw_text": text, "metadata": json.loads(metadata_json), "lane": lane})
        if lane == "candidate_raw_bm25":
            current.update(document_id=document_id, raw_text=text, metadata=json.loads(metadata_json), lane=lane)
    connection.close()
    p0 = {}
    with gzip.open(P0_VIEWS, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            p0[item["case_id"]] = item
    top100 = []
    with gzip.open(TOP100, "rt", encoding="utf-8") as handle:
        top100 = [json.loads(line) for line in handle]
    context_records = []
    view_records = []
    statuses: Counter[str] = Counter()
    grade_a_occurrences = grade_a_context = 0
    missing_evidence_ids: Counter[str] = Counter()
    for case in top100:
        case_id = case["case_id"]
        old_by_key = {item["candidate_key"]: item for item in p0[case_id]["candidates"]}
        views = []
        contexts = []
        for candidate in case["candidates"]:
            key = candidate["candidate_key"]
            meta = candidate_meta[key]
            structured_view = structured.get(key)
            bridge_record = bridge.get(key, {})
            grade = (structured_view or {}).get("bridge_grade") or bridge_record.get("grade", "raw_only")
            hydrated = []
            if structured_view and str(grade).startswith("A"):
                grade_a_occurrences += 1
                status = "authoritative_structured"
                for evidence_id in structured_view.get("semantic_evidence_ids", []):
                    if evidence_id not in evidence:
                        missing_evidence_ids[evidence_id.split(":", 1)[0]] += 1
                        continue
                    evidence_type, payload = evidence[evidence_id]
                    row_id = payload.get("row_id")
                    row = rows.get(row_id, {})
                    table_id = payload.get("table_fragment_id") or payload.get("source_traceback", {}).get("table_fragment_id")
                    table = tables.get(table_id, {})
                    metric = paths.get(row_id, {})
                    context = {
                        "statement_type": table.get("statement_type"),
                        "table_title": table.get("table_title") or structured_view.get("table_title"),
                        "table_fragment_id": table_id,
                        "section_path": structured_view.get("section_path"),
                        "raw_row_label": row.get("raw_label") or metric.get("raw_row_label"),
                        "row_id": row_id,
                        "row_index": row.get("row_index"),
                        "row_type": row.get("row_type"),
                        "metric_path": payload.get("metric_path") or metric.get("metric_path"),
                        "pdf_page": payload.get("source_traceback", {}).get("pdf_page") or structured_view.get("pdf_page"),
                    }
                    hydrated.append({"candidate_key": key, "evidence_id": evidence_id, "evidence_type": evidence_type, "document_id": payload.get("document_id"), "semantic_payload": payload, "context": context})
                if hydrated:
                    grade_a_context += 1
            elif bridge_record.get("grade") == "B_ambiguous":
                status = "ambiguous_not_attached"
            elif bridge_record.get("grade") == "unmapped":
                status = "unmapped"
            else:
                status = "raw_only"
            statuses[status] += 1
            document_view = build_rerank_document_view(meta, hydrated)
            old = old_by_key[key]
            contexts.append({"candidate_key": key, "pre_rerank_rank": candidate["final_candidate_rank"], "context_status": status, "bridge_grade": grade, "authoritative_evidence": hydrated, "document_id": meta["document_id"], "raw_text": meta["raw_text"], "metadata": meta["metadata"], "document_view": document_view, "document_view_sha256": sha256_text(document_view)})
            views.append({"candidate_key": key, "pre_rerank_rank": candidate["final_candidate_rank"], "query_view": old["query_view"], "query_view_sha256": old["query_view_sha256"], "document_view": document_view, "document_view_sha256": sha256_text(document_view), "context_status": status, "authoritative_evidence_count": len(hydrated)})
        context_records.append({"case_id": case_id, "candidates": contexts})
        view_records.append({"case_id": case_id, "query_plan_id": case["query_plan_id"], "query_view_sha256": p0[case_id]["query_view_sha256"], "candidates": views})
    if grade_a_context != grade_a_occurrences or missing_evidence_ids:
        raise RuntimeError(f"grade_a_authoritative_context_incomplete:{grade_a_context}/{grade_a_occurrences}:{dict(missing_evidence_ids)}")
    for path, records in ((CONTEXT, context_records), (VIEWS, view_records)):
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                for record in records:
                    zipped.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    manifest = {"cases": 72, "candidate_occurrences": 7200, "candidate_added": 0, "candidate_removed": 0, "candidate_mutation": 0, "grade_a_occurrences": grade_a_occurrences, "grade_a_with_authoritative_context": grade_a_context, "context_status_counts": dict(statuses), "top100_sha256": EXPECTED_TOP100_SHA, "p0_views_sha256": sha(P0_VIEWS), "context_sha256": sha(CONTEXT), "rerank_input_views_v2_sha256": sha(VIEWS), "instruction_sha256": sha256_text(RERANK_INSTRUCTION), "gate03_input_hashes": {filename: sha(GATE03 / filename) for filename, _ in evidence_files.values()}, "gate05_structured_views_sha256": sha(GATE05 / "structured-views.jsonl"), "gate05_bridge_results_sha256": sha(GATE05 / "bridge-results.jsonl"), "candidate_metadata_sha256": sha(META)}
    protocol = {"gate": "pdf_retrieval_v4_gate_08_r8_r3_1a", "gold_reads_before_seal": 0, "governance_reads_before_seal": 0, "reference_answer_reads": 0, "expected_value_reads": 0, "bridge_runs": 0, "semantic_graph_runs": 0, "retrieval_runs": 0, "bm25_searches": 0, "dense_searches": 0, "embedding_calls": 0, "model_calls": 0, "production_writes": 0}
    write("context-coverage.json", {"status_counts": dict(statuses), "grade_a": f"{grade_a_context}/{grade_a_occurrences}"})
    write("context-source-audit.json", {"allowed_sources_only": True, "missing_evidence_ids": dict(missing_evidence_ids), "grade_a_context_complete": True})
    write("input-integrity.json", manifest)
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**manifest, **protocol, "sealed": True})
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
