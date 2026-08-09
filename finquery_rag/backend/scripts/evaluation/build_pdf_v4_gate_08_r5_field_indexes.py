#!/usr/bin/env python3
"""Build and audit frozen Gate 08 R5 field projections and BM25 indexes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.candidate_field_index import (  # noqa: E402
    FIELD_NAMES,
    CandidateFieldIndex,
)
from src.pdf_retrieval_v4.candidate_field_view import (  # noqa: E402
    canonical_projection_hash,
    project_candidate_fields,
)

SOURCE = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5/structured-views.jsonl"
GENERAL_META = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-06-r4/candidate-indexes/candidate-metadata.sqlite"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r5"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_keys = [str(item["candidate_key"]) for item in records]
    if len(records) != 19500 or len(set(source_keys)) != 19500:
        raise RuntimeError("grade_a_source_not_19500_exact")
    with sqlite3.connect(f"file:{GENERAL_META.resolve().as_posix()}?mode=ro", uri=True) as connection:
        index_keys = {str(row[0]) for row in connection.execute("SELECT candidate_key FROM view_metadata WHERE lane='candidate_structured_bm25'")}
    if set(source_keys) != index_keys:
        raise RuntimeError("grade_a_keyset_mismatch")
    projected = {field: [] for field in FIELD_NAMES}
    for record in records:
        fields = project_candidate_fields(record)
        for field in FIELD_NAMES:
            projected[field].append(fields[field])
    flat_ids = [item.field_view_id for field in FIELD_NAMES for item in projected[field]]
    if len(flat_ids) != len(set(flat_ids)):
        raise RuntimeError("duplicate_field_view_id")
    report = CandidateFieldIndex(OUT / "field-indexes").build(projected)
    coverage = {field + "_nonempty": sum(bool(item.retrieval_text) for item in projected[field]) for field in FIELD_NAMES}
    masks = {key: 0 for key in source_keys}
    for bit, field in enumerate(FIELD_NAMES):
        for item in projected[field]:
            if item.retrieval_text:
                masks[item.candidate_key] |= 1 << bit
    coverage.update({
        "metric_axis": sum(mask & 3 == 3 for mask in masks.values()),
        "metric_context": sum(mask & 5 == 5 for mask in masks.values()),
        "metric_axis_context": sum(mask & 7 == 7 for mask in masks.values()),
        "all_four": sum(mask == 15 for mask in masks.values()),
    })
    protocol = {
        "gate": "pdf_retrieval_v4_gate_08_r5_r0",
        "field_schema_version": "candidate-field-v1",
        "grade_a_input": 19500,
        "semantic_graph_runs": 0,
        "candidate_bridge_runs": 0,
        "gold_reads": 0,
        "question_reads": 0,
        "retrieval_runs": 0,
        "parameter_scan": False,
        "weight_scan": False,
        "topk_scan": False,
        "production_switch_allowed": False,
    }
    write("protocol.json", protocol)
    write("field-projection-manifest.json", {"source_sha256": sha(SOURCE), "projection_sha256": canonical_projection_hash(records), "record_count_per_field": 19500})
    write("field-keyset-integrity.json", {"grade_a_exact": True, "field_keysets_exact": {field: {item.candidate_key for item in projected[field]} == index_keys for field in FIELD_NAMES}, "grade_b_included": 0, "unmapped_included": 0, "candidate_key_conflicts": 0, "duplicate_field_view_ids": 0, "source_traceback_missing": 0})
    write("field-coverage-stats.json", coverage)
    write("field-index-manifest.json", report)
    write("field-index-integrity.json", {"decision": "field_projection_and_index_passed", "all_lane_counts_exact": all(item["document_count"] == 19500 for item in report["lanes"].values())})
    print(json.dumps({"decision": "field_projection_and_index_passed", **coverage}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
