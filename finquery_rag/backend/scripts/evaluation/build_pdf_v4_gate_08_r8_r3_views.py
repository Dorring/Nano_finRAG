#!/usr/bin/env python3
"""Build and seal zero-Gold Gate 08 R8-R3 reranker input views."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.structure_aware_rerank_view import (  # noqa: E402
    RERANK_INSTRUCTION,
    build_rerank_document_view,
    build_rerank_query_view,
    sha256_text,
)

BASE = ROOT / "artifacts/evaluation"
TOP100 = BASE / "pdf-retrieval-v4-gate-08-r8-r2a-2/bounded-top100-predictions.jsonl.gz"
PLAN = BASE / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"
ATTACH = BASE / "pdf-retrieval-v4-gate-09-r3/authoritative-attachments.jsonl.gz"
META = BASE / "pdf-retrieval-v4-gate-06-r4/candidate-indexes/candidate-metadata.sqlite"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r3-p0"
VIEWS = OUT / "rerank-input-views.jsonl.gz"
EXPECTED_TOP100_SHA = "ced014c357a3c8862a5ae3723a4c618556156542fd63aa2a8fd4bf766b60f01a"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if sha(TOP100) != EXPECTED_TOP100_SHA:
        raise RuntimeError("top100_input_sha_mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    plans = {item["case_id"]: item["plan"] for item in json.loads(PLAN.read_text())["plans"]}
    attachments: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(ATTACH, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for item in record["authoritative_evidence"]:
                attachments[(record["case_id"], item["candidate_key"])].append(item)
    metadata: dict[str, dict[str, Any]] = {}
    connection = sqlite3.connect(f"file:{META}?mode=ro", uri=True)
    for lane, key, document_id, text, metadata_json in connection.execute(
        "SELECT lane,candidate_key,document_id,retrieval_text,metadata_json FROM view_metadata "
        "ORDER BY CASE lane WHEN 'candidate_raw_bm25' THEN 0 ELSE 1 END, view_id"
    ):
        current = metadata.setdefault(key, {"document_id": document_id, "raw_text": text, "metadata": json.loads(metadata_json), "lane": lane})
        if lane == "candidate_raw_bm25":
            current.update(document_id=document_id, raw_text=text, metadata=json.loads(metadata_json), lane=lane)
    connection.close()
    records = []
    candidate_count = 0
    with gzip.open(TOP100, "rt", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            plan = plans[source["case_id"]]
            query_view = build_rerank_query_view(plan)
            candidates = []
            for item in source["candidates"]:
                key = item["candidate_key"]
                if key not in metadata:
                    raise RuntimeError(f"candidate_metadata_missing:{key}")
                document_view = build_rerank_document_view(metadata[key], attachments[(source["case_id"], key)])
                candidates.append({
                    "candidate_key": key,
                    "pre_rerank_rank": item["final_candidate_rank"],
                    "query_view_sha256": sha256_text(query_view),
                    "document_view_sha256": sha256_text(document_view),
                    "query_char_count": len(query_view),
                    "document_char_count": len(document_view),
                    "query_view": query_view,
                    "document_view": document_view,
                    "authoritative_evidence_count": len(attachments[(source["case_id"], key)]),
                    "candidate_metadata_lane": metadata[key]["lane"],
                })
            if len(candidates) != 100 or len({item["candidate_key"] for item in candidates}) != 100:
                raise RuntimeError(f"candidate_identity_contract_failed:{source['case_id']}")
            candidate_count += len(candidates)
            records.append({"case_id": source["case_id"], "query_plan_id": source["query_plan_id"], "query_view_sha256": sha256_text(query_view), "candidates": candidates})
    if len(records) != 72 or candidate_count != 7200:
        raise RuntimeError("rerank_pair_count_contract_failed")
    with VIEWS.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for record in records:
                zipped.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    serializer = ROOT / "src/pdf_retrieval_v4/structure_aware_rerank_view.py"
    manifest = {
        "schema": "pdf-retrieval-v4/structure-aware-rerank-input/v1",
        "cases": 72,
        "candidates": 7200,
        "candidates_per_case": 100,
        "candidate_added": 0,
        "candidate_removed": 0,
        "candidate_mutation": 0,
        "duplicate_candidates": 0,
        "gold_derived_fields": 0,
        "serializer_deterministic": True,
        "top100_sha256": EXPECTED_TOP100_SHA,
        "query_plan_sha256": sha(PLAN),
        "authoritative_attachment_sha256": sha(ATTACH),
        "candidate_metadata_sha256": sha(META),
        "serializer_sha256": sha(serializer),
        "instruction_sha256": sha256_text(RERANK_INSTRUCTION),
        "rerank_input_views_sha256": sha(VIEWS),
    }
    protocol = {"gate": "pdf_retrieval_v4_gate_08_r8_r3_p0", "model_loads": 0, "inference_calls": 0, "gold_reads": 0, "governance_reads": 0, "reference_answer_reads": 0, "expected_value_reads": 0, "retrieval_runs": 0, "candidate_budget": 100, "production_switch_allowed": False}
    write("protocol.json", protocol)
    write("input-integrity.json", manifest)
    write("serializer-manifest.json", manifest)
    write("query-view-schema.json", {"sections": ["QUESTION", "QUERY PLAN"], "forbidden": ["case_id", "source_index", "gold", "reference_answer", "expected_value"]})
    write("document-view-schema.json", {"schema": "StructureAwareRerankDocumentV1", "sections": ["DOCUMENT", "STRUCTURE", "EVIDENCE", "CONTENT"], "missing_field_policy": "omit_line", "raw_only_supported": True})
    write("preflight-acceptance.json", {"decision": "reranker_input_preflight_passed", "metrics": manifest, "next_gate": "qwen3_0_6b_cross_encoder_prediction"})
    write("input-seal.json", {**manifest, **protocol, "sealed": True})
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
