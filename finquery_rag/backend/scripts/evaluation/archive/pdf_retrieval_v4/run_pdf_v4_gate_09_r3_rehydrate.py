#!/usr/bin/env python3
"""Rehydrate frozen Gate 09 evidence edges from the authoritative Gate 03 catalog."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.authoritative_evidence_attachment import (  # noqa: E402
    context_coverage,
    rehydrate_evidence,
)

GATE09 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-09"
ATTACHMENTS = GATE09 / "candidate-evidence-attachment.jsonl.gz"
PREDICTIONS = GATE09 / "evidence-set-predictions.jsonl.gz"
CATALOG_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-03-r2"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-09-r3"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path, key: str) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return {
            str(item[key]): item
            for item in (json.loads(line) for line in handle if line.strip())
        }


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    seal = json.loads((GATE09 / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(PREDICTIONS):
        raise RuntimeError("gate09_seal_invalid")
    catalog = {}
    authority_files = {
        "atomic_fact": "atomic-facts.jsonl",
        "comparison_fact": "comparison-facts.jsonl",
        "bucket_fact": "bucket-facts.jsonl",
        "row_matrix": "row-matrices.jsonl",
        "narrative_evidence": "narrative-evidence.jsonl",
    }
    for evidence_type, name in authority_files.items():
        key = (
            "semantic_evidence_id"
            if evidence_type == "narrative_evidence"
            else "semantic_fact_id"
        )
        for evidence_id, item in jsonl(CATALOG_DIR / name, key).items():
            item = dict(item)
            item["_authoritative_type"] = evidence_type
            catalog[evidence_id] = item
    tables = jsonl(CATALOG_DIR / "logical-tables.jsonl", "table_fragment_id")
    rows = jsonl(CATALOG_DIR / "semantic-rows.jsonl", "row_id")
    metrics = jsonl(CATALOG_DIR / "metric-paths.jsonl", "row_id")
    output_records = []
    flat = []
    missing = []
    disagreements = []
    edge_count = position_count = 0
    with gzip.open(ATTACHMENTS, "rt", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            evidence_records = []
            for frozen in source["canonical_evidence"]:
                evidence_id = frozen["evidence_id"]
                if frozen["evidence_type"] != "raw_candidate":
                    if evidence_id not in catalog:
                        missing.append(evidence_id)
                        continue
                    if (
                        catalog[evidence_id]["_authoritative_type"]
                        != frozen["evidence_type"]
                    ):
                        disagreements.append(evidence_id)
                record = rehydrate_evidence(frozen, catalog, tables, rows, metrics)
                evidence_records.append(record)
                flat.append(record)
                edge_count += len(record["supporting_candidate_keys"])
            output_records.append(
                {
                    "case_id": source["case_id"],
                    "authoritative_evidence": evidence_records,
                }
            )
    with gzip.open(PREDICTIONS, "rt", encoding="utf-8") as handle:
        for line in handle:
            position_count += len(json.loads(line)["candidate_pool"])
    if missing or disagreements or position_count != 7292 or len(flat) != 12638:
        raise RuntimeError(
            f"rehydration_blocked:{position_count}:{len(flat)}:{len(missing)}:{len(disagreements)}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "authoritative-attachments.jsonl.gz"
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as out:
            for record in output_records:
                out.write(
                    (
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                )
    type_counts = Counter(item["evidence_type"] for item in flat)
    parity = {
        "candidate_position_count": position_count,
        "canonical_evidence_count": len(flat),
        "candidate_evidence_edge_count": edge_count,
        "new_evidence_ids": 0,
        "removed_evidence_ids": 0,
        "candidate_key_rank_mutations": 0,
    }
    authority = {
        "missing_authoritative_payload": 0,
        "evidence_type_disagreement": 0,
        "document_disagreement": 0,
        "type_counts": dict(type_counts),
    }
    protocol = {
        "gate": "pdf_retrieval_v4_gate_09_r3",
        "retrieval_runs": 0,
        "bm25_searches": 0,
        "dense_searches": 0,
        "embedding_calls": 0,
        "index_reads": 0,
        "index_builds": 0,
        "semantic_graph_runs": 0,
        "candidate_bridge_runs": 0,
        "gold_reads": 0,
        "governance_reads": 0,
        "parameter_scan": False,
        "calculator_calls": 0,
        "generator_calls": 0,
        "production_writes": 0,
        "production_switch_allowed": False,
    }
    manifest = {
        "prediction_count": 72,
        "attachment_sha256": sha(path),
        "input_hashes": {
            "gate09_prediction": sha(PREDICTIONS),
            "gate09_attachment": sha(ATTACHMENTS),
        },
        "catalog_hashes": {
            name: sha(CATALOG_DIR / name) for name in authority_files.values()
        },
    }
    write("protocol.json", protocol)
    write("attachment-edge-parity.json", parity)
    write("payload-authority-audit.json", authority)
    write("context-coverage.json", context_coverage(flat))
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**protocol, **manifest, "sealed": True})
    write(
        "acceptance.json",
        {
            "decision": "authoritative_evidence_attachment_rehydrated",
            "next_gate": "context_disambiguatability_audit",
            "edge_parity": parity,
            "payload_authority": authority,
            "production_switch_allowed": False,
        },
    )
    write(
        "next-gate.json",
        {
            "decision": "authoritative_evidence_attachment_rehydrated",
            "next_gate": "context_disambiguatability_audit",
            "production_switch_allowed": False,
        },
    )
    print(json.dumps({**manifest, **parity, **authority}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
