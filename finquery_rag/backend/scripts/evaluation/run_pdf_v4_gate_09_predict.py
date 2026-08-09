#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.candidate_evidence_attachment import (  # noqa: E402
    attach_candidate,
    canonicalize,
)
from src.pdf_retrieval_v4.evidence_set_generator import generate_evidence_sets  # noqa: E402
from src.pdf_retrieval_v4.evidence_set_validator import validate_prediction  # noqa: E402

R7 = (
    ROOT
    / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r7/field-family-predictions.jsonl.gz"
)
R7_SEAL = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r7/prediction-seal.json"
PLANS = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-07/query-plan-predictions.json"
)
VIEWS = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5/structured-views.jsonl"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-09"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl_gz(name: str, records: list[dict[str, object]]) -> None:
    path = OUT / name
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as out:
            for record in sorted(records, key=lambda item: str(item["case_id"])):
                out.write(
                    (
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                )


def main() -> int:
    seal = json.loads(R7_SEAL.read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(R7):
        raise RuntimeError("r7_seal_invalid")
    plans_raw = json.loads(PLANS.read_text())["plans"]
    plans = {item["case_id"]: item["plan"] for item in plans_raw}
    views = {
        item["candidate_key"]: item
        for item in (json.loads(line) for line in VIEWS.open() if line.strip())
    }
    records = []
    pool_positions = structured_positions = raw_positions = evidence_count = 0
    all_errors = []
    with gzip.open(R7, "rt", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            case_id = source["case_id"]
            plan = plans[case_id]
            pool = source["r7_full_pool"]
            refs = []
            for item in pool:
                key = item["candidate_key"]
                rank = int(item["rank"])
                pool_positions += 1
                if key in views:
                    structured_positions += 1
                else:
                    raw_positions += 1
                refs.extend(
                    attach_candidate(
                        key, rank, views.get(key), tuple(plan["document_scope"])
                    )
                )
            canonical = canonicalize(refs)
            evidence_count += len(canonical)
            result = generate_evidence_sets(plan, canonical)
            record = {
                "case_id": case_id,
                "plan_id": plan["plan_id"],
                "task_type": plan["task_type"],
                "is_multi_slot": len(plan.get("operand_slots") or []) > 1,
                "candidate_pool": pool,
                "canonical_evidence": canonical,
                "evidence_set_result": result,
            }
            record["validation_errors"] = validate_prediction(record)
            all_errors.extend(
                f"{case_id}:{error}" for error in record["validation_errors"]
            )
            records.append(record)
    if len(records) != 72 or all_errors:
        raise RuntimeError(
            f"gate09_prediction_integrity_blocked:{len(records)}:{all_errors[:5]}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    prediction = OUT / "evidence-set-predictions.jsonl.gz"
    with prediction.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as out:
            for record in sorted(records, key=lambda item: item["case_id"]):
                out.write(
                    (
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                )
    protocol = {
        "gate": "pdf_retrieval_v4_gate_09",
        "schema": "deterministic-evidence-set-v1",
        "candidate_pool_source": "gate08_r7_sealed_final_pool",
        "candidate_pool_mutation": False,
        "bm25_searches": 0,
        "dense_searches": 0,
        "embedding_calls": 0,
        "index_reads": 0,
        "index_builds": 0,
        "query_rebuilds": 0,
        "query_plan_mutations": 0,
        "semantic_graph_runs": 0,
        "candidate_bridge_runs": 0,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "parameter_scan": False,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "production_writes": 0,
        "production_switch_allowed": False,
    }
    hashes = {
        "r7_prediction": sha(R7),
        "r7_seal": sha(R7_SEAL),
        "query_plans": sha(PLANS),
        "structured_views": sha(VIEWS),
    }
    integrity = {
        "prediction_count": len(records),
        "candidate_pool_positions": pool_positions,
        "structured_attachment_positions": structured_positions,
        "raw_only_positions": raw_positions,
        "canonical_evidence_count": evidence_count,
        "attachment_coverage": f"{pool_positions}/{pool_positions}",
        "foreign_key_errors": 0,
        "cross_document_bindings": 0,
        "identity_conflicts": 0,
        "validation_errors": 0,
    }
    manifest = {
        "prediction_count": 72,
        "prediction_sha256": sha(prediction),
        "input_hashes": hashes,
    }
    write("protocol.json", protocol)
    write("input-integrity.json", integrity)
    write("attachment-audit.json", integrity)
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**protocol, **manifest, "sealed": True})
    write("gate-09-protocol.json", protocol)
    write("evidence-set-prediction-seal.json", {**protocol, **manifest, "sealed": True})
    write_jsonl_gz(
        "candidate-evidence-attachment.jsonl.gz",
        [
            {
                "case_id": record["case_id"],
                "canonical_evidence": record["canonical_evidence"],
            }
            for record in records
        ],
    )
    write_jsonl_gz(
        "evidence-set-candidates.jsonl.gz",
        [
            {
                "case_id": record["case_id"],
                "sets": record["evidence_set_result"]["sets"],
            }
            for record in records
        ],
    )
    write_jsonl_gz(
        "primary-evidence-sets.jsonl.gz",
        [
            {
                "case_id": record["case_id"],
                "primary_set_ids": record["evidence_set_result"]["primary_set_ids"],
                "status": record["evidence_set_result"]["status"],
            }
            for record in records
        ],
    )
    print(json.dumps({**integrity, "prediction_sha256": sha(prediction)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
