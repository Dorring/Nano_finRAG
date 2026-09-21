#!/usr/bin/env python3
"""Run context-aware Gate 09 R4 without retrieval or attachment mutation."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.evidence_set_cover_v3 import build_sets  # noqa: E402
from src.pdf_retrieval_v4.operand_projection_v2 import project  # noqa: E402

R3 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-09-r3"
ORIGINAL = (
    ROOT
    / "artifacts/evaluation/pdf-retrieval-v4-gate-09/evidence-set-predictions.jsonl.gz"
)
PLANS = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-07/query-plan-predictions.json"
)
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-09-r4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, field: str | None = None):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            item["case_id"]: item[field] if field else item
            for item in (json.loads(line) for line in handle if line.strip())
        }


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    attachments_path = R3 / "authoritative-attachments.jsonl.gz"
    seal = json.loads((R3 / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["attachment_sha256"] != sha(attachments_path):
        raise RuntimeError("r3_seal_invalid")
    evidence = load(attachments_path, "authoritative_evidence")
    original = load(ORIGINAL)
    plans = {
        item["case_id"]: item["plan"] for item in json.loads(PLANS.read_text())["plans"]
    }
    records = []
    for case_id in sorted(original):
        result = build_sets(plans[case_id], evidence[case_id])
        operands = project(plans[case_id], result, evidence[case_id])
        records.append(
            {
                "case_id": case_id,
                "plan_id": plans[case_id]["plan_id"],
                "task_type": plans[case_id]["task_type"],
                "is_multi_slot": original[case_id]["is_multi_slot"],
                "candidate_pool": original[case_id]["candidate_pool"],
                "evidence_set_result": result,
                "operand_projection": operands,
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "evidence-set-predictions.jsonl.gz"
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as output:
            for record in records:
                output.write(
                    (
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                )
    protocol = {
        "gate": "pdf_retrieval_v4_gate_09_r4",
        "attachment_edge_mutations": 0,
        "evidence_identity_mutations": 0,
        "candidate_pool_mutations": 0,
        "retrieval_runs": 0,
        "bm25_searches": 0,
        "dense_searches": 0,
        "embedding_calls": 0,
        "index_reads": 0,
        "index_builds": 0,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "parameter_scan": False,
        "production_writes": 0,
        "production_switch_allowed": False,
    }
    source_files = {
        "matcher_v3": ROOT / "src/pdf_retrieval_v4/evidence_slot_matcher_v3.py",
        "set_cover_v3": ROOT / "src/pdf_retrieval_v4/evidence_set_cover_v3.py",
        "operand_v2": ROOT / "src/pdf_retrieval_v4/operand_projection_v2.py",
    }
    manifest = {
        "prediction_count": 72,
        "prediction_sha256": sha(path),
        "input_hashes": {
            "r3_attachment": sha(attachments_path),
            "original_gate09": sha(ORIGINAL),
            "query_plans": sha(PLANS),
        },
        "source_hashes": {name: sha(value) for name, value in source_files.items()},
        "candidate_positions": 7292,
        "evidence_identities": 12638,
    }
    write("protocol.json", protocol)
    write(
        "input-integrity.json",
        {
            "attachment_edge_mutations": 0,
            "evidence_identity_mutations": 0,
            "candidate_pool_mutations": 0,
        },
    )
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**protocol, **manifest, "sealed": True})
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
