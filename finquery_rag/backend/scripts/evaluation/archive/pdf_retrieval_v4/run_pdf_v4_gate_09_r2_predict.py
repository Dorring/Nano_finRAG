#!/usr/bin/env python3
"""Run zero-retrieval Gate 09 R2 evidence contract repair."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.evidence_set_cover import build_sets  # noqa: E402
from src.pdf_retrieval_v4.operand_projection import project_operands  # noqa: E402

GATE09 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-09"
ORIGINAL = GATE09 / "evidence-set-predictions.jsonl.gz"
ATTACHMENTS = GATE09 / "candidate-evidence-attachment.jsonl.gz"
PLANS = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-07/query-plan-predictions.json"
)
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-09-r2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            item["case_id"]: item
            for item in (json.loads(line) for line in handle if line.strip())
        }


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    seal = json.loads((GATE09 / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(ORIGINAL):
        raise RuntimeError("original_gate09_seal_invalid")
    original = load(ORIGINAL)
    attachments = load(ATTACHMENTS)
    plans = {
        item["case_id"]: item["plan"] for item in json.loads(PLANS.read_text())["plans"]
    }
    if set(original) != set(attachments) or len(original) != 72:
        raise RuntimeError("attachment_case_coverage_blocked")
    records = []
    attachment_positions = canonical_count = 0
    for case_id in sorted(original):
        source = original[case_id]
        evidence = attachments[case_id]["canonical_evidence"]
        attachment_positions += len(source["candidate_pool"])
        canonical_count += len(evidence)
        result = build_sets(plans[case_id], evidence)
        operand_projection = project_operands(plans[case_id], result, evidence)
        records.append(
            {
                "case_id": case_id,
                "plan_id": plans[case_id]["plan_id"],
                "task_type": plans[case_id]["task_type"],
                "is_multi_slot": source["is_multi_slot"],
                "candidate_pool": source["candidate_pool"],
                "canonical_evidence": evidence,
                "evidence_set_result": result,
                "operand_projection": operand_projection,
            }
        )
    if (attachment_positions, canonical_count) != (7292, 12638):
        raise RuntimeError(
            f"frozen_attachment_count_drift:{attachment_positions}:{canonical_count}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    prediction = OUT / "evidence-set-r2-predictions.jsonl.gz"
    with prediction.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as output:
            for record in records:
                output.write(
                    (
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                )
    source_files = {
        "slot_matcher": ROOT / "src/pdf_retrieval_v4/evidence_slot_matcher_v2.py",
        "set_cover": ROOT / "src/pdf_retrieval_v4/evidence_set_cover.py",
        "operand_projection": ROOT / "src/pdf_retrieval_v4/operand_projection.py",
    }
    protocol = {
        "gate": "pdf_retrieval_v4_gate_09_r2",
        "schema": "pdf-retrieval-v4/evidence-set/v2",
        "attachment_reruns": 0,
        "retrieval_runs": 0,
        "bm25_searches": 0,
        "dense_searches": 0,
        "embedding_calls": 0,
        "index_reads": 0,
        "index_builds": 0,
        "semantic_graph_runs": 0,
        "candidate_bridge_runs": 0,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "parameter_scan": False,
        "set_size_scan": False,
        "calculator_calls": 0,
        "generator_calls": 0,
        "production_writes": 0,
        "production_switch_allowed": False,
    }
    manifest = {
        "prediction_count": 72,
        "prediction_sha256": sha(prediction),
        "input_hashes": {
            "original_gate09_prediction": sha(ORIGINAL),
            "original_gate09_attachment": sha(ATTACHMENTS),
            "query_plans": sha(PLANS),
        },
        "source_hashes": {name: sha(path) for name, path in source_files.items()},
        "attachment_positions": attachment_positions,
        "canonical_evidence_count": canonical_count,
    }
    write("protocol.json", protocol)
    write(
        "input-integrity.json",
        {
            "original_gate09_immutable": True,
            "attachment_positions": attachment_positions,
            "canonical_evidence_count": canonical_count,
            "candidate_pool_sequence_mutations": 0,
        },
    )
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**protocol, **manifest, "sealed": True})
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
