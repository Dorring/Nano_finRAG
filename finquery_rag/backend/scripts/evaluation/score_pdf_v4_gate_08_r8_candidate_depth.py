#!/usr/bin/env python3
"""Post-seal Gate 08 R8-R0 candidate-depth scoring."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r8-r0"
PRED = OUT / "candidate-depth-predictions.jsonl.gz"
GOV = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
DEPTHS = (5, 10, 20, 40, 50)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path):
    return {item["case_id"]: item for item in map(json.loads, path.open())}


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(PRED):
        raise RuntimeError("r0_seal_invalid")
    with gzip.open(PRED, "rt", encoding="utf-8") as handle:
        predictions = {
            item["case_id"]: item
            for item in (json.loads(line) for line in handle if line.strip())
        }
    governance = load_jsonl(GOV)
    gold_records = [
        (case_id, candidate_key)
        for case_id, item in governance.items()
        for candidate_key in item["strict_gold_identities"]
    ]
    metrics = {}
    per_case = []
    for depth in DEPTHS:
        recalled = 0
        question_hits = 0
        answerable_questions = 0
        complete_multi = 0
        complete_calculation = 0
        for case_id, prediction in predictions.items():
            pool = {
                item["candidate_key"] for item in prediction[f"candidate_pool_{depth}"]
            }
            expected = set(governance[case_id]["strict_gold_identities"])
            recalled += sum(key in pool for key in expected)
            if expected:
                answerable_questions += 1
                question_hits += bool(expected.intersection(pool))
            if governance[case_id]["requires_multiple_sources"]:
                complete_multi += expected.issubset(pool)
            if governance[case_id]["query_type"] == "calculation_multi_operand":
                complete_calculation += expected.issubset(pool)
        metrics[f"recall_at_{depth}"] = f"{recalled}/80"
        metrics[f"answerable_question_hit_at_{depth}"] = (
            f"{question_hits}/{answerable_questions}"
        )
        metrics[f"benchmark_multi_complete_at_{depth}"] = f"{complete_multi}/16"
        metrics[f"calculation_complete_at_{depth}"] = f"{complete_calculation}/11"
    for case_id, key in gold_records:
        keys = [
            item["candidate_key"] for item in predictions[case_id]["candidate_pool_50"]
        ]
        per_case.append(
            {
                "case_id": case_id,
                "candidate_key": key,
                "rank": keys.index(key) + 1 if key in keys else None,
            }
        )
    c50 = sum(item["rank"] is not None for item in per_case)
    if metrics["recall_at_40"] == "60/80":
        historical_contract = "confirmed"
    else:
        historical_contract = "cutoff_mismatch_detected"
    if c50 >= 68:
        decision = "candidate_depth_ceiling_sufficient_for_recovery"
        next_gate = "candidate_ceiling_recovery"
    elif c50 >= 64:
        decision = "candidate_depth_ceiling_bridge_and_retriever_limited"
        next_gate = "candidate_ceiling_recovery"
    else:
        decision = "candidate_depth_ceiling_insufficient"
        next_gate = "stop_before_reranker"
    write("candidate-depth-metrics.json", metrics)
    write("gold-rank-distribution.json", {"records": per_case})
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_08_r8_r0",
        "decision": decision,
        "next_gate": next_gate,
        "metrics": metrics,
        "candidate_ceiling_at_50": f"{c50}/80",
        "historical_r7_at40_contract": historical_contract,
        "reranker_allowed": c50 >= 68,
        "production_switch_allowed": False,
    }
    write("acceptance.json", acceptance)
    write(
        "next-gate.json",
        {
            "decision": decision,
            "next_gate": next_gate,
            "production_switch_allowed": False,
        },
    )
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
