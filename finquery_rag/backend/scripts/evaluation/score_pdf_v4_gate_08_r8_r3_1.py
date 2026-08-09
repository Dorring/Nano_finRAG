#!/usr/bin/env python3
"""Post-seal 80-binding scoring for the full-context R3.1 replay."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r3-1"
PRED = OUT / "rerank-predictions.jsonl.gz"
TOP100 = BASE / "pdf-retrieval-v4-gate-08-r8-r2a-2/bounded-top100-predictions.jsonl.gz"
SIDECAR = BASE / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
GOV = BASE / "pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
DEPTHS = (1, 3, 5, 10, 20, 100)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {item["case_id"]: item for item in map(json.loads, handle)}


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(PRED):
        raise RuntimeError("r3_1_prediction_seal_invalid")
    predictions, top100 = map(load_gzip, (PRED, TOP100))
    bindings = [json.loads(line) for line in SIDECAR.open(encoding="utf-8")]
    governance = {item["case_id"]: item for item in map(json.loads, GOV.open(encoding="utf-8"))}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in bindings:
        by_case[binding["case_id"]].append(binding)
    status_by_candidate = {}
    for case_id, record in predictions.items():
        before = {item["candidate_key"] for item in top100[case_id]["candidates"]}
        after = {item["candidate_key"] for item in record["ranked_candidates"]}
        if len(after) != 100 or after != before:
            raise RuntimeError(f"candidate_identity_mutation:{case_id}")
        status_by_candidate[case_id] = {item["candidate_key"]: item["context_status"] for item in record["ranked_candidates"]}

    def score(depth: int) -> tuple[set[str], int, int, int]:
        pools = {case_id: {item["candidate_key"] for item in record["ranked_candidates"][:depth]} for case_id, record in predictions.items()}
        hits = {item["binding_id"] for item in bindings if item["candidate_key"] in pools[item["case_id"]]}
        answerable = multi = calculation = 0
        for case_id, expected in by_case.items():
            any_hit = any(item["candidate_key"] in pools[case_id] for item in expected)
            complete = all(item["candidate_key"] in pools[case_id] for item in expected)
            answerable += any_hit
            multi += bool(complete and governance[case_id]["requires_multiple_sources"])
            calculation += bool(complete and governance[case_id]["query_type"] == "calculation_multi_operand")
        return hits, answerable, multi, calculation

    metrics, results = {}, {}
    for depth in DEPTHS:
        hits, answerable, multi, calculation = score(depth)
        results[depth] = hits
        metrics[f"strict_source_binding_recall_at_{depth}"] = f"{len(hits)}/80"
        metrics[f"answerable_question_hit_at_{depth}"] = f"{answerable}/72"
        if depth in (5, 10, 100):
            metrics[f"multi_evidence_complete_at_{depth}"] = f"{multi}/16"
            metrics[f"calculation_complete_at_{depth}"] = f"{calculation}/11"
    if len(results[100]) != 68:
        raise RuntimeError("top100_recall_identity_parity_blocked")
    pre5 = {item["binding_id"] for item in bindings if item["candidate_key"] in {candidate["candidate_key"] for candidate in top100[item["case_id"]]["candidates"][:5]}}
    promoted, demoted = results[5] - pre5, pre5 - results[5]
    context_present = {item["binding_id"] for item in bindings if item["binding_id"] in results[100] and status_by_candidate[item["case_id"]][item["candidate_key"]] == "authoritative_structured"}
    raw_only = results[100] - context_present
    metrics.update({
        "candidate_to_top5_conversion": f"{len(results[5])}/68",
        "candidate_to_top5_conversion_rate": round(len(results[5]) / 68, 6),
        "candidate_to_top10_conversion": f"{len(results[10])}/68",
        "promoted_gold": len(promoted), "demoted_gold": len(demoted),
        "context_present_gold_recall_at_5": f"{len(results[5] & context_present)}/{len(context_present)}",
        "raw_only_gold_recall_at_5": f"{len(results[5] & raw_only)}/{len(raw_only)}",
    })
    migration = []
    for binding in bindings:
        ranked = predictions[binding["case_id"]]["ranked_candidates"]
        item = next((candidate for candidate in ranked if candidate["candidate_key"] == binding["candidate_key"]), None)
        if item:
            migration.append({**binding, "context_status": item["context_status"], "pre_rerank_rank": item["pre_rerank_rank"], "post_rerank_rank": item["post_rerank_rank"], "promotion": binding["binding_id"] in promoted, "demotion": binding["binding_id"] in demoted})
    recall5 = len(results[5])
    multi5 = int(metrics["multi_evidence_complete_at_5"].split("/")[0])
    calc5 = int(metrics["calculation_complete_at_5"].split("/")[0])
    if recall5 >= 60 and multi5 >= 8 and calc5 >= 7:
        decision, next_gate, reached = "structure_aware_cross_encoder_target_reached", "evidence_set_semantic_equivalence", True
    elif 56 <= recall5 <= 59:
        next_gate = "slot_aware_neural_composition" if multi5 < 8 or calc5 < 7 else "qwen3_reranker_4b_capacity_escalation"
        decision, reached = "full_context_cross_encoder_meaningful", False
    else:
        decision, next_gate, reached = "full_context_cross_encoder_insufficient", "qwen3_reranker_4b_capacity_escalation", False
    write("strict-recall.json", metrics)
    write("context-slice-metrics.json", {"context_present_binding_ids": sorted(context_present), "raw_only_binding_ids": sorted(raw_only), "context_present_recall_at_5": metrics["context_present_gold_recall_at_5"], "raw_only_recall_at_5": metrics["raw_only_gold_recall_at_5"]})
    write("gold-promotion-demotion.json", {"promoted": len(promoted), "demoted": len(demoted), "records": migration})
    acceptance = {"gate": "pdf_retrieval_v4_gate_08_r8_r3_1", "decision": decision, "next_gate": next_gate, "metrics": metrics, "candidate_mutation": 0, "gold_reads_before_seal": 0, "grade_a_context_coverage_complete": True, "final_retrieval_target_reached": reached, "production_switch_allowed": False}
    write("acceptance.json", acceptance)
    write("next-gate.json", {"decision": decision, "next_gate": next_gate, "final_retrieval_target_reached": reached, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
