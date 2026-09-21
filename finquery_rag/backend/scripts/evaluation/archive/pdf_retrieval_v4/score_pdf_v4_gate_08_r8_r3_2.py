#!/usr/bin/env python3
"""Post-seal strict 80-binding score and 0.6B-to-4B rank delta."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r3-2"
PRED = OUT / "rerank-predictions.jsonl.gz"
BASELINE = BASE / "pdf-retrieval-v4-gate-08-r8-r3-1/rerank-predictions.jsonl.gz"
TOP100 = BASE / "pdf-retrieval-v4-gate-08-r8-r2a-2/bounded-top100-predictions.jsonl.gz"
SIDECAR = BASE / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
GOV = BASE / "pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
DEPTHS = (1, 3, 5, 10, 20, 50, 100)


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
        raise RuntimeError("r3_2_prediction_seal_invalid")
    predictions, baseline, top100 = map(load_gzip, (PRED, BASELINE, TOP100))
    bindings = [json.loads(line) for line in SIDECAR.open(encoding="utf-8")]
    governance = {item["case_id"]: item for item in map(json.loads, GOV.open(encoding="utf-8"))}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in bindings:
        by_case[binding["case_id"]].append(binding)
    for case_id, record in predictions.items():
        expected = {item["candidate_key"] for item in top100[case_id]["candidates"]}
        actual = {item["candidate_key"] for item in record["ranked_candidates"]}
        if len(actual) != 100 or actual != expected:
            raise RuntimeError(f"candidate_identity_mutation:{case_id}")

    def score(depth: int) -> tuple[set[str], int, int, int, int]:
        pools = {case_id: {item["candidate_key"] for item in record["ranked_candidates"][:depth]} for case_id, record in predictions.items()}
        hits = {item["binding_id"] for item in bindings if item["candidate_key"] in pools[item["case_id"]]}
        answerable = multi = calculation = single = 0
        for case_id, expected in by_case.items():
            any_hit = any(item["candidate_key"] in pools[case_id] for item in expected)
            complete = all(item["candidate_key"] in pools[case_id] for item in expected)
            answerable += any_hit
            multi += bool(complete and governance[case_id]["requires_multiple_sources"])
            calculation += bool(complete and governance[case_id]["query_type"] == "calculation_multi_operand")
            single += bool(any_hit and not governance[case_id]["requires_multiple_sources"])
        return hits, answerable, multi, calculation, single

    metrics, results = {}, {}
    for depth in DEPTHS:
        hits, answerable, multi, calculation, single = score(depth)
        results[depth] = hits
        metrics[f"strict_source_binding_recall_at_{depth}"] = f"{len(hits)}/80"
        metrics[f"answerable_question_hit_at_{depth}"] = f"{answerable}/72"
        if depth in (5, 10, 100):
            metrics[f"multi_evidence_complete_at_{depth}"] = f"{multi}/16"
            metrics[f"calculation_complete_at_{depth}"] = f"{calculation}/11"
            metrics[f"single_source_question_hit_at_{depth}"] = f"{single}/56"
    if len(results[100]) != 68:
        raise RuntimeError("top100_recall_identity_parity_blocked")
    baseline_ranks, four_b_ranks = {}, {}
    for case_id in predictions:
        baseline_ranks[case_id] = {item["candidate_key"]: item["post_rerank_rank"] for item in baseline[case_id]["ranked_candidates"]}
        four_b_ranks[case_id] = {item["candidate_key"]: item["post_rerank_rank"] for item in predictions[case_id]["ranked_candidates"]}
    baseline_top5 = {item["binding_id"] for item in bindings if baseline_ranks[item["case_id"]].get(item["candidate_key"], 101) <= 5}
    promoted = results[5] - baseline_top5
    demoted = baseline_top5 - results[5]
    retained = results[5] & baseline_top5
    migrations = [{**item, "r3_1_0_6b_rank": baseline_ranks[item["case_id"]].get(item["candidate_key"]), "r3_2_4b_rank": four_b_ranks[item["case_id"]].get(item["candidate_key"]), "promoted_by_4b": item["binding_id"] in promoted, "demoted_by_4b": item["binding_id"] in demoted} for item in bindings]
    recall5 = len(results[5])
    multi5 = int(metrics["multi_evidence_complete_at_5"].split("/")[0])
    calc5 = int(metrics["calculation_complete_at_5"].split("/")[0])
    metrics.update({"candidate_to_top5_conversion": f"{recall5}/68", "candidate_to_top5_conversion_rate": round(recall5 / 68, 6), "net_recall_at_5_gain_vs_r3_1": recall5 - 40, "4b_promoted_gold": len(promoted), "4b_demoted_gold": len(demoted), "0_6b_top5_retained_by_4b": len(retained), "0_6b_miss_recovered_by_4b": len(promoted)})
    if recall5 >= 60 and multi5 >= 8 and calc5 >= 7:
        decision, next_gate, reached = "qwen3_reranker_4b_target_reached", "evidence_set_semantic_equivalence", True
    elif 56 <= recall5 <= 59:
        decision, next_gate, reached = "4b_capacity_gain_meaningful_but_multisource_incomplete", "slot_aware_neural_composition", False
    elif 52 <= recall5 <= 55:
        single5 = int(metrics["single_source_question_hit_at_5"].split("/")[0])
        decision = "4b_capacity_gain_partial"
        next_gate = "slot_aware_neural_composition" if single5 >= 40 and (multi5 < 8 or calc5 < 7) else "4b_failure_attribution"
        reached = False
    else:
        decision, next_gate, reached = "qwen3_reranker_4b_insufficient", "4b_failure_attribution", False
    write("strict-recall.json", metrics)
    write("capacity-delta.json", {"baseline_recall_at_5": "40/80", "r3_2_recall_at_5": f"{recall5}/80", "net_gain": recall5 - 40, "promoted": len(promoted), "demoted": len(demoted), "retained": len(retained), "records": migrations})
    write("multi-evidence-metrics.json", {key: value for key, value in metrics.items() if "multi_evidence" in key})
    write("calculation-metrics.json", {key: value for key, value in metrics.items() if "calculation_complete" in key})
    acceptance = {"gate": "pdf_retrieval_v4_gate_08_r8_r3_2", "decision": decision, "next_gate": next_gate, "metrics": metrics, "candidate_mutation": 0, "gold_reads_before_seal": 0, "final_retrieval_target_reached": reached, "production_switch_allowed": False}
    write("acceptance.json", acceptance)
    write("next-gate.json", {"decision": decision, "next_gate": next_gate, "final_retrieval_target_reached": reached, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
