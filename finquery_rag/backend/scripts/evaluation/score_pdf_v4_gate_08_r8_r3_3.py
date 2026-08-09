#!/usr/bin/env python3
"""Post-seal scoring for QueryPlan-guided slot-aware Top5."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r3-3"
FINAL = OUT / "slot_aware_top5_predictions.jsonl.gz"
MAIN = OUT / "main_rerank_predictions.jsonl.gz"
SLOT = OUT / "slot_rerank_predictions.jsonl.gz"
SIDECAR = BASE / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
GOV = BASE / "pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {item["case_id"]: item for item in map(json.loads, handle)}


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["final_prediction_sha256"] != sha(FINAL) or seal["main_prediction_sha256"] != sha(MAIN) or seal["slot_prediction_sha256"] != sha(SLOT):
        raise RuntimeError("r3_3_prediction_seal_invalid")
    final, main = load(FINAL), load(MAIN)
    bindings = [json.loads(line) for line in SIDECAR.open(encoding="utf-8")]
    governance = {item["case_id"]: item for item in map(json.loads, GOV.open(encoding="utf-8"))}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in bindings:
        by_case[binding["case_id"]].append(binding)
    final_pools = {case_id: {item["candidate_key"] for item in record["candidates"]} for case_id, record in final.items()}
    main100 = {case_id: {item["candidate_key"] for item in record["ranked_candidates"]} for case_id, record in main.items()}
    hits = {item["binding_id"] for item in bindings if item["candidate_key"] in final_pools[item["case_id"]]}
    ceiling = {item["binding_id"] for item in bindings if item["candidate_key"] in main100[item["case_id"]]}
    if len(ceiling) != 68:
        raise RuntimeError("top100_identity_parity_blocked")
    multi = calculation = single = answerable = 0
    case_records = []
    for case_id, expected in by_case.items():
        any_hit = any(item["candidate_key"] in final_pools[case_id] for item in expected)
        complete = all(item["candidate_key"] in final_pools[case_id] for item in expected)
        is_multi = governance[case_id]["requires_multiple_sources"]
        is_calc = governance[case_id]["query_type"] == "calculation_multi_operand"
        answerable += any_hit
        multi += bool(complete and is_multi)
        calculation += bool(complete and is_calc)
        single += bool(any_hit and not is_multi)
        case_records.append({"case_id": case_id, "is_multi_evidence": is_multi, "is_calculation": is_calc, "required_source_count": len(expected), "retrieved_source_count": sum(item["candidate_key"] in final_pools[case_id] for item in expected), "complete_at_5": complete, "selection": [{"candidate_key": item["candidate_key"], "selection_source": item["selection_source"], "final_rank": item["final_rank"]} for item in final[case_id]["candidates"]]})
    baseline_main_top5 = {item["binding_id"] for item in bindings if any(candidate["candidate_key"] == item["candidate_key"] for candidate in main[item["case_id"]]["ranked_candidates"][:5])}
    promoted = hits - baseline_main_top5
    demoted = baseline_main_top5 - hits
    metrics = {"strict_source_binding_recall_at_5": f"{len(hits)}/80", "strict_source_binding_recall_at_100": "68/80", "top100_to_top5_conversion": f"{len(hits)}/68", "answerable_question_hit_at_5": f"{answerable}/72", "multi_evidence_complete_at_5": f"{multi}/16", "multi_evidence_input_ceiling": "12/16", "calculation_complete_at_5": f"{calculation}/11", "calculation_input_ceiling": "9/11", "single_source_question_hit_at_5": f"{single}/56", "single_source_regression_vs_r3_2": 30 - single, "slot_aware_promoted_gold": len(promoted), "slot_aware_demoted_gold": len(demoted)}
    if len(hits) >= 60 and multi >= 8 and calculation >= 7 and 30 - single <= 1:
        decision, next_gate, reached = "queryplan_guided_reranking_target_reached", "evidence_set_semantic_equivalence", True
    elif 55 <= len(hits) <= 59:
        decision, next_gate, reached = "queryplan_guided_reranking_near_target", "evidence_aware_candidate_pair_scoring", False
    else:
        decision, next_gate, reached = "queryplan_guided_reranking_below_physical_source_target", "semantic_evidence_recall_contract", False
    write("strict-metrics.json", metrics)
    write("case-completeness.json", {"metrics": metrics, "cases": case_records})
    write("composition-delta.json", {"main_4b_recall_at_5": "43/80", "slot_aware_recall_at_5": f"{len(hits)}/80", "promoted": len(promoted), "demoted": len(demoted), "promoted_binding_ids": sorted(promoted), "demoted_binding_ids": sorted(demoted)})
    acceptance = {"gate": "pdf_retrieval_v4_gate_08_r8_r3_3", "decision": decision, "next_gate": next_gate, "metrics": metrics, "candidate_mutation": 0, "gold_reads_before_seal": 0, "final_retrieval_target_reached": reached, "production_switch_allowed": False}
    write("acceptance.json", acceptance)
    write("next-gate.json", {"decision": decision, "next_gate": next_gate, "final_retrieval_target_reached": reached, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
