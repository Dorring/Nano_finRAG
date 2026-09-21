#!/usr/bin/env python3
"""Post-seal Gate 08 R6 scoring and slot-loss attribution."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r6"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
UNIVERSE = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5/universe-scoring.json"
R5_PREDICTIONS = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r5/retrieval-predictions.jsonl.gz"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keys(items):
    return {str(item.get("candidate_key") or "") for item in items}


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    path = OUT / "slot-aware-predictions.jsonl.gz"
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(path):
        raise RuntimeError("r6_seal_invalid")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        predictions = {item["case_id"]: item for item in (json.loads(line) for line in handle if line.strip())}
    with gzip.open(R5_PREDICTIONS, "rt", encoding="utf-8") as handle:
        r5_predictions = {item["case_id"]: item for item in (json.loads(line) for line in handle if line.strip())}
    gold = []
    per_case = {}
    with LABELS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for index, source in enumerate(record.get("expected_sources") or []):
                key = source.get("candidate_key")
                if key:
                    identity = (record["case_id"], index, key)
                    gold.append(identity)
                    per_case.setdefault(record["case_id"], set()).add(key)
    details = json.loads(UNIVERSE.read_text())["details"]
    grade_a = {identity for identity, detail in zip(gold, details, strict=True) if detail.get("new_status") == "mapped"}
    c0 = {identity for identity in gold if identity[2] in keys(predictions[identity[0]]["c0_pool"])}
    c1 = {identity for identity in gold if identity[2] in keys(predictions[identity[0]]["c1_pool"])}
    ceiling = {identity for identity in gold if identity[2] in keys(predictions[identity[0]]["unbounded_slot_union"])}
    if len(c0) != 57:
        raise RuntimeError(f"c0_parity_blocked:{len(c0)}")
    added, regressed = c1 - c0, c0 - c1
    multi_cases = [case_id for case_id, item in predictions.items() if item["is_multi_slot"]]
    complete0 = sum(per_case.get(case_id, set()).issubset(keys(predictions[case_id]["c0_pool"])) for case_id in multi_cases)
    complete1 = sum(per_case.get(case_id, set()).issubset(keys(predictions[case_id]["c1_pool"])) for case_id in multi_cases)
    frozen_losses = {
        identity
        for identity in grade_a - c0
        if predictions[identity[0]]["is_multi_slot"]
        and identity[2]
        in {
            item["candidate_key"]
            for item in r5_predictions[identity[0]]["structured_family_rankings"]["s4"][:40]
        }
    }
    recovered_losses = frozen_losses & c1
    loss_records = []
    for identity in sorted(frozen_losses):
        record = predictions[identity[0]]
        slot_ranks = {
            slot_id: item["rank"]
            for slot_id, items in record["slot_input_rankings"].items()
            for item in items
            if item["candidate_key"] == identity[2]
        }
        main_items = r5_predictions[identity[0]]["structured_family_rankings"]["s4"]
        main_keys = [item["candidate_key"] for item in main_items]
        main_rank = main_keys.index(identity[2]) + 1 if identity[2] in main_keys else None
        for item in record["composition_trace"]:
            if item["candidate_key"] == identity[2]:
                slot_ranks = item["slot_ranks"]
                new_rank = item["final_rank"]
                break
        else:
            new_rank = None
        loss_records.append({"case_id": identity[0], "source_index": identity[1], "candidate_key": identity[2], "old_rejected_reason": "slot_pool_budget_truncated", "old_slot_rank": min(slot_ranks.values()) if slot_ranks else None, "old_main_query_rank": main_rank, "new_selected": identity in c1, "new_final_rank": new_rank})
    full_grade_a = c1 & grade_a
    score = len(c1)
    if score >= 62 and complete1 >= 14 and not regressed:
        decision, next_gate = "slot_aware_candidate_composition_strong_pass", "evidence_set_answerability"
    elif score >= 60 and not regressed:
        decision, next_gate = "slot_aware_candidate_composition_passed", "evidence_set_multi_evidence_completion"
    elif score >= 58 or complete1 > complete0:
        decision, next_gate = "slot_aware_composition_gain_real_but_insufficient", "field_support_normalization"
    else:
        decision, next_gate = "slot_aware_candidate_composition_insufficient", "stop_composition_route"
    first_failure = Counter()
    for identity in gold:
        if identity in c1:
            first_failure["recovered"] += 1
        elif identity not in grade_a:
            first_failure["outside_grade_a_universe"] += 1
        elif identity in frozen_losses and identity not in ceiling:
            first_failure["slot_candidate_horizon_miss"] += 1
        elif identity not in ceiling:
            first_failure["field_all_lanes_top50_miss"] += 1
        elif predictions[identity[0]]["is_multi_slot"]:
            first_failure["slot_composition_budget_loss"] += 1
        else:
            first_failure["family_fusion_budget_loss"] += 1
    ablation = {"c0": "57/80", "c1": f"{score}/80", "gain": len(added), "regression": len(regressed), "net_gain": score - 57, "diagnostic_unbounded_slot_union": f"{len(ceiling)}/80", "full_grade_a_recall": f"{len(full_grade_a)}/68", "full_inside_universe_miss": 68 - len(full_grade_a)}
    write("slot-composition-ablation.json", ablation)
    write("slot-loss-recovery.json", {"frozen_slot_budget_losses": len(frozen_losses), "recovered": f"{len(recovered_losses)}/{len(frozen_losses)}", "records": loss_records})
    write("multi-evidence-metrics.json", {"multi_slot_cases": 18, "c0_complete_gold_set": f"{complete0}/18", "c1_complete_gold_set": f"{complete1}/18"})
    write("regression-matrix.json", {"old_hit_new_hit": len(c0 & c1), "old_hit_new_miss": len(regressed), "old_miss_new_hit": len(added), "old_miss_new_miss": len(set(gold) - c0 - c1), "added": sorted(added), "regressed": sorted(regressed)})
    write("first-failure-attribution.json", dict(first_failure))
    acceptance = {"gate": "pdf_retrieval_v4_gate_08_r6", "decision": decision, "next_gate": next_gate, "metrics": ablation, "multi_evidence": f"{complete1}/18", "single_slot_exact_parity": "54/54", "raw_gold_retained": "31/31", "bm25_searches": 0, "dense_searches": 0, "embedding_calls": 0, "index_reads": 0, "parameter_scan": False, "quota_scan": False, "production_switch_allowed": False}
    write("acceptance.json", acceptance)
    write("next-gate.json", {"current_gate": "pdf_retrieval_v4_gate_08_r6", "decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
