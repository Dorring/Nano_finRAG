#!/usr/bin/env python3
"""Post-seal scoring for Gate 08 R5 field-aware retrieval."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r5"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
UNIVERSE = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5/universe-scoring.json"
R3_FAILURES = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3/scoring/first-failure-attribution.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def keys(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("candidate_key") or "") for item in items]


def main() -> int:
    pred_path = OUT / "retrieval-predictions.jsonl.gz"
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(pred_path):
        raise RuntimeError("r5_prediction_seal_invalid")
    with gzip.open(pred_path, "rt", encoding="utf-8") as handle:
        predictions = {item["case_id"]: item for item in (json.loads(line) for line in handle if line.strip())}
    gold: list[tuple[str, int, str]] = []
    per_case_gold: dict[str, set[str]] = {}
    with LABELS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            case_id = str(record["case_id"])
            for index, source in enumerate(record.get("expected_sources") or []):
                key = str(source.get("candidate_key") or "")
                if key:
                    gold.append((case_id, index, key))
                    per_case_gold.setdefault(case_id, set()).add(key)
    universe_data = json.loads(UNIVERSE.read_text())
    universe_details = universe_data["details"]
    if len(universe_details) != len(gold):
        raise RuntimeError("universe_gold_record_count_mismatch")
    for identity, detail in zip(gold, universe_details, strict=True):
        if identity[0] != detail["case_id"] or identity[2] != detail["gold_candidate_key"]:
            raise RuntimeError("universe_gold_order_mismatch")
    grade_a = {
        identity
        for identity, detail in zip(gold, universe_details, strict=True)
        if str(detail.get("new_status")) == "mapped"
    }
    if len(grade_a) != 68:
        raise RuntimeError(f"grade_a_gold_not_68:{len(grade_a)}")

    def hit_set(pool_name: str, *, family: bool = False) -> set[tuple[str, int, str]]:
        result = set()
        for identity in gold:
            record = predictions[identity[0]]
            items = record["structured_family_rankings"][pool_name] if family else record[pool_name]
            if identity[2] in set(keys(items)):
                result.add(identity)
        return result

    structured_pool_hits = {name: {identity for identity in gold if identity[2] in set(keys(predictions[identity[0]]["structured_pools"][name]))} for name in [f"s{i}" for i in range(5)]}
    family_hits = {
        name: hit_set(name, family=True) & grade_a for name in [f"s{i}" for i in range(5)]
    }
    full_hits = hit_set("r5_full_pool")
    conversion = {name: f"{len(value)}/68" for name, value in family_hits.items()}
    e2 = {name: f"{len(value)}/80" for name, value in structured_pool_hits.items()}
    if conversion["s0"] != "49/68" or e2["s0"] != "57/80":
        raise RuntimeError(f"s0_scoring_parity_blocked:{conversion['s0']}:{e2['s0']}")
    s4 = len(family_hits["s4"])
    full = len(full_hits)
    if s4 >= 58 and full >= 64:
        decision, next_gate = "field_aware_retrieval_strong_pass", "slot_preserving_evidence_set"
    elif s4 >= 54 and full >= 60:
        decision, next_gate = "field_aware_retrieval_passed", "evidence_set_multi_evidence_completion"
    elif s4 >= 54:
        decision, next_gate = "field_aware_structured_gain_real_but_full_system_insufficient", "slot_aware_candidate_composition"
    elif 51 <= s4 <= 53 and 55 <= full <= 59:
        decision, next_gate = "field_aware_retrieval_small_gain", "field_support_normalization"
    else:
        decision, next_gate = "field_aware_lexical_retrieval_insufficient", "field_aware_dense_representation"
    rank_migration: list[dict[str, Any]] = []
    contribution = Counter()
    recovered = family_hits["s4"] - family_hits["s0"]
    regressed = family_hits["s0"] - family_hits["s4"]
    for identity in sorted(grade_a):
        record = predictions[identity[0]]
        before = keys(record["structured_family_rankings"]["s0"])
        after = keys(record["structured_family_rankings"]["s4"])
        br = before.index(identity[2]) + 1 if identity[2] in before else None
        ar = after.index(identity[2]) + 1 if identity[2] in after else None
        if br is None and ar is None:
            category = "still_missed_top50"
        elif br is None and ar is not None:
            category = "new_entry_top40" if ar <= 40 else "new_entry_top50"
        elif br is not None and ar is None:
            category = "dropped_out_top50"
        elif ar < br:
            category = "improved"
        elif ar == br:
            category = "unchanged"
        else:
            category = "worsened"
        lanes = []
        if ar is not None:
            lanes = record["structured_family_rankings"]["s4"][ar - 1].get("supporting_fields") or []
        rank_migration.append({"case_id": identity[0], "source_index": identity[1], "candidate_key": identity[2], "baseline_best_rank": br, "field_aware_best_rank": ar, "category": category, "supporting_lanes": lanes})
        if identity in recovered:
            for lane in lanes:
                contribution[lane] += 1
    multi_cases = [case_id for case_id, record in predictions.items() if record["is_multi_slot"]]
    multi_complete = sum(per_case_gold.get(case_id, set()).issubset(set(keys(predictions[case_id]["r5_full_pool"]))) for case_id in multi_cases)
    first_fail = Counter()
    for identity in gold:
        if identity in full_hits:
            first_fail["recovered_by_field_aware"] += 1
        elif identity not in grade_a:
            first_fail["outside_grade_a_universe"] += 1
        else:
            rank_items = predictions[identity[0]]["structured_family_rankings"]["s4"]
            candidate_keys = keys(rank_items)
            if identity[2] not in candidate_keys[:50]:
                first_fail["field_all_lanes_top50_miss"] += 1
            elif identity[2] not in candidate_keys[:40]:
                first_fail["field_candidate_rank_41_to_50"] += 1
            elif predictions[identity[0]]["is_multi_slot"]:
                first_fail["slot_pool_budget_truncated"] += 1
            else:
                first_fail["family_fusion_budget_loss"] += 1
    write("structured-ablation.json", {"structured_conversion": conversion, "e2_expanded": e2, "primary": "s4"})
    write("structured-conversion.json", {"baseline": "49/68", "s4": conversion["s4"], "gain": s4 - 49, "inside_universe_miss": 68 - s4})
    write("top50-miss-recovery.json", {"frozen_top50_miss": 12, "new_entries_top50": sum(item["category"].startswith("new_entry") for item in rank_migration), "recovered_into_top40": len(recovered), "still_missed_all_fields": sum(1 for item in rank_migration if item["field_aware_best_rank"] is None)})
    write("field-contribution.json", {"recovered_gold": sorted(recovered), "supporting_lane_counts": dict(contribution)})
    write("rank-migration.json", {"summary": dict(Counter(item["category"] for item in rank_migration)), "records": rank_migration, "regressed": sorted(regressed)})
    write("full-system-metrics.json", {"r4_f2_baseline": "54/80", "r5_full": f"{full}/80", "gain": full - 54, "raw_full_pool": "31/80", "raw_gold_retained": "31/31", "e0": "42/80"})
    write("multi-evidence-metrics.json", {"multi_slot_cases": len(multi_cases), "complete_gold_set_in_r5_full": f"{multi_complete}/{len(multi_cases)}"})
    write("first-failure-attribution.json", dict(first_fail))
    acceptance = {"gate": "pdf_retrieval_v4_gate_08_r5", "decision": decision, "next_gate": next_gate, "structured_conversion": conversion, "e2_expanded": e2, "r5_full": f"{full}/80", "raw_gold_retained": "31/31", "s0_parity": True, "parameter_scan": False, "weight_scan": False, "topk_scan": False, "production_switch_allowed": False}
    write("acceptance.json", acceptance)
    write("next-gate.json", {"current_gate": "pdf_retrieval_v4_gate_08_r5", "decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
