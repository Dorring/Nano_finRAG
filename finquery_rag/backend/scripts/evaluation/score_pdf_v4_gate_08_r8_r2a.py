#!/usr/bin/env python3
"""Post-seal 80-binding scoring for Gate 08 R8-R2A deep supply."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r2a"
PRED = OUT / "deep-supply-predictions.jsonl.gz"
SIDECAR = BASE / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
R7 = BASE / "pdf-retrieval-v4-gate-08-r7/field-family-predictions.jsonl.gz"
GATE08 = BASE / "pdf-retrieval-v4-gate-08/retrieval-predictions.jsonl.gz"
GOV = BASE / "pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
DEPTHS = (5, 10, 20, 40, 50)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {item["case_id"]: item for item in (json.loads(line) for line in handle if line.strip()) if item.get("case_id")}


def keys(items: list[dict[str, Any]]) -> list[str]:
    return [str(item["candidate_key"]) for item in items]


def rank(items: list[dict[str, Any]], candidate_key: str) -> int | None:
    values = keys(items)
    return values.index(candidate_key) + 1 if candidate_key in values else None


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(PRED) or seal["input_hashes"]["strict_source_contract"] != sha(SIDECAR):
        raise RuntimeError("r2a_prediction_seal_invalid")
    predictions, r7, original = map(load_gzip, (PRED, R7, GATE08))
    bindings = [json.loads(line) for line in SIDECAR.open(encoding="utf-8")]
    governance = {item["case_id"]: item for item in map(json.loads, GOV.open(encoding="utf-8"))}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in bindings:
        by_case[binding["case_id"]].append(binding)

    def score(pools: dict[str, set[str]]) -> tuple[int, int, int, set[str]]:
        hit_ids = {item["binding_id"] for item in bindings if item["candidate_key"] in pools[item["case_id"]]}
        multi = calculation = 0
        for case_id, expected in by_case.items():
            complete = all(item["candidate_key"] in pools[case_id] for item in expected)
            if governance[case_id]["requires_multiple_sources"]:
                multi += complete
            if governance[case_id]["query_type"] == "calculation_multi_operand":
                calculation += complete
        return len(hit_ids), multi, calculation, hit_ids

    bounded_metrics = {}
    for depth in DEPTHS:
        pools = {case_id: set(keys(record["bounded_candidate_top50"][:depth])) for case_id, record in predictions.items()}
        recall, multi, calculation, _ = score(pools)
        bounded_metrics[f"recall_at_{depth}"] = f"{recall}/80"
        bounded_metrics[f"multi_complete_at_{depth}"] = f"{multi}/16"
        bounded_metrics[f"calculation_complete_at_{depth}"] = f"{calculation}/11"
    deep_pools = {case_id: set(record["deep_supply_candidate_keys"]) for case_id, record in predictions.items()}
    bounded_pools = {case_id: set(keys(record["bounded_candidate_top50"])) for case_id, record in predictions.items()}
    deep_count, _, _, deep_hits = score(deep_pools)
    bounded_count, multi50, calc50, bounded_hits = score(bounded_pools)
    old_union_pools = {case_id: set(keys(record["r7_full_pool"])) for case_id, record in r7.items()}
    _, _, _, old_union_hits = score(old_union_pools)
    if len(old_union_hits) != 60:
        raise RuntimeError("r2a_baseline_supply_parity_blocked")
    raw_pools = {case_id: set(keys(record["raw_full_rrf_candidates"][:50])) for case_id, record in original.items()}
    _, _, _, raw_hits = score(raw_pools)
    raw_retained = raw_hits & bounded_hits
    raw_regressed = raw_hits - raw_retained
    rank_records = []
    recovery = Counter()
    for binding in bindings:
        case_id, candidate_key = binding["case_id"], binding["candidate_key"]
        record = predictions[case_id]
        lane_ranks: dict[str, int] = {}
        for lane, items in record["main_lane_hits"].items():
            value = rank(items, candidate_key)
            if value is not None:
                lane_ranks[f"main:{lane}"] = value
        for slot_id, trace in record["slot_deep_supply"].items():
            for lane, items in trace["lane_hits"].items():
                value = rank(items, candidate_key)
                if value is not None:
                    lane_ranks[f"slot:{slot_id}:{lane}"] = value
        best_rank = min(lane_ranks.values(), default=None)
        previously_missing = binding["binding_id"] not in old_union_hits
        if previously_missing:
            if best_rank is None:
                category = "still_missing_from_all_deep_lanes"
            elif best_rank <= 100:
                category = "recovered_rank_51_100" if best_rank > 50 else "recovered_rank_1_50_new_supply"
            else:
                category = "recovered_rank_101_200"
            recovery[category] += 1
        else:
            category = "baseline_present"
        rank_records.append({
            **binding,
            "lane_ranks": lane_ranks,
            "best_deep_lane_rank": best_rank,
            "deep_supply_present": binding["binding_id"] in deep_hits,
            "bounded_top50_present": binding["binding_id"] in bounded_hits,
            "baseline_unbounded_present": not previously_missing,
            "recovery_category": category,
        })
    if deep_count >= 72 and bounded_count >= 68 and len(raw_regressed) <= 1 and multi50 >= 10 and calc50 >= 8:
        decision, next_gate, reranker_allowed = "deep_candidate_supply_recovery_passed", "structure_aware_cross_encoder", True
    elif deep_count >= 68 and bounded_count < 68:
        decision, next_gate, reranker_allowed = "deep_supply_recovered_but_top50_compression_insufficient", "supply_aware_compression_audit", False
    elif deep_count < 68:
        decision, next_gate, reranker_allowed = "deep_candidate_supply_insufficient", "structured_bridge_supply_recovery", False
    else:
        decision, next_gate, reranker_allowed = "deep_candidate_supply_partial", "candidate_supply_failure_audit", False
    metrics = {
        "baseline_unbounded_presence": "60/80",
        "deep_supply_presence": f"{deep_count}/80",
        "bounded": bounded_metrics,
        "bounded_recall_at_50": f"{bounded_count}/80",
        "production_raw_own_recall_at_50": f"{len(raw_hits)}/80",
        "raw_retained": f"{len(raw_retained)}/{len(raw_hits)}",
        "raw_regression": len(raw_regressed),
        "multi_evidence_complete_at_50": f"{multi50}/16",
        "calculation_complete_at_50": f"{calc50}/11",
        "new_deep_supply_bindings": len(deep_hits - old_union_hits),
        "old_supply_bindings_lost": len(old_union_hits - deep_hits),
    }
    write("supply-presence.json", {"metrics": metrics, "deep_binding_ids": sorted(deep_hits)})
    write("bounded-recall.json", bounded_metrics)
    write("rank-recovery.json", {"summary": dict(recovery), "records": rank_records})
    write("multi-evidence-metrics.json", {"baseline": "10/16", "r2a": f"{multi50}/16"})
    write("calculation-metrics.json", {"baseline": "8/11", "r2a": f"{calc50}/11"})
    write("raw-retention.json", {"baseline": "23/24", "production_raw": f"{len(raw_hits)}/80", "retained": f"{len(raw_retained)}/{len(raw_hits)}", "regressed_binding_ids": sorted(raw_regressed)})
    first_failure = Counter(
        item["recovery_category"] for item in rank_records if not item["deep_supply_present"]
    )
    write("first-failure-attribution.json", dict(first_failure))
    acceptance = {"gate": "pdf_retrieval_v4_gate_08_r8_r2a", "decision": decision, "next_gate": next_gate, "metrics": metrics, "recovery_summary": dict(recovery), "reranker_allowed": reranker_allowed, "production_switch_allowed": False}
    write("acceptance.json", acceptance)
    write("next-gate.json", {"decision": decision, "next_gate": next_gate, "reranker_allowed": reranker_allowed, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
