#!/usr/bin/env python3
"""Post-seal compression lineage audit for Gate 08 R8-R2A.1."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.bounded_candidate_selector import (  # noqa: E402
    select_multi_slot_top50,
)

BASE = ROOT / "artifacts/evaluation"
R2A = BASE / "pdf-retrieval-v4-gate-08-r8-r2a"
PRED = R2A / "deep-supply-predictions.jsonl.gz"
SIDECAR = BASE / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
GOV = BASE / "pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
R12_RAW = BASE / "pdf-retrieval-v4-gate-08-r8-r1-2/raw-retention.json"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r2a-1"
INF = 10**9


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {item["case_id"]: item for item in (json.loads(line) for line in handle if line.strip()) if item.get("case_id")}


def rank_item(items: list[dict[str, Any]], key: str) -> tuple[int | None, dict[str, Any]]:
    for position, item in enumerate(items, 1):
        if item.get("candidate_key") == key:
            return int(item.get("rank") or item.get("final_candidate_rank") or position), item
    return None, {}


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(values: list[int]) -> dict[str, int | None]:
    return {
        "count": len(values),
        "median": percentile(values, 0.5),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.9),
        "max": max(values) if values else None,
    }


def priority_ranking(
    raw_family: list[dict[str, Any]], structured_family: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    raw = {item["candidate_key"]: item for item in raw_family}
    structured = {item["candidate_key"]: item for item in structured_family}
    records = []
    for key in set(raw) | set(structured):
        raw_item = raw.get(key, {})
        structured_item = structured.get(key, {})
        raw_priority = raw_item.get("best_rank") or raw_item.get("rank")
        structured_priority = structured_item.get("best_rank") or structured_item.get("rank")
        priorities = sorted(value for value in (raw_priority, structured_priority) if value is not None)
        records.append(
            {
                "candidate_key": key,
                "top_priority_rank": priorities[0],
                "second_priority_rank": priorities[1] if len(priorities) > 1 else None,
                "raw_priority_rank": raw_priority,
                "structured_priority_rank": structured_priority,
            }
        )
    records.sort(
        key=lambda item: (
            item["top_priority_rank"],
            item["second_priority_rank"] if item["second_priority_rank"] is not None else INF,
            item["candidate_key"],
        )
    )
    return [{**item, "rank": rank} for rank, item in enumerate(records, 1)]


def classify(lineage: dict[str, Any]) -> str:
    atomic = lineage["best_atomic_lane_rank"]
    if atomic is None:
        return "candidate_present_only_in_non_consumed_lane"
    if atomic > 100:
        return "atomic_lane_rank_above_100"
    if atomic > 50:
        return "atomic_lane_rank_51_to_100"
    if lineage["structured_family_best_rank"] is not None and lineage["structured_family_best_rank"] <= 50 and (lineage["structured_family_ordinal_rank"] or INF) > 50:
        return "structured_family_rank_expansion"
    if lineage["raw_family_best_rank"] is not None and lineage["raw_family_best_rank"] <= 50 and (lineage["raw_family_ordinal_rank"] or INF) > 50:
        return "raw_family_rank_expansion"
    if lineage["main_top_level_best_rank"] is not None and lineage["main_top_level_best_rank"] <= 50 and (lineage["main_top_level_ordinal_rank"] or INF) > 50:
        return "top_level_family_rank_expansion"
    if lineage["is_multi_slot"]:
        if lineage["best_slot_family_best_rank"] is not None and lineage["best_slot_family_best_rank"] <= 50 and (lineage["best_slot_family_ordinal_rank"] or INF) > 50:
            return "slot_family_rank_expansion"
        if lineage["best_slot_family_ordinal_rank"] is not None and lineage["best_slot_family_ordinal_rank"] <= 50:
            return "slot_composition_budget_loss"
        if lineage["main_top_level_ordinal_rank"] is not None and lineage["main_top_level_ordinal_rank"] <= 50:
            return "main_residual_displacement"
    return "other"


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    seal = json.loads((R2A / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(PRED) or seal["input_hashes"]["strict_source_contract"] != sha(SIDECAR):
        raise RuntimeError("r2a_audit_input_seal_invalid")
    predictions = load_gzip(PRED)
    bindings = [json.loads(line) for line in SIDECAR.open(encoding="utf-8")]
    governance = {item["case_id"]: item for item in map(json.loads, GOV.open(encoding="utf-8"))}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in bindings:
        by_case[binding["case_id"]].append(binding)
    diagnostic_pools = {}
    accessibility_pools = {}
    diagnostic_trace = {}
    for case_id, record in predictions.items():
        main_priority = priority_ranking(record["raw_family_v2"], record["structured_family_v2"])
        slot_priority = {}
        for slot_id, trace in record["slot_deep_supply"].items():
            slot_priority[slot_id] = priority_ranking(
                trace["candidate_raw_fused"], trace["structured_family_v2"]
            )
        if slot_priority:
            selected, composition = select_multi_slot_top50(
                slot_priority, main_priority[:50]
            )
        else:
            selected, composition = main_priority[:50], {}
        diagnostic_pools[case_id] = {item["candidate_key"] for item in selected}
        accessible = {item["candidate_key"] for item in main_priority[:100]}
        for items in slot_priority.values():
            accessible.update(item["candidate_key"] for item in items[:100])
        accessibility_pools[case_id] = accessible
        diagnostic_trace[case_id] = {
            "main_priority_ranking": main_priority,
            "slot_priority_rankings": slot_priority,
            "composition_audit": composition,
            "diagnostic_top50": selected,
        }
    lineage_records = []
    deep_present = bounded_present = 0
    atomic_to_family: list[int] = []
    family_to_main: list[int] = []
    main_to_final: list[int] = []
    for binding in bindings:
        case_id, key = binding["case_id"], binding["candidate_key"]
        record = predictions[case_id]
        deep = key in set(record["deep_supply_candidate_keys"])
        final_rank, final_item = rank_item(record["bounded_candidate_top50"], key)
        deep_present += deep
        bounded_present += final_rank is not None
        atomic_ranks = {}
        for lane, items in record["main_lane_hits"].items():
            value, _ = rank_item(items, key)
            if value is not None:
                atomic_ranks[f"main:{lane}"] = value
        slot_lineage = {}
        for slot_id, trace in record["slot_deep_supply"].items():
            slot_atomic = {}
            for lane, items in trace["lane_hits"].items():
                value, _ = rank_item(items, key)
                if value is not None:
                    atomic_ranks[f"slot:{slot_id}:{lane}"] = value
                    slot_atomic[lane] = value
            slot_rank, slot_item = rank_item(trace["slot_candidate_ranking_v2"], key)
            slot_lineage[slot_id] = {
                "atomic_lane_ranks": slot_atomic,
                "best_atomic_rank": min(slot_atomic.values(), default=None),
                "slot_family_best_rank": slot_item.get("best_rank"),
                "slot_family_ordinal_rank": slot_rank,
            }
        candidate_raw_rank, _ = rank_item(record["candidate_raw_fused"], key)
        raw_rank, raw_item = rank_item(record["raw_family_v2"], key)
        h1_rank, _ = rank_item(record["structured_h1"], key)
        structured_rank, structured_item = rank_item(record["structured_family_v2"], key)
        main_rank, main_item = rank_item(record["deep_main_ranking"], key)
        best_atomic_name = min(atomic_ranks, key=atomic_ranks.get) if atomic_ranks else None
        best_atomic_rank = atomic_ranks.get(best_atomic_name) if best_atomic_name else None
        best_slot = min(
            (value for value in slot_lineage.values() if value["slot_family_ordinal_rank"] is not None),
            key=lambda value: value["slot_family_ordinal_rank"],
            default={},
        )
        family_ordinal_candidates = [value for value in (raw_rank, structured_rank) if value is not None]
        best_family_ordinal = min(family_ordinal_candidates, default=None)
        if deep and best_atomic_rank is not None and best_family_ordinal is not None:
            atomic_to_family.append(best_family_ordinal - best_atomic_rank)
        if deep and best_family_ordinal is not None and main_rank is not None:
            family_to_main.append(main_rank - best_family_ordinal)
        if deep and main_rank is not None and final_rank is not None:
            main_to_final.append(final_rank - main_rank)
        lineage = {
            **binding,
            "is_multi_slot": record["is_multi_slot"],
            "deep_supply_present": deep,
            "bounded_top50_present": final_rank is not None,
            "atomic_lane_ranks": atomic_ranks,
            "best_atomic_lane_name": best_atomic_name,
            "best_atomic_lane_rank": best_atomic_rank,
            "candidate_raw_fused_rank": candidate_raw_rank,
            "raw_family_best_rank": raw_item.get("best_rank"),
            "raw_family_ordinal_rank": raw_rank,
            "structured_h1_rank": h1_rank,
            "structured_family_best_rank": structured_item.get("best_rank"),
            "structured_family_ordinal_rank": structured_rank,
            "main_top_level_best_rank": main_item.get("best_rank"),
            "main_top_level_ordinal_rank": main_rank,
            "slot_lineage": slot_lineage,
            "best_slot_family_best_rank": best_slot.get("slot_family_best_rank"),
            "best_slot_family_ordinal_rank": best_slot.get("slot_family_ordinal_rank"),
            "minimum_coverage_selected": final_item.get("minimum_coverage_selected"),
            "residual_rank": final_item.get("residual_lane_ranks"),
            "main_residual_rank": (final_item.get("residual_lane_ranks") or {}).get("main"),
            "final_candidate_rank": final_rank,
        }
        if deep and final_rank is None:
            lineage["first_compression_failure"] = classify(lineage)
        else:
            lineage["first_compression_failure"] = "bounded_present" if final_rank else "deep_supply_missing"
        lineage_records.append(lineage)
    misses = [item for item in lineage_records if item["deep_supply_present"] and not item["bounded_top50_present"]]
    if (deep_present, bounded_present, len(misses)) != (78, 57, 21):
        raise RuntimeError(f"compression_audit_count_mismatch:{deep_present}:{bounded_present}:{len(misses)}")
    failure_summary = Counter(item["first_compression_failure"] for item in misses)
    atomic_thresholds = {
        f"best_atomic_rank_le_{cutoff}": sum((item["best_atomic_lane_rank"] or INF) <= cutoff for item in misses)
        for cutoff in (10, 20, 50, 100, 200)
    }
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
    diagnostic_recall, diagnostic_multi, diagnostic_calc, diagnostic_hits = score(diagnostic_pools)
    accessibility, _, _, accessible_hits = score(accessibility_pools)
    raw_binding_ids = set(json.loads(R12_RAW.read_text())["raw_binding_ids"])
    raw_retained = len(raw_binding_ids & diagnostic_hits)
    if diagnostic_recall >= 68 and raw_retained >= 23 and diagnostic_multi >= 10 and diagnostic_calc >= 8:
        decision, next_gate = "hierarchical_rank_expansion_confirmed", "rank_provenance_preserving_top50"
    elif diagnostic_recall < 68 and accessibility >= 68:
        decision, next_gate = "top50_heuristic_compression_ceiling_reached", "bounded_top100_rerank_input"
    else:
        decision, next_gate = "bounded_top100_compression_contract_required", "bounded_top100_compression_contract"
    write("protocol.json", {"gate": "pdf_retrieval_v4_gate_08_r8_r2a_1", "diagnostic_only": True, "prediction_reruns": 0, "retriever_reruns": 0, "bm25_searches": 0, "dense_searches": 0, "embedding_calls": 0, "index_reads": 0, "index_builds": 0, "bridge_changes": 0, "query_changes": 0, "fusion_changes": 0, "selector_changes": 0, "reranker_calls": 0, "calculator_calls": 0, "generator_calls": 0, "production_switch_allowed": False})
    write("compression-lineage.json", {"records": lineage_records})
    write("rank-inflation.json", {"atomic_to_family_delta": summarize(atomic_to_family), "family_to_main_delta": summarize(family_to_main), "main_to_final_delta": summarize(main_to_final)})
    write("deep-present-top50-misses.json", {"count": len(misses), "atomic_rank_thresholds": atomic_thresholds, "records": misses})
    write("first-compression-failure.json", dict(failure_summary))
    write("provenance-diagnostic.json", {"diagnostic_only": True, "current_bounded_at50": "57/80", "provenance_diagnostic_at50": f"{diagnostic_recall}/80", "raw_retained": f"{raw_retained}/24", "multi_evidence": f"{diagnostic_multi}/16", "calculation": f"{diagnostic_calc}/11", "trace": diagnostic_trace})
    write("rerank-accessibility-at-100.json", {"diagnostic_only": True, "accessible_bindings": f"{accessibility}/80", "binding_ids": sorted(accessible_hits)})
    write("raw-retention-diagnostic.json", {"raw_retained": f"{raw_retained}/24"})
    write("multi-evidence-diagnostic.json", {"complete": f"{diagnostic_multi}/16"})
    write("calculation-diagnostic.json", {"complete": f"{diagnostic_calc}/11"})
    acceptance = {"gate": "pdf_retrieval_v4_gate_08_r8_r2a_1", "decision": decision, "next_gate": next_gate, "deep_present": "78/80", "bounded_top50": "57/80", "compression_gap": 21, "first_failure_classified": "21/21", "provenance_diagnostic_at50": f"{diagnostic_recall}/80", "rerank_input_accessibility_at100": f"{accessibility}/80", "bridge_recovery_needed": False, "embedding_change_allowed": False, "reranker_allowed": False, "production_switch_allowed": False, "r2a_prediction_sha256": sha(PRED)}
    write("acceptance.json", acceptance)
    write("next-gate.json", {"decision": decision, "next_gate": next_gate, "reranker_allowed": False, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2))
    print(json.dumps({"failure_summary": dict(failure_summary), "atomic_thresholds": atomic_thresholds}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
