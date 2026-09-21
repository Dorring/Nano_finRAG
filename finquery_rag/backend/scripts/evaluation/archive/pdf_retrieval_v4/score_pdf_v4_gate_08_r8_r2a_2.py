#!/usr/bin/env python3
"""Post-seal 80-binding scoring for Gate 08 R8-R2A.2."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r2a-2"
PRED = OUT / "bounded-top100-predictions.jsonl.gz"
R2A = BASE / "pdf-retrieval-v4-gate-08-r8-r2a/deep-supply-predictions.jsonl.gz"
SIDECAR = BASE / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
GOV = BASE / "pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
GATE08 = BASE / "pdf-retrieval-v4-gate-08/retrieval-predictions.jsonl.gz"
DEPTHS = (5, 10, 20, 40, 50, 100)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            item["case_id"]: item
            for item in (json.loads(line) for line in handle if line.strip())
            if item.get("case_id")
        }


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(PRED):
        raise RuntimeError("top100_prediction_seal_invalid")
    predictions, deep, original = map(load_gzip, (PRED, R2A, GATE08))
    bindings = [json.loads(line) for line in SIDECAR.open(encoding="utf-8")]
    governance = {item["case_id"]: item for item in map(json.loads, GOV.open(encoding="utf-8"))}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in bindings:
        by_case[binding["case_id"]].append(binding)

    def score(pools: dict[str, set[str]]) -> tuple[set[str], int, int]:
        hits = {b["binding_id"] for b in bindings if b["candidate_key"] in pools[b["case_id"]]}
        multi = calculation = 0
        for case_id, expected in by_case.items():
            complete = all(item["candidate_key"] in pools[case_id] for item in expected)
            multi += bool(complete and governance[case_id]["requires_multiple_sources"])
            calculation += bool(complete and governance[case_id]["query_type"] == "calculation_multi_operand")
        return hits, multi, calculation

    recall_curve = {}
    depth_results = {}
    for depth in DEPTHS:
        pools = {case_id: {item["candidate_key"] for item in record["candidates"][:depth]} for case_id, record in predictions.items()}
        hits, multi, calculation = score(pools)
        depth_results[depth] = (pools, hits, multi, calculation)
        recall_curve[f"recall_at_{depth}"] = f"{len(hits)}/80"
    top100_pools, top100_hits, multi100, calc100 = depth_results[100]
    deep_pools = {case_id: set(record["deep_supply_candidate_keys"]) for case_id, record in deep.items()}
    deep_hits, _, _ = score(deep_pools)
    if len(deep_hits) != 78:
        raise RuntimeError("deep_supply_presence_parity_blocked")
    raw_pools = {case_id: {item["candidate_key"] for item in record["raw_full_rrf_candidates"][:100]} for case_id, record in original.items()}
    raw_hits, _, _ = score(raw_pools)
    raw_retained = raw_hits & top100_hits
    raw_regressed = raw_hits - raw_retained
    conversion = len(top100_hits) / len(deep_hits)

    loss_records = []
    failure_counts: Counter[str] = Counter()
    for binding in bindings:
        if binding["binding_id"] not in deep_hits or binding["binding_id"] in top100_hits:
            continue
        record = predictions[binding["case_id"]]
        key = binding["candidate_key"]
        main = next((item for item in record["main_priority_ranking"] if item["candidate_key"] == key), None)
        slot_ranks = {
            slot: next((int(item["rank"]) for item in items if item["candidate_key"] == key), None)
            for slot, items in record["slot_priority_rankings"].items()
        }
        selected = record["candidates"]
        minimum_count = sum(bool(item.get("minimum_coverage_selected")) for item in selected)
        best_slot = min((rank for rank in slot_ranks.values() if rank is not None), default=None)
        if main and main["rank"] > 100 and (best_slot is None or best_slot > 100):
            category = "priority_rank_above_100"
        elif minimum_count and main and main["rank"] <= 100:
            category = "slot_minimum_coverage_displacement"
        elif best_slot is not None and best_slot <= 100 and main and main["rank"] <= 100:
            category = "main_vs_slot_competition"
        elif best_slot is not None and best_slot <= 100:
            category = "residual_budget_displacement"
        elif main is None and best_slot is None:
            category = "non_consumed_lane_only"
        else:
            category = "other"
        failure_counts[category] += 1
        loss_records.append({**binding, "first_failure": category, "main_priority_rank": main["rank"] if main else None, "slot_priority_ranks": slot_ranks, "minimum_coverage_selected_count": minimum_count})

    healthy = len(raw_regressed) <= 1 and multi100 >= 11 and calc100 >= 9
    if len(top100_hits) >= 72 and healthy:
        decision = "bounded_top100_rerank_input_strong_pass"
        reranker_allowed = True
        next_gate = "structure_aware_cross_encoder"
    elif len(top100_hits) >= 68 and conversion >= 0.87 and healthy:
        decision = "bounded_top100_rerank_input_passed"
        reranker_allowed = True
        next_gate = "structure_aware_cross_encoder"
    else:
        decision = "bounded_top100_rerank_input_blocked"
        reranker_allowed = False
        next_gate = "multislot_top100_budget_contract_audit" if len(top100_hits) >= 64 else "multichannel_rerank_input_contract_audit"
    metrics = {
        **recall_curve,
        "deep_supply_presence": "78/80",
        "deep_to_top100_conversion": f"{len(top100_hits)}/78",
        "deep_to_top100_conversion_rate": round(conversion, 6),
        "production_raw_own_recall_at_100": f"{len(raw_hits)}/80",
        "raw_retained_at_100": f"{len(raw_retained)}/{len(raw_hits)}",
        "raw_regression": len(raw_regressed),
        "multi_evidence_complete_at_100": f"{multi100}/16",
        "calculation_complete_at_100": f"{calc100}/11",
    }
    write("recall-curve.json", recall_curve)
    write("deep-to-top100-conversion.json", {"deep_presence": "78/80", "bounded_recall_at_100": f"{len(top100_hits)}/80", "conversion": f"{len(top100_hits)}/78", "rate": round(conversion, 6)})
    write("raw-retention.json", {"production_raw_own_recall_at_100": f"{len(raw_hits)}/80", "retained": f"{len(raw_retained)}/{len(raw_hits)}", "regression": len(raw_regressed), "regressed_binding_ids": sorted(raw_regressed)})
    write("multi-evidence-metrics.json", {"top50_baseline": "10/16", "complete_at_100": f"{multi100}/16"})
    write("calculation-metrics.json", {"top50_baseline": "8/11", "complete_at_100": f"{calc100}/11"})
    write("top100-loss-attribution.json", {"deep_present_top100_missing": len(loss_records), "counts": dict(failure_counts), "records": loss_records})
    acceptance = {"gate": "pdf_retrieval_v4_gate_08_r8_r2a_2", "decision": decision, "next_gate": next_gate, "metrics": metrics, "reranker_allowed": reranker_allowed, "production_switch_allowed": False}
    write("acceptance.json", acceptance)
    write("next-gate.json", {"decision": decision, "next_gate": next_gate, "reranker_allowed": reranker_allowed, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
