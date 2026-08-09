#!/usr/bin/env python3
"""Post-seal 80-binding scoring for Gate 08 R8-R1.2."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r1-2"
PRED = OUT / "support-invariant-predictions.jsonl.gz"
SIDECAR = BASE / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
R7 = BASE / "pdf-retrieval-v4-gate-08-r7/field-family-predictions.jsonl.gz"
GATE08 = BASE / "pdf-retrieval-v4-gate-08/retrieval-predictions.jsonl.gz"
GOV = BASE / "pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
R11_LOSSES = BASE / "pdf-retrieval-v4-gate-08-r8-r1-1/boundary-loss-audit.json"
DEPTHS = (5, 10, 20, 40, 50)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {item["case_id"]: item for item in (json.loads(line) for line in handle if line.strip()) if item.get("case_id")}


def keys(items: list[dict[str, Any]]) -> list[str]:
    return [str(item["candidate_key"]) for item in items]


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(PRED) or seal["input_hashes"]["strict_source_contract"] != sha(SIDECAR):
        raise RuntimeError("r1_2_prediction_seal_invalid")
    predictions, r7, original = map(load_gzip, (PRED, R7, GATE08))
    bindings = [json.loads(line) for line in SIDECAR.open(encoding="utf-8")]
    governance = {item["case_id"]: item for item in map(json.loads, GOV.open(encoding="utf-8"))}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in bindings:
        by_case[binding["case_id"]].append(binding)

    def pool_map(field: str, depth: int = 50) -> dict[str, set[str]]:
        return {case_id: set(keys(record[field][:depth])) for case_id, record in predictions.items()}

    def score(pools: dict[str, set[str]]) -> tuple[int, int, int, set[str]]:
        hit_ids = {binding["binding_id"] for binding in bindings if binding["candidate_key"] in pools[binding["case_id"]]}
        multi = calculation = 0
        for case_id, expected in by_case.items():
            complete = all(item["candidate_key"] in pools[case_id] for item in expected)
            if governance[case_id]["requires_multiple_sources"]:
                multi += complete
            if governance[case_id]["query_type"] == "calculation_multi_operand":
                calculation += complete
        return len(hit_ids), multi, calculation, hit_ids

    h0_metrics = {}
    h1_metrics = {}
    for depth in DEPTHS:
        for target, field in ((h0_metrics, "h0_bounded_candidate_ranking"), (h1_metrics, "h1_bounded_candidate_ranking")):
            recall, multi, calculation, _ = score(pool_map(field, depth))
            target[f"recall_at_{depth}"] = f"{recall}/80"
            target[f"multi_complete_at_{depth}"] = f"{multi}/16"
            target[f"calculation_complete_at_{depth}"] = f"{calculation}/11"
    if (h0_metrics["recall_at_50"], h0_metrics["multi_complete_at_50"], h0_metrics["calculation_complete_at_50"]) != ("55/80", "9/16", "7/11"):
        raise RuntimeError(f"r1_2_baseline_parity_blocked:{h0_metrics}")
    h0_pools, h1_pools = pool_map("h0_bounded_candidate_ranking"), pool_map("h1_bounded_candidate_ranking")
    _, _, _, h0_hits = score(h0_pools)
    h1_recall, h1_multi, h1_calc, h1_hits = score(h1_pools)
    union_pools = {case_id: set(keys(record["r7_full_pool"])) for case_id, record in r7.items()}
    union_hits = {binding["binding_id"] for binding in bindings if binding["candidate_key"] in union_pools[binding["case_id"]]}
    if len(union_hits) != 60:
        raise RuntimeError(f"candidate_supply_mutated:{len(union_hits)}")
    raw_pools = {case_id: set(keys(record["raw_full_rrf_candidates"][:50])) for case_id, record in original.items()}
    raw_hits = {binding["binding_id"] for binding in bindings if binding["candidate_key"] in raw_pools[binding["case_id"]]}
    raw_retained = raw_hits & h1_hits
    raw_regressed = raw_hits - raw_retained
    gross_loss = union_hits - h1_hits
    synergy = h1_hits - union_hits
    added, regressed = h1_hits - h0_hits, h0_hits - h1_hits
    conversion = h1_recall / 60
    if len(raw_regressed) > 1:
        decision, next_gate = "support_count_invariant_raw_retention_blocked", "residual_candidate_fusion_repair"
    elif h1_recall >= 57 and conversion >= 0.95 and h1_multi >= 9 and h1_calc >= 7 and len(regressed) <= 1:
        decision, next_gate = "support_count_invariant_fusion_strong_pass", "candidate_supply_recovery"
    elif h1_recall >= 54 and conversion >= 0.90 and h1_multi >= 9 and h1_calc >= 7:
        decision, next_gate = "support_count_invariant_fusion_passed", "candidate_supply_recovery"
    else:
        decision, next_gate = "support_count_invariant_fusion_insufficient", "candidate_fusion_failure_audit"
    metrics = {
        "h0": h0_metrics,
        "h1": h1_metrics,
        "unbounded_presence": "60/80",
        "union_to_top50_conversion": f"{h1_recall}/60",
        "union_to_top50_conversion_rate": conversion,
        "h1_gross_loss": len(gross_loss),
        "h1_synergy": len(synergy),
        "h1_net_gap": len(gross_loss) - len(synergy),
        "production_raw_own_recall_at_50": f"{len(raw_hits)}/80",
        "raw_retained": f"{len(raw_retained)}/{len(raw_hits)}",
        "raw_regression": len(raw_regressed),
        "multi_evidence_complete_at_50": f"{h1_multi}/16",
        "calculation_complete_at_50": f"{h1_calc}/11",
        "h0_gold_retained": f"{len(h0_hits & h1_hits)}/55",
        "new_gold_added": len(added),
        "h0_gold_regressed": len(regressed),
        "net_gain": h1_recall - 55,
    }
    original_losses = json.loads(R11_LOSSES.read_text())["records"]
    migration = []
    for loss in original_losses:
        case_id, key = loss["case_id"], loss["candidate_key"]
        h0_keys = keys(predictions[case_id]["h0_bounded_candidate_ranking"])
        h1_keys = keys(predictions[case_id]["h1_bounded_candidate_ranking"])
        migration.append({
            "case_id": case_id,
            "source_index": loss["source_index"],
            "candidate_key": key,
            "original_failure_category": loss["first_failure_stage"],
            "h0_final_rank": h0_keys.index(key) + 1 if key in h0_keys else None,
            "h1_final_rank": h1_keys.index(key) + 1 if key in h1_keys else None,
            "recovered_to_top50": key in h1_keys,
            "still_lost": key not in h1_keys,
        })
    special = {}
    for item in migration:
        case_id = item["case_id"]
        if case_id not in {"pfe_fy2024_005", "tsla_fy2025_007"}:
            continue
        key = item["candidate_key"]
        prediction = predictions[case_id]
        raw_family = keys(prediction["raw_family_v2"])
        main = keys(prediction["main_candidate_ranking_v2"])
        slot_ranks = {
            slot_id: (
                keys(trace["slot_candidate_ranking_v2"]).index(key) + 1
                if key in keys(trace["slot_candidate_ranking_v2"])
                else None
            )
            for slot_id, trace in prediction["slot_family_trace_v2"].items()
        }
        final = keys(prediction["h1_bounded_candidate_ranking"])
        final_item = next(
            (
                candidate
                for candidate in prediction["h1_bounded_candidate_ranking"]
                if candidate["candidate_key"] == key
            ),
            {},
        )
        special[case_id] = {
            **item,
            "h1_raw_family_rank": raw_family.index(key) + 1 if key in raw_family else None,
            "h1_main_top_level_rank": main.index(key) + 1 if key in main else None,
            "h1_slot_local_ranks": slot_ranks,
            "minimum_coverage_selected": final_item.get("minimum_coverage_selected"),
            "residual_rrf_score": final_item.get("residual_rrf_score"),
            "h1_final_rank": final.index(key) + 1 if key in final else None,
        }
    write("fusion-ablation.json", {"h0": h0_metrics, "h1": h1_metrics})
    write("full-system-metrics.json", metrics)
    write("gross-loss-migration.json", {"records": migration})
    write("raw-retention.json", {"raw_binding_ids": sorted(raw_hits), "retained": sorted(raw_retained), "regressed": sorted(raw_regressed)})
    write("pfe-tesla-audit.json", special)
    write("regression-matrix.json", {"h0_hit_h1_hit": len(h0_hits & h1_hits), "h0_miss_h1_hit": len(added), "h0_hit_h1_miss": len(regressed), "h0_miss_h1_miss": 80 - len(h0_hits | h1_hits)})
    acceptance = {"gate": "pdf_retrieval_v4_gate_08_r8_r1_2", "decision": decision, "next_gate": next_gate, "metrics": metrics, "strict_source_contract_sha256": sha(SIDECAR), "reranker_allowed": False, "production_switch_allowed": False}
    write("acceptance.json", acceptance)
    write("next-gate.json", {"decision": decision, "next_gate": next_gate, "reranker_allowed": False, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
