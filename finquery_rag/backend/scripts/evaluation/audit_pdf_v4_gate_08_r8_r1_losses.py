#!/usr/bin/env python3
"""Post-seal causal audit of Gate 08 R8-R1 bounded-selector losses."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
R1 = BASE / "pdf-retrieval-v4-gate-08-r8-r1"
R7 = BASE / "pdf-retrieval-v4-gate-08-r7/field-family-predictions.jsonl.gz"
PRED = R1 / "candidate-top50-predictions.jsonl.gz"
GOV = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r1-1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            item["case_id"]: item
            for item in (json.loads(line) for line in handle if line.strip())
            if item.get("case_id")
        }


def rank_map(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(item["candidate_key"]): int(
            item.get("final_candidate_rank") or item.get("rank") or position
        )
        for position, item in enumerate(items, 1)
    }


def lane_trace(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return next((item for item in items if item.get("candidate_key") == key), {})


def classify(record: dict[str, Any]) -> str:
    raw_best = min(
        (value for value in (record["production_raw_rank"], record["candidate_raw_rank"]) if value),
        default=None,
    )
    structured_best = min(
        (
            value
            for value in (
                record["structured_h1_rank"],
                record["metric_rank"],
                record["existing_structured_rank"],
            )
            if value
        ),
        default=None,
    )
    if record["is_multi_slot"] and (
        record["slot_candidate_rank"] is not None
        or record["top_level_candidate_rank"] is not None
    ):
        if record["slot_candidate_rank"] is not None and record["slot_candidate_rank"] <= 10:
            return "slot_minimum_budget_displacement"
        return "main_query_residual_displacement"
    if raw_best is not None and record["raw_family_rank"] is not None and record["raw_family_rank"] > 50 and raw_best <= 50:
        return "raw_family_internal_demotion"
    if structured_best is not None and record["structured_family_rank"] is not None and record["structured_family_rank"] > 50 and structured_best <= 50:
        return "structured_family_internal_demotion"
    if record["is_multi_slot"]:
        return "slot_local_rank_miss"
    best_family = min(
        (value for value in (record["raw_family_rank"], record["structured_family_rank"]) if value),
        default=None,
    )
    if best_family is not None and best_family <= 50:
        return "top_level_consensus_displacement"
    if record["top_level_candidate_rank"] is not None and 51 <= record["top_level_candidate_rank"] <= 60:
        return "rank_51_to_60_boundary"
    if raw_best is None and structured_best is None:
        return "family_absent_but_union_present"
    return "other"


def main() -> int:
    seal = json.loads((R1 / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(PRED):
        raise RuntimeError("r8_r1_seal_invalid")
    predictions, r7 = load_gzip(PRED), load_gzip(R7)
    governance = {
        item["case_id"]: item for item in map(json.loads, GOV.open(encoding="utf-8"))
    }
    strict_gold: dict[str, list[str]] = {}
    for item in map(json.loads, LABELS.open(encoding="utf-8")):
        strict_gold[item["case_id"]] = [
            source["candidate_key"]
            for source in item.get("expected_sources", [])
            if source.get("candidate_key")
        ]
    losses = []
    union_gold = bounded_gold = 0
    diagnostic_rankings: dict[str, list[str]] = {}
    for case_id, prediction in predictions.items():
        bounded = rank_map(prediction["bounded_candidate_ranking"])
        union = {item["candidate_key"] for item in r7[case_id]["r7_full_pool"]}
        raw = rank_map(prediction["raw_family"])
        structured = rank_map(prediction["structured_family"])
        main = rank_map(prediction["main_candidate_ranking"])
        candidate_universe = set(raw) | set(structured)
        diagnostic_rankings[case_id] = sorted(
            candidate_universe,
            key=lambda key: (
                min(raw.get(key, 10**9), structured.get(key, 10**9)),
                max(raw.get(key, 10**9), structured.get(key, 10**9)),
                key,
            ),
        )[:50]
        for source_index, key in enumerate(strict_gold[case_id]):
            if key in union:
                union_gold += 1
            if key in bounded:
                bounded_gold += 1
            if key not in union or key in bounded:
                continue
            raw_item = lane_trace(prediction["raw_family"], key)
            structured_item = lane_trace(prediction["structured_family"], key)
            raw_lanes = raw_item.get("lane_ranks", {})
            structured_lanes = structured_item.get("lane_ranks", {})
            slot_ranks = {
                slot_id: rank_map(slot["candidate_ranking"]).get(key)
                for slot_id, slot in prediction["slot_rankings"].items()
            }
            slot_ranks = {slot: rank for slot, rank in slot_ranks.items() if rank is not None}
            record = {
                "case_id": case_id,
                "source_index": source_index,
                "candidate_key": key,
                "query_type": governance[case_id]["query_type"],
                "is_multi_slot": prediction["is_multi_slot"],
                "production_raw_rank": raw_lanes.get("production_raw"),
                "candidate_raw_rank": raw_lanes.get("candidate_raw"),
                "raw_family_rank": raw.get(key),
                "structured_h1_rank": structured_lanes.get("structured_h1"),
                "metric_rank": structured_lanes.get("structured_metric"),
                "existing_structured_rank": structured_lanes.get("existing_structured"),
                "structured_family_rank": structured.get(key),
                "top_level_candidate_rank": main.get(key),
                "slot_id": min(slot_ranks, key=slot_ranks.get) if slot_ranks else None,
                "slot_candidate_rank": min(slot_ranks.values()) if slot_ranks else None,
                "slot_candidate_ranks": slot_ranks,
                "final_candidate_rank": None,
                "raw_support_count": len(raw_lanes),
                "structured_support_count": len(structured_lanes),
                "top_level_family_support_count": int(key in raw) + int(key in structured),
            }
            record["first_failure_stage"] = classify(record)
            losses.append(record)
    governance_union = governance_bounded = 0
    labels_synergy = 0
    governance_gross_loss = governance_synergy = 0
    identity_disagreements = []
    for case_id, gov in governance.items():
        union = {item["candidate_key"] for item in r7[case_id]["r7_full_pool"]}
        bounded = rank_map(predictions[case_id]["bounded_candidate_ranking"])
        governance_union += sum(key in union for key in gov["strict_gold_identities"])
        governance_bounded += sum(key in bounded for key in gov["strict_gold_identities"])
        governance_gross_loss += sum(
            key in union and key not in bounded for key in gov["strict_gold_identities"]
        )
        governance_synergy += sum(
            key not in union and key in bounded for key in gov["strict_gold_identities"]
        )
        labels_synergy += sum(key not in union and key in bounded for key in strict_gold[case_id])
        label_keys = strict_gold[case_id]
        gov_keys = gov["strict_gold_identities"]
        for source_index in range(max(len(label_keys), len(gov_keys))):
            label_key = label_keys[source_index] if source_index < len(label_keys) else None
            governance_key = gov_keys[source_index] if source_index < len(gov_keys) else None
            if label_key != governance_key:
                identity_disagreements.append(
                    {
                        "case_id": case_id,
                        "source_index": source_index,
                        "labels_candidate_key": label_key,
                        "governance_candidate_key": governance_key,
                    }
                )
    identity_contract_consistent = (
        union_gold == 60
        and bounded_gold == 50
        and len(losses) == 10
        and governance_union == union_gold
        and governance_bounded == bounded_gold
        and not identity_disagreements
    )
    diagnostic_gold = sum(
        key in diagnostic_rankings[case_id]
        for case_id, keys in strict_gold.items()
        for key in keys
    )
    boundary_records = []
    for loss in losses:
        case_id = loss["case_id"]
        boundary_key = predictions[case_id]["bounded_candidate_ranking"][-1]["candidate_key"]
        raw = rank_map(predictions[case_id]["raw_family"])
        structured = rank_map(predictions[case_id]["structured_family"])
        gold_best = min(raw.get(loss["candidate_key"], 10**9), structured.get(loss["candidate_key"], 10**9))
        boundary_best = min(raw.get(boundary_key, 10**9), structured.get(boundary_key, 10**9))
        boundary_records.append(
            {
                "case_id": case_id,
                "candidate_key": loss["candidate_key"],
                "gold_best_family_rank": None if gold_best == 10**9 else gold_best,
                "gold_support_count": loss["top_level_family_support_count"],
                "boundary_candidate_key": boundary_key,
                "boundary_best_family_rank": None if boundary_best == 10**9 else boundary_best,
                "boundary_support_count": int(boundary_key in raw) + int(boundary_key in structured),
                "lost_gold_with_better_best_rank_than_boundary": gold_best < boundary_best,
            }
        )
    summary = Counter(item["first_failure_stage"] for item in losses)
    family_bias_count = sum(
        summary[name]
        for name in (
            "raw_family_internal_demotion",
            "structured_family_internal_demotion",
            "top_level_consensus_displacement",
        )
    )
    bias_confirmed = family_bias_count >= 4 or diagnostic_gold >= 54
    if not identity_contract_consistent:
        decision, next_gate = (
            "boundary_loss_identity_contract_blocked",
            "stop_and_fix_strict_gold_identity_contract",
        )
    elif bias_confirmed:
        decision, next_gate = (
            "support_count_bias_confirmed",
            "support_count_invariant_candidate_fusion",
        )
    elif summary["slot_minimum_budget_displacement"] + summary["main_query_residual_displacement"] >= 5:
        decision, next_gate = "slot_boundary_bias_confirmed", "slot_candidate_boundary_repair"
    else:
        decision, next_gate = "candidate_supply_limited", "candidate_supply_recovery"
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {
        "boundary-loss-audit.json": {
            "requested_loss_count": 10,
            "auditable_same_identity_loss_count": len(losses),
            "summary": dict(summary),
            "records": losses,
        },
        "strict-identity-contract-audit.json": {
            "identity_contract_consistent": identity_contract_consistent,
            "labels_golden": {
                "identity_count": sum(map(len, strict_gold.values())),
                "unbounded_presence": f"{union_gold}/80",
                "bounded_top50": f"{bounded_gold}/80",
                "net_gap": union_gold - bounded_gold,
                "gross_loss": len(losses),
                "selector_synergy": labels_synergy,
            },
            "gate1_governance": {
                "identity_count": sum(len(item["strict_gold_identities"]) for item in governance.values()),
                "unbounded_presence": f"{governance_union}/80",
                "bounded_top50": f"{governance_bounded}/80",
                "net_gap": governance_union - governance_bounded,
                "gross_loss": governance_gross_loss,
                "selector_synergy": governance_synergy,
            },
            "identity_disagreement_count": len(identity_disagreements),
            "identity_disagreements": identity_disagreements,
            "reason": "60/80 and 50/80 were scored from different strict identity catalogs",
        },
        "support-count-bias.json": {
            "lost_gold_with_better_best_rank_than_boundary": sum(item["lost_gold_with_better_best_rank_than_boundary"] for item in boundary_records),
            "support_count_bias_confirmed": bias_confirmed,
            "records": boundary_records,
        },
        "best-family-rank-diagnostic-ceiling.json": {
            "diagnostic_only": True,
            "strict_gold_top50": f"{diagnostic_gold}/80",
        },
        "acceptance.json": {
            "gate": "pdf_retrieval_v4_gate_08_r8_r1_1",
            "decision": decision,
            "next_gate": next_gate,
            "losses_classified": f"{len(losses)}/{len(losses)}_within_labels_identity_contract",
            "requested_ten_loss_audit_completed": False,
            "identity_contract_consistent": identity_contract_consistent,
            "original_r8_r1_prediction_sha256": sha(PRED),
            "prediction_reruns": 0,
            "fusion_reruns": 0,
            "retrieval_runs": 0,
            "gold_reads_before_original_seal": 0,
            "production_switch_allowed": False,
        },
        "next-gate.json": {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False},
    }
    for name, payload in payloads.items():
        (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payloads["acceptance.json"], indent=2))
    print(json.dumps({"summary": dict(summary), "diagnostic_gold": diagnostic_gold}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
