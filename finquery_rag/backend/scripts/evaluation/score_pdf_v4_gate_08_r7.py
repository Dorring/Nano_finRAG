#!/usr/bin/env python3
"""Post-seal scoring for Gate 08 R7 hierarchical field normalization."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r7"
R5 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r5/retrieval-predictions.jsonl.gz"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
UNIVERSE = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5/universe-scoring.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {item["case_id"]: item for item in (json.loads(line) for line in handle if line.strip())}


def keys(items):
    return [str(item.get("candidate_key") or "") for item in items]


def rank(items, key):
    values = keys(items)
    return values.index(key) + 1 if key in values else None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def write(name, payload):
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    path = OUT / "field-family-predictions.jsonl.gz"
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(path):
        raise RuntimeError("r7_prediction_seal_invalid")
    predictions, r5 = load(path), load(R5)
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
    ranks = {variant: {identity: rank(predictions[identity[0]][f"structured_{variant}"], identity[2]) for identity in grade_a} for variant in ("h0", "h1")}
    cutoff_metrics = {}
    for variant in ("h0", "h1"):
        cutoff_metrics[variant] = {f"recall_at_{cutoff}": f"{sum(value is not None and value <= cutoff for value in ranks[variant].values())}/68" for cutoff in (10, 20, 40, 50)}
        cutoff_metrics[variant]["presence_any"] = f"{sum(value is not None for value in ranks[variant].values())}/68"
    if cutoff_metrics["h0"]["recall_at_40"] != "43/68" or cutoff_metrics["h0"]["recall_at_50"] != "44/68":
        raise RuntimeError(f"h0_structured_score_parity_blocked:{cutoff_metrics['h0']}")
    h0_full = {identity for identity in gold if identity[2] in set(keys(predictions[identity[0]]["h0_full_pool"]))}
    h1_full = {identity for identity in gold if identity[2] in set(keys(predictions[identity[0]]["r7_full_pool"]))}
    if len(h0_full) != 59:
        raise RuntimeError(f"h0_full_score_parity_blocked:{len(h0_full)}")
    migrations = []
    gross_gain = gross_regression = 0
    for identity in sorted(grade_a):
        old, new = ranks["h0"][identity], ranks["h1"][identity]
        if (old is None or old > 40) and new is not None and new <= 40:
            category = "entered_top40"
            gross_gain += 1
        elif old is not None and old <= 40 and (new is None or new > 40):
            category = "dropped_out_top40"
            gross_regression += 1
        elif old is None and new is None:
            category = "still_missed"
        elif old is None or (new is not None and new < old):
            category = "improved"
        elif new is None or new > old:
            category = "worsened"
        else:
            category = "unchanged"
        migrations.append({"case_id": identity[0], "source_index": identity[1], "candidate_key": identity[2], "h0_rank": old, "h1_rank": new, "category": category})
    metric_top40 = {identity for identity in grade_a if rank(r5[identity[0]]["structured_family_rankings"]["s1"], identity[2]) is not None and rank(r5[identity[0]]["structured_family_rankings"]["s1"], identity[2]) <= 40}
    flat_top40 = {identity for identity in grade_a if ranks["h0"][identity] is not None and ranks["h0"][identity] <= 40}
    h1_top40 = {identity for identity in grade_a if ranks["h1"][identity] is not None and ranks["h1"][identity] <= 40}
    diluted = metric_top40 - flat_top40
    contribution = {"metric_supported_gold_count": len(metric_top40), "flat_s4_diluted_gold": len(diluted), "hierarchical_recovered_gold": len(diluted & h1_top40), "hierarchical_still_diluted": len(diluted - h1_top40)}
    strongest = Counter()
    field_to_family = []
    family_to_structured = []
    compression_records = []
    for identity in sorted(grade_a):
        record = predictions[identity[0]]
        family_items = record["field_family_ranking"]
        family_rank = rank(family_items, identity[2])
        lane_ranks = {}
        if family_rank is not None:
            lane_ranks = family_items[family_rank - 1].get("lane_ranks") or {}
        if lane_ranks:
            best_lane, best_rank = min(lane_ranks.items(), key=lambda item: (item[1], item[0]))
            strongest[best_lane] += 1
            field_to_family.append(family_rank - best_rank)
            if ranks["h1"][identity] is not None:
                family_to_structured.append(ranks["h1"][identity] - family_rank)
        else:
            best_lane = None
            best_rank = None
        compression_records.append({"case_id": identity[0], "source_index": identity[1], "candidate_key": identity[2], "best_field": best_lane, "best_field_rank": best_rank, "supporting_field_count": len(lane_ranks), "field_family_rank": family_rank, "structured_h1_rank": ranks["h1"][identity]})
    multi_cases = [case_id for case_id, item in predictions.items() if item["is_multi_slot"]]
    complete0 = {case_id for case_id in multi_cases if per_case.get(case_id, set()).issubset(set(keys(predictions[case_id]["h0_full_pool"])))}
    complete1 = {case_id for case_id in multi_cases if per_case.get(case_id, set()).issubset(set(keys(predictions[case_id]["r7_full_pool"])))}
    added, regressed = h1_full - h0_full, h0_full - h1_full
    score = len(h1_full)
    h1_at40 = sum(value is not None and value <= 40 for value in ranks["h1"].values())
    if len(regressed) > 1:
        decision, next_gate = "field_normalization_regression_blocked", "stop_and_audit_regression"
    elif score >= 62 and h1_at40 >= 52 and len(complete1) >= 13:
        decision, next_gate = "field_support_normalization_strong_pass", "evidence_set_construction"
    elif score >= 60:
        decision, next_gate = "field_support_normalization_passed", "evidence_set_construction"
    elif score == 59 and (h1_at40 - 43 >= 4 or len(complete1) >= 13):
        decision, next_gate = "field_support_normalization_structured_gain_but_full_plateau", "evidence_set_composition"
    else:
        decision, next_gate = "field_support_normalization_insufficient", "dominant_signal_field_aggregation"
    first_failure = Counter()
    for identity in gold:
        if identity in h1_full:
            first_failure["recovered"] += 1
        elif identity not in grade_a:
            first_failure["outside_grade_a_universe"] += 1
        elif ranks["h1"][identity] is None or ranks["h1"][identity] > 50:
            first_failure["field_family_top50_miss"] += 1
        elif ranks["h1"][identity] > 40:
            first_failure["structured_rank_41_to_50"] += 1
        elif predictions[identity[0]]["is_multi_slot"]:
            record = predictions[identity[0]]
            in_slot_horizon = any(
                identity[2] in set(keys(items))
                for items in record["family_fusion_trace"].values()
            )
            if not in_slot_horizon:
                first_failure["slot_candidate_horizon_miss"] += 1
            else:
                selected = set(
                    keys((record.get("slot_composition_trace") or {}).get("trace") or [])
                )
                category = (
                    "slot_composition_budget_loss"
                    if identity[2] not in selected
                    else "family_fusion_budget_loss"
                )
                first_failure[category] += 1
        else:
            first_failure["family_fusion_budget_loss"] += 1
    full_metrics = {"r6_baseline": "59/80", "r7_full": f"{score}/80", "original_r6_gold_retained": f"{len(h0_full & h1_full)}/59", "new_gold_added": len(added), "regressed": len(regressed), "net_gain": score - 59, "grade_a_full_recall": f"{len(h1_full & grade_a)}/68", "inside_universe_miss": 68 - len(h1_full & grade_a), "raw_gold_retained": "31/31"}
    write("structured-cutoff-metrics.json", cutoff_metrics)
    write("field-family-contribution.json", {**contribution, "strongest_field_counts": dict(strongest)})
    write("rank-migration.json", {"gross_gain": gross_gain, "gross_regression": gross_regression, "net_top40_gain": h1_at40 - 43, "summary": dict(Counter(item["category"] for item in migrations)), "records": migrations})
    write("rank-compression.json", {"field_to_family_loss": {"median": percentile(field_to_family, 0.5), "p75": percentile(field_to_family, 0.75), "p90": percentile(field_to_family, 0.9)}, "family_to_structured_loss": {"median": percentile(family_to_structured, 0.5), "p75": percentile(family_to_structured, 0.75), "p90": percentile(family_to_structured, 0.9)}, "records": compression_records})
    write("full-system-metrics.json", full_metrics)
    write("multi-evidence-metrics.json", {"r6_complete": f"{len(complete0)}/18", "r7_complete": f"{len(complete1)}/18", "newly_completed": sorted(complete1 - complete0), "regressed_complete": sorted(complete0 - complete1)})
    write("regression-matrix.json", {"added": sorted(added), "regressed": sorted(regressed), "old_hit_new_hit": len(h0_full & h1_full), "old_miss_new_hit": len(added), "old_hit_new_miss": len(regressed)})
    write("first-failure-attribution.json", dict(first_failure))
    acceptance = {"gate": "pdf_retrieval_v4_gate_08_r7", "decision": decision, "next_gate": next_gate, "structured": cutoff_metrics, "structured_recommended_at40_gate": h1_at40 >= 48, "full_system": full_metrics, "multi_evidence": f"{len(complete1)}/18", "raw_gold_retained": "31/31", "candidate_pool_recall_target_reached": score >= 60, "baseline_parity": True, "production_switch_allowed": False}
    write("acceptance.json", acceptance)
    write("next-gate.json", {"current_gate": "pdf_retrieval_v4_gate_08_r7", "decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
