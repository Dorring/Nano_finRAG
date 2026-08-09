#!/usr/bin/env python3
"""Post-seal strict 80-source-binding scoring for Gate 08 R8-R3."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r3"
PRED = OUT / "rerank-predictions.jsonl.gz"
P0 = BASE / "pdf-retrieval-v4-gate-08-r8-r3-p0/rerank-input-views.jsonl.gz"
TOP100 = BASE / "pdf-retrieval-v4-gate-08-r8-r2a-2/bounded-top100-predictions.jsonl.gz"
SIDECAR = BASE / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
GOV = BASE / "pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
DEPTHS = (1, 3, 5, 10, 20, 100)


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
        raise RuntimeError("rerank_prediction_seal_invalid")
    predictions, views, top100 = map(load_gzip, (PRED, P0, TOP100))
    bindings = [json.loads(line) for line in SIDECAR.open(encoding="utf-8")]
    governance = {item["case_id"]: item for item in map(json.loads, GOV.open(encoding="utf-8"))}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in bindings:
        by_case[binding["case_id"]].append(binding)
    for case_id, record in predictions.items():
        before = [item["candidate_key"] for item in top100[case_id]["candidates"]]
        after = [item["candidate_key"] for item in record["ranked_candidates"]]
        if len(after) != 100 or set(after) != set(before):
            raise RuntimeError(f"candidate_identity_mutation:{case_id}")

    def score(depth: int) -> tuple[set[str], int, int, int]:
        pools = {case_id: {item["candidate_key"] for item in record["ranked_candidates"][:depth]} for case_id, record in predictions.items()}
        hits = {item["binding_id"] for item in bindings if item["candidate_key"] in pools[item["case_id"]]}
        answerable = multi = calculation = 0
        for case_id, expected in by_case.items():
            any_hit = any(item["candidate_key"] in pools[case_id] for item in expected)
            complete = all(item["candidate_key"] in pools[case_id] for item in expected)
            answerable += any_hit
            multi += bool(complete and governance[case_id]["requires_multiple_sources"])
            calculation += bool(complete and governance[case_id]["query_type"] == "calculation_multi_operand")
        return hits, answerable, multi, calculation

    metrics = {}
    results = {}
    for depth in DEPTHS:
        hits, answerable, multi, calculation = score(depth)
        results[depth] = hits
        metrics[f"strict_source_binding_recall_at_{depth}"] = f"{len(hits)}/80"
        metrics[f"answerable_question_hit_at_{depth}"] = f"{answerable}/72"
        if depth in (5, 10, 100):
            metrics[f"multi_evidence_complete_at_{depth}"] = f"{multi}/16"
            metrics[f"calculation_complete_at_{depth}"] = f"{calculation}/11"
    if len(results[100]) != 68:
        raise RuntimeError("top100_recall_identity_parity_blocked")
    pre5 = {
        item["binding_id"] for item in bindings
        if item["candidate_key"] in {candidate["candidate_key"] for candidate in top100[item["case_id"]]["candidates"][:5]}
    }
    promoted = results[5] - pre5
    demoted = pre5 - results[5]
    metrics["candidate_to_top5_conversion"] = f"{len(results[5])}/68"
    metrics["candidate_to_top5_conversion_rate"] = round(len(results[5]) / 68, 6)
    metrics["candidate_to_top10_conversion"] = f"{len(results[10])}/68"
    metrics["promoted_gold"] = len(promoted)
    metrics["demoted_gold"] = len(demoted)

    view_by_key = {
        case_id: {item["candidate_key"]: item for item in record["candidates"]}
        for case_id, record in views.items()
    }
    migrations = []
    failures = Counter()
    for binding in bindings:
        case_id, key = binding["case_id"], binding["candidate_key"]
        ranked = predictions[case_id]["ranked_candidates"]
        item = next((candidate for candidate in ranked if candidate["candidate_key"] == key), None)
        if item is None:
            continue
        pre_rank, post_rank = item["pre_rerank_rank"], item["post_rerank_rank"]
        if post_rank <= 5:
            category = "recovered"
        elif item["truncated"]:
            category = "truncation_related"
        elif view_by_key[case_id][key]["authoritative_evidence_count"] == 0:
            category = "serialization_context_missing"
        elif governance[case_id]["query_type"] == "calculation_multi_operand":
            category = "calculation_operand_competition"
        elif governance[case_id]["requires_multiple_sources"]:
            category = "multi_slot_competition"
        else:
            category = "generic_financial_similarity"
        if binding["binding_id"] in results[100] and post_rank > 5:
            failures[category] += 1
        if pre_rank <= 10 and post_rank > 5:
            migration_bucket = "1_10_to_above_5"
        elif pre_rank <= 20 and post_rank > 5:
            migration_bucket = "11_20_to_above_5"
        elif pre_rank <= 50 and post_rank > 5:
            migration_bucket = "21_50_to_above_5"
        elif pre_rank > 50 and post_rank <= 5:
            migration_bucket = "51_100_to_top5"
        else:
            migration_bucket = "other"
        migrations.append({**binding, "pre_rerank_rank": pre_rank, "post_rerank_rank": post_rank, "promotion": binding["binding_id"] in promoted, "demotion": binding["binding_id"] in demoted, "migration_bucket": migration_bucket, "first_failure": category})
    multi5 = int(metrics["multi_evidence_complete_at_5"].split("/")[0])
    calc5 = int(metrics["calculation_complete_at_5"].split("/")[0])
    recall5 = len(results[5])
    if recall5 >= 60 and multi5 >= 8 and calc5 >= 7:
        decision, next_gate, reached = "structure_aware_cross_encoder_target_reached", "evidence_set_semantic_equivalence", True
    elif 56 <= recall5 <= 59:
        if failures["multi_slot_competition"] + failures["calculation_operand_competition"] >= failures["generic_financial_similarity"]:
            decision, next_gate = "structure_aware_cross_encoder_meaningful", "slot_aware_neural_composition"
        else:
            decision, next_gate = "structure_aware_cross_encoder_meaningful", "qwen3_reranker_4b_capacity_escalation"
        reached = False
    else:
        decision, next_gate, reached = "structure_aware_cross_encoder_insufficient", "rerank_failure_audit", False
    write("strict-recall.json", metrics)
    write("question-level-metrics.json", {key: value for key, value in metrics.items() if "question" in key or "multi" in key or "calculation" in key})
    write("gold-promotion-demotion.json", {"promoted": len(promoted), "demoted": len(demoted), "promoted_binding_ids": sorted(promoted), "demoted_binding_ids": sorted(demoted), "records": migrations})
    write("failure-attribution.json", {"counts": dict(failures), "records": [item for item in migrations if item["post_rerank_rank"] > 5]})
    acceptance = {"gate": "pdf_retrieval_v4_gate_08_r8_r3", "decision": decision, "next_gate": next_gate, "metrics": metrics, "candidate_mutation": 0, "gold_reads_before_seal": 0, "final_retrieval_target_reached": reached, "production_switch_allowed": False}
    write("acceptance.json", acceptance)
    write("next-gate.json", {"decision": decision, "next_gate": next_gate, "final_retrieval_target_reached": reached, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
