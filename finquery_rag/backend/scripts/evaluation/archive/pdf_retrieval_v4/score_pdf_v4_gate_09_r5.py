#!/usr/bin/env python3
"""Post-seal 80-binding scoring for Gate09 R5 semantic evidence sets."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "artifacts/evaluation"
OUT = EVAL / "pdf-retrieval-v4-gate-09-r5"
SE1 = EVAL / "pdf-retrieval-v4-gate-08-r8-se1"
SE1_P0 = EVAL / "pdf-retrieval-v4-gate-08-r8-se1-p0"
STRICT = EVAL / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
GOVERNANCE = EVAL / "pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def semantic_hit(
    target: dict[str, Any], candidate_keys: set[str], semantic_fact_ids: set[str]
) -> bool:
    if str(target["candidate_key"]) in candidate_keys:
        return True
    return bool(
        target.get("semantic_target_status") == "resolved"
        and target.get("gold_semantic_fact_id") in semantic_fact_ids
    )


def facts_for_candidates(candidate_keys: set[str], registry: dict[str, dict[str, Any]]) -> set[str]:
    return {
        str(fact_id)
        for key in candidate_keys
        for fact_id in (registry.get(key) or {}).get("semantic_fact_ids") or []
    }


def main() -> int:
    seal = json.loads((OUT / "prediction-seal.json").read_text(encoding="utf-8"))
    if not seal.get("sealed") or seal["gold_reads_before_seal"] != 0 or seal["strict_binding_reads_before_seal"] != 0:
        raise RuntimeError("gate09_r5_prediction_seal_invalid")
    files = {
        "access": OUT / "evidence-access-universe.jsonl.gz",
        "classes": OUT / "semantic-evidence-classes.jsonl.gz",
        "matches": OUT / "slot-semantic-matches.jsonl.gz",
        "projections": OUT / "operand-projections.jsonl.gz",
        "sets": OUT / "evidence-set-predictions.jsonl.gz",
    }
    for name, path in files.items():
        if seal["output_sha256"][name] != sha256(path):
            raise RuntimeError(f"gate09_r5_sealed_output_mutation:{name}")

    access = {str(row["case_id"]): row for row in read_jsonl(files["access"])}
    matches = {str(row["case_id"]): row for row in read_jsonl(files["matches"])}
    projections = {str(row["case_id"]): row for row in read_jsonl(files["projections"])}
    sets = {str(row["case_id"]): row for row in read_jsonl(files["sets"])}
    registry = {str(row["candidate_key"]): row for row in read_jsonl(SE1_P0 / "candidate-semantic-fact-registry.jsonl.gz")}
    targets = list(read_jsonl(SE1 / "gold-semantic-targets.jsonl"))
    strict_bindings = list(read_jsonl(STRICT))
    governance = {str(row["case_id"]): row for row in read_jsonl(GOVERNANCE)}
    if len(targets) != len(strict_bindings) or len(targets) != 80 or len(sets) != 72:
        raise RuntimeError("gate09_r5_scoring_denominator_blocked")

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        by_case[str(target["case_id"])].append(target)

    main_hits: set[str] = set()
    augmented_hits: set[str] = set()
    final_hits: set[str] = set()
    case_records: list[dict[str, Any]] = []
    multi_access = multi_final = calculation_access = calculation_final = 0
    false_bindings: list[dict[str, Any]] = []
    for case_id, case_targets in sorted(by_case.items()):
        access_record = access[case_id]
        main_keys = {
            str(item["candidate_key"])
            for item in access_record["candidates"]
            if item.get("main_rank") is not None and int(item["main_rank"]) <= 10
        }
        augmented_keys = {str(item["candidate_key"]) for item in access_record["candidates"]}
        selected_keys = {str(key) for key in sets[case_id]["selected_candidate_keys"]}
        main_facts = facts_for_candidates(main_keys, registry)
        augmented_facts = facts_for_candidates(augmented_keys, registry)
        selected_facts = facts_for_candidates(selected_keys, registry)
        for target in case_targets:
            binding_id = str(target["binding_id"])
            if semantic_hit(target, main_keys, main_facts):
                main_hits.add(binding_id)
            if semantic_hit(target, augmented_keys, augmented_facts):
                augmented_hits.add(binding_id)
            if semantic_hit(target, selected_keys, selected_facts):
                final_hits.add(binding_id)
        required_ids = {str(target["binding_id"]) for target in case_targets}
        access_complete = required_ids <= augmented_hits
        final_complete = required_ids <= final_hits
        is_multi = bool(governance[case_id]["requires_multiple_sources"])
        is_calculation = governance[case_id]["query_type"] == "calculation_multi_operand"
        multi_access += bool(is_multi and access_complete)
        multi_final += bool(is_multi and final_complete)
        calculation_access += bool(is_calculation and access_complete)
        calculation_final += bool(is_calculation and final_complete)
        case_records.append(
            {
                "case_id": case_id,
                "is_multi_evidence": is_multi,
                "is_calculation": is_calculation,
                "main_only_semantic_complete": required_ids <= main_hits,
                "slot_augmented_semantic_complete": access_complete,
                "evidence_set_semantic_complete": final_complete,
                "selected_candidate_keys": sorted(selected_keys),
                "selected_semantic_fact_ids": sorted(selected_facts),
                "calculation_runtime_ready": projections[case_id]["calculation_runtime_ready"],
            }
        )

        slots = governance[case_id].get("operand_slots") or []
        match_by_slot = {str(item["slot_id"]): item for item in matches[case_id]["slot_matches"]}
        for source_index, target in enumerate(case_targets):
            if target.get("semantic_target_status") != "resolved" or source_index >= len(slots):
                continue
            slot_id = str(slots[source_index]["slot_id"])
            slot_match = match_by_slot.get(slot_id)
            if not slot_match or slot_match["slot_status"] != "deterministic":
                continue
            selected_fact_id = slot_match["compatible_semantic_fact_ids"][0]
            if selected_fact_id != target["gold_semantic_fact_id"]:
                false_bindings.append(
                    {
                        "case_id": case_id,
                        "source_index": source_index,
                        "slot_id": slot_id,
                        "gold_semantic_fact_id": target["gold_semantic_fact_id"],
                        "selected_semantic_fact_id": selected_fact_id,
                        "failure": "false_slot_binding",
                    }
                )

    runtime_ready_cases = sorted(
        case_id
        for case_id, projection in projections.items()
        if governance[case_id]["query_type"] == "calculation_multi_operand" and projection["calculation_runtime_ready"]
    )
    conversion = len(final_hits) / len(augmented_hits) if augmented_hits else 0.0
    retention = {
        "main_only_semantic_access": f"{len(main_hits)}/80",
        "slot_augmented_semantic_access": f"{len(augmented_hits)}/80",
        "semantic_evidence_set_recall": f"{len(final_hits)}/80",
        "top10_to_evidence_set_retention": f"{len(final_hits)}/{len(augmented_hits)}",
        "top10_to_evidence_set_conversion": round(conversion, 6),
        "main_to_slot_augmented_gain": len(augmented_hits) - len(main_hits),
    }
    multi_metrics = {
        "main_input_complete_at_10": "9/16",
        "slot_augmented_complete": f"{multi_access}/16",
        "evidence_set_complete": f"{multi_final}/16",
    }
    calculation = {
        "main_input_semantic_complete_at_10": "8/11",
        "slot_augmented_semantic_complete": f"{calculation_access}/11",
        "evidence_set_semantic_complete": f"{calculation_final}/11",
        "calculation_runtime_ready": f"{len(runtime_ready_cases)}/11",
        "runtime_ready_case_ids": runtime_ready_cases,
        "runtime_operand_ambiguity_count": sum(
            item["slot_status"] == "runtime_operand_ambiguity"
            for record in matches.values()
            for item in record["slot_matches"]
        ),
    }
    false_binding_audit = {
        "false_slot_binding": len(false_bindings),
        "records": false_bindings,
        "decision": "pass" if not false_bindings else "blocked",
        "unresolved_gold_targets_not_used_to_claim_false_binding_zero": True,
    }
    write_json(OUT / "semantic-retention.json", retention)
    write_json(OUT / "multi-evidence-metrics.json", multi_metrics)
    write_json(OUT / "calculation-readiness.json", calculation)
    write_json(OUT / "false-binding-audit.json", false_binding_audit)
    write_json(OUT / "postseal-case-metrics.json", {"cases": case_records})

    max_set_size = max(record["evidence_item_count"] for record in sets.values())
    if (
        len(final_hits) >= 58
        and conversion >= 0.90
        and multi_final >= 9
        and len(runtime_ready_cases) >= 9
        and not false_bindings
        and max_set_size <= 5
    ):
        decision = "top10_semantic_evidence_set_strong_pass"
        next_gate = "equivalence_aware_deterministic_calculator"
    elif (
        len(final_hits) >= 55
        and conversion >= 0.90
        and multi_final >= 8
        and calculation_final >= 8
        and len(runtime_ready_cases) >= 8
        and not false_bindings
        and max_set_size <= 5
    ):
        decision = "top10_semantic_evidence_set_passed"
        next_gate = "equivalence_aware_deterministic_calculator"
    else:
        decision = "top10_semantic_evidence_set_insufficient"
        next_gate = "deterministic_operand_binding_contract_repair"
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_09_r5",
        "decision": decision,
        "next_gate": next_gate,
        **retention,
        "multi_semantic_complete": f"{multi_final}/16",
        "calculation_semantic_complete": f"{calculation_final}/11",
        "calculation_runtime_ready": f"{len(runtime_ready_cases)}/11",
        "false_slot_binding": len(false_bindings),
        "runtime_operand_ambiguity_explicit": True,
        "max_evidence_set_size": max_set_size,
        "candidate_mutation": 0,
        "semantic_registry_mutation": 0,
        "retrieval_optimization_stopped": True,
        "production_switch_allowed": False,
    }
    write_json(OUT / "acceptance.json", acceptance)
    write_json(OUT / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
