#!/usr/bin/env python3
"""Post-seal scoring for Gate09 R5.1 operand binding contract V2."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "artifacts/evaluation"
OUT = EVAL / "pdf-retrieval-v4-gate-09-r5-1"
R5 = EVAL / "pdf-retrieval-v4-gate-09-r5"
SE1 = EVAL / "pdf-retrieval-v4-gate-08-r8-se1"
SE1_P0 = EVAL / "pdf-retrieval-v4-gate-08-r8-se1-p0"
STRICT = EVAL / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
GOVERNANCE = EVAL / "pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
QUERY_PLAN = EVAL / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"


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
    return str(target["candidate_key"]) in candidate_keys or bool(
        target.get("semantic_target_status") == "resolved"
        and target.get("gold_semantic_fact_id") in semantic_fact_ids
    )


def facts_for_candidates(
    candidate_keys: set[str], registry: dict[str, dict[str, Any]]
) -> set[str]:
    return {
        str(fact_id)
        for key in candidate_keys
        for fact_id in (registry.get(key) or {}).get("semantic_fact_ids") or []
    }


def main() -> int:
    seal = json.loads((OUT / "prediction-seal.json").read_text(encoding="utf-8"))
    if (
        not seal.get("sealed")
        or seal["gold_reads_before_seal"] != 0
        or seal["strict_binding_reads_before_seal"] != 0
    ):
        raise RuntimeError("gate09_r5_1_prediction_seal_invalid")
    files = {
        "classes_v2": OUT / "semantic-evidence-classes-v2.jsonl.gz",
        "metric_bindings": OUT / "metric-binding-candidates.jsonl.gz",
        "joint_bindings": OUT / "joint-operand-bindings.jsonl.gz",
        "projections_v2": OUT / "operand-projections-v2.jsonl.gz",
        "sets_v2": OUT / "evidence-set-predictions-v2.jsonl.gz",
    }
    for name, path in files.items():
        if seal["output_sha256"][name] != sha256(path):
            raise RuntimeError(f"gate09_r5_1_sealed_output_mutation:{name}")

    access = {
        str(row["case_id"]): row
        for row in read_jsonl(R5 / "evidence-access-universe.jsonl.gz")
    }
    projections = {
        str(row["case_id"]): row for row in read_jsonl(files["projections_v2"])
    }
    bindings = {
        str(row["case_id"]): row for row in read_jsonl(files["joint_bindings"])
    }
    sets = {str(row["case_id"]): row for row in read_jsonl(files["sets_v2"])}
    registry = {
        str(row["candidate_key"]): row
        for row in read_jsonl(SE1_P0 / "candidate-semantic-fact-registry.jsonl.gz")
    }
    targets = list(read_jsonl(SE1 / "gold-semantic-targets.jsonl"))
    strict_bindings = list(read_jsonl(STRICT))
    governance = {str(row["case_id"]): row for row in read_jsonl(GOVERNANCE)}
    plan_payload = json.loads(QUERY_PLAN.read_text(encoding="utf-8"))
    plans = {str(row["case_id"]): row["plan"] for row in plan_payload["plans"]}
    if len(targets) != len(strict_bindings) or len(targets) != 80 or len(sets) != 72:
        raise RuntimeError("gate09_r5_1_scoring_denominator_blocked")

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        by_case[str(target["case_id"])].append(target)
    for case_targets in by_case.values():
        case_targets.sort(key=lambda item: int(item["source_index"]))

    augmented_hits: set[str] = set()
    final_hits: set[str] = set()
    false_bindings: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    multi_access = multi_final = calculation_access = calculation_final = 0
    for case_id, case_targets in sorted(by_case.items()):
        augmented_keys = {
            str(item["candidate_key"]) for item in access[case_id]["candidates"]
        }
        selected_keys = {str(key) for key in sets[case_id]["selected_candidate_keys"]}
        augmented_facts = facts_for_candidates(augmented_keys, registry)
        selected_facts = {str(value) for value in sets[case_id]["selected_semantic_fact_ids"]}
        for target in case_targets:
            binding_id = str(target["binding_id"])
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
                "binding_status": projections[case_id]["binding_status"],
                "slot_augmented_semantic_complete": access_complete,
                "evidence_set_semantic_complete": final_complete,
                "calculation_runtime_ready": projections[case_id][
                    "calculation_runtime_ready"
                ],
            }
        )

        selected_assignment = bindings[case_id].get("selected_assignment") or {}
        equivalent_ids = selected_assignment.get("equivalent_semantic_fact_ids") or []
        source_slots = governance[case_id].get("operand_slots") or []
        plan_slots = plans[case_id].get("operand_slots") or []
        plan_slot_index = {
            str(slot["slot_id"]): index for index, slot in enumerate(plan_slots)
        }
        for source_index, target in enumerate(case_targets):
            source_slot = source_slots[source_index] if source_index < len(source_slots) else None
            selected_index = plan_slot_index.get(str((source_slot or {}).get("slot_id")))
            if (
                target.get("semantic_target_status") != "resolved"
                or source_slot is None
                or selected_index is None
                or selected_index >= len(equivalent_ids)
            ):
                continue
            allowed_ids = {str(value) for value in equivalent_ids[selected_index]}
            if str(target["gold_semantic_fact_id"]) not in allowed_ids:
                false_bindings.append(
                    {
                        "case_id": case_id,
                        "source_index": source_index,
                        "slot_id": source_slot["slot_id"],
                        "gold_semantic_fact_id": target["gold_semantic_fact_id"],
                        "selected_equivalent_semantic_fact_ids": sorted(allowed_ids),
                        "failure": "false_slot_binding",
                    }
                )

    ready_cases = sorted(
        case_id
        for case_id, projection in projections.items()
        if governance[case_id]["query_type"] == "calculation_multi_operand"
        and projection["calculation_runtime_ready"]
    )
    status_counts = Counter(item["binding_status"] for item in projections.values())
    max_set_size = max(item["evidence_item_count"] for item in sets.values())
    false_binding_audit = {
        "false_slot_binding": len(false_bindings),
        "records": false_bindings,
        "decision": "pass" if not false_bindings else "blocked",
        "unresolved_gold_targets_not_used_to_claim_correct_binding": True,
    }
    retention = {
        "slot_augmented_semantic_access": f"{len(augmented_hits)}/80",
        "semantic_evidence_set_recall": f"{len(final_hits)}/80",
        "top10_to_evidence_set_retention": f"{len(final_hits)}/{len(augmented_hits)}",
    }
    multi_metrics = {
        "slot_augmented_complete": f"{multi_access}/16",
        "evidence_set_complete": f"{multi_final}/16",
    }
    calculation = {
        "slot_augmented_semantic_complete": f"{calculation_access}/11",
        "evidence_set_semantic_complete": f"{calculation_final}/11",
        "calculation_runtime_ready": f"{len(ready_cases)}/11",
        "runtime_ready_case_ids": ready_cases,
        "binding_status_counts": dict(sorted(status_counts.items())),
    }
    ambiguity_audit = {
        "binding_status_counts": dict(sorted(status_counts.items())),
        "runtime_operand_ambiguity_case_ids": sorted(
            case_id
            for case_id, projection in projections.items()
            if projection["binding_status"] == "runtime_operand_ambiguity"
        ),
        "deterministic_unit_blocked_case_ids": sorted(
            case_id
            for case_id, projection in projections.items()
            if projection["binding_status"] == "deterministic_unit_blocked"
        ),
        "undercovered_case_ids": sorted(
            case_id
            for case_id, projection in projections.items()
            if projection["binding_status"] == "undercovered"
        ),
        "rank_used_to_resolve_ambiguity": False,
    }
    write_json(OUT / "semantic-retention-v2.json", retention)
    write_json(OUT / "multi-evidence-metrics-v2.json", multi_metrics)
    write_json(OUT / "calculation-readiness-v2.json", calculation)
    write_json(OUT / "ambiguity-audit.json", ambiguity_audit)
    write_json(OUT / "false-binding-audit.json", false_binding_audit)
    write_json(OUT / "postseal-case-metrics.json", {"cases": case_records})

    if not false_bindings and len(ready_cases) >= 8:
        decision = "deterministic_operand_binding_strong_pass"
        next_gate = "deterministic_calculator_execution"
    elif not false_bindings and len(ready_cases) >= 7:
        decision = "deterministic_operand_binding_passed"
        next_gate = "deterministic_calculator_execution"
    elif not false_bindings and len(ready_cases) >= 5:
        decision = "deterministic_operand_binding_meaningful"
        next_gate = "unit_context_resolver_repair"
    else:
        decision = "deterministic_operand_binding_insufficient"
        if status_counts["deterministic_unit_blocked"]:
            next_gate = "unit_context_resolver_repair"
        elif status_counts["runtime_operand_ambiguity"]:
            next_gate = "joint_operand_ambiguity_contract_repair"
        else:
            next_gate = "operand_undercoverage_contract_audit"
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_09_r5_1",
        "decision": decision,
        "next_gate": next_gate,
        **retention,
        "multi_semantic_complete": f"{multi_final}/16",
        "calculation_semantic_complete": f"{calculation_final}/11",
        "calculation_runtime_ready": f"{len(ready_cases)}/11",
        "false_slot_binding": len(false_bindings),
        "binding_status_counts": dict(sorted(status_counts.items())),
        "max_evidence_set_size": max_set_size,
        "candidate_mutation": 0,
        "semantic_registry_mutation": 0,
        "retrieval_runs": 0,
        "production_switch_allowed": False,
    }
    write_json(OUT / "acceptance.json", acceptance)
    write_json(
        OUT / "next-gate.json",
        {
            "decision": decision,
            "next_gate": next_gate,
            "production_switch_allowed": False,
        },
    )
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
