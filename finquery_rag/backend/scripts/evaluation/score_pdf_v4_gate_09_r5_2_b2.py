#!/usr/bin/env python3
"""Post-seal formal scoring for Gate09 R5.2 B2 structural binding."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "artifacts/evaluation"
OUT = EVAL / "pdf-retrieval-v4-gate-09-r5-2-b2"
R0 = EVAL / "pdf-retrieval-v4-gate-09-r5-2-r0"
SE1 = EVAL / "pdf-retrieval-v4-gate-08-r8-se1"
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


def main() -> int:
    seal = json.loads((OUT / "prediction-seal.json").read_text(encoding="utf-8"))
    if (
        not seal.get("sealed")
        or seal["gold_reads_before_seal"] != 0
        or seal["strict_binding_reads_before_seal"] != 0
    ):
        raise RuntimeError("b2_prediction_seal_invalid")
    files = {
        "hydrated_classes": OUT / "semantic-classes-structural-b2.jsonl.gz",
        "joint_bindings_b2": OUT / "joint-bindings-b2.jsonl.gz",
        "projections_b2": OUT / "operand-projections-b2.jsonl.gz",
        "sets_b2": OUT / "evidence-sets-b2.jsonl.gz",
    }
    for name, path in files.items():
        if seal["output_sha256"][name] != sha256(path):
            raise RuntimeError(f"b2_prediction_mutation:{name}")

    bindings = {
        str(row["case_id"]): row for row in read_jsonl(files["joint_bindings_b2"])
    }
    projections = {
        str(row["case_id"]): row for row in read_jsonl(files["projections_b2"])
    }
    sets = {str(row["case_id"]): row for row in read_jsonl(files["sets_b2"])}
    targets = list(read_jsonl(SE1 / "gold-semantic-targets.jsonl"))
    strict_bindings = list(read_jsonl(STRICT))
    governance = {str(row["case_id"]): row for row in read_jsonl(GOVERNANCE)}
    plans_payload = json.loads(QUERY_PLAN.read_text(encoding="utf-8"))
    plans = {str(row["case_id"]): row["plan"] for row in plans_payload["plans"]}
    r0_slots = list(read_jsonl(R0 / "calculation-slot-audit.jsonl.gz"))
    if len(targets) != len(strict_bindings) or len(targets) != 80 or len(bindings) != 72:
        raise RuntimeError("b2_scoring_denominator_blocked")

    targets_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        targets_by_case[str(target["case_id"])].append(target)
    for values in targets_by_case.values():
        values.sort(key=lambda item: int(item["source_index"]))

    false_bindings: list[dict[str, Any]] = []
    for case_id, case_targets in sorted(targets_by_case.items()):
        selected = (bindings[case_id].get("selected_assignment") or {}).get(
            "equivalent_semantic_fact_ids"
        ) or []
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
                or selected_index >= len(selected)
            ):
                continue
            allowed = {str(value) for value in selected[selected_index]}
            if str(target["gold_semantic_fact_id"]) not in allowed:
                false_bindings.append(
                    {
                        "case_id": case_id,
                        "source_index": source_index,
                        "slot_id": source_slot["slot_id"],
                        "gold_semantic_fact_id": target["gold_semantic_fact_id"],
                        "selected_equivalent_semantic_fact_ids": sorted(allowed),
                        "failure": "false_slot_binding",
                    }
                )

    calculation_case_ids = sorted(
        case_id
        for case_id, record in governance.items()
        if record["query_type"] == "calculation_multi_operand"
    )
    calculation_status = Counter(
        projections[case_id]["binding_status"] for case_id in calculation_case_ids
    )
    ready_cases = sorted(
        case_id
        for case_id in calculation_case_ids
        if projections[case_id]["calculation_runtime_ready"]
    )
    slot_status = Counter()
    for row in r0_slots:
        case_status = projections[str(row["case_id"])]["binding_status"]
        if case_status in {"deterministic_ready", "deterministic_unit_blocked"}:
            slot_status["deterministic"] += 1
        elif row["first_failure"] == "true_metric_absence":
            slot_status["undercovered"] += 1
        elif row["first_failure"] == "multiple_operand_tuples":
            slot_status["ambiguous"] += 1
        else:
            slot_status["deterministic"] += 1

    false_audit = {
        "false_slot_binding": len(false_bindings),
        "records": false_bindings,
        "decision": "pass" if not false_bindings else "blocked",
        "unresolved_gold_targets_not_used_to_claim_correct_binding": True,
    }
    write_json(OUT / "false-binding-audit.json", false_audit)
    formal_metrics = {
        "calculation_runtime_ready": f"{len(ready_cases)}/11",
        "calculation_runtime_ambiguous": f"{calculation_status['runtime_operand_ambiguity']}/11",
        "calculation_undercovered": f"{calculation_status['undercovered']}/11",
        "calculation_unit_blocked": f"{calculation_status['deterministic_unit_blocked']}/11",
        "runtime_ready_case_ids": ready_cases,
        "required_slot_status_counts": dict(sorted(slot_status.items())),
        "false_slot_binding": len(false_bindings),
        "max_evidence_set_size": max(row["evidence_item_count"] for row in sets.values()),
    }
    write_json(OUT / "formal-metrics.json", formal_metrics)

    ready_count = len(ready_cases)
    ambiguous_count = calculation_status["runtime_operand_ambiguity"]
    if false_bindings:
        decision = "structural_joint_binder_regressed"
        next_gate = "blocked"
    elif ready_count >= 5:
        decision = "structural_joint_binding_strong_pass"
        next_gate = "deterministic_calculator_shadow"
    elif ready_count >= 3 and ambiguous_count < 6:
        decision = "structural_joint_binding_gain_confirmed"
        next_gate = "deterministic_calculator_shadow"
    else:
        decision = "structural_coherence_insufficient"
        next_gate = "remaining_operand_tuple_discriminator_audit"
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_09_r5_2_b2",
        "decision": decision,
        "next_gate": next_gate,
        **formal_metrics,
        "r5_1_runtime_ready": "2/11",
        "r5_1_runtime_ambiguous": "6/11",
        "candidate_mutation": 0,
        "semantic_registry_mutation": 0,
        "metric_contract_mutation": 0,
        "query_plan_mutation": 0,
        "unit_contract_mutation": 0,
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
