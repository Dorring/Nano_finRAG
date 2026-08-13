#!/usr/bin/env python3
"""NF-V2-03 R7: seal the final selective, fail-closed Binder architecture.

The script intentionally performs no provider calls.  It verifies the sealed
R6 policy, projects deterministic eligibility over the frozen 72 requests,
and records that no BOUND result can be released without a sealed semantic
selection plus the structural admission checks.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.evidence.shortlist_comparative_binder import build_shortlists  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r6_shortlist_comparative as r6  # noqa: E402


BASE_COMMIT = "fbc2335555904cf0ce929b94b929ae2fe3f0f95d"
GATE = "NF-V2-03-R7"
MODEL = "qwen3.7-plus"
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r7-selective-binder-freeze"
R6_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r6-shortlist-comparative"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_r6_policy() -> tuple[dict[str, Any], dict[str, Any]]:
    policy = read_json(R6_OUT / "selective-freeze-policy.json")
    seal = read_json(R6_OUT / "shortlist-seal.json")
    if policy.get("policy") != "selective_fail_closed_v1":
        raise RuntimeError("R6 selective policy is not the frozen fail-closed policy")
    if not seal.get("sealed") or int(seal.get("gold_reads_before_shortlist_seal", -1)) != 0:
        raise RuntimeError("R6 shortlist seal is not valid")
    if int(policy.get("released_bound_queries", -1)) != 0:
        raise RuntimeError("R6 policy unexpectedly contains released BOUND queries")
    return policy, seal


def request_rows(frozen: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_id, request in sorted(frozen["requests"].items()):
        shortlists, _, _ = build_shortlists(request, fact_view_version="v2", source_by_candidate=source_map)
        sizes = {slot_id: len(item.candidates) for slot_id, item in shortlists.items()}
        eligible = bool(sizes) and all(size == 1 for size in sizes.values())
        # R6 sealed no provider selection.  A unique shortlist is only
        # potential eligibility; it is never released as BOUND by itself.
        status = "MISSING" if any(size == 0 for size in sizes.values()) or eligible else "AMBIGUOUS"
        rows.append({
            "question_id": question_id,
            "intent": request.plan.intent.value,
            "operation": request.plan.operation,
            "required_slot_count": len(sizes),
            "shortlist_sizes": sizes,
            "eligible_for_bound": eligible,
            "selectively_bound": False,
            "status": status,
            "provider_calls": 0,
            "semantic_response_count": 0,
        })
    return rows


def counts(rows: list[dict[str, Any]], intent: str) -> dict[str, Any]:
    cohort = [row for row in rows if row["intent"] == intent]
    return {
        "total": len(cohort),
        "eligible_for_bound": sum(int(row["eligible_for_bound"]) for row in cohort),
        "selectively_bound": 0,
        "missing": sum(row["status"] == "MISSING" for row in cohort),
        "ambiguous": sum(row["status"] == "AMBIGUOUS" for row in cohort),
        "false_binding": 0,
        "strict_correct_bound": 0,
        "released_bound_denominator": 0,
    }


def calculation_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cohort = [row for row in rows if row["intent"] == "CALCULATION"]
    return {
        "total": len(cohort),
        "all_operands_selectively_admitted": 0,
        "partial_operand_admission": 0,
        "not_admitted": len(cohort),
        "eligible_for_bound": sum(int(row["eligible_for_bound"]) for row in cohort),
        "false_operand_binding": 0,
        "strict_correct_operand_slots": 0,
        "released_operand_denominator": 0,
    }


def multi_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cohort = [row for row in rows if row["intent"] == "MULTI_EVIDENCE"]
    return {
        "total": len(cohort),
        "complete_selective_binding": 0,
        "partial": 0,
        "not_admitted": len(cohort),
        "eligible_for_bound": sum(int(row["eligible_for_bound"]) for row in cohort),
        "false_binding": 0,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    policy, r6_seal = verify_r6_policy()
    frozen = r1d.load_r1c_frozen_inputs()
    source_map = r6.r1c.candidate_source_map(r6.nf02.verify_frozen_top100())
    rows = request_rows(frozen, source_map)
    direct = counts(rows, "DIRECT_FACT")
    calculation = calculation_counts(rows)
    multi = multi_counts(rows)
    status_by_intent = {
        intent: dict(Counter(row["status"] for row in rows if row["intent"] == intent))
        for intent in ("DIRECT_FACT", "CALCULATION", "MULTI_EVIDENCE")
    }
    policy_sha = sha256_file(R6_OUT / "selective-freeze-policy.json")
    r6_seal_sha = sha256_file(R6_OUT / "shortlist-seal.json")

    contract = {
        "contract": "SelectiveBindingAdmissionV1",
        "model": MODEL,
        "fact_view": "BinderFactViewV2",
        "fail_closed": True,
        "release_bound_requires": [
            "selected_slot_id_valid",
            "selected_fact_id_valid",
            "fact_belongs_to_query_packet",
            "provenance_complete",
            "structural_relation_valid",
            "no_cardinality_violation",
            "canonical_source_identity_valid",
            "unique_admissible_binder_selection",
            "binding_validator_pass",
        ],
        "failure_result": ["MISSING", "AMBIGUOUS"],
        "semantic_accuracy_claim": False,
        "false_binding_definition": "false binding under selective admission",
        "question_specific_rules": 0,
        "gold_conditioned_rules": 0,
        "model_calls": 0,
        "production_default": "V1",
        "production_switch_allowed": False,
    }
    replay = {
        "gate": GATE,
        "model_calls": 0,
        "gold_reads": 0,
        "r6_policy_sha256": policy_sha,
        "r6_shortlist_seal_sha256": r6_seal_sha,
        "r6_policy_released_bound_queries": policy["released_bound_queries"],
        "r6_shortlist_sealed": bool(r6_seal["sealed"]),
        "questions": len(rows),
        "rows": rows,
        "status_by_intent": status_by_intent,
        "direct": direct,
        "calculation": calculation,
        "multi_evidence": multi,
        "false_binding_under_selective_admission": 0,
        "semantic_accuracy_evaluated": False,
    }
    ablation = {
        "global": {"direct_visible_unique": "8/21", "calculation_operands": "1/12"},
        "slotwise": {"direct": "9/21", "calculation": "5/12", "indistinguishable_abstention": "0/6", "unbindable_false_binding": "6/7"},
        "pairwise": {"direct": "7/21", "calculation": "1/12", "indistinguishable_abstention": "2/6", "unbindable_false_binding": "6/7"},
        "fact_view_v2": {"direct_visible_uniqueness": "4/27 -> 21/27", "calculation_visible_uniqueness": "0/12 -> 12/12"},
        "qwen3_7_max_ablation": {"plus_direct": "8/21", "max_direct": "8/21", "plus_calculation": "1/12", "max_calculation": "0/12"},
        "shortlist_precheck": {"direct_retention": "19/21", "calculation_retention": "11/12"},
        "selective_admission": {"direct_selectively_bound": f"{direct['selectively_bound']}/{direct['total']}", "false_binding": 0},
    }
    v2_handoff = {
        "next_gate": "v2_04_missing_evidence_supply_repair",
        "binder_frozen": True,
        "direct": {"total": 56, "gold_source_admitted": "43/56", "gold_source_financial_fact": "33/56", "reviewed_strict_bindable": "27/56", "fact_view_v2_visible_unique": "21/27", "selectively_admitted": f"{direct['selectively_bound']}/56"},
        "calculation": {"total": 11, "fact_supply_complete": "6/11", "strict_bindable": "6/11", "selectively_complete": f"{calculation['all_operands_selectively_admitted']}/11"},
        "multi_evidence": {"total": 5, "strict_bindable": "0/5", "selectively_complete": f"{multi['complete_selective_binding']}/5"},
        "coverage_is_not_precision": True,
        "false_binding_under_selective_admission": 0,
    }
    decision = {
        "gate": GATE,
        "base_commit": BASE_COMMIT,
        "model_calls": 0,
        "binder_model": MODEL,
        "binder_fact_view": "BinderFactViewV2",
        "binder_admission": "SelectiveBindingAdmissionV1",
        "binder_safety_policy": "fail_closed",
        "binder_model_frozen": True,
        "binder_semantic_policy_frozen": True,
        "binder_fact_view_v2_frozen": True,
        "nf_v2_03_closed": True,
        "production_default": "V1",
        "production_switch_allowed": False,
        "false_binding_under_selective_admission": 0,
        "semantic_accuracy_claim": False,
        "dominant_conclusion": "selective_fail_closed_binding",
        "next_gate": "v2_04_missing_evidence_supply_repair",
    }
    write_json(OUT / "selective-admission-contract.json", contract)
    write_json(OUT / "selective-offline-replay.json", replay)
    write_json(OUT / "direct-selective-metrics.json", direct)
    write_json(OUT / "calculation-selective-metrics.json", calculation)
    write_json(OUT / "multi-evidence-selective-metrics.json", multi)
    write_json(OUT / "binder-formulation-ablation.json", ablation)
    write_json(OUT / "v2-03-final-metrics.json", {"direct": direct, "calculation": calculation, "multi_evidence": multi, "historical": {"direct_supply": "43/56", "gold_source_fact": "33/56", "reviewed_strict_bindable": "27/56"}, "false_binding_under_selective_admission": 0, "semantic_accuracy_claim": False})
    write_json(OUT / "v2-04-handoff-baseline.json", v2_handoff)
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {"gate": GATE, "summary": "NF-V2-03 is closed with SelectiveBindingAdmissionV1. No model calls were made; the sealed R6 shortlist policy released no BOUND queries, so offline replay reports zero coverage and zero false binding by construction.", "interpretation": ["BinderFactViewV2 improved source-derived distinguishability.", "The larger Qwen ablation did not improve Binder accuracy.", "Global, slot-wise, pairwise, and shortlist formulations did not justify unsafe coverage maximization.", "Missing evidence is delegated to V2-04 evidence repair."], "decision": decision, "artifacts": {"r6_policy_sha256": policy_sha, "r6_shortlist_seal_sha256": r6_seal_sha}})
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

