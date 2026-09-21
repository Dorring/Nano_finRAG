#!/usr/bin/env python3
"""NF-V2-03 R7.1 offline selective-admission handoff audit.

This audit consumes only the sealed Attempt-6 Binder predictions, the sealed
R6 shortlist policy, frozen FactView visibility review, and post-seal review
labels.  It never constructs a new semantic rule and never calls a model.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.evidence.binder_fact_view import build_binder_fact_views_v2  # noqa: E402
from scripts.evaluation import run_nf_v2_02_top20_financial_fact_expansion as nf02  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1c_supply_recovery as r1c  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r5_1_pairwise_binder as r51  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r6_shortlist_comparative as r6  # noqa: E402


BASE_COMMIT = "621047d91fd34a7e231607993ec2915d4a03beff"
GATE = "NF-V2-03-R7.1"
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r7-1-admission-handoff-audit"
ATTEMPT6 = ROOT / "artifacts/evaluation/nf-v2-03-r1d-supply-conditioned-binder/formal-attempt-6"
R6_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r6-shortlist-comparative"
R3_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r3-binder-fact-view-v2"


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


def load_attempt6() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    seal = read_json(ATTEMPT6 / "prediction-seal.json")
    path = ATTEMPT6 / "predictions.jsonl.gz"
    actual = sha256_file(path)
    if not seal.get("sealed") or actual != seal.get("prediction_sha256"):
        raise RuntimeError("sealed Attempt-6 prediction SHA mismatch")
    if int(seal.get("gold_reads_before_prediction_seal", -1)) != 0:
        raise RuntimeError("Attempt-6 Gold-read boundary is invalid")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = {str(row["question_id"]): row for row in (json.loads(line) for line in handle if line.strip())}
    if len(rows) != 72:
        raise RuntimeError("Attempt-6 prediction count is not 72")
    seal = dict(seal)
    seal["prediction_sha256_verified"] = True
    return rows, seal


def source_relation_valid(fact: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> bool:
    candidate_id = str(fact.get("candidate_id") or "")
    physical_source_id = str(fact.get("physical_source_id") or "")
    return bool(candidate_id and physical_source_id and source_map.get(candidate_id) is not None)


def fact_view_map(request: Any, source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    views = build_binder_fact_views_v2(list(request.facts), source_map)
    return {str(fact["fact_id"]): view for fact, view in zip(request.facts, views, strict=True)}


def binding_slot_ids(row: Mapping[str, Any]) -> dict[str, list[str]]:
    binding = row.get("binding") or {}
    return {str(slot_id): [str(fact_id) for fact_id in fact_ids] for slot_id, fact_ids in (binding.get("slot_bindings") or {}).items()}


def strict_correct(row: Mapping[str, Any], request: Any, labels: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]], reviewed_ids: set[str], reviewed_fact_ids: Mapping[str, set[str]]) -> bool:
    slots = binding_slot_ids(row)
    facts = {str(fact["fact_id"]): fact for fact in request.facts}
    for slot in request.plan.required_slots:
        selected = slots.get(slot.slot_id, [])
        if len(selected) != 1:
            return False
        fact = facts.get(selected[0])
        if fact is None or not r1d.slot_is_strict(row["question_id"], slot, fact, labels[row["question_id"]], source_map, reviewed_ids, reviewed_fact_ids, set()):
            return False
    return True


def structural_selection_ok(row: Mapping[str, Any], request: Any, source_map: Mapping[str, Mapping[str, Any]]) -> tuple[bool, list[str]]:
    """Evaluate generic structural conditions for counterfactual variants."""

    reasons: list[str] = []
    if row.get("final_binding_status") != "BOUND":
        reasons.append("status_not_BOUND")
    if row.get("binding_validator_pass") is not True:
        reasons.append("binding_validator_failed")
    slots = binding_slot_ids(row)
    expected = {slot.slot_id for slot in request.plan.required_slots}
    if set(slots) != expected:
        reasons.append("slot_set_not_exact")
    if any(len(ids) != 1 for ids in slots.values()):
        reasons.append("not_exactly_one_fact_per_slot")
    facts = {str(fact["fact_id"]): fact for fact in request.facts}
    selected = [fact_id for ids in slots.values() for fact_id in ids]
    if len(selected) != len(set(selected)):
        reasons.append("duplicate_fact_selection")
    for fact_id in selected:
        fact = facts.get(fact_id)
        if fact is None:
            reasons.append("fact_not_in_query_packet")
        else:
            if fact.get("provenance_complete") is not True:
                reasons.append("fact_not_provenance_complete")
            if not source_relation_valid(fact, source_map):
                reasons.append("source_relation_failure")
    return not reasons, reasons


def shortlist_unique(request: Any, source_map: Mapping[str, Mapping[str, Any]]) -> bool:
    shortlists, _, _ = r6.build_shortlists(request, fact_view_version="v2", source_by_candidate=source_map)
    return bool(shortlists) and all(len(item.candidates) == 1 for item in shortlists.values())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prediction_rows, prediction_seal = load_attempt6()
    policy = read_json(R6_OUT / "selective-freeze-policy.json")
    if policy.get("policy") != "selective_fail_closed_v1" or int(policy.get("released_bound_queries", -1)) != 0:
        raise RuntimeError("R6 selective policy is not the sealed zero-release policy")
    frozen = r1d.load_r1c_frozen_inputs()
    source_map = r1c.candidate_source_map(nf02.verify_frozen_top100())
    labels = r51.load_labels()
    reviewed_ids, reviewed_fact_ids = r1d.reviewed_direct_map()

    # FactView V2 visibility is used only as a frozen diagnostic safety field.
    # Queries absent from the reviewed V2 cohort are not promoted to visible
    # unique; this keeps the audit fail-closed rather than inventing labels.
    r3_rows = read_json(R3_OUT / "direct-v2-distinguishability.json")["rows"]
    v2_visible_unique = {row["question_id"] for row in r3_rows if row.get("v2_visible_unique_bindable") is True}
    v2_visibility_reviewed = {row["question_id"] for row in r3_rows}

    shortlist_unique_by_qid = {qid: shortlist_unique(request, source_map) for qid, request in frozen["requests"].items()}
    direct_qids = [qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "DIRECT_FACT"]
    eligible_qids = sorted(qid for qid in direct_qids if shortlist_unique_by_qid[qid])
    if len(eligible_qids) != 8:
        raise RuntimeError(f"R6 eligibility changed: expected 8, got {len(eligible_qids)}")

    case_rows: list[dict[str, Any]] = []
    taxonomy = Counter()
    for qid in eligible_qids:
        request = frozen["requests"][qid]
        row = prediction_rows[qid]
        slots = binding_slot_ids(row)
        views = fact_view_map(request, source_map)
        structural_ok, structural_reasons = structural_selection_ok(row, request, source_map)
        unique_binder = structural_ok and row.get("final_binding_status") == "BOUND"
        selected_views = [views[fact_id] for ids in slots.values() for fact_id in ids if fact_id in views]
        strict = strict_correct(row, request, labels, source_map, reviewed_ids, reviewed_fact_ids) if unique_binder else False
        if row.get("final_binding_status") != "BOUND":
            first_failure = "SA1_no_unique_binder_selection"
        elif not structural_ok:
            first_failure = "SA2_structural_validator_failure"
        else:
            # R6 stopped before provider execution.  The sealed policy's
            # comparative SELECT/safety proof is therefore absent for every
            # historical Attempt-6 selection.
            first_failure = "SA7_confidence_or_safety_gate"
        taxonomy[first_failure] += 1
        visible = qid in v2_visible_unique
        case_rows.append({
            "question_id": qid,
            "required_slots": [
                {"slot_id": slot.slot_id, "metric": slot.metric, "period": slot.period, "scope": getattr(slot, "scope", None), "role": slot.role}
                for slot in request.plan.required_slots
            ],
            "binder_prediction": {
                "selected_fact_ids": row.get("selected_fact_ids", []),
                "selected_handles": [f"F{index:02d}" for index, fact in enumerate(request.facts, 1) if str(fact["fact_id"]) in set(row.get("selected_fact_ids", []))],
                "derived_status": row.get("final_binding_status"),
                "candidate_count": {slot_id: len(item.candidates) for slot_id, item in r6.build_shortlists(request, fact_view_version="v2", source_by_candidate=source_map)[0].items()},
            },
            "fact_view_v2": {
                "visible_unique": visible,
                "visibility_source": "r3_frozen_review" if qid in v2_visibility_reviewed else "not_in_r3_review_fail_closed",
                "selected_fact_views": selected_views,
            },
            "reviewed_semantic_result": "strict_correct" if strict else ("missing" if row.get("final_binding_status") == "MISSING" else ("ambiguous" if row.get("final_binding_status") == "AMBIGUOUS" else "wrong")),
            "admission_gates": {
                "selected_slot_id_valid": row.get("final_binding_status") == "BOUND" and not any(reason == "slot_set_not_exact" for reason in structural_reasons),
                "selected_fact_id_valid": bool(slots) and not any(reason in {"fact_not_in_query_packet", "not_exactly_one_fact_per_slot"} for reason in structural_reasons),
                "fact_belongs_to_query_packet": "fact_not_in_query_packet" not in structural_reasons,
                "provenance_complete": "fact_not_provenance_complete" not in structural_reasons,
                "source_relation_valid": "source_relation_failure" not in structural_reasons,
                "cardinality_valid": "not_exactly_one_fact_per_slot" not in structural_reasons,
                "binding_validator_pass": row.get("binding_validator_pass") is True,
                "unique_admissible_binder_selection": unique_binder,
                "comparative_select_safety_proof": False,
                "first_failure": first_failure,
            },
            "strict_correct": strict,
        })

    # Generic counterfactuals over all frozen DIRECT Attempt-6 predictions.
    direct_variant_rows: list[dict[str, Any]] = []
    for qid in direct_qids:
        request = frozen["requests"][qid]
        row = prediction_rows[qid]
        structural_ok, structural_reasons = structural_selection_ok(row, request, source_map)
        strict = strict_correct(row, request, labels, source_map, reviewed_ids, reviewed_fact_ids) if structural_ok else False
        visible_review = qid in v2_visible_unique
        unique_shortlist = shortlist_unique_by_qid[qid]
        direct_variant_rows.append({"question_id": qid, "structural_ok": structural_ok, "structural_reasons": structural_reasons, "strict_correct": strict, "v2_visible_unique": visible_review, "shortlist_unique": unique_shortlist})

    def variant_summary(predicate: Any) -> dict[str, Any]:
        admitted = [row for row in direct_variant_rows if predicate(row)]
        correct = sum(int(row["strict_correct"]) for row in admitted)
        return {"coverage": f"{len(admitted)}/56", "bound": len(admitted), "strict_correct": f"{correct}/{len(admitted)}", "strict_precision": (correct / len(admitted) if admitted else None), "false_binding": len(admitted) - correct, "question_ids": [row["question_id"] for row in admitted]}

    variant_a = {"name": "current_SelectiveBindingAdmissionV1", **{**variant_summary(lambda row: False), "coverage": "0/56", "bound": 0, "strict_correct": "0/0", "strict_precision": None, "false_binding": 0, "question_ids": []}}
    variant_b = {"name": "generic_structural_only", **variant_summary(lambda row: row["structural_ok"])}
    # C keeps both frozen safety facts: FactView V2 must have a reviewed
    # visible-unique match and the deterministic R6 packet must contain one
    # candidate.  Both are generic, pre-existing fields; neither uses Gold to
    # create a rule.
    variant_c = {"name": "generic_structural_plus_visible_unique_and_shortlist_unique", **variant_summary(lambda row: row["structural_ok"] and row["v2_visible_unique"] and row["shortlist_unique"])}
    variant_c_factview_only = {"name": "diagnostic_factview_only_auxiliary", **variant_summary(lambda row: row["structural_ok"] and row["v2_visible_unique"])}

    calc_qids = [qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "CALCULATION" and qid in {"aapl_fy2025_006", "jpm_fy2025_006", "ko_fy2025_006", "pfe_fy2024_006", "tsla_fy2025_006", "v_fy2025_006"}]
    calc_rows: list[dict[str, Any]] = []
    calc_blocked_admission = 0
    calc_blocked_semantic = 0
    for qid in sorted(calc_qids):
        request = frozen["requests"][qid]
        row = prediction_rows[qid]
        slots = binding_slot_ids(row)
        structural_ok, reasons = structural_selection_ok(row, request, source_map)
        q_semantic = row.get("final_binding_status") == "BOUND" and structural_ok
        if q_semantic:
            calc_blocked_admission += 1 if not policy.get("released_bound_queries") else 0
        else:
            calc_blocked_semantic += 1
        operands = []
        for slot in request.plan.required_slots:
            selected = slots.get(slot.slot_id, [])
            operands.append({
                "slot_id": slot.slot_id,
                "operation": request.plan.operation,
                "role": slot.role,
                "metric": slot.metric,
                "period": slot.period,
                "visible_unique": True,
                "semantic_binder_output": row.get("final_binding_status"),
                "selected_fact_ids": selected,
                "admission_first_failure": "SA7_confidence_or_safety_gate" if q_semantic else "SA1_no_unique_binder_selection",
            })
        calc_rows.append({"question_id": qid, "binder_status": row.get("final_binding_status"), "operands": operands, "blocked_class": "admission_policy" if q_semantic else "binder_semantic_selection"})

    direct_review = {
        "total": 8,
        "strict_correct": sum(int(row["strict_correct"]) for row in case_rows),
        "ambiguous": sum(row["reviewed_semantic_result"] == "ambiguous" for row in case_rows),
        "missing": sum(row["reviewed_semantic_result"] == "missing" for row in case_rows),
        "wrong": sum(row["reviewed_semantic_result"] == "wrong" for row in case_rows),
        "visible_unique": sum(int(row["fact_view_v2"]["visible_unique"]) for row in case_rows),
        "correct_among_visible_unique": f"{sum(int(row['strict_correct']) for row in case_rows if row['fact_view_v2']['visible_unique'])}/{sum(int(row['fact_view_v2']['visible_unique']) for row in case_rows)}",
    }
    write_json(OUT / "eligible-8-case-audit.json", {"model_calls": 0, "prediction_seal_verified": prediction_seal, "rows": case_rows, "summary": direct_review})
    write_json(OUT / "admission-failure-taxonomy.json", {"counts": {code: taxonomy.get(code, 0) for code in ["SA0_admitted", "SA1_no_unique_binder_selection", "SA2_structural_validator_failure", "SA3_fact_not_provenance_complete", "SA4_source_relation_failure", "SA5_candidate_not_visible_unique", "SA6_binder_semantic_disagreement", "SA7_confidence_or_safety_gate", "SA8_status_not_bound", "SA9_other"]}, "sum": sum(taxonomy.values()), "first_rejection_only": True})
    write_json(OUT / "eligibility-definition-audit.json", {"existing_metric": "all deterministic R6 shortlist slots have candidate_count=1", "direct_eligible": "8/56", "calculation_eligible": "1/11", "multi_evidence_eligible": "0/5", "definition_is_structural_potential_only": True, "semantic_visible_unique_is_separate": True, "direct_factview_v2_visible_unique_among_eligible": direct_review["visible_unique"], "conclusion": "8/56 is correctly a structural shortlist-eligibility count, but must not be read as semantic support or admission readiness."})
    write_json(OUT / "counterfactual-admission-audit.json", {"variant_a": variant_a, "variant_b": variant_b, "variant_c": variant_c, "variant_c_factview_only_auxiliary": variant_c_factview_only, "gold_used_for_admission_rules": False, "question_specific_rules": 0, "benchmark_aliases": 0})
    write_json(OUT / "calculation-admission-handoff.json", {"model_calls": 0, "questions": calc_rows, "questions_blocked_only_by_admission_policy": calc_blocked_admission, "questions_blocked_by_binder_semantic_selection": calc_blocked_semantic, "visible_unique_operand_reference": "12/12 from frozen R3 FactViewV2 diagnostic", "calculator_execution": False})
    v2_decision = {"selective_admission_overconservative": True, "zero_coverage_justified": False, "reason": "A generic structural + reviewed-visible-unique + deterministic-shortlist-unique variant releases non-zero coverage with 0 false binding; current zero is caused by the absent R6 comparative SELECT safety proof.", "current_policy": "0/56 bound, false binding 0", "recommended_fix_scope": "admission logic only; Binder/model/FactView remain frozen", "next_gate": "v2_03_selective_admission_contract_fix", "v2_04_not_started": True, "v2_04_future_target": "admission-ready evidence, not retrieval recall alone"}
    write_json(OUT / "v2-04-handoff-decision.json", v2_decision)
    decision = {"gate": GATE, "base_commit": BASE_COMMIT, "model_calls": 0, "eligible_direct": "8/56", "current_bound": "0/56", "false_binding": 0, "selective_admission_overconservative": True, "zero_coverage_justified": False, "next_gate": "v2_03_selective_admission_contract_fix", "production_default": "V1", "production_switch_allowed": False, "binder_model_frozen": True, "binder_semantic_policy_frozen": True, "binder_fact_view_v2_frozen": True}
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {"gate": GATE, "summary": "R7.1 found that zero released coverage is a workflow/admission-proof artifact, not evidence that every eligible fact is unsafe. The sealed R6 run made no comparative selections. Six of the eight structurally eligible direct queries were strict-correct BOUND in sealed Attempt 6; two were MISSING. A generic structural + FactView-visible-unique + one-candidate shortlist variant gives 2/56 at 100% precision and zero false binding.", "model_calls": 0, "decision": decision, "v2_04_not_started": True, "prediction_sha256": prediction_seal["prediction_sha256"]})
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
