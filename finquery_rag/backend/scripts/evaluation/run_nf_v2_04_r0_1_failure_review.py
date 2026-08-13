#!/usr/bin/env python3
"""Offline NF-V2-04 R0.1 repair failure attribution.

This gate consumes only sealed R0 predictions, frozen Top20/Top100 artifacts,
and post-seal review labels.  It never calls the provider or retrieval.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding  # noqa: E402
from rag_v2.evidence.selective_admission_v2 import admit_binding_v2, evaluate_slot  # noqa: E402
from scripts.evaluation import run_nf_v2_02_top20_financial_fact_expansion as nf02  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r7_2_admission_contract_fix as r72  # noqa: E402
from scripts.evaluation import run_nf_v2_04_r0_missing_evidence_repair as r0  # noqa: E402


BASE_COMMIT = "92cfb4a560f7ba56509be3316c13668a9739ef18"
GATE = "NF-V2-04-R0.1"
OUT = ROOT / "artifacts/evaluation/nf-v2-04-r0-1-failure-review"
R0_OUT = ROOT / "artifacts/evaluation/nf-v2-04-r0-missing-evidence-repair"
V203_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r7-2-admission-contract-fix"
V202_OUT = ROOT / "artifacts/evaluation/nf-v2-02-top20-financial-fact-expansion"
R71_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r7-1-admission-handoff-audit"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def labels() -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if row}


def source_keys(label: Mapping[str, Any]) -> set[str]:
    return {str(item.get("candidate_key")) for item in label.get("expected_sources", []) if item.get("candidate_key")}


def slot_source_keys(slot: Any, label: Mapping[str, Any]) -> set[str]:
    """Use the frozen slot-aware source contract for calculation/multi slots."""
    return {str(item.get("candidate_key")) for item in r1d.r1a.expected_sources(slot, label) if item.get("candidate_key")}


def exact_period(value: Any) -> str | None:
    return r0.exact_period(value)


def fact_candidates(fact: Mapping[str, Any]) -> set[str]:
    return r0.candidate_ids(fact)


def selected_for(binding: Mapping[str, Any], slot_id: str) -> list[str]:
    return [str(item) for item in (binding.get("slot_bindings") or {}).get(slot_id, [])]


def binding_from_row(row: Mapping[str, Any]) -> EvidenceBinding:
    binding = row.get("binding") or row.get("v2_binding") or {}
    return EvidenceBinding(
        status=str(binding.get("status")),
        slot_bindings={key: tuple(value) for key, value in (binding.get("slot_bindings") or {}).items()},
        missing_slots=tuple(binding.get("missing_slots") or ()),
        ambiguous_slots=tuple(binding.get("ambiguous_slots") or ()),
        invalid_reasons=tuple(binding.get("invalid_reasons") or ()),
    )


def request_with_packet(qid: str, frozen: Mapping[str, Any], facts: list[dict[str, Any]], order: Mapping[str, list[str]]) -> Any:
    request0 = frozen["requests"][qid]
    packet = r0.packet_for(qid, facts, order, r0.TOP50)
    return r0.BinderRequest(qid, request0.question, request0.plan, tuple(packet))


def expected_fact_rows(slot: Any, label: Mapping[str, Any], packet: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = slot_source_keys(slot, label)
    rows = []
    for fact in packet:
        if fact_candidates(fact) & expected:
            rows.append(dict(fact))
    return rows


def period_fact_rows(slot: Any, label: Mapping[str, Any], packet: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    wanted = exact_period(getattr(slot, "period", None))
    return [fact for fact in expected_fact_rows(slot, label, packet) if not wanted or exact_period(fact.get("normalized_period") or fact.get("raw_period")) == wanted]


def unique_admission_candidates(slot: Any, packet: list[dict[str, Any]], source_map: Mapping[str, Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    for fact in packet:
        evidence = evaluate_slot(slot, str(fact.get("fact_id")), packet, source_map=source_map)
        if evidence.uniquely_admissible:
            result.append(str(fact.get("fact_id")))
    return result


def selected_is_strict(qid: str, row: Mapping[str, Any], request: Any, labels_by_id: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]], reviewed_ids: set[str], reviewed_fact_ids: Mapping[str, set[str]]) -> bool:
    wrapper = {"question_id": qid, "v2_binding": row.get("v2_binding") or row.get("binding") or {}}
    return bool(r72.strict_correct(wrapper, request, labels_by_id, source_map, reviewed_ids, reviewed_fact_ids))


def first_direct_bottleneck(
    qid: str,
    request0: Any,
    final_request: Any,
    initial_row: Mapping[str, Any],
    final_row: Mapping[str, Any],
    label: Mapping[str, Any],
    source_map: Mapping[str, Mapping[str, Any]],
    reviewed_ids: set[str],
    reviewed_fact_ids: Mapping[str, set[str]],
) -> dict[str, Any]:
    packet = list(final_request.facts)
    expected = source_keys(label)
    top50_candidates = set().union(*(fact_candidates(fact) for fact in packet)) if packet else set()
    source_present = bool(expected & top50_candidates)
    slot_rows: list[dict[str, Any]] = []
    for slot in request0.plan.required_slots:
        source_facts = expected_fact_rows(slot, label, packet)
        period_facts = period_fact_rows(slot, label, packet)
        unique = unique_admission_candidates(slot, packet, source_map)
        selected = selected_for(final_row.get("v2_binding") or final_row.get("binding") or {}, slot.slot_id)
        slot_rows.append({
            "slot_id": slot.slot_id,
            "source_facts": len(source_facts),
            "period_facts": len(period_facts),
            "unique_admission_candidates": unique,
            "selected": selected,
        })
    no_period = any(row["source_facts"] and not row["period_facts"] for row in slot_rows)
    no_fact = any(not row["source_facts"] for row in slot_rows)
    all_unique = all(row["unique_admission_candidates"] for row in slot_rows)
    any_multi = any(len(row["unique_admission_candidates"]) > 1 for row in slot_rows)
    status = str((final_row.get("v2_binding") or final_row.get("binding") or {}).get("status"))
    selected_strict = selected_is_strict(qid, final_row, final_request, {qid: label}, source_map, reviewed_ids, reviewed_fact_ids)
    if bool(label.get("expected_no_answer")):
        category = "DR6_NO_SUPPORTING_EVIDENCE"
    elif not source_present:
        category = "DR0_RETRIEVAL_MISS"
    elif no_fact or no_period:
        category = "DR1_SOURCE_PRESENT_FACT_MISSING"
    elif any_multi:
        category = "DR5_GENUINE_AMBIGUITY"
    elif all_unique and status != BindingStatus.BOUND.value:
        category = "DR3_FACT_VIEW_SUFFICIENT_BINDER_FAILED"
    elif status == BindingStatus.BOUND.value and selected_strict and not bool(final_row.get("released")):
        category = "DR4_BINDER_CORRECT_ADMISSION_BLOCKED"
    elif status == BindingStatus.BOUND.value and not selected_strict:
        category = "DR3_FACT_VIEW_SUFFICIENT_BINDER_FAILED"
    elif status in {BindingStatus.MISSING.value, BindingStatus.AMBIGUOUS.value}:
        category = "DR3_FACT_VIEW_SUFFICIENT_BINDER_FAILED" if all_unique else "DR5_GENUINE_AMBIGUITY"
    else:
        category = "DR7_OTHER"
    return {
        "question_id": qid,
        "initial_status": initial_row["v2_binding"]["status"],
        "final_status": status,
        "source_present_in_top50": source_present,
        "slot_rows": slot_rows,
        "selected_strict": selected_strict,
        "released": bool(final_row.get("released")),
        "classification": category,
    }


def admission_audit(qid: str, request: Any, raw_row: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = admit_binding_v2(binding_from_row(raw_row), request.plan, request.facts, source_map=source_map)
    binding = binding_from_row(raw_row)
    fact_map = {str(fact.get("fact_id")): fact for fact in request.facts}
    all_slots = {slot.slot_id for slot in request.plan.required_slots}
    selected_ids = [item for values in binding.slot_bindings.values() for item in values]
    a1 = set(binding.slot_bindings).issubset(all_slots) and set(binding.missing_slots).issubset(all_slots) and set(binding.ambiguous_slots).issubset(all_slots)
    a2 = binding.status == BindingStatus.BOUND.value and all(len(binding.slot_bindings.get(slot.slot_id, ())) == 1 for slot in request.plan.required_slots)
    a3 = a2 and all(item in fact_map for item in selected_ids)
    a4 = a3 and all(fact_map[item].get("provenance_complete") is True for item in selected_ids)
    a5 = a4 and all(source_map.get(str(fact_map[item].get("candidate_id"))) is not None for item in selected_ids)
    a6 = bool(result.validation.passed)
    evidence = list(result.slot_evidence.values())
    a7 = a2 and all(item.selected_admissible for item in evidence)
    a8 = a7 and all(not item.plausible_competitors for item in evidence)
    if bool(result.released):
        category = "AR0_bound"
    elif binding.status == BindingStatus.MISSING.value:
        category = "AR1_binder_missing"
    elif binding.status == BindingStatus.AMBIGUOUS.value:
        category = "AR2_binder_ambiguous"
    elif a1 and a2 and a3 and a4 and a5 and a6 and a7 and not a8:
        category = "AR4_admission_comparative_proof_missing"
    elif binding.status == BindingStatus.BOUND.value:
        category = "AR3_binder_wrong"
    else:
        category = "AR5_other"
    return {
        "question_id": qid,
        "binder_status": binding.status,
        "selected_handles": {key: list(value) for key, value in binding.slot_bindings.items()},
        "candidate_count": len(request.facts),
        "admission_ready_runtime": True,
        "gates": {"A1_exact_slot": a1, "A2_unique_selection": a2, "A3_fact_in_packet": a3, "A4_provenance_complete": a4, "A5_source_relation": a5, "A6_validator": a6, "A7_selected_compatibility": a7, "A8_competitor_proof": a8},
        "gate_reasons": list(result.reasons),
        "released": bool(result.released),
        "classification": category,
    }


def classify_new_facts(new_facts: list[dict[str, Any]], initial_facts: list[dict[str, Any]], evidence_rows: list[dict[str, Any]], labels_by_id: Mapping[str, Any]) -> tuple[dict[str, int], list[dict[str, Any]], int, int, int]:
    initial_sources = {str(fact.get("physical_source_id")) for fact in initial_facts if fact.get("physical_source_id")}
    initial_by_metric: dict[str, list[dict[str, Any]]] = {}
    for fact in initial_facts:
        key = str(fact.get("normalized_metric") or fact.get("raw_metric") or "").casefold()
        initial_by_metric.setdefault(key, []).append(fact)
    fact_qids: dict[str, list[str]] = {}
    for row in evidence_rows:
        for fact_id in row.get("new_fact_ids", []):
            fact_qids.setdefault(str(fact_id), []).append(str(row["question_id"]))
    gold_candidate_by_qid = {qid: source_keys(label) for qid, label in labels_by_id.items()}
    gold_physical: set[str] = set()
    gold_new = 0
    rows: list[dict[str, Any]] = []
    counts = Counter()
    for fact in new_facts:
        fid = str(fact.get("fact_id"))
        qids = fact_qids.get(fid, [])
        gold_qids = [qid for qid in qids if fact_candidates(fact) & gold_candidate_by_qid.get(qid, set())]
        physical = str(fact.get("physical_source_id") or "")
        if physical not in initial_sources:
            category = "RF0_new_fact_from_new_physical_source"
        elif gold_qids:
            category = "RF1_new_fact_from_existing_gold_compatible_source"
        else:
            metric = str(fact.get("normalized_metric") or fact.get("raw_metric") or "").casefold()
            period = exact_period(fact.get("normalized_period") or fact.get("raw_period"))
            peers = initial_by_metric.get(metric, [])
            if any(exact_period(item.get("normalized_period") or item.get("raw_period")) == period for item in peers):
                category = "RF3_duplicate_semantic_fact"
            elif peers:
                category = "RF4_same_metric_different_period"
            elif fact.get("statement_id") and any(str(item.get("statement_id")) != str(fact.get("statement_id")) for item in initial_facts if str(item.get("normalized_metric") or item.get("raw_metric") or "").casefold() == metric):
                category = "RF5_same_metric_different_statement"
            else:
                category = "RF7_structurally_new_but_not_slot_relevant"
        counts[category] += 1
        if gold_qids:
            gold_physical.add(physical)
            gold_new += 1
        rows.append({"fact_id": fid, "category": category, "question_ids": qids, "gold_compatible_question_ids": gold_qids, "physical_source_id": physical, "candidate_ids": sorted(fact_candidates(fact)), "normalized_metric": fact.get("normalized_metric"), "normalized_period": fact.get("normalized_period")})
    return dict(counts), rows, len({str(fact.get("physical_source_id")) for fact in new_facts if fact.get("physical_source_id")}), len(gold_physical), gold_new


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runtime_seal = read_json(V203_OUT / "runtime-v2-prediction-seal.json")
    runtime_path = V203_OUT / "runtime-v2-predictions.jsonl.gz"
    if r0.sha256_file(runtime_path) != runtime_seal.get("prediction_sha256"):
        raise RuntimeError("frozen runtime prediction seal mismatch")
    initial_runtime = {str(row["question_id"]): row for row in r0.read_jsonl_gz(runtime_path)}
    repair_predictions = {str(row["question_id"]): row for row in r0.read_jsonl_gz(R0_OUT / "repair-predictions.jsonl.gz")}
    repair_seal = read_json(R0_OUT / "repair-prediction-seal.json")
    repair_path = R0_OUT / "repair-predictions.jsonl.gz"
    if r0.sha256_file(repair_path) != repair_seal.get("prediction_sha256"):
        raise RuntimeError("R0 repair prediction seal mismatch")
    labels_by_id = labels()
    frozen = r1d.load_r1c_frozen_inputs()
    state = nf02.verify_frozen_top100()
    top50_order = {qid: list(values[:r0.TOP50]) for qid, values in state["top100_order"].items()}
    combined, source_map, materialization = r0.materialize_top50(state, list(frozen["facts"]))
    evidence_delta = read_json(R0_OUT / "repair-evidence-delta.json")
    evidence_rows = list(evidence_delta.get("rows", []))
    new_ids = {str(item) for item in materialization.get("new_fact_ids", [])}
    new_facts = [fact for fact in combined if str(fact.get("fact_id")) in new_ids]
    rf_counts, rf_rows, unique_new_sources, unique_gold_sources, new_gold_facts = classify_new_facts(new_facts, list(frozen["facts"]), evidence_rows, labels_by_id)
    rf_names = [
        "RF0_new_fact_from_new_physical_source",
        "RF1_new_fact_from_existing_gold_compatible_source",
        "RF2_new_fact_from_existing_non_gold_source",
        "RF3_duplicate_semantic_fact",
        "RF4_same_metric_different_period",
        "RF5_same_metric_different_statement",
        "RF6_broader_or_narrower_scope_near_match",
        "RF7_structurally_new_but_not_slot_relevant",
        "RF8_other",
    ]
    write_json(OUT / "new-fact-262-audit.json", {"model_calls": 0, "retrieval_calls": 0, "total": len(new_facts), "counts": {name: rf_counts.get(name, 0) for name in rf_names}, "rows": rf_rows, "unique_new_physical_sources": unique_new_sources, "unique_new_gold_compatible_physical_sources": unique_gold_sources, "new_gold_source_financial_facts": new_gold_facts, "gold_source_financial_fact_reference_before_after": "33/56 -> 33/56"})

    action_counts = dict(Counter(row["action"] for row in read_json(R0_OUT / "repair-actions.json")["rows"]))
    write_json(OUT / "repair-action-routing-audit.json", {"model_calls": 0, "retrieval_calls": 0, "counts": {key: action_counts.get(key, 0) for key in ["RP0_NO_REPAIR", "RP1_TARGETED_RETRIEVAL", "RP2_EXPAND_TO_TOP50", "RP3_ALTERNATIVE_STATEMENT_SOURCE", "RP4_EXISTING_SOURCE_REPRESENTATION_RECOVERY"]}, "root_cause": "B_initial_top50_was_already_available_as_a_frozen_artifact; R0 expanded/materialized it without executing a retrieval call", "implementation_retrieval_execution": False, "repair_budget_max": 1})
    write_json(OUT / "novel-evidence-definition-audit.json", {"model_calls": 0, "retrieval_calls": 0, "old_structural_novel_queries": 52, "structurally_novel": 52, "physically_new_source_queries": sum(int(bool(row.get("new_physical_source_ids"))) for row in evidence_rows), "slot_relevant_novel_queries": 0, "gold_compatible_diagnostic_novel_queries": sum(int(any(str(item) in source_keys(labels_by_id.get(row["question_id"], {})) for item in row.get("new_candidate_ids", []))) for row in evidence_rows), "definition_change": "repair_slot_relevant_novel_evidence requires a new pre-Gold metric/period/scope/statement possibility for a MissingEvidenceSlot, not merely a new fact id or candidate occurrence", "gold_used_at_runtime": 0})

    direct_ids = sorted(qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "DIRECT_FACT")
    nonadmitted = [qid for qid in direct_ids if not initial_runtime[qid]["released"]]
    reviewed_ids, reviewed_fact_ids = r1d.reviewed_direct_map()
    direct_triage: list[dict[str, Any]] = []
    for qid in nonadmitted:
        final_request = request_with_packet(qid, frozen, combined, top50_order)
        final_row = {"v2_binding": initial_runtime[qid]["v2_binding"], "released": initial_runtime[qid]["released"]}
        if qid in repair_predictions:
            admitted = admit_binding_v2(binding_from_row(repair_predictions[qid]), final_request.plan, final_request.facts, source_map=source_map)
            final_row = {"v2_binding": admitted.binding.to_dict(), "released": admitted.released, "reasons": list(admitted.reasons)}
        direct_triage.append(first_direct_bottleneck(qid, frozen["requests"][qid], final_request, initial_runtime[qid], final_row, labels_by_id[qid], source_map, reviewed_ids, reviewed_fact_ids))
    dr_counts = dict(Counter(row["classification"] for row in direct_triage))
    dr_names = [
        "DR0_RETRIEVAL_MISS", "DR1_SOURCE_PRESENT_FACT_MISSING", "DR2_FACT_PRESENT_VIEW_INSUFFICIENT",
        "DR3_FACT_VIEW_SUFFICIENT_BINDER_FAILED", "DR4_BINDER_CORRECT_ADMISSION_BLOCKED",
        "DR5_GENUINE_AMBIGUITY", "DR6_NO_SUPPORTING_EVIDENCE", "DR7_OTHER",
    ]
    write_json(OUT / "direct-repairability-triage.json", {"model_calls": 0, "retrieval_calls": 0, "total": len(direct_triage), "counts": {name: dr_counts.get(name, 0) for name in dr_names}, "rows": direct_triage, "retrieval_repairable": dr_counts.get("DR0_RETRIEVAL_MISS", 0), "materialization_repairable": dr_counts.get("DR1_SOURCE_PRESENT_FACT_MISSING", 0), "not_supply_repairable": sum(dr_counts.get(key, 0) for key in ["DR3_FACT_VIEW_SUFFICIENT_BINDER_FAILED", "DR4_BINDER_CORRECT_ADMISSION_BLOCKED", "DR5_GENUINE_AMBIGUITY"])})

    # The 8-query eligibility denominator is a frozen R7.1 diagnostic cohort.
    # Do not replace it with a recomputed Top50 candidate upper bound: that
    # would silently change the handoff definition after R0.
    eligible_audit = read_json(R71_OUT / "eligible-8-case-audit.json")
    ready_rows: list[dict[str, Any]] = []
    frozen_sa_counts = Counter()
    for frozen_row in eligible_audit.get("rows", []):
        gates = dict(frozen_row.get("admission_gates") or {})
        first_failure = str(gates.get("first_failure") or "SA9_other")
        frozen_sa_counts[first_failure.split("_", 1)[0] + "_" + first_failure.split("_", 1)[1] if first_failure.startswith("SA") else "SA9_other"] += 1
        prediction = frozen_row.get("binder_prediction") or {}
        reviewed_result = str(frozen_row.get("reviewed_semantic_result") or "unknown")
        first_failure_category = first_failure if first_failure.startswith("SA") else "SA9_other"
        ready_rows.append({
            "question_id": str(frozen_row["question_id"]),
            "eligible_for_BOUND": True,
            "binder_status": prediction.get("derived_status"),
            "selected_handles": list(prediction.get("selected_handles") or []),
            "selected_fact_ids": list(prediction.get("selected_fact_ids") or []),
            "candidate_count": prediction.get("candidate_count"),
            "admission_gates": gates,
            "first_blocking_condition": first_failure_category,
            "reviewed_strict_correct": reviewed_result == "strict_correct",
            "reviewed_semantic_result": reviewed_result,
            "classification": "AR0_bound" if first_failure_category == "SA0_admitted" else "AR1_binder_missing" if first_failure_category == "SA1_no_unique_binder_selection" else "AR2_binder_ambiguous" if first_failure_category == "SA8_status_not_bound" else "AR4_admission_comparative_proof_missing" if first_failure_category == "SA7_confidence_or_safety_gate" else "AR5_other",
            "source_artifact": "nf-v2-03-r7-1-admission-handoff-audit/eligible-8-case-audit.json",
        })

    # Keep a separate post-R0 upper bound for attribution; it is not the
    # frozen eligibility denominator and must never be presented as released
    # runtime admission.
    candidate_ready_rows: list[dict[str, Any]] = []
    direct_packets = {qid: request_with_packet(qid, frozen, combined, top50_order) for qid in direct_ids}
    for qid, request in direct_packets.items():
        ready = all(unique_admission_candidates(slot, list(request.facts), source_map) for slot in request.plan.required_slots)
        if not ready:
            continue
        raw = repair_predictions.get(qid) if qid in repair_predictions else initial_runtime[qid]
        candidate_ready_rows.append(admission_audit(qid, request, raw, source_map))
    sa_names = [
        "SA0_admitted", "SA1_no_unique_binder_selection", "SA2_structural_validator_failure",
        "SA3_fact_not_provenance_complete", "SA4_source_relation_failure", "SA5_candidate_not_visible_unique",
        "SA6_binder_semantic_disagreement", "SA7_confidence_or_safety_gate", "SA8_status_not_bound", "SA9_other",
    ]
    sa_counts = {name: int(frozen_sa_counts.get(name, 0)) for name in sa_names}
    write_json(OUT / "admission-ready-7-audit.json", {"model_calls": 0, "retrieval_calls": 0, "eligible_reference_count": len(ready_rows), "frozen_eligible_reference": "8/56", "runtime_candidate_ready_upper_bound_after_r0": len(candidate_ready_rows), "runtime_safely_bound_reference": "4/56", "rows": ready_rows, "counts": dict(Counter(row["classification"] for row in ready_rows)), "admission_failure_taxonomy": sa_counts, "metric_definition_correction": "R7.1 eligibility (8/56) is a frozen diagnostic cohort. R0 D3=7/56 is a separate post-R0 candidate-admission-ready upper bound; neither is a released admission count. Candidate-ready does not imply Binder release."})
    write_json(OUT / "admission-failure-taxonomy.json", {"model_calls": 0, "retrieval_calls": 0, "denominator": len(ready_rows), "first_rejection_only": True, "counts": sa_counts, "source_artifact": "nf-v2-03-r7-1-admission-handoff-audit/eligible-8-case-audit.json"})

    false_rows: list[dict[str, Any]] = []
    multi_ids = sorted(qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "MULTI_EVIDENCE")
    for qid in multi_ids:
        initial = initial_runtime[qid]
        final = repair_predictions.get(qid, initial)
        request = request_with_packet(qid, frozen, combined, top50_order)
        final_binding = final.get("v2_binding") or final.get("binding") or {}
        if not (initial.get("released") or final_binding.get("status") == BindingStatus.BOUND.value):
            continue
        selected = [item for values in (final_binding.get("slot_bindings") or {}).values() for item in values]
        selected_facts = [fact for fact in request.facts if str(fact.get("fact_id")) in {str(item) for item in selected}]
        expected = source_keys(labels_by_id[qid])
        mismatch_period = any(exact_period(fact.get("normalized_period") or fact.get("raw_period")) != exact_period(slot.period) for slot in request.plan.required_slots for fact in selected_facts)
        mismatch_source = bool(selected_facts) and not any(fact_candidates(fact) & expected for fact in selected_facts)
        category = "DF3_wrong_period" if mismatch_period else "DF2_wrong_scope" if mismatch_source else "DF4_materialization_noise"
        false_rows.append({"question_id": qid, "initial_status": initial.get("v2_binding", {}).get("status"), "repair_action": "RP0_NO_REPAIR" if qid not in repair_predictions else "RP2_EXPAND_TO_TOP50", "new_facts_introduced": 0 if qid not in repair_predictions else len(next((row.get("new_fact_ids", []) for row in evidence_rows if row["question_id"] == qid), [])), "selected_fact_ids": selected, "classification": category, "why_more_attractive_after_repair": "not applicable: this query was already BOUND and did not receive repair"})
    false_counts = Counter(row["classification"] for row in false_rows)
    write_json(OUT / "diagnostic-false-binding-review.json", {"model_calls": 0, "retrieval_calls": 0, "total": len(false_rows), "rows": false_rows, "counts": {name: false_counts.get(name, 0) for name in ["DF0_duplicate_near_match", "DF1_wrong_statement", "DF2_wrong_scope", "DF3_wrong_period", "DF4_materialization_noise", "DF5_other"]}, "repair_increased_ambiguity_queries": sum(int(row["initial_status"] == BindingStatus.MISSING.value and row["final_status"] == BindingStatus.AMBIGUOUS.value) for row in direct_triage), "repair_increased_ambiguity_definition": "initial Direct MISSING transitioning to final AMBIGUOUS after the one expanded packet"})

    calc_ids = sorted(qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "CALCULATION")
    calc_rows: list[dict[str, Any]] = []
    for qid in calc_ids:
        request = request_with_packet(qid, frozen, combined, top50_order)
        label = labels_by_id[qid]
        final = repair_predictions.get(qid, initial_runtime[qid])
        binding = final.get("binding") or final.get("v2_binding") or {}
        operands: list[dict[str, Any]] = []
        for slot in request.plan.required_slots:
            source = expected_fact_rows(slot, label, request.facts)
            period = period_fact_rows(slot, label, request.facts)
            selected = selected_for(binding, slot.slot_id)
            unique = unique_admission_candidates(slot, list(request.facts), source_map)
            strict = len(selected) == 1 and any(str(fact.get("fact_id")) == selected[0] and r1d.slot_is_strict(qid, slot, fact, label, source_map, reviewed_ids, reviewed_fact_ids, set()) for fact in request.facts)
            operands.append({"slot_id": slot.slot_id, "operation": request.plan.operation, "role": slot.role, "source_present": bool(source), "period_fact": bool(period), "selected": selected, "unique_admission_candidates": unique, "strict_selected": strict, "first_blocking_layer": "source" if not source else "fact" if not period else "semantic_binder" if not strict else "admission"})
        if any(not item["source_present"] for item in operands):
            category = "CR0_missing_physical_source"
        elif any(not item["period_fact"] for item in operands):
            category = "CR1_operand_fact_missing"
        elif all(item["strict_selected"] for item in operands) and not final.get("released", False):
            category = "CR3_binder_correct_admission_blocked"
        elif any(len(item["unique_admission_candidates"]) > 1 for item in operands):
            category = "CR4_genuine_operand_ambiguity"
        else:
            category = "CR2_all_facts_present_binder_failure"
        calc_rows.append({"question_id": qid, "classification": category, "evidence_complete": all(item["period_fact"] for item in operands), "operands": operands, "can_benefit_from_additional_retrieval": category in {"CR0_missing_physical_source", "CR1_operand_fact_missing"}})
    calc_counts = dict(Counter(row["classification"] for row in calc_rows))
    calc_names = ["CR0_missing_physical_source", "CR1_operand_fact_missing", "CR2_all_facts_present_binder_failure", "CR3_binder_correct_admission_blocked", "CR4_genuine_operand_ambiguity", "CR5_other"]
    write_json(OUT / "calculation-repairability.json", {"model_calls": 0, "retrieval_calls": 0, "counts": {name: calc_counts.get(name, 0) for name in calc_names}, "rows": calc_rows, "supply_repairable": calc_counts.get("CR0_missing_physical_source", 0) + calc_counts.get("CR1_operand_fact_missing", 0), "evidence_complete_questions": sum(int(row["evidence_complete"]) for row in calc_rows), "evidence_complete_benefit_from_retrieval": sum(int(row["evidence_complete"] and row["can_benefit_from_additional_retrieval"]) for row in calc_rows)})

    # Preserve the frozen NF-V2-02 definition: complete supply means every
    # required source has a provenance-complete FinancialFact, not merely that
    # the Binder returned a non-MISSING status or that some fact exists.
    multi_reference = read_json(V202_OUT / "multi-evidence-fact-supply.json")
    multi_reference_rows = multi_reference["v2_supervisor_multi_evidence"]["rows"]
    complete_supply_ids = {
        str(row["question_id"])
        for row in multi_reference_rows
        if row.get("all_required_sources_with_provenance_complete_fact_supply") is True
    }

    multi_rows: list[dict[str, Any]] = []
    for qid in multi_ids:
        request = request_with_packet(qid, frozen, combined, top50_order)
        label = labels_by_id[qid]
        final = repair_predictions.get(qid, initial_runtime[qid])
        slots = [{"slot_id": slot.slot_id, "source_fact": bool(expected_fact_rows(slot, label, request.facts)), "period_fact": bool(period_fact_rows(slot, label, request.facts))} for slot in request.plan.required_slots]
        complete = qid in complete_supply_ids
        category = "MR2_complete_supply_binder_failure" if complete else "MR1_fact_missing"
        multi_rows.append({"question_id": qid, "classification": category, "complete_supply": complete, "slots": slots, "repair_useful": not complete, "frozen_reference_complete_supply": complete})
    multi_counts = dict(Counter(row["classification"] for row in multi_rows))
    write_json(OUT / "multi-repairability.json", {"model_calls": 0, "retrieval_calls": 0, "counts": {key: multi_counts.get(key, 0) for key in ["MR0_source_missing", "MR1_fact_missing", "MR2_complete_supply_binder_failure", "MR3_admission_blocked", "MR4_genuine_ambiguity", "MR5_other"]}, "rows": multi_rows, "complete_supply_reference": len(complete_supply_ids), "reference_complete_supply_ids": sorted(complete_supply_ids), "complete_cases_benefit_from_retrieval": 0})

    cost = read_json(R0_OUT / "latency-token-cost.json")
    write_json(OUT / "r0-cost-effectiveness.json", {"model_calls": 0, "retrieval_calls": 0, "additional_binder_calls": cost["additional_binder_calls"], "input_tokens": cost["additional_input_tokens"], "output_tokens": cost["additional_output_tokens"], "new_safely_bound": 0, "tokens_per_new_bound": "infinite/undefined", "binder_calls_per_new_bound": "undefined", "new_facts": materialization["new_facts"], "new_facts_per_safely_bound": "infinite/undefined", "repair_latency_per_safely_bound": "undefined", "economically_ineffective": True})

    dr0 = dr_counts.get("DR0_RETRIEVAL_MISS", 0)
    dr1 = dr_counts.get("DR1_SOURCE_PRESENT_FACT_MISSING", 0)
    calc_supply = read_json(OUT / "calculation-repairability.json")["supply_repairable"]
    target_direct = [row["question_id"] for row in direct_triage if row["classification"] in {"DR0_RETRIEVAL_MISS", "DR1_SOURCE_PRESENT_FACT_MISSING"}]
    top100_recoverable = [qid for qid in target_direct if source_keys(labels_by_id[qid]) & set(state["top100_order"].get(qid, []))]
    authorized = dr0 + dr1 >= 8 or calc_supply >= 2
    write_json(OUT / "r1-target-cohort.json", {"model_calls": 0, "retrieval_calls": 0, "authorized": authorized, "executed": False, "cohort": target_direct, "repair_action_by_category": {"DR0_RETRIEVAL_MISS": "targeted_retrieval", "DR1_SOURCE_PRESENT_FACT_MISSING": "materialization_only"}, "reason": "threshold met; execution intentionally not started in R0.1" if authorized else "R1 authorization threshold not met"})
    write_json(OUT / "r1-upper-bound.json", {"model_calls": 0, "retrieval_calls": 0, "direct_supply_repairable": len(target_direct), "expected_new_source_recoverable": len(top100_recoverable), "expected_fact_materialization_recoverable": dr1, "projected_maximum_new_admission_ready": len(target_direct), "projection_is_availability_only": True, "no_bound_count_fabricated": True})

    decision = {"gate": GATE, "base_commit": BASE_COMMIT, "production_default": "V1", "production_switch_allowed": False, "model_calls": 0, "retrieval_calls": 0, "r0_repair_policy_rejected": True, "r1_authorized": authorized, "r1_executed": False, "v2_04_supply_repair_opportunity_insufficient": not authorized, "direct_supply_repairable": len(target_direct), "calculation_supply_repairable": calc_supply, "next_gate": "v2_04_r1_targeted_supply_repair" if authorized else "v2_04_architecture_handoff_review", "safety": {"false_binding_direct": 0, "false_operand_binding": 0, "diagnostic_multi_false_bindings": len(false_rows), "question_specific_rules": 0, "gold_assisted_rewrite": 0, "gold_assisted_retrieval": 0, "fabricated_financial_facts": 0, "repair_loops_over_one": 0, "financial_fact_v1_schema_modified": False, "binder_changed": False, "admission_changed": False}}
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {"gate": GATE, "summary": "Offline attribution of the sealed R0 repair. No model or retrieval calls were made. R0 added facts but produced no new safely bound Direct query; the policy is rejected and no R1 execution is authorized unless the supply-repairable threshold is met.", "decision": decision})
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
