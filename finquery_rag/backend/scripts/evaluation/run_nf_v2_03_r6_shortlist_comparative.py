#!/usr/bin/env python3
"""NF-V2-03 R6 deterministic shortlist and comparative Binder diagnostic."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.evidence import BindingStatus  # noqa: E402
from rag_v2.evidence.binder_service import BinderRequest, BinderRun, SemanticBinderService, empty_fact_binding  # noqa: E402
from rag_v2.evidence.binding_validator import validate_binding  # noqa: E402
from rag_v2.evidence.shortlist_comparative_binder import (  # noqa: E402
    COMPARATIVE_SYSTEM_PROMPT,
    SHORTLIST_FORMULATION,
    BailianShortlistComparativeBinderProvider,
    CandidateShortlist,
    build_shortlists,
)
from scripts.evaluation import run_nf_v2_02_top20_financial_fact_expansion as nf02  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1c_supply_recovery as r1c  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r5_1_pairwise_binder as r51  # noqa: E402


BASE_COMMIT = "bda77108dfaf545854041584336c8ea86767aac5"
MODEL = "qwen3.7-plus"
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r6-shortlist-comparative"
R5_1_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r5-1-pairwise-binder"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    frozen = r1d.load_r1c_frozen_inputs()
    return frozen, r1c.candidate_source_map(nf02.verify_frozen_top100())


def load_cohort() -> dict[str, Any]:
    path = r51.R4_OUT / "model-review-cohort.json"
    actual = sha256_file(path)
    expected = (r51.R4_OUT / "model-review-cohort.sha256").read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError("R4 frozen diagnostic cohort SHA mismatch")
    value = read_json(path)
    value["cohort_sha256"] = actual
    return value


def load_labels() -> dict[str, dict[str, Any]]:
    return r51.load_labels()


def shortlist_record(request: BinderRequest, shortlists: Mapping[str, CandidateShortlist]) -> dict[str, Any]:
    return {
        "question_id": request.question_id,
        "fact_count": len(request.facts),
        "slots": {
            slot_id: {
                "candidate_count": len(item.candidates),
                "candidate_handles": list(item.handles),
                "candidates": [
                    {"handle": candidate["handle"], "fact_id": candidate["fact_id"], "score": candidate["score"], "signals": candidate["signals"]}
                    for candidate in item.candidates
                ],
                "hard_rejected": list(item.hard_rejected),
                "pre_status": "NO_ELIGIBLE_CANDIDATE" if not item.candidates else None,
            }
            for slot_id, item in shortlists.items()
        },
    }


def build_shortlist_artifact(frozen: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]], cohort: Mapping[str, Any]) -> tuple[dict[str, CandidateShortlist], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, CandidateShortlist]] = {}
    for qid in cohort["unique_question_ids"]:
        request = frozen["requests"][qid]
        shortlists, _, _ = build_shortlists(request, fact_view_version="v2", source_by_candidate=source_map)
        cache[qid] = shortlists
        rows.append(shortlist_record(request, shortlists))
    path = OUT / "shortlist-results.jsonl.gz"
    write_jsonl_gz(path, rows)
    digest = sha256_file(path)
    write_json(OUT / "shortlist-seal.json", {"gate": "NF-V2-03-R6", "formulation": SHORTLIST_FORMULATION, "prediction_count": len(rows), "shortlist_sha256": digest, "sealed": True, "gold_reads_before_shortlist_seal": 0})
    return {}, {}, cache


def gold_compatible_handles(qid: str, slot: Any, request: BinderRequest, labels: Mapping[str, Mapping[str, Any]], source_map: Mapping[str, Mapping[str, Any]]) -> set[str]:
    reviewed_ids, reviewed_fact_ids = r1d.reviewed_direct_map()
    handles = {str(fact["fact_id"]): f"F{index:02d}" for index, fact in enumerate(request.facts, 1)}
    result: set[str] = set()
    for fact in request.facts:
        if r1d.slot_is_strict(qid, slot, fact, labels[qid], source_map, reviewed_ids, reviewed_fact_ids, set()):
            result.add(handles[str(fact["fact_id"])])
    return result


def shortlist_recall_audit(frozen: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]], cohort: Mapping[str, Any], cache: Mapping[str, Mapping[str, CandidateShortlist]]) -> dict[str, Any]:
    labels = load_labels()
    groups = cohort["groups"]
    direct_rows: list[dict[str, Any]] = []
    calc_rows: list[dict[str, Any]] = []
    size_values: list[int] = []
    rank_counts = {name: 0 for name in ("Top1", "Top2", "Top3", "Top5")}
    for qid in cohort["unique_question_ids"]:
        request = frozen["requests"][qid]
        for slot in request.plan.required_slots:
            item = cache[qid][slot.slot_id]
            size_values.append(len(item.candidates))
            gold = gold_compatible_handles(qid, slot, request, labels, source_map)
            ranked = [candidate["handle"] for candidate in item.candidates]
            retained = sorted(gold & set(ranked), key=lambda handle: ranked.index(handle))
            rank = ranked.index(retained[0]) + 1 if retained else None
            if rank == 1:
                rank_counts["Top1"] += 1
            if rank is not None and rank <= 2:
                rank_counts["Top2"] += 1
            if rank is not None and rank <= 3:
                rank_counts["Top3"] += 1
            if rank is not None and rank <= 5:
                rank_counts["Top5"] += 1
            record = {"question_id": qid, "slot_id": slot.slot_id, "shortlist_size": len(item.candidates), "gold_handles": sorted(gold), "retained_gold_handles": retained, "correct_candidate_rank": rank}
            if qid in groups["A_direct_visible_unique"]:
                direct_rows.append(record)
            if qid in groups["B_calculation_visible_unique"]:
                calc_rows.append(record)
    indist_rows = [{"question_id": qid, "shortlist_size": len(cache[qid][frozen["requests"][qid].plan.required_slots[0].slot_id].candidates), "plausible_candidates_retained": len(cache[qid][frozen["requests"][qid].plan.required_slots[0].slot_id].candidates) >= 2} for qid in groups["C_indistinguishable"]]
    unbind_rows = [{"question_id": qid, "zero_eligible": all(not cache[qid][slot.slot_id].candidates for slot in frozen["requests"][qid].plan.required_slots)} for qid in groups["D_unbindable_safety"]]
    size_distribution = {str(size): size_values.count(size) for size in range(6)}
    size_summary = {"distribution": size_distribution, "mean": statistics.mean(size_values) if size_values else 0, "median": statistics.median(size_values) if size_values else 0, "p95": r1d.percentile([float(value) for value in size_values]), "max": max(size_values) if size_values else 0}
    result = {
        "direct": {"questions": 21, "gold_compatible_candidate_retained": sum(bool(row["retained_gold_handles"]) for row in direct_rows), "rows": direct_rows},
        "calculation": {"operand_slots": 12, "gold_compatible_candidate_retained": sum(bool(row["retained_gold_handles"]) for row in calc_rows), "rows": calc_rows},
        "indistinguishable": {"questions": 6, "plausible_candidates_retained": sum(int(row["plausible_candidates_retained"]) for row in indist_rows), "rows": indist_rows},
        "unbindable": {"questions": 7, "zero_eligible": sum(int(row["zero_eligible"]) for row in unbind_rows), "rows": unbind_rows},
        "shortlist_size": size_summary,
        "correct_candidate_rank": rank_counts,
        "hard_gate": bool(sum(bool(row["retained_gold_handles"]) for row in direct_rows) >= 20 and sum(bool(row["retained_gold_handles"]) for row in calc_rows) == 12),
        "gold_reads_after_shortlist_seal": True,
    }
    write_json(OUT / "shortlist-recall-audit.json", result)
    write_json(OUT / "shortlist-size-distribution.json", size_summary)
    return result


def provider(source_map: Mapping[str, Mapping[str, Any]]) -> BailianShortlistComparativeBinderProvider:
    config = r1d.legacy.load_config()
    return BailianShortlistComparativeBinderProvider(base_url=os.getenv("V2_SUPERVISOR_BASE_URL", "").strip(), api_key=config["api_key"], model_name=MODEL, enable_thinking=False, temperature=0.0, timeout=180.0, max_retries=0, system_prompt=COMPARATIVE_SYSTEM_PROMPT, fact_view_version="v2", source_metadata_by_candidate=source_map)


def no_eligible_run(request: BinderRequest) -> BinderRun:
    binding = empty_fact_binding(request.plan)
    validation = validate_binding(binding, request.plan, request.facts)
    return BinderRun(request=request, binding=binding, validation=validation, metadata=None, skipped_no_fact_supply=True, schema_valid=True)


def summarize_run(request: BinderRequest, run: BinderRun, provider_obj: BailianShortlistComparativeBinderProvider, shortlists: Mapping[str, CandidateShortlist]) -> dict[str, Any]:
    row = run.to_dict()
    row.update({"question_id": request.question_id, "question": request.question, "intent": request.plan.intent.value, "operation": request.plan.operation, "fact_count": len(request.facts), "provider_calls": int(run.metadata is not None), "provider_response_success": bool(run.metadata and run.metadata.provider_response_success) if run.metadata else True, "structured_output_success": bool(run.metadata and run.metadata.structured_output_success) if run.metadata else True, "dto_valid": bool(run.schema_valid), "adapter_valid": bool(run.schema_valid and run.binding and run.binding.status != BindingStatus.INVALID.value), "binding_validator_pass": bool(run.validation.passed), "comparative_decisions": provider_obj.last_comparative_outcomes or {}, "shortlist_handles": {slot_id: list(item.handles) for slot_id, item in shortlists.items()}, "shortlist_sizes": {slot_id: len(item.candidates) for slot_id, item in shortlists.items()}, "raw_response": run.raw_response or ""})
    return row


def actual_handles(row: Mapping[str, Any], request: BinderRequest) -> dict[str, list[str]]:
    ids_to_handles = {str(fact["fact_id"]): f"F{index:02d}" for index, fact in enumerate(request.facts, 1)}
    binding = row.get("binding") or {}
    return {slot_id: [ids_to_handles.get(str(fact_id), str(fact_id)) for fact_id in ids] for slot_id, ids in (binding.get("slot_bindings") or {}).items()}


def expected_correct(row: Mapping[str, Any], request: BinderRequest, expected: Mapping[str, Any]) -> bool:
    actual = actual_handles(row, request)
    for slot_id, target in expected.items():
        if isinstance(target, list) and sorted(actual.get(slot_id, [])) != sorted(target):
            return False
        if target == "MISSING" and row.get("final_binding_status") != "MISSING":
            return False
        if target == "AMBIGUOUS" and row.get("final_binding_status") != "AMBIGUOUS":
            return False
    return True


def synthetic_run() -> dict[str, Any]:
    cases = r51.synthetic_cases()
    comparative = provider({})
    service = SemanticBinderService(comparative)
    rows: list[dict[str, Any]] = []
    try:
        for request, expected, tags in cases:
            shortlists, _, _ = build_shortlists(request, fact_view_version="v2", source_by_candidate={})
            run = no_eligible_run(request) if all(not item.candidates for item in shortlists.values()) else service.bind(request)
            row = summarize_run(request, run, comparative, shortlists)
            row.update({"expected": expected, "tags": sorted(tags), "semantic_correct": expected_correct(row, request, expected), "indistinguishable_safe": "indistinguishable" in tags and row["final_binding_status"] in {"MISSING", "AMBIGUOUS"}, "false_binding": "unbindable" in tags and row["final_binding_status"] == "BOUND"})
            rows.append(row)
    finally:
        comparative.close()
    calc_rows = [row for row in rows if "calculation" in row["tags"]]
    summary = {"gate": "NF-V2-03-R6", "formulation": SHORTLIST_FORMULATION, "model": MODEL, "benchmark_questions_used": 0, "provider_calls": sum(row["provider_calls"] for row in rows), "provider_success": sum(int(row["provider_response_success"]) for row in rows), "structured_output": sum(int(row["structured_output_success"]) for row in rows), "dto_valid": sum(int(row["dto_valid"]) for row in rows), "adapter_valid": sum(int(row["adapter_valid"]) for row in rows), "binding_validator": sum(int(row["binding_validator_pass"]) for row in rows), "semantic_correct": sum(int(row["semantic_correct"]) for row in rows), "semantic_total": len(rows), "indistinguishable_safe": sum(int(row["indistinguishable_safe"]) for row in rows), "indistinguishable_total": sum("indistinguishable" in row["tags"] for row in rows), "unbindable_false_binding": sum(int(row["false_binding"]) for row in rows), "unbindable_total": sum("unbindable" in row["tags"] for row in rows), "calculation_groups_correct": sum(int(row["semantic_correct"]) for row in calc_rows), "calculation_groups_total": len(calc_rows), "structural_healthy": all(row["provider_response_success"] and row["structured_output_success"] and row["dto_valid"] and row["adapter_valid"] and row["binding_validator_pass"] for row in rows), "rows": rows}
    summary["pass"] = bool(summary["semantic_correct"] >= 12 and summary["indistinguishable_safe"] >= 2 and summary["unbindable_false_binding"] == 0 and summary["calculation_groups_correct"] >= 5 and summary["structural_healthy"])
    write_json(OUT / "synthetic-comparative-suite.json", summary)
    return summary


def diagnostic_run(frozen: dict[str, Any], source_map: Mapping[str, Mapping[str, Any]], cohort: Mapping[str, Any], cache: Mapping[str, Mapping[str, CandidateShortlist]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    comparative = provider(source_map)
    service = SemanticBinderService(comparative)
    rows: list[dict[str, Any]] = []
    try:
        for qid in cohort["unique_question_ids"]:
            request = frozen["requests"][qid]
            shortlists = cache[qid]
            run = no_eligible_run(request) if all(not item.candidates for item in shortlists.values()) else service.bind(request)
            rows.append(summarize_run(request, run, comparative, shortlists))
    finally:
        comparative.close()
    path = OUT / "diagnostic-predictions.jsonl.gz"
    write_jsonl_gz(path, rows)
    digest = sha256_file(path)
    write_json(OUT / "diagnostic-seal.json", {"gate": "NF-V2-03-R6", "formulation": SHORTLIST_FORMULATION, "prediction_count": len(rows), "prediction_sha256": digest, "sealed": True, "gold_reads_before_prediction_seal": 0, "sealed_before_gold": True})
    if sha256_file(path) != digest:
        raise RuntimeError("R6 diagnostic prediction seal verification failed")
    scored = r51.diagnostic_score(rows, frozen, source_map, cohort)
    metadata = [row["metadata"] for row in rows if row.get("metadata")]
    latencies = [float(item.get("latency_ms") or 0) for item in metadata]
    inputs = [int(item.get("input_tokens") or 0) for item in metadata]
    outputs = [int(item.get("output_tokens") or 0) for item in metadata]
    decisions = {name: 0 for name in ("SELECT", "NONE", "AMBIGUOUS")}
    for row in rows:
        for decision in (row.get("comparative_decisions") or {}).values():
            if decision.get("decision") in decisions:
                decisions[decision["decision"]] += 1
    runtime = {"provider_calls": sum(row["provider_calls"] for row in rows), "provider_calls_per_query": 1, "input_tokens": sum(inputs), "output_tokens": sum(outputs), "average_latency_ms": statistics.mean(latencies) if latencies else 0, "p50_latency_ms": statistics.median(latencies) if latencies else 0, "p95_latency_ms": r1d.percentile(latencies), "max_latency_ms": max(latencies) if latencies else 0, "raw_comparative_decisions": decisions}
    old_path = R5_1_OUT / "diagnostic-predictions.jsonl.gz"
    if old_path.exists():
        old = {row["question_id"]: row for row in read_jsonl_gz(old_path)}
        old_rows = [old[qid] for qid in cohort["unique_question_ids"]]
        old_inputs = [int(item.get("metadata", {}).get("input_tokens") or 0) for item in old_rows if item.get("metadata")]
        runtime["pairwise_r5_1_same_cohort"] = {"provider_calls": len(old_rows), "input_tokens": sum(old_inputs), "average_input_tokens": statistics.mean(old_inputs) if old_inputs else 0, "p50_input_tokens": statistics.median(old_inputs) if old_inputs else 0, "p95_input_tokens": r1d.percentile([float(x) for x in old_inputs]), "max_input_tokens": max(old_inputs) if old_inputs else 0}
        runtime["token_delta_shortlist_comparative_minus_pairwise"] = {"total_input_tokens": sum(inputs) - sum(old_inputs), "mean_input_tokens": statistics.mean(inputs) - statistics.mean(old_inputs), "p50_input_tokens": statistics.median(inputs) - statistics.median(old_inputs), "p95_input_tokens": r1d.percentile([float(x) for x in inputs]) - r1d.percentile([float(x) for x in old_inputs]), "max_input_tokens": max(inputs) - max(old_inputs)}
    write_json(OUT / "token-latency-cost.json", runtime)
    return rows, scored


def selective_freeze(rows: list[dict[str, Any]], *, reason: str) -> dict[str, Any]:
    released = []
    for row in rows:
        unique_all = all(size == 1 for size in (row.get("shortlist_sizes") or {}).values())
        if unique_all and row.get("final_binding_status") == BindingStatus.BOUND.value and row.get("binding_validator_pass"):
            released.append(row["question_id"])
    policy = {"policy": "selective_fail_closed_v1", "reason": reason, "release_bound_requires": ["all_required_slots_have_exactly_one_shortlist_candidate", "comparative_decision_SELECT", "selected_handle_in_shortlist", "Binding Validator pass"], "model": MODEL, "question_specific_rules": 0, "gold_conditioned_rules": 0, "released_bound_queries": len(released), "released_question_ids": released, "otherwise": "MISSING_OR_AMBIGUOUS", "production_default": "V1", "production_switch_allowed": False}
    write_json(OUT / "selective-freeze-policy.json", policy)
    return policy


def main() -> int:
    if os.getenv("V2_SUPERVISOR_MODEL", "").strip() not in ("", MODEL):
        raise SystemExit("V2_SUPERVISOR_MODEL must remain qwen3.7-plus")
    OUT.mkdir(parents=True, exist_ok=True)
    frozen, source_map = load_frozen()
    cohort = load_cohort()
    write_json(OUT / "shortlist-contract.json", {"component": "BinderCandidateShortlistV1", "formulation": SHORTLIST_FORMULATION, "max_candidates_per_slot": 5, "preferred_candidates_per_slot": 3, "model_calls": 0, "gold_used": False, "hard_rejection_rules": ["provenance_invalid", "explicit_period_conflict", "explicit_unit_conflict"], "ranking_weights": r1c.__dict__.get("SHORTLIST_WEIGHTS", None) or {"normalized_metric_overlap": 5.0, "raw_metric_overlap": 3.0, "row_header_overlap": 3.0, "period_exactness": 4.0, "scope_overlap": 3.0, "statement_table_overlap": 2.0, "section_overlap": 1.0}})
    write_json(OUT / "comparative-task-contract.json", {"formulation": SHORTLIST_FORMULATION, "decisions": ["SELECT", "NONE", "AMBIGUOUS"], "one_provider_request_per_query": True, "query_level_status": False, "reasoning": False, "additional_properties": False, "model": MODEL})
    _, _, cache = build_shortlist_artifact(frozen, source_map, cohort)
    audit = shortlist_recall_audit(frozen, source_map, cohort, cache)
    if not audit["hard_gate"]:
        # The recall gate is evaluated before any provider call.  Freeze a
        # fail-closed policy even on this early-stop path so the artifact
        # explicitly records that no shortlist decision is released.
        selective_freeze([], reason="R6 shortlist recall gate failure")
        decision = {
            "gate": "NF-V2-03-R6",
            "base_commit": BASE_COMMIT,
            "model": MODEL,
            "formulation": SHORTLIST_FORMULATION,
            "shortlist_hard_gate": False,
            "shortlist_gate_thresholds": {"direct": "20/21", "calculation": "12/12"},
            "shortlist_direct_retention": f"{audit['direct']['gold_compatible_candidate_retained']}/21",
            "shortlist_calculation_retention": f"{audit['calculation']['gold_compatible_candidate_retained']}/12",
            "shortlist_mean": audit["shortlist_size"]["mean"],
            "shortlist_median": audit["shortlist_size"]["median"],
            "shortlist_p95": audit["shortlist_size"]["p95"],
            "shortlist_distribution": audit["shortlist_size"]["distribution"],
            "shortlist_unbindable_zero_eligible": f"{audit['unbindable']['zero_eligible']}/7",
            "model_calls": 0,
            "provider_calls_per_query": 0,
            "gold_reads_before_prediction_seal": 0,
            "synthetic": "not_run",
            "diagnostic": "not_run",
            "indistinguishable_safe": "not_run",
            "unbindable_false_binding": "not_run",
            "token_delta": "not_run",
            "structural_violations": "not_run",
            "binder_formulation_effective": False,
            "binder_model_frozen": MODEL,
            "binder_task_formulation_frozen": False,
            "formal_attempt_9": "not_run",
            "dominant_failure": "shortlist_recall_failure",
            "next_gate": "v2_03_selective_binder_freeze_review",
            "production_default": "V1",
            "production_switch_allowed": False,
        }
        write_json(OUT / "decision.json", decision)
        write_json(OUT / "README.md", {"gate": "NF-V2-03 R6", "external_model_review": "cancelled_by_design", "reason": "shortlist hard gate failed before provider calls", "decision": decision, "shortlist_audit": audit, "selective_freeze_policy": "selective-freeze-policy.json"})
        print(json.dumps(decision, sort_keys=True))
        return 3
    synthetic = synthetic_run()
    if not synthetic["pass"]:
        selective_freeze([], reason="R6 comparative synthetic gate failure")
        decision = {"gate": "NF-V2-03-R6", "base_commit": BASE_COMMIT, "model": MODEL, "formulation": SHORTLIST_FORMULATION, "shortlist_hard_gate": True, "synthetic": f"{synthetic['semantic_correct']}/{synthetic['semantic_total']}", "synthetic_calculation": f"{synthetic['calculation_groups_correct']}/{synthetic['calculation_groups_total']}", "binder_formulation_effective": False, "binder_model_frozen": MODEL, "binder_task_formulation_frozen": False, "formal_attempt_9": "not_run", "dominant_failure": "comparative_synthetic_gate_failure", "next_gate": "v2_03_selective_binder_freeze_review", "production_default": "V1", "production_switch_allowed": False}
        write_json(OUT / "decision.json", decision)
        write_json(OUT / "README.md", {"gate": "NF-V2-03 R6", "external_model_review": "cancelled_by_design", "reason": "comparative synthetic gate failed", "decision": decision, "shortlist_audit": audit, "synthetic": synthetic, "selective_freeze_policy": "selective-freeze-policy.json"})
        print(json.dumps(decision, sort_keys=True))
        return 3
    rows, scored = diagnostic_run(frozen, source_map, cohort, cache)
    direct = scored["direct"]
    calc = scored["calculation"]
    indist = scored["indistinguishable"]
    unbindable = scored["unbindable"]
    structural = all(row["provider_response_success"] and row["structured_output_success"] and row["dto_valid"] and row["adapter_valid"] and row["binding_validator_pass"] for row in rows)
    runtime = read_json(OUT / "token-latency-cost.json")
    pass_gate = bool(direct["correct"] >= 15 and calc["correct_operand_slots"] >= 9 and indist["appropriate_abstention"] >= 5 and unbindable["false_binding"] <= 1 and structural)
    selective_pass = bool(direct["correct"] >= 13 and calc["correct_operand_slots"] >= 8 and indist["appropriate_abstention"] >= 5 and unbindable["false_binding"] == 0 and structural)
    decision = {"gate": "NF-V2-03-R6", "base_commit": BASE_COMMIT, "model": MODEL, "formulation": SHORTLIST_FORMULATION, "shortlist_direct_retention": f"{audit['direct']['gold_compatible_candidate_retained']}/21", "shortlist_calculation_retention": f"{audit['calculation']['gold_compatible_candidate_retained']}/12", "shortlist_mean": audit["shortlist_size"]["mean"], "shortlist_median": audit["shortlist_size"]["median"], "shortlist_p95": audit["shortlist_size"]["p95"], "shortlist_unbindable_zero_eligible": f"{audit['unbindable']['zero_eligible']}/7", "synthetic": f"{synthetic['semantic_correct']}/{synthetic['semantic_total']}", "diagnostic_direct": f"{direct['correct']}/21", "diagnostic_calculation": f"{calc['correct_operand_slots']}/12", "diagnostic_all_operands": f"{calc['all_operands_correct']}/6", "indistinguishable_safe": f"{indist['appropriate_abstention']}/6", "unbindable_false_binding": f"{unbindable['false_binding']}/7", "provider_calls_per_query": 1, "token_delta": runtime.get("token_delta_shortlist_comparative_minus_pairwise", {}), "structural_violations": 0 if structural else 1, "binder_formulation_effective": True if pass_gate else ("selective_safe" if selective_pass else False), "binder_model_frozen": MODEL, "binder_task_formulation_frozen": SHORTLIST_FORMULATION if pass_gate or selective_pass else False, "formal_attempt_9": "pending" if pass_gate or selective_pass else "not_run", "dominant_failure": "none" if pass_gate else ("selective_safety_failure" if not selective_pass else "comparative_selection_quality"), "next_gate": "formal_attempt_9" if pass_gate or selective_pass else "v2_03_selective_binder_freeze_review", "production_default": "V1", "production_switch_allowed": False}
    write_json(OUT / "direct-results.json", direct)
    write_json(OUT / "calculation-results.json", calc)
    write_json(OUT / "indistinguishable-safety.json", indist)
    write_json(OUT / "unbindable-safety.json", unbindable)
    write_json(OUT / "formulation-ablation.json", {"global": {"direct": "8/21", "calculation": "1/12"}, "slotwise": {"direct": "9/21", "calculation": "5/12", "indistinguishable_abstention": "0/6", "unbindable_false_binding": "6/7"}, "pairwise": {"direct": "7/21", "calculation": "1/12", "indistinguishable_abstention": "2/6", "unbindable_false_binding": "6/7"}, "shortlist_comparative": {"direct": f"{direct['correct']}/21", "calculation": f"{calc['correct_operand_slots']}/12", "indistinguishable_abstention": f"{indist['appropriate_abstention']}/6", "unbindable_false_binding": f"{unbindable['false_binding']}/7"}})
    if not pass_gate:
        selective_freeze(rows, reason="R6 comparative safety or quality gate not met")
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {"gate": "NF-V2-03 R6", "external_model_review": "cancelled_by_design", "reason": "cost_and_project_scope", "decision": decision, "shortlist_audit": audit, "synthetic": synthetic, "scored": scored, "formal_attempt_9": "not_run"})
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
