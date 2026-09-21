#!/usr/bin/env python3
"""NF-V2-03 R0E transport resilience gate and formal Attempt 3."""

from __future__ import annotations

import gzip
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.evidence.binder_provider import BailianBinderProvider  # noqa: E402
from rag_v2.evidence.binder_service import BinderRequest, SemanticBinderService  # noqa: E402
from rag_v2.evidence.transport_retry import (  # noqa: E402
    TransportRetryPolicy,
    TransportRetryResult,
    bind_with_transport_retry,
    retry_contract_dict,
)
from scripts.evaluation import run_nf_v2_03_formal_semantic_evidence_binder as formal  # noqa: E402
from scripts.evaluation import run_nf_v2_03_semantic_evidence_binder as legacy  # noqa: E402


BASE_COMMIT = "0b744b990b01d8772897d43b07cf527b3a3c9051"
MODEL = "qwen3.7-plus"
POLICY = TransportRetryPolicy()
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r0e-transport-resilience"
FORMAL_OUT = ROOT / "artifacts/evaluation/nf-v2-03-formal-attempt-3"
Q2 = "aapl_fy2025_002"
QUESTION_TOTAL = 72


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def percentile(values: list[float], q: float = 0.95) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * q) - 1))]


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provider_config() -> dict[str, Any]:
    config = legacy.load_config()
    config["base_url"] = os.getenv("V2_SUPERVISOR_BASE_URL", "").strip()
    if config["model"] != MODEL:
        raise RuntimeError("V2_SUPERVISOR_MODEL must be qwen3.7-plus for this run")
    if config["max_retries"] != 0 or not config["base_url"]:
        raise RuntimeError("frozen provider configuration mismatch")
    return config


def make_provider(config: dict[str, Any]) -> BailianBinderProvider:
    return BailianBinderProvider(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_name=MODEL,
        enable_thinking=False,
        temperature=0.0,
        timeout=180.0,
        max_retries=0,
    )


def audit_row(request: BinderRequest, result: TransportRetryResult, *, pass_number: int | None = None, sequence: int | None = None) -> dict[str, Any]:
    final = result.run
    metadata = final.metadata.to_dict() if final.metadata else {}
    row = {
        "question_id": request.question_id,
        "fact_count": len(request.facts),
        "request_sha256": result.request_sha256,
        "request_sha_match": result.retry_request_sha_matches_original,
        "attempt_1": result.attempt_1.to_dict(),
        "attempt_2": result.attempt_2.to_dict() if result.attempt_2 else {"attempted": False, "retry_reason": result.attempt_1.failure_class},
        "recovered_by_transport_retry": result.recovered_by_transport_retry,
        "semantic_response_count": result.semantic_response_count,
        "final_provider_completion": result.final_provider_completion,
        "final_structured_output_success": bool(metadata.get("structured_output_success")),
        "final_schema_valid": bool(final.schema_valid),
        "final_binding_validator_pass": bool(final.validation.passed),
        "final_latency_ms": metadata.get("latency_ms"),
        "pass_number": pass_number,
        "sequence": sequence,
    }
    return row


def run_sequence(provider: BailianBinderProvider, service: SemanticBinderService, request: BinderRequest, *, pass_number: int | None = None, sequence: int | None = None) -> tuple[TransportRetryResult, dict[str, Any]]:
    result = bind_with_transport_retry(service, request, policy=POLICY)
    return result, audit_row(request, result, pass_number=pass_number, sequence=sequence)


def reliability(rows: list[dict[str, Any]], *, model_required: int | None = None) -> dict[str, Any]:
    attempted_first = [row for row in rows if row["fact_count"] > 0 and row["attempt_1"]["attempted"]]
    first_success = sum(int(row["attempt_1"]["provider_success"] and row["attempt_1"]["structured_output_success"] and row["attempt_1"]["schema_valid"]) for row in attempted_first)
    first_failures = len(attempted_first) - first_success
    retry_attempts = sum(int(row["attempt_2"].get("attempted", False)) for row in rows)
    retry_recovered = sum(int(row["recovered_by_transport_retry"]) for row in rows)
    retry_failed = sum(int(row["attempt_2"].get("attempted", False) and not row["recovered_by_transport_retry"]) for row in rows)
    final_completion = sum(int(row["final_provider_completion"]) for row in rows)
    schema_completed = sum(int(row["final_provider_completion"] and row["final_schema_valid"]) for row in rows)
    return {
        "model_required_queries": model_required if model_required is not None else len(rows),
        "first_attempt_success": first_success,
        "first_attempt_transport_failures": first_failures,
        "first_attempt_timeout": sum(int(row["attempt_1"].get("failure_class") in {"APITimeoutError", "ReadTimeout", "ConnectTimeout"}) for row in attempted_first),
        "first_attempt_connection_failure": sum(int(row["attempt_1"].get("failure_class") == "APIConnectionError") for row in attempted_first),
        "first_attempt_429": sum(int(row["attempt_1"].get("failure_class") == "HTTP_429") for row in attempted_first),
        "first_attempt_5xx": sum(int(row["attempt_1"].get("failure_class") in {"HTTP_502", "HTTP_503", "HTTP_504"}) for row in attempted_first),
        "transport_retry_attempts": retry_attempts,
        "transport_retry_recovered": retry_recovered,
        "transport_retry_failed": retry_failed,
        "final_provider_completion": final_completion,
        "schema_valid_among_completed": schema_completed,
        "semantic_response_count_max": max((int(row["semantic_response_count"]) for row in rows), default=0),
        "first_attempt_reliability_percent": round(100 * first_success / len(attempted_first), 4) if attempted_first else 0.0,
        "final_transport_assisted_completion_percent": round(100 * final_completion / len(rows), 4) if rows else 0.0,
    }


def transport_contract_seal() -> dict[str, Any]:
    contract = retry_contract_dict(POLICY)
    contract.update({"gate": "NF-V2-03-R0E", "model": MODEL, "provider": "Alibaba Bailian", "production_switch_allowed": False})
    write_json(OUT / "transport-resilience-contract.json", contract)
    (OUT / "transport-resilience-contract.sha256").write_text(sha256_file(OUT / "transport-resilience-contract.json") + "\n", encoding="utf-8")
    return contract


def exact_q2_resilience(config: dict[str, Any], request: BinderRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence in range(1, 21):
        provider = make_provider(config)
        service = SemanticBinderService(provider)
        try:
            _, row = run_sequence(provider, service, request, sequence=sequence)
            rows.append(row)
        finally:
            provider.close()
    summary = reliability(rows)
    summary["final_completion_target_met"] = summary["final_provider_completion"] == 20
    summary["schema_valid_target_met"] = summary["schema_valid_among_completed"] == 20
    return rows, {"summary": summary, "rows": rows, "gold_reads": 0, "semantic_inspection": False}


def first10_stability(config: dict[str, Any], requests: dict[str, BinderRequest]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question_ids = sorted(requests)[:10]
    rows: list[dict[str, Any]] = []
    for pass_number in range(1, 4):
        provider = make_provider(config)
        service = SemanticBinderService(provider)
        try:
            for sequence, question_id in enumerate(question_ids, 1):
                _, row = run_sequence(provider, service, requests[question_id], pass_number=pass_number, sequence=sequence)
                rows.append(row)
                if not row["final_provider_completion"]:
                    return rows, {"summary": reliability(rows), "rows": rows, "gold_reads": 0, "semantic_inspection": False, "stopped_on_failure": True}
        finally:
            provider.close()
    summary = reliability(rows)
    summary["final_completion_target_met"] = summary["final_provider_completion"] == 30
    summary["structured_output_target_met"] = all(row["final_structured_output_success"] for row in rows) and len(rows) == 30
    summary["schema_valid_target_met"] = all(row["final_schema_valid"] for row in rows) and len(rows) == 30
    summary["binding_validator_target_met"] = all(row["final_binding_validator_pass"] for row in rows) and len(rows) == 30
    return rows, {"summary": summary, "rows": rows, "gold_reads": 0, "semantic_inspection": False}


def formal_attempt_3(config: dict[str, Any], frozen: dict[str, Any], verification: dict[str, Any]) -> int:
    FORMAL_OUT.mkdir(parents=True, exist_ok=True)
    write_json(FORMAL_OUT / "formal-run-config.json", {
        "gate": "NF-V2-03",
        "attempt_number": 3,
        "base_commit": BASE_COMMIT,
        "provider": "Alibaba Bailian",
        "model": MODEL,
        "provider_role": "evidence_binder",
        "model_role": "strong_general_llm",
        "thinking": False,
        "temperature": 0.0,
        "max_retries": 0,
        "http_timeout_seconds": 180,
        "concurrency": 1,
        "transport_retry_budget": 1,
        "semantic_attempt_budget": 1,
        "retry_delay_seconds": 3,
        "prompt_sha256": verification["binder_prompt_sha256"],
        "schema_sha256": verification["binder_schema_sha256"],
        "supervisor_prediction_sha256": verification["supervisor_prediction_sha256"],
        "top20_fact_artifact_sha256": verification["top20_fact_artifact_sha256"],
        "transport_resilience_contract_sha256": sha256_file(OUT / "transport-resilience-contract.json"),
        "gold_reads_before_prediction_seal": 0,
        "production_default": "V1",
        "production_switch_allowed": False,
    })
    write_json(FORMAL_OUT / "formal-attempt-history.json", {
        "attempt_1": {"status": "invalidated", "reason": "provider_read_timeout", "gold_reads": 0, "semantic_metrics": "none"},
        "attempt_2": {"status": "invalidated", "reason": "provider_read_timeout", "gold_reads": 0, "semantic_metrics": "none"},
        "attempt_3": {"status": "transport_resilience_contract_pre_frozen", "gold_reads_before_seal": 0},
    })
    provider = make_provider(config)
    service = SemanticBinderService(provider)
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    started_all = time.perf_counter()
    try:
        for call_index, question_id in enumerate(sorted(frozen["requests"]), 1):
            request = frozen["requests"][question_id]
            result, audit = run_sequence(provider, service, request, sequence=call_index)
            audits.append(audit)
            if not result.final_provider_completion and not request.facts:
                row = result.run.to_dict()
            elif not result.final_provider_completion:
                failure = {"question_id": question_id, "call_index": call_index, "transport_audit": audit, "gold_reads_before_prediction_seal": 0}
                break
            else:
                row = result.run.to_dict()
            flags = formal.leak_flags(result.run.raw_response)
            row.update({
                "call_index": call_index,
                "question": request.question,
                "intent": request.plan.intent.value,
                "operation": request.plan.operation,
                "required_slots": [slot.to_dict() for slot in request.plan.required_slots],
                "fact_count": len(request.facts),
                "candidate_ranks": sorted({fact.get("candidate_rank") for fact in request.facts}),
                "answer_leakage": flags["answer_leakage"],
                "invented_numeric_values": flags["invented_numeric_values"],
                "calculation_outputs": flags["calculation_outputs"],
                "invented_fact_ids": flags["invented_fact_ids"],
                "invented_source_ids": flags["invented_source_ids"],
                "new_slots": flags["new_slots"],
                "role_mutation": flags.get("role_mutation", 0),
                "transport_audit": audit,
                "semantic_response_count": result.semantic_response_count,
                "recovered_by_transport_retry": result.recovered_by_transport_retry,
                "request_sha_match": result.retry_request_sha_matches_original,
            })
            predictions.append(row)
    finally:
        provider.close()
    if failure is not None or len(predictions) != QUESTION_TOTAL:
        write_json(FORMAL_OUT / "formal-failure.json", failure or {"failure": "prediction_count_mismatch", "predictions": len(predictions), "gold_reads_before_prediction_seal": 0})
        write_json(FORMAL_OUT / "transport-reliability.json", {"formal_run_complete": False, "audit_rows": audits, "summary": reliability(audits, model_required=sum(int(bool(row["fact_count"])) for row in audits)), "gold_reads_before_prediction_seal": 0})
        write_json(FORMAL_OUT / "decision.json", {"gate": "NF-V2-03", "attempt": 3, "formal_run_complete": False, "prediction_sealed": False, "gold_reads_before_prediction_seal": 0, "semantic_evidence_binder_effective": "not_evaluated", "semantic_binder_frozen": False, "dominant_failure": "provider_transport_failure", "production_switch_allowed": False, "next_gate": "nf_v2_03_transport_resilience_failure_review"})
        return 1
    prediction_path = FORMAL_OUT / "binder-predictions.jsonl.gz"
    write_jsonl_gz(prediction_path, predictions)
    prediction_sha = sha256_file(prediction_path)
    seal = {
        "gate": "NF-V2-03",
        "attempt": 3,
        "sealed": True,
        "predictions_written": len(predictions),
        "questions_expected": QUESTION_TOTAL,
        "prediction_sha256": prediction_sha,
        "binder_prompt_sha256": verification["binder_prompt_sha256"],
        "binder_schema_sha256": verification["binder_schema_sha256"],
        "transport_resilience_contract_sha256": sha256_file(OUT / "transport-resilience-contract.json"),
        "gold_reads_before_prediction_seal": 0,
        "sealed_before_gold": True,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
    }
    write_json(FORMAL_OUT / "binder-prediction-seal.json", seal)
    if sha256_file(prediction_path) != prediction_sha:
        raise RuntimeError("Attempt 3 prediction seal verification failed")
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in legacy.LABELS.read_text(encoding="utf-8").splitlines()) if row}
    scored = legacy.score_predictions(frozen, predictions, labels)
    metadata_rows = [row["metadata"] for row in predictions if row.get("metadata")]
    eligible = [row for row in predictions if not row["skipped_no_fact_supply"]]
    provider_success = sum(int(bool(row.get("metadata") and row["metadata"].get("provider_response_success"))) for row in eligible)
    structured = sum(int(row.get("binding_schema_valid") and bool(row.get("metadata") and row["metadata"].get("structured_output_success"))) for row in eligible)
    schema_valid = sum(int(row.get("binding_schema_valid")) for row in eligible)
    validator_pass = sum(int(row["binding_validator_pass"]) for row in predictions)
    latencies = [float(row.get("latency_ms") or 0.0) for row in metadata_rows]
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in metadata_rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in metadata_rows)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in metadata_rows)
    attempt_rows = [row for audit in audits for row in ([audit["attempt_1"]] if audit["attempt_1"].get("attempted") else []) + ([audit["attempt_2"]] if audit["attempt_2"].get("attempted") else [])]
    reliability_summary = reliability(audits, model_required=len(eligible))
    largest = max(metadata_rows, key=lambda row: int(row.get("input_tokens") or 0), default=None)
    write_json(FORMAL_OUT / "transport-reliability.json", {"summary": reliability_summary, "rows": audits, "total_provider_attempts": len(attempt_rows), "total_semantic_responses": sum(int(row["semantic_response_count"]) for row in audits), "retry_overhead_wall_time_seconds": sum(3 for row in audits if row["attempt_2"].get("attempted")), "gold_reads_before_prediction_seal": 0})
    write_json(FORMAL_OUT / "binding-validator-results.json", {"rows": [{"question_id": row["question_id"], "binding_validator_pass": row["binding_validator_pass"], "final_status": row["final_binding_status"], "reasons": row["validation_reasons"]} for row in predictions], "passed": validator_pass, "failed": len(predictions) - validator_pass})
    write_json(FORMAL_OUT / "binding-status-metrics.json", {"query_status": scored["query_status"], "slot_status": scored["slot_status"], "binder_calls": len(eligible), "queries_skipped_no_supply": len(predictions) - len(eligible)})
    write_json(FORMAL_OUT / "direct-fact-binding.json", scored["direct"])
    write_json(FORMAL_OUT / "calculation-binding.json", scored["calculation"])
    write_json(FORMAL_OUT / "multi-evidence-binding.json", scored["multi"])
    write_json(FORMAL_OUT / "historical-46-binding.json", {"top20_fact_supply": "42/46", **scored["historical"]})
    write_json(FORMAL_OUT / "first-loss-funnel.json", {"direct_fact": {"questions": 56, "supervisor_slot_valid": 56, "financial_fact_supply": scored["direct"]["strict_bindable"], "strict_bindable": scored["direct"]["strict_bindable"], "binder_bound": scored["direct"]["bound"], "strict_correct": scored["direct"]["strict_complete"]}, "calculation": {"questions": 11, "supervisor_operands": 11, "all_required_source_supply": "8/11", "fact_supply_complete": "6/11", "binder_all_operands": scored["calculation"]["strict_complete"], "strict_all_operand": scored["calculation"]["strict_complete"]}, "multi_evidence": {"questions": 5, "supervisor_slots": 5, "complete_supply": "4/5", "binder_complete": scored["multi"]["bound"], "strict_complete": scored["multi"]["strict_complete"]}})
    failure_keys = [f"EB{i}_{name}" for i, name in enumerate(["correct", "no_relevant_fact_in_packet", "correct_source_present_but_no_matching_financial_fact", "metric_semantic_mismatch", "period_mismatch", "scope_or_segment_ambiguity", "multiple_statement_ambiguity", "multi_slot_association_error", "wrong_fact_selected", "wrong_physical_source_selected", "model_missing_despite_bindable_fact", "model_ambiguous_despite_unique_fact", "schema_or_binding_validator_failure", "supervisor_required_slot_defect", "other"])]
    write_json(FORMAL_OUT / "failure-taxonomy.json", {key: scored["failure_counts"].get(key, 0) for key in failure_keys})
    safety = {"false_binding_queries": scored["false_binding_queries"], "false_binding_slots": scored["false_binding_slots"], "invented_fact_ids": sum(row["invented_fact_ids"] for row in predictions), "invented_source_ids": sum(row["invented_source_ids"] for row in predictions), "new_slots": sum(row["new_slots"] for row in predictions), "role_mutation": sum(row["role_mutation"] for row in predictions), "answer_leakage": sum(row["answer_leakage"] for row in predictions), "calculation_leakage": sum(row["calculation_outputs"] for row in predictions)}
    write_json(FORMAL_OUT / "safety-analysis.json", safety)
    write_json(FORMAL_OUT / "latency-token-cost.json", {"total_provider_attempts": len(attempt_rows), "total_semantic_responses": sum(int(row["semantic_response_count"]) for row in audits), "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens, "average_semantic_latency_ms": statistics.mean(latencies) if latencies else 0.0, "p50_semantic_latency_ms": statistics.median(latencies) if latencies else 0.0, "p95_semantic_latency_ms": percentile(latencies), "max_semantic_latency_ms": max(latencies) if latencies else 0.0, "retry_overhead_wall_time_seconds": sum(3 for row in audits if row["attempt_2"].get("attempted")), "largest_input_question": largest.get("question_id") if largest else None, "largest_input_tokens": largest.get("input_tokens") if largest else None, "largest_input_latency_ms": largest.get("latency_ms") if largest else None, "estimated_cost": "not_configured", "formal_wall_time_ms": round((time.perf_counter() - started_all) * 1000.0, 3)})
    effective = bool(structured >= 0.98 * len(eligible) and validator_pass >= 0.98 * QUESTION_TOTAL and not any(safety[key] for key in ("false_binding_queries", "invented_fact_ids", "invented_source_ids", "answer_leakage", "calculation_leakage")) and scored["direct"]["strict_complete"] >= 40 and scored["calculation"]["strict_complete"] >= 6 and scored["multi"]["strict_complete"] >= 4)
    partial = bool(not effective and structured >= 0.95 * len(eligible) and validator_pass >= 0.95 * QUESTION_TOTAL and not any(safety[key] for key in ("false_binding_queries", "invented_fact_ids", "invented_source_ids", "answer_leakage", "calculation_leakage")) and scored["direct"]["strict_complete"] >= 36 and scored["calculation"]["strict_complete"] >= 5 and scored["multi"]["strict_complete"] >= 3)
    decision = {"gate": "NF-V2-03", "attempt": 3, "base_commit": BASE_COMMIT, "provider": "Alibaba Bailian", "model": MODEL, "formal_run_complete": True, "prediction_sealed": True, "gold_reads_before_prediction_seal": 0, "binder_calls": len(eligible), "provider_attempts": len(attempt_rows), "semantic_responses": sum(int(row["semantic_response_count"]) for row in audits), "provider_response_success": provider_success, "structured_output_success": structured, "schema_valid": schema_valid, "binding_validator_pass": validator_pass, "transport_reliability": reliability_summary, "direct_fact_strict_bindable": scored["direct"]["strict_bindable"], "direct_fact_strict_complete": scored["direct"]["strict_complete"], "direct_fact_success_given_bindable": scored["direct"]["success_given_bindable"], "calculation_fact_supply_complete": "6/11", "calculation_all_operand_bound": scored["calculation"]["strict_complete"], "calculation_success_given_complete_supply": scored["calculation"]["success_given_bindable"], "multi_evidence_complete_supply": "4/5", "multi_evidence_complete_bound": scored["multi"]["strict_complete"], "multi_evidence_success_given_complete_supply": scored["multi"]["success_given_bindable"], "historical_46_fact_supply": "42/46", "historical_46_strict_complete": scored["historical"]["strict_complete"], "false_binding_queries": safety["false_binding_queries"], "false_binding_slots": safety["false_binding_slots"], "invented_fact_ids": safety["invented_fact_ids"], "invented_source_ids": safety["invented_source_ids"], "new_slots": safety["new_slots"], "role_mutation": safety["role_mutation"], "answer_leakage": safety["answer_leakage"], "calculation_leakage": safety["calculation_leakage"], "semantic_evidence_binder_effective": True if effective else ("partial" if partial else False), "semantic_binder_frozen": effective, "dominant_failure": "none" if effective else ("coverage" if partial else "binding_safety_or_semantics"), "production_default": "V1", "production_switch_allowed": False, "next_gate": "v2_04_missing_slot_retrieval_repair" if effective else "v2_03_semantic_binder_failure_review"}
    write_json(FORMAL_OUT / "decision.json", decision)
    (FORMAL_OUT / "README.md").write_text("# NF-V2-03 Formal Attempt 3\n\nTransport-only retry policy was sealed before this run. Predictions were sealed before Gold scoring. Production remains V1.\n", encoding="utf-8")
    return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if os.getenv("V2_SUPERVISOR_MODEL", "").strip() != MODEL:
        write_json(OUT / "decision.json", {"gate": "NF-V2-03-R0E", "formal_evaluation_status": "configuration_blocked", "reason": "V2_SUPERVISOR_MODEL must be qwen3.7-plus", "production_switch_allowed": False})
        return 2
    config = provider_config()
    frozen = legacy.load_frozen_inputs()
    verification = formal.verify_frozen_inputs(frozen)
    requests = frozen["requests"]
    write_json(OUT / "transport-retry-tests.json", {"policy": retry_contract_dict(POLICY), "unit_tests_required": ["attempt_1_success_no_retry", "transport_failure_recovery", "schema_and_semantic_status_no_retry", "request_sha_mismatch_fail_closed"], "gold_reads": 0})
    if len(sys.argv) > 1 and sys.argv[1] == "--formal-attempt-3":
        transport_contract_seal()
        write_json(OUT / "decision.json", {
            "gate": "NF-V2-03-R0E",
            "base_commit": BASE_COMMIT,
            "stability_replay": "skipped_by_explicit_user_instruction",
            "transport_contract_sealed": True,
            "transport_contract_sha256": sha256_file(OUT / "transport-resilience-contract.json"),
            "formal_attempt_3_started": True,
            "gold_reads": 0,
            "production_switch_allowed": False,
        })
        return formal_attempt_3(config, frozen, verification)
    q2_request = requests[Q2]
    _, q2_artifact = exact_q2_resilience(config, q2_request)
    write_json(OUT / "exact-request-20.json", q2_artifact)
    if q2_artifact["summary"]["final_provider_completion"] < 19 or q2_artifact["summary"]["schema_valid_among_completed"] < 19:
        write_json(OUT / "decision.json", {"gate": "NF-V2-03-R0E", "base_commit": BASE_COMMIT, "exact_q2": q2_artifact["summary"], "transport_contract_sealed": False, "binder_provider_contract_ready": False, "formal_attempt_3_executed": False, "gold_reads": 0, "production_switch_allowed": False, "next_gate": "nf_v2_03_transport_resilience_failure_review"})
        return 1
    first10_rows, first10_artifact = first10_stability(config, requests)
    write_json(OUT / "first10-three-pass.json", first10_artifact)
    if first10_artifact["summary"]["final_provider_completion"] != 30 or not first10_artifact["summary"].get("structured_output_target_met") or not first10_artifact["summary"].get("schema_valid_target_met") or not first10_artifact["summary"].get("binding_validator_target_met"):
        write_json(OUT / "decision.json", {"gate": "NF-V2-03-R0E", "base_commit": BASE_COMMIT, "exact_q2": q2_artifact["summary"], "first10": first10_artifact["summary"], "transport_contract_sealed": False, "binder_provider_contract_ready": False, "formal_attempt_3_executed": False, "gold_reads": 0, "production_switch_allowed": False, "next_gate": "nf_v2_03_transport_resilience_failure_review"})
        return 1
    transport_contract_seal()
    write_json(OUT / "decision.json", {"gate": "NF-V2-03-R0E", "base_commit": BASE_COMMIT, "exact_q2": q2_artifact["summary"], "first10": first10_artifact["summary"], "transport_contract_sealed": True, "transport_contract_sha256": sha256_file(OUT / "transport-resilience-contract.json"), "binder_provider_contract_ready": True, "formal_attempt_3_started": True, "gold_reads": 0, "production_switch_allowed": False})
    return formal_attempt_3(config, frozen, verification)


if __name__ == "__main__":
    raise SystemExit(main())
