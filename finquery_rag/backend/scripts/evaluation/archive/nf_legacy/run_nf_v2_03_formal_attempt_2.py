#!/usr/bin/env python3
"""NF-V2-03 formal Semantic Evidence Binder evaluation, attempt 2."""

from __future__ import annotations

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

from scripts.evaluation import run_nf_v2_03_formal_semantic_evidence_binder as formal  # noqa: E402
from scripts.evaluation import run_nf_v2_03_semantic_evidence_binder as legacy  # noqa: E402
from rag_v2.evidence.binder_provider import BailianBinderProvider  # noqa: E402
from rag_v2.evidence.binder_service import BinderRun, SemanticBinderService  # noqa: E402


OUT = ROOT / "artifacts/evaluation/nf-v2-03-formal-attempt-2"
BASE_COMMIT = "b64ccd9645fffe7a48e4aa189d7f6f40d082c138"
MODEL = "qwen3.7-max"
QUESTION_TOTAL = 72


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: list[float], q: float = 0.95) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * q) - 1))]


def history(attempt_2: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_1": {
            "base_commit": "ce69c29",
            "calls_attempted": "2/72",
            "formal_run_complete": False,
            "failure": "intermittent_provider_read_timeout",
            "failed_question": "aapl_fy2025_002",
            "gold_reads": 0,
            "prediction_seal": "not_created",
            "valid_for_semantic_metrics": False,
        },
        "attempt_2": attempt_2,
    }


def failure_record(exc: BaseException, *, call_count: int) -> dict[str, Any]:
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        return {
            **details,
            "calls_attempted": details.get("call_index", call_count),
            "formal_run_complete": False,
            "gold_reads_before_prediction_seal": 0,
            "prediction_seal": "not_created",
            "semantic_scoring": False,
        }
    return {
        "formal_run_complete": False,
        "failure_class": "provider_or_runner_failure",
        "exception_type": type(exc).__name__,
        "exception_message": str(exc)[:500],
        "calls_attempted": call_count,
        "gold_reads_before_prediction_seal": 0,
        "prediction_seal": "not_created",
        "semantic_scoring": False,
    }


class FormalInfrastructureFailure(RuntimeError):
    def __init__(self, details: dict[str, Any]) -> None:
        self.details = details
        super().__init__(details.get("error", "formal provider failure"))


def run_formal_attempt(config: dict[str, Any], frozen: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run once per frozen request and stop on the first provider boundary failure."""
    provider = BailianBinderProvider(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_name=MODEL,
        enable_thinking=False,
        temperature=0.0,
        timeout=180.0,
        max_retries=0,
    )
    service = SemanticBinderService(provider)
    predictions: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    try:
        for index, question_id in enumerate(sorted(frozen["requests"]), 1):
            request = frozen["requests"][question_id]
            run: BinderRun = service.bind(request)
            metadata = run.metadata.to_dict() if run.metadata else None
            flags = formal.leak_flags(run.raw_response)
            row = run.to_dict()
            row.update({
                "call_index": index,
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
            })
            predictions.append(row)
            if metadata and (not metadata.get("provider_response_success") or not metadata.get("structured_output_success")):
                details = {
                    "question_id": question_id,
                    "call_index": index,
                    "fact_count": len(request.facts),
                    "input_tokens": metadata.get("input_tokens"),
                    "latency_ms": metadata.get("latency_ms"),
                    "exception_type": metadata.get("exception_type"),
                    "exception_cause_type": metadata.get("exception_cause_type"),
                    "exception_cause_message": metadata.get("exception_cause_message"),
                    "exception_chain": metadata.get("exception_chain", []),
                    "http_status": metadata.get("http_status"),
                    "finish_reason": metadata.get("finish_reason"),
                    "raw_content_length": metadata.get("raw_content_length"),
                    "error": metadata.get("error"),
                }
                raise FormalInfrastructureFailure(details)
    finally:
        provider.close()
    return predictions, {"formal_wall_time_ms": round((time.perf_counter() - started_all) * 1000.0, 3)}


def finalize_failure_artifacts() -> int:
    failure_path = OUT / "formal-failure.json"
    if not failure_path.exists():
        raise RuntimeError("formal-failure.json is required")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    attempt = {
        "base_commit": BASE_COMMIT,
        "calls_attempted": failure.get("calls_attempted", failure.get("call_index", 0)),
        "formal_run_complete": False,
        "prediction_seal": "not_created",
        "gold_reads_before_seal": 0,
        "semantic_scoring": False,
        "valid_for_semantic_metrics": False,
        "failure": failure,
    }
    write_json(OUT / "formal-attempt-history.json", history(attempt))
    decision = {
        "gate": "NF-V2-03",
        "evaluation_role": "development_shadow_v2_semantic_evidence_binder_attempt_2",
        "base_commit": BASE_COMMIT,
        "provider": "Alibaba Bailian",
        "model": MODEL,
        "formal_run_complete": False,
        "prediction_sealed": False,
        "binder_calls_attempted": failure.get("calls_attempted", failure.get("call_index", 0)),
        "failed_question": failure.get("question_id"),
        "gold_reads_before_prediction_seal": 0,
        "semantic_scoring": False,
        "semantic_evidence_binder_effective": "not_evaluated",
        "semantic_binder_frozen": False,
        "dominant_failure": "provider_read_timeout",
        "production_default": "V1",
        "production_switch_allowed": False,
        "next_gate": "nf_v2_03_provider_failure_review",
    }
    write_json(OUT / "decision.json", decision)
    (OUT / "README.md").write_text(
        "# NF-V2-03 Formal Semantic Evidence Binder — Attempt 2\n\n"
        "Attempt 2 stopped at the first provider read timeout, before prediction\n"
        "serialization/seal and before any Gold or semantic scoring. No retry was\n"
        "performed. The partial attempt is invalid for Binder capability metrics.\n"
        "Production remains V1.\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--finalize-failure" in sys.argv[1:]:
        return finalize_failure_artifacts()
    if os.getenv("V2_SUPERVISOR_MODEL", "").strip() != MODEL:
        decision = {
            "gate": "NF-V2-03",
            "evaluation_role": "development_shadow_v2_semantic_evidence_binder_attempt_2",
            "base_commit": BASE_COMMIT,
            "formal_run_complete": False,
            "formal_evaluation_status": "configuration_blocked",
            "reason": "V2_SUPERVISOR_MODEL must be qwen3.7-max",
            "production_switch_allowed": False,
        }
        write_json(OUT / "decision.json", decision)
        return 2

    config = legacy.load_config()
    config["base_url"] = os.getenv("V2_SUPERVISOR_BASE_URL", "").strip()
    if not config["base_url"]:
        raise RuntimeError("V2_SUPERVISOR_BASE_URL is not configured")
    if config["model"] != MODEL or config["max_retries"] != 0:
        raise RuntimeError("frozen Binder provider configuration mismatch")
    frozen = legacy.load_frozen_inputs()
    verification = formal.verify_frozen_inputs(frozen)
    write_json(OUT / "frozen-input-verification.json", verification)
    run_config = {
        "gate": "NF-V2-03",
        "evaluation_role": "development_shadow_v2_semantic_evidence_binder_attempt_2",
        "base_commit": BASE_COMMIT,
        "provider": "Alibaba Bailian",
        "provider_id": "bailian",
        "model": MODEL,
        "provider_role": "evidence_binder",
        "model_role": "strong_general_llm",
        "thinking": False,
        "temperature": 0.0,
        "max_retries": 0,
        "http_timeout_seconds": 180,
        "concurrency": 1,
        "prompt_sha256": verification["binder_prompt_sha256"],
        "schema_sha256": verification["binder_schema_sha256"],
        "supervisor_prediction_sha256": verification["supervisor_prediction_sha256"],
        "top20_fact_artifact_sha256": verification["top20_fact_artifact_sha256"],
        "top20_order_sha256": verification["top20_order_sha256"],
        "transport_gate": "NF-V2-03 R0C",
        "transport_contract_ready": True,
        "attempt_number": 2,
        "previous_attempt_invalidated": True,
        "production_default": "V1",
        "production_switch_allowed": False,
        "gold_reads_before_prediction_seal": 0,
        "credentials_persisted": False,
    }
    write_json(OUT / "formal-run-config.json", run_config)
    write_json(OUT / "formal-attempt-history.json", history({
        "base_commit": BASE_COMMIT,
        "calls_attempted": 0,
        "formal_run_complete": None,
        "prediction_seal": None,
        "gold_reads_before_seal": 0,
        "semantic_scoring": None,
        "valid_for_semantic_metrics": None,
    }))

    try:
        predictions, runtime = run_formal_attempt(config, frozen)
    except Exception as exc:
        record = failure_record(exc, call_count=0)
        write_json(OUT / "formal-failure.json", record)
        write_json(OUT / "formal-attempt-history.json", history({
            "base_commit": BASE_COMMIT,
            "calls_attempted": record["calls_attempted"],
            "formal_run_complete": False,
            "prediction_seal": "not_created",
            "gold_reads_before_seal": 0,
            "semantic_scoring": False,
            "valid_for_semantic_metrics": False,
            "failure": record,
        }))
        write_json(OUT / "decision.json", {
            "gate": "NF-V2-03",
            "evaluation_role": "development_shadow_v2_semantic_evidence_binder_attempt_2",
            "base_commit": BASE_COMMIT,
            "provider": "Alibaba Bailian",
            "model": MODEL,
            "formal_run_complete": False,
            "prediction_sealed": False,
            "gold_reads_before_prediction_seal": 0,
            "semantic_evidence_binder_effective": "not_evaluated",
            "dominant_failure": "provider_or_runner_failure",
            "production_switch_allowed": False,
            "next_gate": "nf_v2_03_provider_failure_review",
        })
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        return 1

    if len(predictions) != QUESTION_TOTAL:
        raise RuntimeError(f"formal prediction count mismatch: {len(predictions)}/{QUESTION_TOTAL}")
    prediction_path = OUT / "binder-predictions.jsonl.gz"
    formal.write_jsonl_gz(prediction_path, predictions)
    prediction_sha = formal.sha256_file(prediction_path)
    metadata_rows = [row["metadata"] for row in predictions if row.get("metadata")]
    eligible = [row for row in predictions if not row["skipped_no_fact_supply"]]
    binder_calls = len(eligible)
    seal = {
        "gate": "NF-V2-03",
        "attempt": 2,
        "sealed": True,
        "predictions_written": len(predictions),
        "questions_expected": QUESTION_TOTAL,
        "binder_model_calls": binder_calls,
        "max_calls_per_query": 1,
        "retry": 0,
        "concurrency": 1,
        "gold_reads_before_prediction_seal": 0,
        "reference_answer_reads_before_prediction_seal": 0,
        "prediction_sha256": prediction_sha,
        "supervisor_prediction_sha256": verification["supervisor_prediction_sha256"],
        "financial_facts_sha256": verification["top20_fact_artifact_sha256"],
        "binder_prompt_sha256": verification["binder_prompt_sha256"],
        "binder_schema_sha256": verification["binder_schema_sha256"],
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "validator_repair_calls": 0,
        "sealed_before_gold": True,
    }
    write_json(OUT / "binder-prediction-seal.json", seal)
    if formal.sha256_file(prediction_path) != prediction_sha:
        raise RuntimeError("prediction seal verification failed")

    # Gold is intentionally opened only after the prediction artifact is sealed.
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in legacy.LABELS.read_text(encoding="utf-8").splitlines()) if row}
    scored = legacy.score_predictions(frozen, predictions, labels)
    latencies = [float(row.get("latency_ms") or 0.0) for row in metadata_rows]
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in metadata_rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in metadata_rows)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in metadata_rows)
    reasoning_tokens = sum(int(row.get("reasoning_tokens") or 0) for row in metadata_rows)
    facts_per_call = [float(row["fact_count"]) for row in eligible]
    provider_success = sum(int(bool(row.get("metadata") and row["metadata"].get("provider_response_success"))) for row in eligible)
    structured = sum(int(row.get("binding_schema_valid") and bool(row.get("metadata") and row["metadata"].get("structured_output_success"))) for row in eligible)
    schema_valid = sum(int(row.get("binding_schema_valid")) for row in eligible)
    validator_pass = sum(int(row["binding_validator_pass"]) for row in predictions)
    largest = max((row for row in predictions if row.get("metadata")), key=lambda row: int(row["metadata"].get("input_tokens") or 0), default=None)

    write_json(OUT / "binding-validator-results.json", {
        "rows": [{"question_id": row["question_id"], "binding_validator_pass": row["binding_validator_pass"], "final_status": row["final_binding_status"], "reasons": row["validation_reasons"]} for row in predictions],
        "passed": validator_pass,
        "failed": len(predictions) - validator_pass,
    })
    write_json(OUT / "binding-status-metrics.json", {
        "query_status": scored["query_status"],
        "slot_status": scored["slot_status"],
        "binder_calls": binder_calls,
        "queries_skipped_no_supply": len(predictions) - len(eligible),
    })
    write_json(OUT / "direct-fact-binding.json", scored["direct"])
    write_json(OUT / "calculation-binding.json", scored["calculation"])
    write_json(OUT / "multi-evidence-binding.json", scored["multi"])
    write_json(OUT / "historical-46-binding.json", {"top20_fact_supply": "42/46", **scored["historical"]})
    write_json(OUT / "first-loss-funnel.json", {
        "direct_fact": {"questions": 56, "supervisor_slot_valid": 56, "financial_fact_supply": scored["direct"]["strict_bindable"], "strict_bindable": scored["direct"]["strict_bindable"], "binder_bound": scored["direct"]["bound"], "strict_correct": scored["direct"]["strict_complete"]},
        "calculation": {"questions": 11, "supervisor_operands": 11, "all_required_source_supply": "8/11", "fact_supply_complete": "6/11", "binder_all_operands": scored["calculation"]["strict_complete"], "strict_all_operand": scored["calculation"]["strict_complete"]},
        "multi_evidence": {"questions": 5, "supervisor_slots": 5, "complete_supply": "4/5", "binder_complete": scored["multi"]["bound"], "strict_complete": scored["multi"]["strict_complete"]},
    })
    failure_keys = [f"EB{i}_{name}" for i, name in enumerate(["correct", "no_relevant_fact_in_packet", "correct_source_present_but_no_matching_financial_fact", "metric_semantic_mismatch", "period_mismatch", "scope_or_segment_ambiguity", "multiple_statement_ambiguity", "multi_slot_association_error", "wrong_fact_selected", "wrong_physical_source_selected", "model_missing_despite_bindable_fact", "model_ambiguous_despite_unique_fact", "schema_or_binding_validator_failure", "supervisor_required_slot_defect", "other"])]
    write_json(OUT / "failure-taxonomy.json", {key: scored["failure_counts"].get(key, 0) for key in failure_keys})
    safety = {
        "false_binding_queries": scored["false_binding_queries"],
        "false_binding_slots": scored["false_binding_slots"],
        "invented_fact_ids": sum(row["invented_fact_ids"] for row in predictions),
        "invented_source_ids": sum(row["invented_source_ids"] for row in predictions),
        "new_slots": sum(row["new_slots"] for row in predictions),
        "role_mutation": sum(row["role_mutation"] for row in predictions),
        "answer_leakage": sum(row["answer_leakage"] for row in predictions),
        "calculation_leakage": sum(row["calculation_outputs"] for row in predictions),
        "numeric_hallucination": sum(row["invented_numeric_values"] for row in predictions),
    }
    write_json(OUT / "safety-analysis.json", safety)
    write_json(OUT / "latency-token-cost.json", {
        "binder_calls": binder_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "average_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies),
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "total_wall_time_ms": runtime["formal_wall_time_ms"],
        "estimated_cost": "not_configured",
        "median_facts_per_call": statistics.median(facts_per_call) if facts_per_call else 0.0,
        "p95_facts_per_call": percentile(facts_per_call),
        "max_facts_per_call": max(facts_per_call) if facts_per_call else 0.0,
        "largest_input_token_query": largest["question_id"] if largest else None,
        "largest_input_tokens": int(largest["metadata"].get("input_tokens") or 0) if largest else None,
        "largest_input_latency_ms": float(largest["metadata"].get("latency_ms") or 0.0) if largest else None,
    })
    structured_rate = structured / max(1, binder_calls)
    effective = bool(
        structured_rate >= 0.98
        and validator_pass >= 0.98 * QUESTION_TOTAL
        and not any(safety[key] for key in ("false_binding_queries", "invented_fact_ids", "invented_source_ids", "new_slots", "role_mutation", "answer_leakage", "calculation_leakage"))
        and scored["direct"]["strict_complete"] >= 40
        and scored["calculation"]["strict_complete"] >= 6
        and scored["multi"]["strict_complete"] >= 4
    )
    partial = bool(
        not effective
        and structured_rate >= 0.95
        and validator_pass >= 0.95 * QUESTION_TOTAL
        and not any(safety[key] for key in ("false_binding_queries", "invented_fact_ids", "invented_source_ids", "answer_leakage", "calculation_leakage"))
        and scored["direct"]["strict_complete"] >= 36
        and scored["calculation"]["strict_complete"] >= 5
        and scored["multi"]["strict_complete"] >= 3
    )
    decision = {
        "gate": "NF-V2-03",
        "evaluation_role": "development_shadow_v2_semantic_evidence_binder_attempt_2",
        "base_commit": BASE_COMMIT,
        "production_default": "V1",
        "production_switch_allowed": False,
        "provider": "Alibaba Bailian",
        "model": MODEL,
        "model_role": "strong_general_llm",
        "provider_role": "evidence_binder",
        "thinking": False,
        "temperature": 0.0,
        "max_retries": 0,
        "http_timeout_seconds": 180,
        "concurrency": 1,
        "formal_run_complete": True,
        "prediction_sealed": True,
        "gold_reads_before_prediction_seal": 0,
        "binder_calls": binder_calls,
        "queries_skipped_no_supply": len(predictions) - len(eligible),
        "provider_response_success": provider_success,
        "structured_output_success": structured,
        "schema_valid": schema_valid,
        "binding_validator_pass": validator_pass,
        "direct_fact_questions": 56,
        "direct_fact_strict_bindable": scored["direct"]["strict_bindable"],
        "direct_fact_strict_complete": scored["direct"]["strict_complete"],
        "direct_fact_success_given_bindable": scored["direct"]["success_given_bindable"],
        "calculation_questions": 11,
        "calculation_fact_supply_complete": "6/11",
        "calculation_all_operand_bound": scored["calculation"]["strict_complete"],
        "calculation_success_given_complete_supply": scored["calculation"]["success_given_bindable"],
        "multi_evidence_questions": 5,
        "multi_evidence_complete_supply": "4/5",
        "multi_evidence_complete_bound": scored["multi"]["strict_complete"],
        "multi_evidence_success_given_complete_supply": scored["multi"]["success_given_bindable"],
        "historical_46_fact_supply": "42/46",
        "historical_46_strict_complete": scored["historical"]["strict_complete"],
        "false_binding_queries": safety["false_binding_queries"],
        "false_binding_slots": safety["false_binding_slots"],
        "invented_fact_ids": safety["invented_fact_ids"],
        "invented_source_ids": safety["invented_source_ids"],
        "new_slots": safety["new_slots"],
        "role_mutation": safety["role_mutation"],
        "answer_leakage": safety["answer_leakage"],
        "calculation_leakage": safety["calculation_leakage"],
        "semantic_evidence_binder_effective": True if effective else ("partial" if partial else False),
        "semantic_binder_frozen": effective,
        "dominant_failure": "none" if effective else ("coverage" if partial else "binding_safety_or_semantics"),
        "next_gate": "v2_04_missing_slot_retrieval_repair" if effective else "v2_03_semantic_binder_failure_review",
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "validator_repair_calls": 0,
    }
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "formal-attempt-history.json", history({
        "base_commit": BASE_COMMIT,
        "calls_attempted": QUESTION_TOTAL,
        "formal_run_complete": True,
        "prediction_seal": "pass",
        "prediction_sha256": prediction_sha,
        "gold_reads_before_seal": 0,
        "semantic_scoring": True,
        "valid_for_semantic_metrics": True,
    }))
    (OUT / "README.md").write_text(
        "# NF-V2-03 Formal Semantic Evidence Binder — Attempt 2\n\n"
        "Attempt 1 was invalidated after a provider read timeout before seal.\n"
        "Attempt 2 ran the frozen 72-question Binder path once per eligible query,\n"
        "sealed predictions before opening Gold labels, and then scored the frozen\n"
        "semantic contracts. No retrieval, reranker, calculator, generator, or repair\n"
        "execution occurred. Production remains V1.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "gate": "NF-V2-03",
        "formal_run_complete": True,
        "binder_calls": binder_calls,
        "structured_output": structured,
        "schema_valid": schema_valid,
        "validator_pass": validator_pass,
        "direct_strict_complete": scored["direct"]["strict_complete"],
        "calculation_all_operand": scored["calculation"]["strict_complete"],
        "multi_complete": scored["multi"]["strict_complete"],
        "false_binding": safety["false_binding_queries"],
        "effective": decision["semantic_evidence_binder_effective"],
        "next_gate": decision["next_gate"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
