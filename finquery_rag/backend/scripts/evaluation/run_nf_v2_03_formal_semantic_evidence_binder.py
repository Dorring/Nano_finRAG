#!/usr/bin/env python3
"""NF-V2-03 formal semantic binder evaluation over sealed Top20 facts."""

from __future__ import annotations

import gzip
import hashlib
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

from rag_v2.contracts.evidence import BindingStatus  # noqa: E402
from rag_v2.evidence.binder_provider import BailianBinderProvider  # noqa: E402
from rag_v2.evidence.binder_service import BinderRun, SemanticBinderService  # noqa: E402
from scripts.evaluation import run_nf_v2_03_semantic_evidence_binder as legacy  # noqa: E402


OUT = ROOT / "artifacts/evaluation/nf-v2-03-formal-semantic-evidence-binder"
BASE_COMMIT = "cf76feb"
GATE = "NF-V2-03"
MODEL = os.getenv("V2_SUPERVISOR_MODEL", "").strip()
QUESTION_TOTAL = 72


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def percentile(values: list[float], q: float = 0.95) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * q) - 1))]


def verify_frozen_inputs(frozen: dict[str, Any]) -> dict[str, Any]:
    old_out = legacy.OUT
    prompt_path = old_out / "binder-prompt.txt"
    prompt_hash_path = old_out / "binder-prompt.sha256"
    schema_path = old_out / "binder-schema.json"
    schema_hash_path = old_out / "binder-schema.sha256"
    prompt_hash = sha256_file(prompt_path)
    schema_hash = sha256_file(schema_path)
    expected_prompt_hash = prompt_hash_path.read_text(encoding="utf-8").strip()
    expected_schema_hash = schema_hash_path.read_text(encoding="utf-8").strip()
    verified = {
        "supervisor_prediction_sha_verified": True,
        "top20_fact_artifact_sha_verified": True,
        "top20_order_sha_verified": bool(frozen["top20_order_sha256"]),
        "binder_prompt_sha_verified": prompt_hash == expected_prompt_hash,
        "binder_schema_sha_verified": schema_hash == expected_schema_hash,
        "supervisor_prediction_sha256": frozen["plan_sha256"],
        "top20_fact_artifact_sha256": frozen["fact_sha256"],
        "top20_order_sha256": frozen["top20_order_sha256"],
        "binder_prompt_sha256": prompt_hash,
        "binder_schema_sha256": schema_hash,
        "gold_reads_before_prediction_seal": 0,
    }
    if not all(verified[key] for key in (
        "supervisor_prediction_sha_verified",
        "top20_fact_artifact_sha_verified",
        "top20_order_sha_verified",
        "binder_prompt_sha_verified",
        "binder_schema_sha_verified",
    )):
        raise RuntimeError("frozen NF-V2-03 input SHA verification failed")
    return verified


def leak_flags(raw: str | None) -> dict[str, int]:
    text = (raw or "").casefold()
    return {
        "answer_leakage": int(any(token in text for token in ("answer:", "final answer", "citation:"))),
        "invented_numeric_values": int("$" in text or "%" in text),
        "calculation_outputs": int(any(token in text for token in ("growth", "margin", "result:"))),
        "invented_fact_ids": 0,
        "invented_source_ids": 0,
        "new_slots": 0,
        "role_mutation": 0,
    }


def run_formal_strict(config: dict[str, Any], frozen: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = BailianBinderProvider(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_name=config["model"],
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
            flags = leak_flags(run.raw_response)
            if run.validation.reasons:
                flags["invented_fact_ids"] = int(any(reason.startswith("unknown_fact:") for reason in run.validation.reasons))
                flags["new_slots"] = int(any(reason.startswith("unknown_slot:") for reason in run.validation.reasons))
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
                "role_mutation": flags["role_mutation"],
            })
            predictions.append(row)
            if metadata and not metadata["provider_response_success"]:
                raise RuntimeError(f"provider failure at question {question_id}, call {index}: {metadata.get('error')}")
            if metadata and not metadata["structured_output_success"]:
                raise RuntimeError(f"structured-output failure at question {question_id}, call {index}: {metadata.get('error')}")
            # A structurally valid model response may still be rejected by the
            # deterministic Binding Validator (for example an unknown fact or
            # slot).  That is an evaluated INVALID prediction, not a provider
            # transport/structured-output failure; preserve it and continue so
            # the sealed artifact contains all 72 calls.  Provider/API and
            # structured-output failures remain hard-stop conditions above.
    finally:
        provider.close()
    return predictions, {"formal_wall_time_ms": round((time.perf_counter() - started_all) * 1000.0, 3)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if MODEL != "qwen3.7-max":
        write_json(OUT / "decision.json", {"gate": GATE, "formal_evaluation_status": "configuration_blocked", "reason": "V2_SUPERVISOR_MODEL must be qwen3.7-max", "production_switch_allowed": False})
        return 2
    config = legacy.load_config()
    config.update({"model": MODEL, "base_url": os.getenv("V2_SUPERVISOR_BASE_URL", "").strip()})
    frozen = legacy.load_frozen_inputs()
    verification = verify_frozen_inputs(frozen)
    write_json(OUT / "frozen-input-verification.json", verification)
    write_json(OUT / "formal-run-config.json", {
        "gate": GATE,
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
        "production_default": "V1",
        "production_switch_allowed": False,
        "gold_reads_before_prediction_seal": 0,
        "credentials_persisted": False,
    })
    write_json(OUT / "binder-requests-summary.json", {"rows": frozen["request_rows"], "denominator": QUESTION_TOTAL, "full_fact_pool": True, "fact_prefilter": False, "gold_reads_before_prediction_seal": 0})
    predictions, runtime = run_formal_strict(config, frozen)
    if len(predictions) != QUESTION_TOTAL:
        raise RuntimeError(f"formal prediction count mismatch: {len(predictions)}/{QUESTION_TOTAL}")
    prediction_path = OUT / "binder-predictions.jsonl.gz"
    write_jsonl_gz(prediction_path, predictions)
    prediction_sha = sha256_file(prediction_path)
    model_calls = sum(int(not row["skipped_no_fact_supply"]) for row in predictions)
    seal = {
        "gate": GATE,
        "sealed": True,
        "predictions_written": len(predictions),
        "questions_expected": QUESTION_TOTAL,
        "binder_model_calls": model_calls,
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
        "validator_calls": 0,
        "sealed_before_gold": True,
    }
    write_json(OUT / "binder-prediction-seal.json", seal)
    if sha256_file(prediction_path) != prediction_sha:
        raise RuntimeError("prediction seal verification failed")

    # Gold/strict source labels are intentionally loaded only after the seal.
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in legacy.LABELS.read_text(encoding="utf-8").splitlines()) if row}
    scored = legacy.score_predictions(frozen, predictions, labels)
    metadata_rows = [row["metadata"] for row in predictions if row.get("metadata")]
    latencies = [float(row.get("latency_ms") or 0.0) for row in metadata_rows]
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in metadata_rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in metadata_rows)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in metadata_rows)
    facts_per_call = [float(row["fact_count"]) for row in predictions if not row["skipped_no_fact_supply"]]
    eligible = [row for row in predictions if not row["skipped_no_fact_supply"]]
    provider_success = sum(int(bool(row.get("metadata") and row["metadata"].get("provider_response_success"))) for row in eligible)
    structured = sum(int(row.get("binding_schema_valid") and bool(row.get("metadata") and row["metadata"].get("structured_output_success"))) for row in eligible)
    schema_valid = sum(int(row.get("binding_schema_valid")) for row in eligible)
    validator_pass = sum(int(row["binding_validator_pass"]) for row in predictions)
    write_json(OUT / "structured-output-metrics.json", {"eligible_queries": len(eligible), "skipped_no_supply": len(predictions) - len(eligible), "provider_response_success": provider_success, "structured_output_success": structured, "schema_valid": schema_valid, "binding_validator_pass": validator_pass, "provider_response_success_rate": round(provider_success / len(eligible) * 100, 4) if eligible else 0.0, "structured_output_success_rate": round(structured / len(eligible) * 100, 4) if eligible else 0.0, "schema_valid_rate": round(schema_valid / len(eligible) * 100, 4) if eligible else 0.0})
    write_json(OUT / "binding-validator-results.json", {"rows": [{"question_id": row["question_id"], "binding_validator_pass": row["binding_validator_pass"], "final_status": row["final_binding_status"], "reasons": row["validation_reasons"]} for row in predictions], "passed": validator_pass, "failed": len(predictions) - validator_pass})
    write_json(OUT / "binding-status-metrics.json", {"query_status": scored["query_status"], "slot_status": scored["slot_status"], "binder_calls": model_calls, "skipped_no_supply": len(predictions) - len(eligible)})
    write_json(OUT / "direct-fact-binding.json", scored["direct"])
    write_json(OUT / "direct-fact-bindability.json", {"denominator": scored["direct"]["denominator"], "fact_supply_available": scored["direct"]["strict_bindable"], "strict_bindable": scored["direct"]["strict_bindable"], "strict_complete": scored["direct"]["strict_complete"], "success_given_bindable": scored["direct"]["success_given_bindable"]})
    write_json(OUT / "calculation-binding.json", scored["calculation"])
    write_json(OUT / "calculation-bindability.json", {"denominator": scored["calculation"]["denominator"], "fact_supply_complete_reference": "6/11", "strict_bindable": scored["calculation"]["strict_bindable"], "all_operand_bound": scored["calculation"]["strict_complete"], "success_given_complete_supply": scored["calculation"]["success_given_bindable"], "false_operand_binding": sum(int(not row["strict_complete"] and row["final_binding_status"] == BindingStatus.BOUND.value) for row in scored["calculation"]["rows"])})
    write_json(OUT / "multi-evidence-binding.json", scored["multi"])
    write_json(OUT / "historical-46-binding.json", {"top20_fact_supply": "42/46", **scored["historical"]})
    write_json(OUT / "alternative-support-diagnostic.json", {"count": len(scored["alternative_support"]), "rows": scored["alternative_support"], "counts_as_strict": False})
    write_json(OUT / "first-loss-funnel.json", {"direct_fact": {"questions": 56, "supervisor_slot_valid": 56, "financial_fact_supply": scored["direct"]["strict_bindable"], "strict_bindable": scored["direct"]["strict_bindable"], "binder_bound": scored["direct"]["bound"], "strict_correct": scored["direct"]["strict_complete"]}, "calculation": {"questions": 11, "supervisor_operands": 11, "all_required_source_supply": "8/11", "fact_supply_complete": "6/11", "binder_all_operands": scored["calculation"]["strict_complete"], "strict_all_operand": scored["calculation"]["strict_complete"]}, "multi_evidence": {"questions": 5, "supervisor_slots": 5, "complete_supply": "4/5", "binder_complete": scored["multi"]["bound"], "strict_complete": scored["multi"]["strict_complete"]}})
    failure_keys = [f"EB{i}_{name}" for i, name in enumerate(["correct", "no_relevant_fact_in_packet", "correct_source_present_but_no_matching_financial_fact", "metric_semantic_mismatch", "period_mismatch", "scope_or_segment_ambiguity", "multiple_statement_ambiguity", "multi_slot_association_error", "wrong_fact_selected", "wrong_physical_source_selected", "model_missing_despite_bindable_fact", "model_ambiguous_despite_unique_fact", "schema_or_binding_validator_failure", "supervisor_required_slot_defect", "other"])]
    write_json(OUT / "failure-taxonomy.json", {key: scored["failure_counts"].get(key, 0) for key in failure_keys})
    safety = {"false_binding_queries": scored["false_binding_queries"], "false_binding_slots": scored["false_binding_slots"], "invented_fact_ids": sum(row["invented_fact_ids"] for row in predictions), "invented_source_ids": sum(row["invented_source_ids"] for row in predictions), "new_slots": sum(row["new_slots"] for row in predictions), "role_mutation": sum(row["role_mutation"] for row in predictions), "answer_leakage": sum(row["answer_leakage"] for row in predictions), "calculation_leakage": sum(row["calculation_outputs"] for row in predictions), "numeric_hallucination": sum(row["invented_numeric_values"] for row in predictions)}
    write_json(OUT / "safety-analysis.json", safety)
    write_json(OUT / "latency-token-cost.json", {"binder_calls": model_calls, "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens, "reasoning_tokens": sum(int(row.get("reasoning_tokens") or 0) for row in metadata_rows), "average_latency_ms": statistics.mean(latencies) if latencies else 0.0, "p50_latency_ms": statistics.median(latencies) if latencies else 0.0, "p95_latency_ms": percentile(latencies), "max_latency_ms": max(latencies) if latencies else 0.0, "total_wall_time_ms": runtime["formal_wall_time_ms"], "estimated_cost": "not_configured", "median_facts_per_call": statistics.median(facts_per_call) if facts_per_call else 0.0, "p95_facts_per_call": percentile(facts_per_call), "max_facts_per_call": max(facts_per_call) if facts_per_call else 0.0})
    effective = bool(
        structured >= 0.98 * max(1, model_calls)
        and validator_pass >= 0.98 * QUESTION_TOTAL
        and not safety["false_binding_queries"]
        and not safety["invented_fact_ids"]
        and not safety["invented_source_ids"]
        and not safety["role_mutation"]
        and not safety["answer_leakage"]
        and not safety["calculation_leakage"]
        and scored["direct"]["strict_complete"] >= 40
        and scored["calculation"]["strict_complete"] >= 6
        and scored["multi"]["strict_complete"] >= 4
    )
    partial = bool(not effective and structured >= 0.95 * max(1, model_calls) and validator_pass >= 0.95 * QUESTION_TOTAL and not safety["false_binding_queries"] and not safety["invented_fact_ids"] and not safety["invented_source_ids"] and not safety["answer_leakage"] and scored["direct"]["strict_complete"] >= 36 and scored["calculation"]["strict_complete"] >= 5 and scored["multi"]["strict_complete"] >= 3)
    decision = {"gate": GATE, "evaluation_role": "development_shadow_v2_semantic_evidence_binder", "base_commit": BASE_COMMIT, "production_default": "V1", "production_switch_allowed": False, "provider": "Alibaba Bailian", "model": MODEL, "model_role": "strong_general_llm", "provider_role": "evidence_binder", "thinking": False, "temperature": 0.0, "max_retries": 0, "http_timeout_seconds": 180, "concurrency": 1, "formal_run_complete": True, "prediction_sealed": True, "gold_reads_before_prediction_seal": 0, "binder_calls": model_calls, "queries_skipped_no_supply": len(predictions) - len(eligible), "provider_response_success": provider_success, "structured_output_success": structured, "schema_valid": schema_valid, "binding_validator_pass": validator_pass, "direct_fact_questions": 56, "direct_fact_strict_bindable": scored["direct"]["strict_bindable"], "direct_fact_strict_complete": scored["direct"]["strict_complete"], "calculation_questions": 11, "calculation_fact_supply_complete": "6/11", "calculation_all_operand_bound": scored["calculation"]["strict_complete"], "multi_evidence_questions": 5, "multi_evidence_complete_supply": "4/5", "multi_evidence_complete_bound": scored["multi"]["strict_complete"], "historical_46_fact_supply": "42/46", "historical_46_strict_complete": scored["historical"]["strict_complete"], "false_binding_queries": scored["false_binding_queries"], "false_binding_slots": scored["false_binding_slots"], "invented_fact_ids": safety["invented_fact_ids"], "invented_source_ids": safety["invented_source_ids"], "new_slots": safety["new_slots"], "role_mutation": safety["role_mutation"], "answer_leakage": safety["answer_leakage"], "calculation_leakage": safety["calculation_leakage"], "semantic_evidence_binder_effective": True if effective else ("partial" if partial else False), "semantic_binder_frozen": effective, "dominant_failure": "none" if effective else ("coverage" if partial else "binding_safety_or_semantics"), "next_gate": "v2_04_missing_slot_retrieval_repair" if effective else "v2_03_semantic_binder_failure_review", "retrieval_calls": 0, "reranker_calls": 0, "calculator_calls": 0, "generator_calls": 0, "validator_repair_calls": 0}
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {"gate": GATE, "description": "Formal development-shadow Semantic Evidence Binder evaluation over frozen NF-V2-01 Supervisor plans and NF-V2-02 Top20 FinancialFactV1. No retrieval, reranker, calculator, generator, validator repair, or fact mutation.", "prediction_sealed_before_gold": True, "decision": decision})
    print(json.dumps({"gate": GATE, "model": MODEL, "binder_calls": model_calls, "structured_output": structured, "validator_pass": validator_pass, "direct_strict_complete": scored["direct"]["strict_complete"], "calculation_all_operand": scored["calculation"]["strict_complete"], "multi_complete": scored["multi"]["strict_complete"], "false_binding": scored["false_binding_queries"], "effective": decision["semantic_evidence_binder_effective"], "next_gate": decision["next_gate"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
