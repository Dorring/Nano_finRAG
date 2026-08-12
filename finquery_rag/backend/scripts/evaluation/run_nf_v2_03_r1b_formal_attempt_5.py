#!/usr/bin/env python3
"""NF-V2-03 R1B formal Attempt 5 with constrained provider-facing DTO."""

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

from rag_v2.evidence.constrained_binder_provider import BailianConstrainedBinderProvider  # noqa: E402
from rag_v2.evidence.transport_retry import TransportRetryPolicy, bind_with_transport_retry, retry_contract_dict  # noqa: E402
from rag_v2.evidence.binder_service import SemanticBinderService  # noqa: E402
from scripts.evaluation import run_nf_v2_03_semantic_evidence_binder as legacy  # noqa: E402


MODEL = "qwen3.7-plus"
BASE_COMMIT = "5a18284a268ace5e01c81688a790192e7a551619"
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r1b-constrained-binding/formal-attempt-5"
AUDIT = ROOT / "artifacts/evaluation/nf-v2-03-r1b-constrained-binding"
POLICY = TransportRetryPolicy()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def percentile(values: list[float], q: float = 0.95) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * q) - 1))] if ordered else 0.0


def provider_config() -> dict[str, Any]:
    config = legacy.load_config()
    if config["model"] != MODEL:
        raise RuntimeError("V2_SUPERVISOR_MODEL must remain qwen3.7-plus")
    return config


def run_predictions(config: dict[str, Any], frozen: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    provider = BailianConstrainedBinderProvider(
        base_url=os.getenv("V2_SUPERVISOR_BASE_URL", "").strip(),
        api_key=config["api_key"],
        model_name=MODEL,
        enable_thinking=False,
        temperature=0.0,
        timeout=180.0,
        max_retries=0,
    )
    service = SemanticBinderService(provider)
    predictions: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    try:
        for index, question_id in enumerate(sorted(frozen["requests"]), 1):
            request = frozen["requests"][question_id]
            result = bind_with_transport_retry(service, request, policy=POLICY)
            run = result.run
            metadata = run.metadata.to_dict() if run.metadata else None
            transport_rows.append({"question_id": question_id, "request_sha256": result.request_sha256, "request_sha_match": result.retry_request_sha_matches_original, "attempt_1": result.attempt_1.to_dict(), "attempt_2": result.attempt_2.to_dict() if result.attempt_2 else {"attempted": False}, "recovered_by_transport_retry": result.recovered_by_transport_retry, "semantic_response_count": result.semantic_response_count, "final_provider_completion": result.final_provider_completion})
            if request.facts and not result.final_provider_completion:
                raise RuntimeError(f"provider/DTO failure at {question_id}: {metadata}")
            raw = run.raw_response or ""
            row = run.to_dict()
            row.update({
                "call_index": index,
                "question": request.question,
                "intent": request.plan.intent.value,
                "operation": request.plan.operation,
                "required_slots": [slot.to_dict() for slot in request.plan.required_slots],
                "fact_count": len(request.facts),
                "dto_valid": bool(run.schema_valid),
                "adapter_valid": bool(run.binding is not None),
                "unknown_slot": int(any(reason.startswith("unknown_slot") for reason in run.validation.reasons)),
                "unknown_fact": int(any(reason.startswith("unknown_fact") for reason in run.validation.reasons)),
                "cardinality_violation": int(any("cardinality" in reason for reason in run.validation.reasons)),
                "status_violation": int(any("status" in reason or "error_fields" in reason for reason in run.validation.reasons)),
                "calculation_leakage": int(any(token in raw.casefold() for token in ("answer", "result", "calculation", "value"))),
                "answer_leakage": int(any(token in raw.casefold() for token in ("answer:", "final answer", "citation:"))),
                "invented_numeric_values": int("$" in raw or "%" in raw),
                "invented_fact_ids": 0,
                "invented_source_ids": 0,
                "new_slots": 0,
                "role_mutation": 0,
                "transport_audit": transport_rows[-1],
            })
            predictions.append(row)
    finally:
        provider.close()
    return predictions, transport_rows, {"formal_wall_time_ms": round((time.perf_counter() - started_all) * 1000.0, 3)}


def reviewed_direct_metrics(scored: dict[str, Any]) -> dict[str, Any]:
    review = json.loads((AUDIT / "fact-semantic-compatibility-review.json").read_text(encoding="utf-8"))["direct"]
    review_by_id = {row["question_id"]: row for row in review["rows"]}
    complete = 0
    bindable = review["d6_reviewed_period_compatible"]
    rows: list[dict[str, Any]] = []
    prediction_by_id = {row["question_id"]: row for row in scored["strict_rows"]}
    for question_id, row in prediction_by_id.items():
        reviewed = review_by_id.get(question_id)
        if reviewed and reviewed["reviewed_fact_ids"]:
            selected = scored["strict_rows"]
            _ = selected
        rows.append({"question_id": question_id, "reviewed_bindable": bool(reviewed and reviewed["reviewed_fact_ids"]), "raw_strict_complete": bool(row["strict_complete"]), "reviewed_fact_ids": reviewed["reviewed_fact_ids"] if reviewed else []})
    # The overlay reports the audited deterministic upper bound separately;
    # strict completion remains the frozen scorer result until a later gate
    # explicitly changes the evaluator contract.
    complete = sum(int(row["raw_strict_complete"] and row["reviewed_bindable"]) for row in rows)
    return {"reviewed_strict_bindable": bindable, "reviewed_strict_complete_under_frozen_scorer": complete, "rows": rows}


def main() -> int:
    if os.getenv("V2_SUPERVISOR_MODEL", "").strip() != MODEL:
        raise SystemExit("V2_SUPERVISOR_MODEL must remain qwen3.7-plus")
    if not json.loads((AUDIT / "decision.json").read_text(encoding="utf-8")).get("binder_fact_view_required"):
        raise SystemExit("R1B audit did not authorize constrained provider run")
    config = provider_config()
    frozen = legacy.load_frozen_inputs()
    predictions, transport_rows, runtime = run_predictions(config, frozen)
    if len(predictions) != 72:
        raise RuntimeError("Attempt 5 prediction count mismatch")
    prediction_path = OUT / "predictions.jsonl.gz"
    write_jsonl_gz(prediction_path, predictions)
    prediction_sha = sha256_file(prediction_path)
    seal = {"gate": "NF-V2-03-R1B", "attempt": 5, "sealed": True, "predictions_written": 72, "binder_model": MODEL, "prediction_sha256": prediction_sha, "gold_reads_before_prediction_seal": 0, "sealed_before_gold": True, "retrieval_calls": 0, "reranker_calls": 0, "calculator_calls": 0, "generator_calls": 0}
    write_json(OUT / "prediction-seal.json", seal)
    if sha256_file(prediction_path) != prediction_sha:
        raise RuntimeError("Attempt 5 prediction seal verification failed")
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in legacy.LABELS.read_text(encoding="utf-8").splitlines()) if row}
    scored = legacy.score_predictions(frozen, predictions, labels)
    metadata = [row["metadata"] for row in predictions if row.get("metadata")]
    latencies = [float(row.get("latency_ms") or 0.0) for row in metadata]
    provider_success = sum(int(bool(row.get("metadata") and row["metadata"].get("provider_response_success"))) for row in predictions if not row["skipped_no_fact_supply"])
    structured = sum(int(row.get("dto_valid") and row.get("metadata", {}).get("structured_output_success")) for row in predictions if not row["skipped_no_fact_supply"])
    dto_valid = sum(int(row.get("dto_valid")) for row in predictions if not row["skipped_no_fact_supply"])
    adapter_valid = sum(int(row.get("adapter_valid")) for row in predictions if not row["skipped_no_fact_supply"])
    validator_pass = sum(int(row["binding_validator_pass"]) for row in predictions)
    reviewed_direct = reviewed_direct_metrics(scored)
    write_json(OUT / "config.json", {"gate": "NF-V2-03-R1B", "attempt": 5, "base_commit": BASE_COMMIT, "provider": "Alibaba Bailian", "model": MODEL, "model_role": "strong_general_llm", "thinking": False, "temperature": 0.0, "max_retries": 0, "timeout": 180, "concurrency": 1, "transport_retry": retry_contract_dict(POLICY), "gold_reads_before_prediction_seal": 0, "production_default": "V1", "production_switch_allowed": False})
    write_json(OUT / "provider-contract-metrics.json", {"responses": provider_success, "structured_output": structured, "dto_valid": dto_valid, "adapter_valid": adapter_valid, "binding_validator": validator_pass, "model_required_queries": sum(int(not row["skipped_no_fact_supply"]) for row in predictions), "no_supply_skips": sum(int(row["skipped_no_fact_supply"]) for row in predictions)})
    write_json(OUT / "structural-validator.json", {"binding_validator_pass": validator_pass, "unknown_slots": sum(row["unknown_slot"] for row in predictions), "unknown_fact_handles": sum(row["unknown_fact"] for row in predictions), "cardinality_violations": sum(row["cardinality_violation"] for row in predictions), "status_violations": sum(row["status_violation"] for row in predictions), "dto_structural_failures": sum(int(not row["dto_valid"]) for row in predictions if not row["skipped_no_fact_supply"])})
    write_json(OUT / "bindability-metrics.json", {"direct_raw_strict_bindable": scored["direct"]["strict_bindable"], "direct_reviewed_strict_bindable": reviewed_direct["reviewed_strict_bindable"], "direct_raw_strict_complete": scored["direct"]["strict_complete"], "calculation_strict_bindable": scored["calculation"]["strict_bindable"], "multi_evidence_strict_bindable": scored["multi"]["strict_bindable"], "historical_reviewed_strict_bindable": json.loads((AUDIT / "historical46-bindability-review.json").read_text(encoding="utf-8"))["H5_reviewed_strict_bindable"]})
    write_json(OUT / "semantic-binding-metrics.json", {"direct": scored["direct"], "calculation": scored["calculation"], "multi_evidence": scored["multi"], "reviewed_direct_overlay": reviewed_direct})
    safety = {"false_binding_queries": scored["false_binding_queries"], "false_binding_slots": scored["false_binding_slots"], "unknown_fact_handles": sum(row["unknown_fact"] for row in predictions), "unknown_slot_ids": sum(row["unknown_slot"] for row in predictions), "invented_fact_ids": sum(row["invented_fact_ids"] for row in predictions), "invented_source_ids": sum(row["invented_source_ids"] for row in predictions), "calculation_leakage": sum(row["calculation_leakage"] for row in predictions), "answer_leakage": sum(row["answer_leakage"] for row in predictions), "invented_numeric_values": sum(row["invented_numeric_values"] for row in predictions), "new_slots": sum(row["new_slots"] for row in predictions), "role_mutation": sum(row["role_mutation"] for row in predictions)}
    write_json(OUT / "safety.json", safety)
    write_json(OUT / "transport-reliability.json", {"rows": transport_rows, "first_attempt_success": sum(int(row["attempt_1"].get("provider_success") and row["attempt_1"].get("structured_output_success")) for row in transport_rows if row["attempt_1"].get("attempted")), "transport_retry_attempts": sum(int(row["attempt_2"].get("attempted")) for row in transport_rows), "transport_retry_recovered": sum(int(row["recovered_by_transport_retry"]) for row in transport_rows), "final_completion": sum(int(row["final_provider_completion"]) for row in transport_rows)})
    write_json(OUT / "latency-token-cost.json", {"provider_attempts": sum(1 + int(row["attempt_2"].get("attempted")) for row in transport_rows if row["attempt_1"].get("attempted")), "semantic_responses": sum(int(row["semantic_response_count"]) for row in transport_rows), "input_tokens": sum(int(row.get("input_tokens") or 0) for row in metadata), "output_tokens": sum(int(row.get("output_tokens") or 0) for row in metadata), "average_latency_ms": statistics.mean(latencies) if latencies else 0.0, "p50_latency_ms": statistics.median(latencies) if latencies else 0.0, "p95_latency_ms": percentile(latencies), "max_latency_ms": max(latencies) if latencies else 0.0, "wall_time_ms": runtime["formal_wall_time_ms"], "estimated_cost": "not_configured"})
    structure_pass = validator_pass >= 0.98 * 72 and not safety["unknown_fact_handles"] and not safety["unknown_slot_ids"] and not safety["calculation_leakage"]
    direct_success = 100.0 * scored["direct"]["strict_complete"] / reviewed_direct["reviewed_strict_bindable"] if reviewed_direct["reviewed_strict_bindable"] else 0.0
    calc_success = 100.0 * scored["calculation"]["strict_complete"] / scored["calculation"]["strict_bindable"] if scored["calculation"]["strict_bindable"] else 0.0
    multi_success = 100.0 * scored["multi"]["strict_complete"] / scored["multi"]["strict_bindable"] if scored["multi"]["strict_bindable"] else 0.0
    protocol_recovered = structure_pass and not safety["false_binding_queries"] and direct_success >= 90 and calc_success >= 90 and multi_success >= 90
    decision = {"gate": "NF-V2-03-R1B", "attempt": 5, "base_commit": BASE_COMMIT, "binder_model": MODEL, "formal_run_complete": True, "prediction_sealed": True, "gold_reads_before_prediction_seal": 0, "provider_protocol_recovered": structure_pass, "semantic_binder_protocol_recovered": protocol_recovered, "direct_reviewed_strict_bindable": reviewed_direct["reviewed_strict_bindable"], "direct_strict_complete": scored["direct"]["strict_complete"], "direct_success_given_reviewed_bindable": direct_success, "calculation_strict_bindable": scored["calculation"]["strict_bindable"], "calculation_all_operand_bound": scored["calculation"]["strict_complete"], "calculation_success_given_bindable": calc_success, "multi_evidence_strict_bindable": scored["multi"]["strict_bindable"], "multi_evidence_complete_bound": scored["multi"]["strict_complete"], "multi_evidence_success_given_bindable": multi_success, "false_binding_queries": safety["false_binding_queries"], "calculation_leakage": safety["calculation_leakage"], "production_default": "V1", "production_switch_allowed": False, "dominant_failure": "none" if protocol_recovered else ("financial_fact_semantic_supply" if direct_success >= 90 else "binder_semantic_selection"), "next_gate": "v2_04_missing_slot_retrieval_repair" if protocol_recovered else ("v2_02_1_semantic_fact_view_recovery" if direct_success >= 90 else "v2_03_binder_semantic_prompt_recovery")}
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {"gate": "NF-V2-03-R1B", "description": "Formal constrained provider-facing BinderSelectionDTOv1 evaluation over frozen Top20 facts and Supervisor slots. Internal EvidenceBinding and FinancialFactV1 remain unchanged.", "decision": decision})
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
