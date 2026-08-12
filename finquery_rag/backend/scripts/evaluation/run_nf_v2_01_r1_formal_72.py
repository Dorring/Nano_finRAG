#!/usr/bin/env python3
"""NF-V2-01 R1 formal, question-only 72-question Supervisor replay.

This runner deliberately has no smoke path and no downstream execution.  It
seals the question-only input and frozen configuration before making the first
model call, then opens evaluation annotations only after the prediction seal.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

ROOT = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/pdf-sr-v2/finquery_rag/backend")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.plan import Action, Intent  # noqa: E402
from rag_v2.orchestration.loader import load_question_envelopes  # noqa: E402
from rag_v2.supervisor.bailian_provider import BailianProvider  # noqa: E402
from rag_v2.supervisor.prompt import (  # noqa: E402
    SUPERVISOR_PLAN_JSON_SCHEMA,
    SUPERVISOR_SYSTEM_PROMPT_V1,
)
from rag_v2.supervisor.service import SupervisorService  # noqa: E402
from scripts.evaluation.run_nf_v2_01_general_llm_supervisor import (  # noqa: E402
    QUESTIONS,
    QUERY_REQUIREMENTS,
    V2_CONTRACT,
    evaluate,
    sha256_bytes,
    sha256_file,
    write_json,
)
from scripts.evaluation.run_nf_v2_01_r1_bailian_strong_general_supervisor import (  # noqa: E402
    load_env_config,
)

GATE = "NF-V2-01-R1"
BASE_COMMIT = "2861ac1d8494afb800f1a90f102dd42c0cfd1abb"
ROLE = "development_shadow_v2_strong_general_llm_supervisor"
PROVIDER = "bailian"
MODEL = "qwen3.7-max-2026-06-08"
OUT = ROOT / "artifacts/evaluation/nf-v2-01-r1-bailian-formal-72"
TRANSPORT_SEAL = ROOT / "artifacts/evaluation/nf-v2-01-r1-transport-isolation/transport-seal.json"
FROZEN_BENCHMARK_CONTRACT = ROOT / "artifacts/evaluation/nf-opt-26-r0-internal-retrieval-freeze/benchmark-freeze-contract.json"
QUESTION_COUNT = 72
MAX_RETRIES = 0


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sanitized_endpoint(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def question_payload(envelopes: tuple[Any, ...]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "case_id": item.question_id,
                "question": item.question,
                "document_scope": sorted(item.document_scope),
            }
            for item in envelopes
        ],
        key=lambda item: item["case_id"],
    )


def schema_payload() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "SupervisorPlan",
            "strict": True,
            "schema": SUPERVISOR_PLAN_JSON_SCHEMA,
        },
    }


def redact_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return text[:500]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_transport_seal() -> tuple[bool, dict[str, Any], str | None]:
    if not TRANSPORT_SEAL.exists():
        return False, {}, "transport seal is missing"
    seal = read_json(TRANSPORT_SEAL)
    checks = {
        "provider": seal.get("provider") == PROVIDER,
        "model": seal.get("model") == MODEL,
        "max_retries": seal.get("max_retries") == MAX_RETRIES,
        "formal_runner_transport_ready": seal.get("formal_runner_transport_ready") is True,
        "synthetic_runner_success": seal.get("synthetic_runner_success") == "30/30",
        "api_connection_errors": seal.get("api_connection_errors") == 0,
        "production_switch_allowed": seal.get("production_switch_allowed") is False,
    }
    bad = [name for name, passed in checks.items() if not passed]
    return not bad, seal, ("transport seal checks failed: " + ", ".join(bad)) if bad else None


def question_set_hash(envelopes: tuple[Any, ...]) -> str:
    return stable_json_hash(question_payload(envelopes))


def write_question_only_input(envelopes: tuple[Any, ...]) -> tuple[Path, str]:
    path = OUT / "question-only-input.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for item in envelopes:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return path, sha256_file(path)


def pre_run_seal(config: dict[str, Any], envelopes: tuple[Any, ...]) -> tuple[bool, dict[str, Any], str | None]:
    transport_ok, transport, transport_error = verify_transport_seal()
    prompt_hash = sha256_bytes(SUPERVISOR_SYSTEM_PROMPT_V1.encode("utf-8"))
    schema_bytes = (json.dumps(schema_payload(), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    schema_hash = sha256_bytes(schema_bytes)
    previous = ROOT / "artifacts/evaluation/nf-v2-01-r1-bailian-strong-general-supervisor"
    previous_prompt = (previous / "supervisor-prompt.sha256").read_text(encoding="ascii").strip() if (previous / "supervisor-prompt.sha256").exists() else None
    previous_schema = (previous / "supervisor-schema.sha256").read_text(encoding="ascii").strip() if (previous / "supervisor-schema.sha256").exists() else None
    q_hash = question_set_hash(envelopes)
    frozen_contract = read_json(FROZEN_BENCHMARK_CONTRACT) if FROZEN_BENCHMARK_CONTRACT.exists() else {}
    frozen_q_hash = frozen_contract.get("frozen_hashes", {}).get("question_hash")
    checks = {
        "transport_seal_valid": transport_ok,
        "prompt_hash_matches": previous_prompt == prompt_hash,
        "schema_hash_matches": previous_schema == schema_hash,
        "question_hash_matches": q_hash == frozen_q_hash,
        "question_count_matches": len(envelopes) == QUESTION_COUNT,
    }
    errors = [name for name, passed in checks.items() if not passed]
    reason = "; ".join(errors) if errors else transport_error
    question_path, question_only_hash = write_question_only_input(envelopes)
    artifact = {
        "gate": GATE,
        "base_commit": BASE_COMMIT,
        "provider": "Alibaba Bailian",
        "provider_id": PROVIDER,
        "model": MODEL,
        "provider_role": "supervisor",
        "model_role": "strong_general_llm",
        "thinking": False,
        "temperature": 0.0,
        "max_retries": MAX_RETRIES,
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_hash,
        "question_set_sha256": q_hash,
        "question_only_input_sha256": question_only_hash,
        "question_only_input_path": str(question_path.relative_to(ROOT)),
        "frozen_question_hash_reference": frozen_q_hash,
        "transport_seal_reference": str(TRANSPORT_SEAL.relative_to(ROOT)),
        "transport_seal_sha256": sha256_file(TRANSPORT_SEAL),
        "production_default": "V1",
        "production_switch_allowed": False,
        "gold_reads_before_prediction_seal": 0,
        "evaluation_annotations_loaded_before_prediction_seal": False,
        "checks": checks,
        "formal_evaluation_status": "formal_ready" if not errors else "blocked_pre_run_seal",
        "block_reason": reason,
    }
    write_json(OUT / "formal-run-config.json", artifact)
    write_json(OUT / "frozen-v2-contract.json", {
        "gate": GATE,
        "base_commit": BASE_COMMIT,
        "evaluation_role": ROLE,
        "fresh_blind_evaluation": False,
        "question_count": len(envelopes),
        "question_fields_loaded": ["question_id", "question", "document_scope"],
        "gold_fields_loaded_before_seal": False,
        "production_default": "V1",
        "production_switch_allowed": False,
        "v2_00_contract_artifact": str(V2_CONTRACT.relative_to(ROOT)),
        "v2_00_contract_sha256": sha256_file(V2_CONTRACT),
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "binder_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "validator_calls": 0,
    })
    (OUT / "supervisor-prompt.txt").write_text(SUPERVISOR_SYSTEM_PROMPT_V1, encoding="utf-8")
    (OUT / "supervisor-prompt.sha256").write_text(prompt_hash + "\n", encoding="ascii")
    write_json(OUT / "supervisor-schema.json", schema_payload())
    (OUT / "supervisor-schema.sha256").write_text(schema_hash + "\n", encoding="ascii")
    write_json(OUT / "supervisor-model-role.json", {
        "provider_role": "supervisor",
        "model_role": "strong_general_llm",
        "provider": PROVIDER,
        "model": MODEL,
        "financial_sft_fallback_forbidden": True,
    })
    return not errors, artifact, reason


def record_from_run(envelope: Any, run: Any, provider: BailianProvider, started: float) -> dict[str, Any]:
    metadata = run.metadata
    plan = run.plan.to_dict() if run.plan is not None else None
    raw = metadata.raw_response if metadata else None
    return {
        "question_id": envelope.question_id,
        "question": envelope.question,
        "document_scope": list(envelope.document_scope),
        "intent": plan.get("intent") if plan else None,
        "required_slots": plan.get("required_slots", []) if plan else [],
        "operation": plan.get("operation") if plan else None,
        "next_action": plan.get("next_action") if plan else None,
        "raw_next_action": plan.get("next_action") if plan else None,
        "plan": plan,
        "schema_valid": plan is not None,
        "plan_valid": bool(run.plan_valid),
        "error": redact_text(run.error),
        "error_type": "parse_failure" if metadata and (metadata.parse_failure or (metadata.provider_response_success and not metadata.structured_output_success)) else None,
        "provider": metadata.provider if metadata else PROVIDER,
        "provider_role": metadata.provider_role if metadata else "supervisor",
        "model": metadata.model if metadata else MODEL,
        "model_role": metadata.model_role if metadata else "strong_general_llm",
        "provider_response_success": metadata.provider_response_success if metadata else False,
        "structured_output_success": metadata.structured_output_success if metadata else False,
        "reasoning_tokens": metadata.reasoning_tokens if metadata else None,
        "latency_ms": metadata.latency_ms if metadata else (time.perf_counter() - started) * 1000,
        "input_tokens": metadata.input_tokens if metadata else None,
        "output_tokens": metadata.output_tokens if metadata else None,
        "total_tokens": metadata.total_tokens if metadata else None,
        "raw_response": raw,
        "exception_chain": getattr(provider, "last_exception_chain", []),
    }


def write_prediction_seal(records: list[dict[str, Any]], *, complete: bool, status: str, failure: dict[str, Any] | None = None) -> None:
    path = OUT / "supervisor-plans.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(OUT / "supervisor-prediction-seal.json", {
        "gate": GATE,
        "formal_run_complete": complete,
        "formal_evaluation_status": status,
        "questions_expected": QUESTION_COUNT,
        "predictions_written": len(records),
        "model_calls": len(records),
        "max_calls_per_question": 1,
        "retry": 0,
        "concurrency": 1,
        "gold_reads_before_prediction_seal": 0,
        "reference_answer_reads_before_prediction_seal": 0,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "binder_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "validator_calls": 0,
        "plans_sha256": sha256_file(path),
        "infrastructure_failure": failure,
        "sealed": True,
    })


def empty_blocked_artifacts(reason: str, records: list[dict[str, Any]]) -> None:
    write_json(OUT / "structured-output-metrics.json", {"formal_evaluation_status": "infrastructure_regression", "provider_response_success": None, "structured_output_success": None, "schema_valid": None, "plan_validator_pass": None, "parse_failures": None})
    write_json(OUT / "plan-validity.json", {"formal_evaluation_status": "infrastructure_regression", "schema_valid": None, "plan_validator_pass": None, "parse_failure": None})
    for name in [
        "routing-confusion-matrix.json", "routing-metrics.json", "calculation-routing.json",
        "calculation-operation.json", "calculation-operands.json", "slot-metrics.json",
        "safety-analysis.json", "failure-taxonomy.json", "routing-slot-attribution.json",
    ]:
        write_json(OUT / name, {"formal_evaluation_status": "infrastructure_regression", "reason": reason})
    write_json(OUT / "latency-token-cost.json", {"formal_evaluation_status": "infrastructure_regression", "calls": len(records)})
    write_json(OUT / "r0-r1-model-role-ablation.json", {
        "financial_sft_r0": {"schema_valid": "0/72", "plan_validator_pass": "0/72", "parse_failure": "72/72", "answer_leakage": "55/72", "semantic_metrics": "NOT EVALUABLE"},
        "bailian_r1": {"formal_evaluation_status": "infrastructure_regression", "reason": reason},
    })
    decision = {
        "gate": GATE,
        "evaluation_role": ROLE,
        "base_commit": BASE_COMMIT,
        "provider": "Alibaba Bailian",
        "model": MODEL,
        "model_role": "strong_general_llm",
        "thinking": False,
        "temperature": 0.0,
        "max_retries": MAX_RETRIES,
        "production_default": "V1",
        "production_switch_allowed": False,
        "formal_run_complete": False,
        "formal_evaluation_status": "infrastructure_regression",
        "supervisor_calls": len(records),
        "general_llm_supervisor_effective": None,
        "supervisor_frozen": False,
        "dominant_failure": "runner_transport_instability",
        "next_gate": "v2_01_r1_failure_review",
    }
    write_json(OUT / "decision.json", decision)
    (OUT / "README.md").write_text("# NF-V2-01 R1 formal Bailian supervisor\n\nFormal replay stopped after a new infrastructure failure; no incomplete metrics were scored.\n", encoding="utf-8")


def raw_numeric_leakage(raw: str | None) -> bool:
    if not raw:
        return False
    return bool(re.search(r"(?:[$€£]\s*\d|\d+(?:\.\d+)?\s*%)", raw))


def raw_action_operation_safety(records: list[dict[str, Any]]) -> dict[str, int]:
    invalid_action = unknown_operation = premature_calculate = premature_generate = premature_abstain = 0
    allowed_actions = {Action.RETRIEVE.value, Action.ABSTAIN.value}
    allowed_operations = {"difference", "growth_rate", "percentage_share", "sum", "average", "gross_margin", "net_margin", "debt_ratio", "scale_conversion", None}
    for record in records:
        try:
            payload = json.loads(record.get("raw_response") or "{}")
        except json.JSONDecodeError:
            continue
        action = payload.get("next_action")
        operation = payload.get("operation")
        if action not in allowed_actions:
            invalid_action += 1
        premature_calculate += action == Action.CALCULATE.value
        premature_generate += action == Action.GENERATE.value
        premature_abstain += action == Action.ABSTAIN.value
        unknown_operation += operation not in allowed_operations
    return {
        "invalid_action": invalid_action,
        "unknown_operation": unknown_operation,
        "premature_calculate": premature_calculate,
        "premature_generate": premature_generate,
        "premature_abstain": premature_abstain,
    }


def routing_slot_attribution(records: list[dict[str, Any]], rows: list[dict[str, Any]], requirements: dict[str, Any]) -> dict[str, int]:
    expected = {str(row["case_id"]): row for row in rows}
    output = Counter()
    for record in records:
        row = expected[record["question_id"]]
        expected_intent = (Intent.CALCULATION.value if row.get("requires_calculation") else Intent.MULTI_EVIDENCE.value if row.get("requires_multiple_sources") else Intent.DIRECT_FACT.value)
        plan = record.get("plan") or {}
        routing_ok = plan.get("intent") == expected_intent
        required = requirements[record["question_id"]].get("required_slots", [])
        predicted = plan.get("required_slots", []) if isinstance(plan, dict) else []
        slots_ok = len(required) == len(predicted) and all(
            str(item.get("metric", "")).strip().casefold() == str(gold.get("target", "")).strip().casefold()
            and str(item.get("period", "")).strip().casefold() == str(gold.get("period", "")).strip().casefold()
            and str(item.get("role", "")).strip().casefold() == str(gold.get("role", "")).strip().casefold()
            for item, gold in zip(predicted, required)
        )
        useful = any(
            str(item.get("metric", "")).strip().casefold() == str(gold.get("target", "")).strip().casefold()
            or str(item.get("period", "")).strip().casefold() == str(gold.get("period", "")).strip().casefold()
            for item, gold in zip(predicted, required)
        )
        if routing_ok and slots_ok:
            output["routing_correct_and_slots_correct"] += 1
        elif routing_ok:
            output["routing_correct_but_slot_incomplete"] += 1
        elif useful:
            output["routing_wrong_but_slots_semantically_useful"] += 1
        else:
            output["routing_and_slots_wrong"] += 1
    return dict(output)


def class_metrics(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [Intent.DIRECT_FACT.value, Intent.MULTI_EVIDENCE.value, Intent.CALCULATION.value]
    expected: dict[str, str] = {}
    predicted: dict[str, str] = {}
    for row in rows:
        qid = str(row["case_id"])
        expected[qid] = Intent.CALCULATION.value if row.get("requires_calculation") else Intent.MULTI_EVIDENCE.value if row.get("requires_multiple_sources") else Intent.DIRECT_FACT.value
    for record in records:
        predicted[record["question_id"]] = str((record.get("plan") or {}).get("intent") or "INVALID")
    matrix = {gold: {label: sum(expected[qid] == gold and predicted.get(qid) == label for qid in expected) for label in labels + ["INVALID"]} for gold in labels}
    metrics: dict[str, Any] = {"confusion_matrix": matrix}
    for label in labels:
        tp = matrix[label][label]
        actual = sum(matrix[label].values())
        predicted_count = sum(row.get(label, 0) for row in matrix.values())
        metrics[label] = {"correct": tp, "actual": actual, "predicted": predicted_count, "precision": tp / predicted_count if predicted_count else 0.0, "recall": tp / actual if actual else 0.0}
    return metrics


def run_formal(config: dict[str, Any], envelopes: tuple[Any, ...]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    provider = BailianProvider(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_name=config["model"],
        enable_thinking=False,
        temperature=0.0,
        max_retries=MAX_RETRIES,
    )
    service = SupervisorService(provider)
    records: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    started_all = time.perf_counter()
    try:
        for index, envelope in enumerate(envelopes, start=1):
            started = time.perf_counter()
            run = service.plan(envelope.question)
            record = record_from_run(envelope, run, provider, started)
            record["call_index"] = index
            records.append(record)
            if record["provider_response_success"] is False:
                failure = {
                    "question_id": envelope.question_id,
                    "call_index": index,
                    "exception_type": (record["exception_chain"] or [{"type": "provider_failure"}])[0].get("type"),
                    "sanitized_cause_chain": record["exception_chain"],
                }
                break
    finally:
        provider.close()
    elapsed_ms = (time.perf_counter() - started_all) * 1000
    return records, failure, elapsed_ms


def score_and_write(records: list[dict[str, Any]], rows: list[dict[str, Any]], requirements: dict[str, Any], config: dict[str, Any], wall_time_ms: float) -> dict[str, Any]:
    metrics = evaluate(records, rows, requirements)
    safety_extra = raw_action_operation_safety(records)
    invented = sum(raw_numeric_leakage(record.get("raw_response")) for record in records)
    class_info = class_metrics(records, rows)
    attribution = routing_slot_attribution(records, rows, requirements)
    schema_valid = metrics["plan_validity"]["schema_valid"]
    validator_pass = metrics["plan_validity"]["plan_validator_pass"]
    structured_success = sum(bool(record.get("structured_output_success")) for record in records)
    provider_success = sum(bool(record.get("provider_response_success")) for record in records)
    parse_failures = sum(record.get("error_type") == "parse_failure" for record in records)
    safety = metrics["safety"] | safety_extra | {"invented_numeric_values": invented}
    latency = metrics["latency_cost"] | {
        "max_latency_ms": max((float(record.get("latency_ms") or 0.0) for record in records), default=None),
        "total_wall_time_ms": wall_time_ms,
        "estimated_cost_usd": metrics["latency_cost"].get("estimated_cost_usd"),
    }
    write_json(OUT / "structured-output-metrics.json", {
        "provider_response_success": provider_success,
        "structured_output_success": structured_success,
        "schema_valid": schema_valid,
        "plan_validator_pass": validator_pass,
        "parse_failures": parse_failures,
        "questions": len(records),
    })
    write_json(OUT / "plan-validity.json", metrics["plan_validity"] | {"schema_valid_rate": schema_valid / len(records), "plan_validator_pass_rate": validator_pass / len(records)})
    write_json(OUT / "routing-confusion-matrix.json", class_info["confusion_matrix"])
    write_json(OUT / "routing-metrics.json", {label: class_info[label] for label in [Intent.DIRECT_FACT.value, Intent.MULTI_EVIDENCE.value, Intent.CALCULATION.value]})
    write_json(OUT / "calculation-routing.json", metrics["calculation"])
    write_json(OUT / "calculation-operation.json", {"operation_correct": metrics["calculation"]["operation_correct"], "cohort": 11, "wrong_operation": 11 - metrics["calculation"]["operation_correct"]})
    write_json(OUT / "calculation-operands.json", {"all_operand_slots_correct": metrics["calculation"]["all_operand_slots_correct"], "cohort": 11})
    write_json(OUT / "slot-metrics.json", metrics["slots"])
    write_json(OUT / "safety-analysis.json", safety)
    write_json(OUT / "failure-taxonomy.json", metrics["failure_taxonomy"] | {
        "provider_failure": len(records) - provider_success,
        "structured_output_failure": len(records) - structured_success,
        "schema_failure": len(records) - schema_valid,
        "plan_validator_failure": schema_valid - validator_pass,
        "semantic_plan_failure": sum(value for key, value in metrics["failure_taxonomy"].items() if key not in {"SP0_correct", "SP10_schema_invalid", "SP11_answer_leakage"}),
        "invalid_action": safety_extra["invalid_action"],
        "unknown_operation": safety_extra["unknown_operation"],
    })
    write_json(OUT / "routing-slot-attribution.json", attribution)
    write_json(OUT / "latency-token-cost.json", latency)
    write_json(OUT / "r0-r1-model-role-ablation.json", {
        "financial_sft_r0": {
            "schema_valid": "0/72",
            "plan_validator_pass": "0/72",
            "parse_failure": "72/72",
            "answer_leakage": "55/72",
            "calculation_recall": "N/A",
            "metric_slot": "N/A",
            "period_slot": "N/A",
            "semantic_metrics": "NOT EVALUABLE: 0/72 valid SupervisorPlans",
        },
        "bailian_r1": {
            "schema_valid": f"{schema_valid}/72",
            "plan_validator_pass": f"{validator_pass}/72",
            "answer_leakage": f"{safety['answer_leakage']}/72",
            "calculation_recall": f"{metrics['calculation']['true_positive']}/11",
            "metric_slot": metrics["slots"]["metric_accuracy"],
            "period_slot": metrics["slots"]["period_accuracy"],
        },
    })
    calc = metrics["calculation"]
    slots = metrics["slots"]
    strong = (
        schema_valid / len(records) >= 0.98
        and validator_pass / len(records) >= 0.98
        and calc["recall"] == 1.0
        and calc["false_positive"] == 0
        and calc["operation_correct"] == 11
        and slots["metric_accuracy"] >= 0.90
        and slots["period_accuracy"] >= 0.95
        and calc["all_operand_slots_correct"] >= 10
        and safety["premature_calculate"] == 0
        and safety["premature_generate"] == 0
        and safety["answer_leakage"] == 0
        and safety["invented_numeric_values"] == 0
    )
    partial = (
        schema_valid / len(records) >= 0.95
        and validator_pass / len(records) >= 0.95
        and calc["recall"] >= 10 / 11
        and calc["false_positive"] == 0
        and slots["metric_accuracy"] >= 0.85
        and slots["period_accuracy"] >= 0.90
        and safety["premature_calculate"] == 0
        and safety["premature_generate"] == 0
        and safety["answer_leakage"] == 0
        and safety["invented_numeric_values"] == 0
    )
    status = "true" if strong else "partial" if partial else "false"
    decision = {
        "gate": GATE,
        "evaluation_role": ROLE,
        "base_commit": BASE_COMMIT,
        "provider": "Alibaba Bailian",
        "model": MODEL,
        "model_role": "strong_general_llm",
        "thinking": False,
        "temperature": 0.0,
        "max_retries": MAX_RETRIES,
        "production_default": "V1",
        "production_switch_allowed": False,
        "formal_run_complete": True,
        "gold_reads_before_prediction_seal": 0,
        "supervisor_calls": len(records),
        "provider_response_success": provider_success,
        "structured_output_success": structured_success,
        "schema_valid": schema_valid,
        "plan_validator_pass": validator_pass,
        "calculation_recall": calc["recall"],
        "calculation_precision": calc["precision"],
        "false_calculation_routing": calc["false_positive"],
        "operation_accuracy": calc["operation_correct"],
        "metric_slot_accuracy": slots["metric_accuracy"],
        "period_slot_accuracy": slots["period_accuracy"],
        "slot_count_accuracy": slots["slot_count_accuracy"],
        "role_accuracy": slots["role_accuracy"],
        "all_operand_slots_correct": calc["all_operand_slots_correct"],
        "premature_calculate": safety["premature_calculate"],
        "premature_generate": safety["premature_generate"],
        "premature_abstain": safety["premature_abstain"],
        "invalid_action": safety["invalid_action"],
        "unknown_operation": safety["unknown_operation"],
        "answer_leakage": safety["answer_leakage"],
        "invented_numeric_values": safety["invented_numeric_values"],
        "general_llm_supervisor_effective": status,
        "supervisor_frozen": strong,
        "dominant_failure": "none" if strong else "bailian_supervisor_failure",
        "next_gate": "v2_02_top20_financial_fact_expansion" if strong else "v2_01_r1_failure_review",
    }
    write_json(OUT / "decision.json", decision)
    (OUT / "README.md").write_text(
        "# NF-V2-01 R1 — Formal Bailian Strong-General Supervisor\n\n"
        "Exactly one sequential temperature-zero structured-output call was made per frozen question. "
        "Gold/reference annotations were opened only after the prediction seal. "
        "No retrieval, reranking, binding, calculation, generation, validation, or repair executed.\n",
        encoding="utf-8",
    )
    return decision


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config, error = load_env_config()
    if error or config is None:
        write_json(OUT / "formal-run-config.json", {"gate": GATE, "base_commit": BASE_COMMIT, "formal_evaluation_status": "infrastructure_blocked", "block_reason": error or "missing config", "production_switch_allowed": False})
        empty_blocked_artifacts(error or "missing config", [])
        print(json.dumps({"formal_evaluation_status": "infrastructure_blocked", "block_reason": error or "missing config"}, sort_keys=True))
        return 0
    envelopes = load_question_envelopes(QUESTIONS)
    if len(envelopes) != QUESTION_COUNT:
        reason = f"expected {QUESTION_COUNT} question envelopes, got {len(envelopes)}"
        empty_blocked_artifacts(reason, [])
        print(json.dumps({"formal_evaluation_status": "infrastructure_blocked", "block_reason": reason}, sort_keys=True))
        return 0
    ready, _, reason = pre_run_seal(config, envelopes)
    if not ready:
        empty_blocked_artifacts(reason or "pre-run seal failed", [])
        print(json.dumps({"formal_evaluation_status": "infrastructure_blocked", "block_reason": reason}, sort_keys=True))
        return 0
    records, failure, wall_time_ms = run_formal(config, envelopes)
    write_prediction_seal(records, complete=failure is None and len(records) == QUESTION_COUNT, status="completed" if failure is None else "infrastructure_regression", failure=failure)
    if failure is not None or len(records) != QUESTION_COUNT:
        empty_blocked_artifacts("new infrastructure failure during formal replay", records)
        # Restore the prediction seal after blocked-artifact writes so it remains
        # the authoritative sealed execution record.
        write_prediction_seal(records, complete=False, status="infrastructure_regression", failure=failure)
        print(json.dumps({"formal_evaluation_status": "infrastructure_regression", "formal_run_complete": False, "supervisor_calls": len(records), "failure": failure}, sort_keys=True))
        return 0
    # Only this branch reads frozen evaluation annotations, after prediction seal.
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    requirements = json.loads(QUERY_REQUIREMENTS.read_text(encoding="utf-8"))
    decision = score_and_write(records, rows, requirements, config, wall_time_ms)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
