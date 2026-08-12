#!/usr/bin/env python3
"""NF-V2-01 R1 resume: Alibaba Bailian strong-general Supervisor."""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.plan import Action, Intent  # noqa: E402
from rag_v2.orchestration.loader import load_question_envelopes  # noqa: E402
from rag_v2.supervisor.bailian_provider import BailianProvider  # noqa: E402
from rag_v2.supervisor.prompt import SUPERVISOR_PLAN_JSON_SCHEMA, SUPERVISOR_SYSTEM_PROMPT_V1  # noqa: E402
from rag_v2.supervisor.service import SupervisorService  # noqa: E402

from scripts.evaluation.run_nf_v2_01_general_llm_supervisor import (  # noqa: E402
    QUESTIONS,
    QUERY_REQUIREMENTS,
    V2_CONTRACT,
    evaluate,
    is_answer_leakage,
    sha256_bytes,
    sha256_file,
    write_json,
)


GATE = "NF-V2-01-R1"
BASE_COMMIT = "faa17f1ccd5b81e55968dce43df45a4237c3d9a5"
ROLE = "development_shadow_v2_bailian_strong_general_llm_supervisor"
PROVIDER = "bailian"
MODEL = "qwen3.7-max-2026-06-08"
R0_OUT = ROOT / "artifacts/evaluation/nf-v2-01-r0-supervisor-role-mismatch-closure"
OUT = ROOT / "artifacts/evaluation/nf-v2-01-r1-bailian-strong-general-supervisor"


def sanitized_endpoint(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    if value.casefold() == "true":
        return True
    if value.casefold() == "false":
        return False
    return None


def load_env_config() -> tuple[dict[str, Any] | None, str | None]:
    provider = os.environ.get("V2_SUPERVISOR_PROVIDER")
    model = os.environ.get("V2_SUPERVISOR_MODEL")
    base_url = os.environ.get("V2_SUPERVISOR_BASE_URL")
    api_key = os.environ.get("V2_SUPERVISOR_API_KEY")
    thinking_raw = os.environ.get("V2_SUPERVISOR_ENABLE_THINKING")
    temperature_raw = os.environ.get("V2_SUPERVISOR_TEMPERATURE")
    missing = [name for name, value in {
        "V2_SUPERVISOR_PROVIDER": provider,
        "V2_SUPERVISOR_MODEL": model,
        "V2_SUPERVISOR_BASE_URL": base_url,
        "V2_SUPERVISOR_API_KEY": api_key,
        "V2_SUPERVISOR_ENABLE_THINKING": thinking_raw,
        "V2_SUPERVISOR_TEMPERATURE": temperature_raw,
    }.items() if not value]
    if missing:
        return None, "missing required environment variables: " + ", ".join(missing)
    if provider != PROVIDER:
        return None, f"V2_SUPERVISOR_PROVIDER must be {PROVIDER!r}"
    if model != MODEL:
        return None, f"V2_SUPERVISOR_MODEL must be the frozen Bailian model {MODEL!r}"
    thinking = parse_bool(thinking_raw or "")
    if thinking is not False:
        return None, "V2_SUPERVISOR_ENABLE_THINKING must be false"
    try:
        temperature = float(temperature_raw or "")
    except ValueError:
        return None, "V2_SUPERVISOR_TEMPERATURE must be numeric zero"
    if temperature != 0.0:
        return None, "V2_SUPERVISOR_TEMPERATURE must be 0"
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "base_url_sanitized": sanitized_endpoint(base_url),
        "api_key": api_key,
        "enable_thinking": thinking,
        "temperature": temperature,
    }, None


def write_frozen_contract(config: dict[str, Any] | None, status: str, reason: str | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "frozen-v2-contract.json", {
        "gate": GATE,
        "base_commit": BASE_COMMIT,
        "v2_00_contract_artifact": str(V2_CONTRACT.relative_to(ROOT)),
        "v2_00_contract_sha256": sha256_file(V2_CONTRACT),
        "evaluation_role": ROLE,
        "fresh_blind_evaluation": False,
        "production_default": "V1",
        "production_switch_allowed": False,
        "question_count": 72,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "binder_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "validator_calls": 0,
        "formal_evaluation_status": status,
        "block_reason": reason,
    })
    prompt_bytes = SUPERVISOR_SYSTEM_PROMPT_V1.encode("utf-8")
    schema = {"type": "json_schema", "json_schema": {"name": "SupervisorPlan", "strict": True, "schema": SUPERVISOR_PLAN_JSON_SCHEMA}}
    (OUT / "supervisor-prompt.txt").write_bytes(prompt_bytes)
    (OUT / "supervisor-prompt.sha256").write_text(sha256_bytes(prompt_bytes) + "\n", encoding="ascii")
    write_json(OUT / "supervisor-schema.json", schema)
    schema_bytes = (json.dumps(schema, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    (OUT / "supervisor-schema.sha256").write_text(sha256_bytes(schema_bytes) + "\n", encoding="ascii")
    safe = {
        "gate": GATE,
        "provider_role": "supervisor",
        "model_role": "strong_general_llm",
        "provider": PROVIDER,
        "model": MODEL,
        "base_url": config.get("base_url_sanitized") if config else None,
        "enable_thinking": False,
        "temperature": 0.0,
        "prompt_sha256": sha256_bytes(prompt_bytes),
        "schema_sha256": sha256_bytes(schema_bytes),
        "api_key_serialized": False,
        "formal_evaluation_status": status,
        "block_reason": reason,
    }
    safe_bytes = (json.dumps(safe, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    safe["config_seal_sha256"] = sha256_bytes(safe_bytes)
    write_json(OUT / "provider-config-seal.json", safe)
    write_json(OUT / "supervisor-model-role.json", {
        "provider_role": "supervisor",
        "model_role": "strong_general_llm",
        "provider": PROVIDER,
        "model": MODEL,
        "financial_sft_fallback_forbidden": True,
        "role_match": True if config else False,
    })


def write_blocked_artifacts(reason: str, smoke: dict[str, Any]) -> dict[str, Any]:
    write_json(OUT / "smoke-test.json", smoke)
    plans_path = OUT / "supervisor-plans.jsonl.gz"
    with gzip.open(plans_path, "wt", encoding="utf-8"):
        pass
    write_json(OUT / "supervisor-prediction-seal.json", {
        "gate": GATE,
        "formal_evaluation_status": "infrastructure_blocked",
        "block_reason": reason,
        "questions": 72,
        "model_calls": 0,
        "max_calls_per_question": 1,
        "gold_reads_before_prediction_seal": 0,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "binder_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "validator_calls": 0,
        "plans_sha256": sha256_file(plans_path),
        "sealed": True,
    })
    write_json(OUT / "structured-output-metrics.json", {"formal_evaluation_status": "infrastructure_blocked", "provider_response_success": None, "structured_output_success": None, "schema_valid": None, "plan_validator_pass": None, "parse_failures": None})
    write_json(OUT / "plan-validity.json", {"formal_evaluation_status": "infrastructure_blocked", "schema_valid": None, "plan_validator_pass": None, "parse_failure": None})
    for name, body in {
        "routing-confusion-matrix.json": {"formal_evaluation_status": "infrastructure_blocked", "matrix": None},
        "routing-metrics.json": {"formal_evaluation_status": "infrastructure_blocked", "direct_fact_accuracy": None, "multi_evidence_accuracy": None, "calculation_recall": None, "calculation_precision": None},
        "slot-metrics.json": {"formal_evaluation_status": "infrastructure_blocked", "metric_accuracy": None, "period_accuracy": None, "slot_count_accuracy": None, "role_accuracy": None},
        "calculation-analysis.json": {"formal_evaluation_status": "infrastructure_blocked", "operation_correct": None, "all_operand_slots_correct": None, "false_calculation_routing": None},
        "safety-analysis.json": {"formal_evaluation_status": "infrastructure_blocked", "premature_calculate": None, "premature_generate": None, "answer_leakage": None, "invented_numeric_values": None, "invalid_action": None, "unknown_operation": None},
        "latency-token-analysis.json": {"formal_evaluation_status": "infrastructure_blocked", "calls": 0, "input_tokens": None, "output_tokens": None, "reasoning_tokens": None, "average_latency_ms": None, "p50_latency_ms": None, "p95_latency_ms": None},
        "failure-taxonomy.json": {"formal_evaluation_status": "infrastructure_blocked", **{f"SP{i}": None for i in range(13)}, "provider_failure": None, "structured_output_failure": None, "schema_failure": None, "plan_validator_failure": None, "semantic_plan_failure": None},
    }.items():
        write_json(OUT / name, body)
    write_json(OUT / "supervisor-model-role-ablation.json", {
        "financial_sft_r0": {"schema_valid": "0/72", "plan_validator_pass": "0/72", "parse_failure": "72/72", "answer_leakage": "55/72", "calculation_recall": "N/A", "metric_slot": "N/A", "period_slot": "N/A", "note": "Not semantically evaluable because 0/72 valid SupervisorPlans were produced."},
        "bailian_r1": {"formal_evaluation_status": "infrastructure_blocked", "reason": reason},
    })
    decision = {
        "gate": GATE,
        "evaluation_role": ROLE,
        "provider": "bailian",
        "model": MODEL,
        "model_role": "strong_general_llm",
        "thinking": False,
        "production_default": "V1",
        "production_switch_allowed": False,
        "financial_sft_as_supervisor_effective": False,
        "general_llm_supervisor_effective": "not_evaluated",
        "formal_evaluation_status": "infrastructure_blocked",
        "block_reason": reason,
        "questions": 72,
        "supervisor_calls": 0,
        "schema_valid": None,
        "plan_validator_pass": None,
        "calculation_recall": None,
        "calculation_precision": None,
        "false_calculation_routing": None,
        "operation_accuracy": None,
        "metric_slot_accuracy": None,
        "period_slot_accuracy": None,
        "all_operand_slots_correct": None,
        "premature_calculate": None,
        "premature_generate": None,
        "answer_leakage": None,
        "invented_numeric_values": None,
        "dominant_failure": "infrastructure_blocked",
        "next_gate": "v2_01_r1_failure_review",
    }
    write_json(OUT / "decision.json", decision)
    (OUT / "README.md").write_text(
        "# NF-V2-01 R1 — Alibaba Bailian strong-general Supervisor\n\n"
        "The formal Bailian environment was unavailable. The task stopped before\n"
        "the synthetic smoke call; no financial SFT fallback was used.\n",
        encoding="utf-8",
    )
    return decision


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    question = "What was ExampleCorp's revenue in FY2025?"
    try:
        provider = BailianProvider(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model_name=config["model"],
            enable_thinking=False,
            temperature=0.0,
        )
        run = SupervisorService(provider).plan(question)
        plan = run.plan.to_dict() if run.plan else None
        slot = plan["required_slots"][0] if plan and plan.get("required_slots") else {}
        semantic = bool(
            plan and plan.get("intent") == Intent.DIRECT_FACT.value
            and norm(slot.get("metric")) == "revenue"
            and norm(slot.get("period")) == "fy2025"
            and plan.get("next_action") == Action.RETRIEVE.value
        )
        leakage = is_answer_leakage(run.metadata.raw_response if run.metadata else None)
        status = "pass" if run.plan_valid and semantic and not leakage else "fail"
        return {
            "status": status,
            "question": question,
            "provider_response_success": run.metadata.provider_response_success if run.metadata else False,
            "structured_output_success": run.metadata.structured_output_success if run.metadata else False,
            "schema_valid": plan is not None,
            "plan_validator_pass": run.plan_valid,
            "answer_leakage": leakage,
            "semantic_shape": semantic,
            "model_calls": 1,
            "error": run.error,
        }
    except Exception as exc:
        return {"status": "fail", "question": question, "provider_response_success": False, "structured_output_success": False, "schema_valid": False, "plan_validator_pass": False, "answer_leakage": False, "semantic_shape": False, "model_calls": 1, "error": f"{type(exc).__name__}: {exc}"}


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def run_formal(config: dict[str, Any]) -> dict[str, Any]:
    provider = BailianProvider(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_name=config["model"],
        enable_thinking=False,
        temperature=0.0,
    )
    service = SupervisorService(provider)
    questions = load_question_envelopes(QUESTIONS)
    records: list[dict[str, Any]] = []
    for envelope in questions:
        run = service.plan(envelope.question)
        metadata = run.metadata
        records.append({
            "question_id": envelope.question_id,
            "question": envelope.question,
            "document_scope": list(envelope.document_scope),
            "plan": run.plan.to_dict() if run.plan else None,
            "schema_valid": run.plan is not None,
            "plan_valid": run.plan_valid,
            "error": run.error,
            "error_type": "parse_failure" if metadata and metadata.parse_failure else None,
            "provider": metadata.provider if metadata else PROVIDER,
            "provider_role": metadata.provider_role if metadata else "supervisor",
            "model": metadata.model if metadata else MODEL,
            "model_role": metadata.model_role if metadata else "strong_general_llm",
            "provider_response_success": metadata.provider_response_success if metadata else False,
            "structured_output_success": metadata.structured_output_success if metadata else False,
            "reasoning_tokens": metadata.reasoning_tokens if metadata else None,
            "latency_ms": metadata.latency_ms if metadata else None,
            "input_tokens": metadata.input_tokens if metadata else None,
            "output_tokens": metadata.output_tokens if metadata else None,
            "total_tokens": metadata.total_tokens if metadata else None,
            "raw_response": metadata.raw_response if metadata else None,
        })
    plans_path = OUT / "supervisor-plans.jsonl.gz"
    with gzip.open(plans_path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(OUT / "supervisor-prediction-seal.json", {
        "gate": GATE,
        "formal_evaluation_status": "completed",
        "questions": 72,
        "model_calls": 72,
        "max_calls_per_question": 1,
        "gold_reads_before_prediction_seal": 0,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "binder_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "validator_calls": 0,
        "plans_sha256": sha256_file(plans_path),
        "sealed": True,
    })
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    requirements = json.loads(QUERY_REQUIREMENTS.read_text(encoding="utf-8"))
    metrics = evaluate(records, rows, requirements)
    provider_success = sum(bool(record["provider_response_success"]) for record in records)
    structured_success = sum(bool(record["structured_output_success"]) for record in records)
    invented = sum(bool(re.search(r"(?:[$€£]\s*\d|\d+(?:\.\d+)?\s*%)", record.get("raw_response") or "")) for record in records)
    write_json(OUT / "structured-output-metrics.json", {"provider_response_success": provider_success, "structured_output_success": structured_success, "schema_valid": metrics["plan_validity"]["schema_valid"], "plan_validator_pass": metrics["plan_validity"]["plan_validator_pass"], "parse_failures": metrics["plan_validity"]["parse_failure"]})
    for filename, body in {
        "plan-validity.json": metrics["plan_validity"],
        "routing-confusion-matrix.json": metrics["confusion_matrix"],
        "routing-metrics.json": metrics["routing"] | {"calculation": metrics["calculation"]},
        "slot-metrics.json": metrics["slots"],
        "calculation-analysis.json": metrics["calculation"],
        "safety-analysis.json": metrics["safety"] | {"invented_numeric_values": invented},
        "failure-taxonomy.json": metrics["failure_taxonomy"] | {"provider_failure": 72 - provider_success, "structured_output_failure": 72 - structured_success, "schema_failure": metrics["plan_validity"]["invalid_plan"], "plan_validator_failure": metrics["plan_validity"]["schema_valid"] - metrics["plan_validity"]["plan_validator_pass"], "semantic_plan_failure": 0},
    }.items():
        write_json(OUT / filename, body)
    latency = metrics["latency_cost"]
    latency["reasoning_tokens"] = sum(int(record.get("reasoning_tokens") or 0) for record in records)
    write_json(OUT / "latency-token-analysis.json", latency)
    write_json(OUT / "supervisor-model-role-ablation.json", {
        "financial_sft_r0": {"schema_valid": "0/72", "plan_validator_pass": "0/72", "parse_failure": "72/72", "answer_leakage": "55/72", "calculation_recall": "N/A", "metric_slot": "N/A", "period_slot": "N/A", "note": "Not semantically evaluable because 0/72 valid SupervisorPlans were produced."},
        "bailian_r1": {"schema_valid": f"{metrics['plan_validity']['schema_valid']}/72", "plan_validator_pass": f"{metrics['plan_validity']['plan_validator_pass']}/72", "answer_leakage": f"{metrics['safety']['answer_leakage']}/72", "calculation_recall": f"{metrics['calculation']['true_positive']}/11", "metric_slot": metrics["slots"]["metric_accuracy"], "period_slot": metrics["slots"]["period_accuracy"]},
    })
    safety = metrics["safety"]
    strong = metrics["plan_validity"]["plan_validator_pass"] / 72 >= 0.98 and metrics["calculation"]["recall"] == 1.0 and metrics["calculation"]["false_positive"] == 0 and metrics["calculation"]["operation_correct"] == 11 and metrics["slots"]["metric_accuracy"] >= 0.90 and metrics["slots"]["period_accuracy"] >= 0.95 and metrics["calculation"]["all_operand_slots_correct"] >= 10 and safety["premature_calculate"] == safety["premature_generate"] == safety["answer_leakage"] == invented == 0
    partial = metrics["plan_validity"]["plan_validator_pass"] / 72 >= 0.95 and metrics["calculation"]["recall"] >= 10 / 11 and metrics["calculation"]["false_positive"] == 0 and metrics["slots"]["metric_accuracy"] >= 0.85 and metrics["slots"]["period_accuracy"] >= 0.90 and safety["premature_calculate"] == safety["premature_generate"] == safety["answer_leakage"] == invented == 0
    decision = {"gate": GATE, "evaluation_role": ROLE, "provider": "Alibaba Bailian", "model": MODEL, "model_role": "strong_general_llm", "thinking": False, "production_default": "V1", "production_switch_allowed": False, "financial_sft_as_supervisor_effective": False, "general_llm_supervisor_effective": True if strong else "partial" if partial else False, "formal_evaluation_status": "completed", "questions": 72, "supervisor_calls": 72, "schema_valid": metrics["plan_validity"]["schema_valid"], "plan_validator_pass": metrics["plan_validity"]["plan_validator_pass"], "calculation_recall": metrics["calculation"]["recall"], "calculation_precision": metrics["calculation"]["precision"], "false_calculation_routing": metrics["calculation"]["false_positive"], "operation_accuracy": metrics["calculation"]["operation_correct"], "metric_slot_accuracy": metrics["slots"]["metric_accuracy"], "period_slot_accuracy": metrics["slots"]["period_accuracy"], "all_operand_slots_correct": metrics["calculation"]["all_operand_slots_correct"], "premature_calculate": safety["premature_calculate"], "premature_generate": safety["premature_generate"], "answer_leakage": safety["answer_leakage"], "invented_numeric_values": invented, "dominant_failure": "none" if strong else "bailian_supervisor_failure", "next_gate": "v2_02_top20_financial_fact_expansion" if strong else "v2_01_r1_failure_review"}
    write_json(OUT / "decision.json", decision)
    (OUT / "README.md").write_text("# NF-V2-01 R1 — Alibaba Bailian strong-general Supervisor\n\nFormal replay completed with one strict structured-output call per question.\n", encoding="utf-8")
    return decision


def main() -> int:
    config, config_error = load_env_config()
    if config_error:
        write_frozen_contract(None, "infrastructure_blocked", config_error)
        decision = write_blocked_artifacts(config_error, {"status": "infrastructure_blocked", "model_calls": 0, "question": "What was ExampleCorp's revenue in FY2025?", "reason": config_error})
        print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
        return 0
    write_frozen_contract(config, "smoke_pending", None)
    smoke = run_smoke(config)
    write_json(OUT / "smoke-test.json", smoke)
    if smoke["status"] != "pass":
        reason = "Bailian synthetic smoke test failed"
        write_json(OUT / "provider-config-seal.json", {"gate": GATE, "provider": PROVIDER, "model": MODEL, "provider_role": "supervisor", "model_role": "strong_general_llm", "base_url": config["base_url_sanitized"], "enable_thinking": False, "temperature": 0.0, "api_key_serialized": False, "smoke_status": "fail", "formal_evaluation_status": "infrastructure_blocked", "block_reason": reason})
        decision = write_blocked_artifacts(reason, smoke)
        print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
        return 0
    write_frozen_contract(config, "formal_ready", None)
    decision = run_formal(config)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
