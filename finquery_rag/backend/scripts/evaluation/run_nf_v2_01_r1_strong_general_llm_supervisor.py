#!/usr/bin/env python3
"""NF-V2-01 R1: role-isolated Strong General LLM Supervisor evaluation."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.orchestration.loader import load_question_envelopes  # noqa: E402
from rag_v2.supervisor.prompt import SUPERVISOR_SYSTEM_PROMPT_V1  # noqa: E402
from rag_v2.supervisor.service import SupervisorService  # noqa: E402
from rag_v2.supervisor.strong_general_provider import StrongGeneralAPIProvider  # noqa: E402

from scripts.evaluation.run_nf_v2_01_general_llm_supervisor import (  # noqa: E402
    BASE_COMMIT as R0_BASE_COMMIT,
    QUESTIONS,
    QUERY_REQUIREMENTS,
    evaluate,
    sha256_file,
    write_json,
)


GATE = "NF-V2-01-R1"
BASE_COMMIT = "df771c08d2deea23f4c0ed451856a386202899e7"
ROLE = "development_shadow_v2_strong_general_llm_supervisor"
R0_OUT = ROOT / "artifacts/evaluation/nf-v2-01-general-llm-supervisor"
CLOSURE_OUT = ROOT / "artifacts/evaluation/nf-v2-01-r0-supervisor-role-mismatch-closure"
OUT = ROOT / "artifacts/evaluation/nf-v2-01-r1-strong-general-llm-supervisor"
FINANCIAL_SFT_MODEL = "finquery-finance-v2-lr010-150"


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def write_r0_closure() -> None:
    """Stage A: close R0 from sealed artifacts without any model call."""

    R0_OUT.mkdir(parents=True, exist_ok=True)
    plan_validity = json.loads((R0_OUT / "plan-validity.json").read_text(encoding="utf-8"))
    safety = json.loads((R0_OUT / "safety-analysis.json").read_text(encoding="utf-8"))
    failure = json.loads((R0_OUT / "failure-taxonomy.json").read_text(encoding="utf-8"))
    seal = json.loads((R0_OUT / "supervisor-prediction-seal.json").read_text(encoding="utf-8"))
    reference = {
        "gate": "NF-V2-01-R0",
        "model_calls": 0,
        "schema_valid": f"{plan_validity['schema_valid']}/72",
        "plan_validator_pass": f"{plan_validity['plan_validator_pass']}/72",
        "parse_failure": f"{plan_validity['parse_failure']}/72",
        "answer_leakage": f"{safety['answer_leakage']}/72",
        "SP10_schema_invalid": failure.get("SP10_schema_invalid", 0),
        "SP11_answer_leakage": failure.get("SP11_answer_leakage", 0),
        "gold_reads_before_prediction_seal": seal.get("gold_reads_before_prediction_seal"),
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "downstream_calls": 0,
    }
    write_json(CLOSURE_OUT / "r0-metrics-reference.json", reference)
    write_json(CLOSURE_OUT / "role-mismatch-analysis.json", {
        "gate": "NF-V2-01-R0-closure",
        "financial_sft_as_supervisor_effective": False,
        "general_llm_supervisor_effective": "not_evaluated",
        "supervisor_model_role_match": False,
        "dominant_failure": "supervisor_model_role_mismatch",
        "evidence": reference,
        "interpretation": "The Financial SFT model was trained/positioned as a domain generator rather than an Agent control-plane model. Because no valid SupervisorPlan was produced, downstream routing and slot metrics are not valid estimates of a Strong General LLM Supervisor's semantic capability.",
        "model_calls": 0,
        "retrieval_calls": 0,
        "fresh_blind_evaluation": False,
        "production_switch_allowed": False,
    })
    write_json(CLOSURE_OUT / "decision.json", {
        "gate": "NF-V2-01-R0-closure",
        "evaluation_role": "development_shadow_v2_r0_role_mismatch_closure",
        "financial_sft_as_supervisor_effective": False,
        "general_llm_supervisor_effective": "not_evaluated",
        "supervisor_model_role_match": False,
        "dominant_failure": "supervisor_model_role_mismatch",
        "model_calls": 0,
        "retrieval_calls": 0,
        "production_default": "V1",
        "production_switch_allowed": False,
        "next_gate": "NF-V2-01-R1",
    })
    (CLOSURE_OUT / "README.md").write_text(
        "# NF-V2-01 R0 role-mismatch closure\n\n"
        "R0 is closed from its sealed artifacts without replay. The financial SFT\n"
        "checkpoint is not treated as a General LLM control-plane model.\n",
        encoding="utf-8",
    )


def discover_model(base_url: str) -> str:
    request = urllib.request.Request(base_url.rstrip("/") + "/models", method="GET")
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("data", []) if isinstance(payload, dict) else []
    if models and isinstance(models[0], dict) and models[0].get("id"):
        return str(models[0]["id"])
    raise RuntimeError("strong-general endpoint returned no model")


def endpoint_available(base_url: str) -> tuple[bool, str | None]:
    try:
        discover_model(base_url)
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def is_financial_sft_model(model: str | None) -> bool:
    if not model:
        return False
    value = model.casefold()
    return value == FINANCIAL_SFT_MODEL.casefold() or "finquery-finance" in value or "d24_finance" in value


def write_common_artifacts(
    *,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    structured_output: bool,
    status: str,
    reason: str | None,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v2_contract = {
        "gate": GATE,
        "base_commit": BASE_COMMIT,
        "v2_00_base_contract": R0_BASE_COMMIT,
        "evaluation_role": ROLE,
        "fresh_blind_evaluation": False,
        "production_default": "V1",
        "production_switch_allowed": False,
        "model_role": "strong_general_llm",
        "question_count": 72,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "binder_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "validator_calls": 0,
        "formal_evaluation_status": status,
        "block_reason": reason,
    }
    write_json(OUT / "frozen-v2-contract.json", v2_contract)
    write_json(OUT / "supervisor-provider-contract.json", {
        "gate": GATE,
        "provider": provider,
        "provider_role": "supervisor",
        "model": model,
        "model_role": "strong_general_llm",
        "base_url": base_url,
        "temperature": 0.0,
        "max_tokens": 512,
        "max_supervisor_calls_per_question": 1,
        "retry": False,
        "structured_output": "json_object" if structured_output else "strict_json_parsing",
        "fallback_provider": None,
        "financial_sft_fallback_forbidden": True,
        "formal_evaluation_status": status,
    })
    write_json(OUT / "supervisor-model-role.json", {
        "provider_role": "supervisor",
        "model_role": "strong_general_llm",
        "provider": provider,
        "model": model,
        "financial_sft_model_rejected": is_financial_sft_model(model),
        "role_match": bool(model) and not is_financial_sft_model(model),
    })
    prompt_bytes = SUPERVISOR_SYSTEM_PROMPT_V1.encode("utf-8")
    (OUT / "supervisor-prompt.txt").write_bytes(prompt_bytes)
    (OUT / "supervisor-prompt.sha256").write_text(sha256_bytes(prompt_bytes) + "\n", encoding="ascii")


def write_blocked_outputs(reason: str) -> dict[str, Any]:
    plans_path = OUT / "supervisor-plans.jsonl.gz"
    with gzip.open(plans_path, "wt", encoding="utf-8"):
        pass
    seal = {
        "gate": GATE,
        "formal_evaluation_status": "infrastructure_blocked",
        "block_reason": reason,
        "questions": 72,
        "model_calls": 0,
        "gold_reads_before_prediction_seal": 0,
        "reference_answer_reads_before_prediction_seal": 0,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "binder_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "validator_calls": 0,
        "prompt_sha256": sha256_file(OUT / "supervisor-prompt.txt"),
        "plans_sha256": sha256_file(plans_path),
        "sealed": True,
    }
    write_json(OUT / "supervisor-prediction-seal.json", seal)
    write_json(OUT / "structured-output-metrics.json", {"formal_evaluation_status": "infrastructure_blocked", "provider_structured_output_success": None, "reason": reason})
    write_json(OUT / "plan-validity.json", {"formal_evaluation_status": "infrastructure_blocked", "schema_valid": None, "plan_validator_pass": None, "parse_failure": None})
    write_json(OUT / "routing-confusion-matrix.json", {"formal_evaluation_status": "infrastructure_blocked", "matrix": None})
    write_json(OUT / "routing-metrics.json", {"formal_evaluation_status": "infrastructure_blocked", "direct_fact_accuracy": None, "multi_evidence_accuracy": None, "calculation_recall": None, "calculation_precision": None})
    write_json(OUT / "slot-metrics.json", {"formal_evaluation_status": "infrastructure_blocked", "metric_accuracy": None, "period_accuracy": None, "slot_count_accuracy": None, "role_accuracy": None})
    write_json(OUT / "calculation-analysis.json", {"formal_evaluation_status": "infrastructure_blocked", "calculation_recall": None, "false_calculation_routing": None, "operation_accuracy": None, "all_operand_slots_correct": None})
    write_json(OUT / "safety-analysis.json", {"formal_evaluation_status": "infrastructure_blocked", "premature_calculate": None, "premature_generate": None, "answer_leakage": None, "invented_numeric_values": None, "invalid_action": None, "unknown_operation": None})
    write_json(OUT / "failure-taxonomy.json", {f"SP{i}": None for i in range(13)} | {"formal_evaluation_status": "infrastructure_blocked"})
    write_json(OUT / "latency-cost-analysis.json", {"formal_evaluation_status": "infrastructure_blocked", "calls": 0, "input_tokens": None, "output_tokens": None, "average_latency_ms": None, "p50_latency_ms": None, "p95_latency_ms": None, "estimated_cost_usd": None})
    write_json(OUT / "supervisor-model-role-ablation.json", {
        "financial_sft_r0": {"schema_valid": "0/72", "plan_validator_pass": "0/72", "answer_leakage": "55/72", "calculation_recall": "N/A", "metric_slot": "N/A", "period_slot": "N/A", "note": "Not semantically evaluable because 0/72 valid SupervisorPlans were produced."},
        "strong_general_r1": {"status": "infrastructure_blocked", "reason": reason},
    })
    decision = {
        "gate": GATE,
        "evaluation_role": ROLE,
        "production_default": "V1",
        "production_switch_allowed": False,
        "supervisor_model_role": "strong_general_llm",
        "financial_sft_as_supervisor_effective": False,
        "general_llm_supervisor_effective": None,
        "formal_evaluation_status": "infrastructure_blocked",
        "block_reason": reason,
        "questions": 72,
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
        "# NF-V2-01 R1 — Strong General LLM Supervisor\n\n"
        "The R0 role-mismatch closure was completed without model calls. R1 was\n"
        "preflighted with an explicit strong-general role and was blocked because\n"
        f"no eligible endpoint/model was available: {reason}\n\n"
        "The running financial SFT service was not reused or treated as a fallback.\n",
        encoding="utf-8",
    )
    return decision


def run_available_replay(base_url: str, model: str, api_key: str, structured_output: bool, timeout: float) -> dict[str, Any]:
    provider = StrongGeneralAPIProvider(
        base_url=base_url,
        api_key=api_key,
        model_name=model,
        temperature=0.0,
        max_tokens=512,
        timeout=timeout,
        structured_output=structured_output,
    )
    service = SupervisorService(provider)
    questions = load_question_envelopes(QUESTIONS)
    records: list[dict[str, Any]] = []
    for envelope in questions:
        run = service.plan(envelope.question)
        metadata = run.metadata
        plan_dict = run.plan.to_dict() if run.plan else None
        raw = metadata.raw_response if metadata else None
        records.append({
            "question_id": envelope.question_id,
            "question": envelope.question,
            "document_scope": list(envelope.document_scope),
            "plan": plan_dict,
            "schema_valid": plan_dict is not None,
            "plan_valid": run.plan_valid,
            "error": run.error,
            "error_type": "parse_failure" if metadata and metadata.parse_failure else None,
            "provider": metadata.provider if metadata else "strong_general_api",
            "provider_role": metadata.provider_role if metadata else "supervisor",
            "model": metadata.model if metadata else model,
            "model_role": metadata.model_role if metadata else "strong_general_llm",
            "latency_ms": metadata.latency_ms if metadata else None,
            "input_tokens": metadata.input_tokens if metadata else None,
            "output_tokens": metadata.output_tokens if metadata else None,
            "total_tokens": metadata.total_tokens if metadata else None,
            "raw_response": raw,
        })
    plans_path = OUT / "supervisor-plans.jsonl.gz"
    with gzip.open(plans_path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    seal = {
        "gate": GATE,
        "formal_evaluation_status": "completed",
        "questions": 72,
        "model_calls": 72,
        "gold_reads_before_prediction_seal": 0,
        "reference_answer_reads_before_prediction_seal": 0,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "binder_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "validator_calls": 0,
        "prompt_sha256": sha256_file(OUT / "supervisor-prompt.txt"),
        "plans_sha256": sha256_file(plans_path),
        "sealed": True,
    }
    write_json(OUT / "supervisor-prediction-seal.json", seal)
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    requirements = json.loads(QUERY_REQUIREMENTS.read_text(encoding="utf-8"))
    metrics = evaluate(records, rows, requirements)
    write_json(OUT / "structured-output-metrics.json", {"provider_structured_output_success": sum(bool(r.get("plan_valid")) for r in records), "schema_parse_success": sum(bool(r.get("schema_valid")) for r in records), "mode": "json_object" if structured_output else "strict_json_parsing"})
    write_json(OUT / "plan-validity.json", metrics["plan_validity"])
    write_json(OUT / "routing-confusion-matrix.json", metrics["confusion_matrix"])
    write_json(OUT / "routing-metrics.json", metrics["routing"] | {"calculation": metrics["calculation"]})
    write_json(OUT / "slot-metrics.json", metrics["slots"])
    write_json(OUT / "calculation-analysis.json", metrics["calculation"])
    safety = metrics["safety"] | {"invented_numeric_values": sum(bool(re.search(r"(?:[$€£]\\s*\\d|\\d+(?:\\.\\d+)?\\s*%)", r.get("raw_response") or "")) for r in records)}
    write_json(OUT / "safety-analysis.json", safety)
    write_json(OUT / "failure-taxonomy.json", metrics["failure_taxonomy"])
    write_json(OUT / "latency-cost-analysis.json", metrics["latency_cost"])
    baseline = {"schema_valid": "0/72", "plan_validator_pass": "0/72", "answer_leakage": "55/72", "calculation_recall": "N/A", "metric_slot": "N/A", "period_slot": "N/A", "note": "Not semantically evaluable because 0/72 valid SupervisorPlans were produced."}
    write_json(OUT / "supervisor-model-role-ablation.json", {"financial_sft_r0": baseline, "strong_general_r1": {"schema_valid": f"{metrics['plan_validity']['schema_valid']}/72", "plan_validator_pass": f"{metrics['plan_validity']['plan_validator_pass']}/72", "answer_leakage": f"{safety['answer_leakage']}/72", "calculation_recall": f"{metrics['calculation']['true_positive']}/11", "metric_slot": metrics["slots"]["metric_accuracy"], "period_slot": metrics["slots"]["period_accuracy"]}})
    strong_validity = metrics["plan_validity"]["plan_validator_pass"] / 72
    strong = (
        strong_validity >= 0.98 and metrics["calculation"]["recall"] == 1.0
        and metrics["calculation"]["false_positive"] == 0 and metrics["calculation"]["operation_correct"] == 11
        and metrics["slots"]["metric_accuracy"] >= 0.90 and metrics["slots"]["period_accuracy"] >= 0.95
        and metrics["calculation"]["all_operand_slots_correct"] >= 10
        and safety["premature_calculate"] == 0 and safety["premature_generate"] == 0
        and safety["answer_leakage"] == 0 and safety["invented_numeric_values"] == 0
    )
    partial = (
        strong_validity >= 0.95 and metrics["calculation"]["recall"] >= 10 / 11
        and metrics["calculation"]["false_positive"] == 0 and metrics["slots"]["metric_accuracy"] >= 0.85
        and metrics["slots"]["period_accuracy"] >= 0.90 and safety["premature_calculate"] == 0
        and safety["premature_generate"] == 0 and safety["answer_leakage"] == 0
        and safety["invented_numeric_values"] == 0
    )
    decision = {
        "gate": GATE,
        "evaluation_role": ROLE,
        "production_default": "V1",
        "production_switch_allowed": False,
        "supervisor_model_role": "strong_general_llm",
        "financial_sft_as_supervisor_effective": False,
        "general_llm_supervisor_effective": True if strong else "partial" if partial else False,
        "formal_evaluation_status": "completed",
        "questions": 72,
        "schema_valid": metrics["plan_validity"]["schema_valid"],
        "plan_validator_pass": metrics["plan_validity"]["plan_validator_pass"],
        "calculation_recall": metrics["calculation"]["recall"],
        "calculation_precision": metrics["calculation"]["precision"],
        "false_calculation_routing": metrics["calculation"]["false_positive"],
        "operation_accuracy": metrics["calculation"]["operation_correct"],
        "metric_slot_accuracy": metrics["slots"]["metric_accuracy"],
        "period_slot_accuracy": metrics["slots"]["period_accuracy"],
        "all_operand_slots_correct": metrics["calculation"]["all_operand_slots_correct"],
        "premature_calculate": safety["premature_calculate"],
        "premature_generate": safety["premature_generate"],
        "answer_leakage": safety["answer_leakage"],
        "invented_numeric_values": safety["invented_numeric_values"],
        "dominant_failure": "none" if strong else "r1_supervisor_failure_review",
        "next_gate": "v2_02_top20_financial_fact_expansion" if strong else "v2_01_r1_failure_review",
    }
    write_json(OUT / "decision.json", decision)
    (OUT / "README.md").write_text("# NF-V2-01 R1 — Strong General LLM Supervisor\n\nCompleted role-isolated replay. No downstream component was executed.\n", encoding="utf-8")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--structured-output", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    # Stage A is always performed first and has no model/retrieval calls.
    write_r0_closure()
    base_url = args.base_url or os.getenv("V2_SUPERVISOR_BASE_URL")
    model = args.model or os.getenv("V2_SUPERVISOR_MODEL")
    structured_output = args.structured_output or os.getenv("V2_SUPERVISOR_STRUCTURED_OUTPUT", "false").casefold() == "true"
    provider_name = os.getenv("V2_SUPERVISOR_PROVIDER", "strong_general_api")
    api_key = args.api_key or os.getenv("V2_SUPERVISOR_API_KEY") or "not-needed"

    if not base_url:
        reason = "V2_SUPERVISOR_BASE_URL is not configured; no Strong General LLM endpoint is available"
        write_common_artifacts(provider=provider_name, model=model, base_url=base_url, structured_output=structured_output, status="infrastructure_blocked", reason=reason)
        decision = write_blocked_outputs(reason)
        print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
        return 0
    available, error = endpoint_available(base_url)
    if not available:
        reason = f"strong-general endpoint unavailable: {error}"
        write_common_artifacts(provider=provider_name, model=model, base_url=base_url, structured_output=structured_output, status="infrastructure_blocked", reason=reason)
        decision = write_blocked_outputs(reason)
        print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
        return 0
    if not model:
        model = discover_model(base_url)
    if is_financial_sft_model(model):
        reason = f"configured endpoint advertises financial SFT model {model}; R1 refuses role-mismatched fallback"
        write_common_artifacts(provider=provider_name, model=model, base_url=base_url, structured_output=structured_output, status="infrastructure_blocked", reason=reason)
        decision = write_blocked_outputs(reason)
        print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
        return 0
    write_common_artifacts(provider=provider_name, model=model, base_url=base_url, structured_output=structured_output, status="ready", reason=None)
    decision = run_available_replay(base_url, model, api_key, structured_output, args.timeout)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
