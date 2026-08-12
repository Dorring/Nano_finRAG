#!/usr/bin/env python3
"""NF-V2-01: question-only General LLM Supervisor shadow evaluation."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.plan import Action, Intent  # noqa: E402
from rag_v2.orchestration.loader import load_question_envelopes  # noqa: E402
from rag_v2.supervisor.api_provider import APIProvider  # noqa: E402
from rag_v2.supervisor.local_provider import LocalProvider  # noqa: E402
from rag_v2.supervisor.prompt import SUPERVISOR_SYSTEM_PROMPT_V1  # noqa: E402
from rag_v2.supervisor.service import SupervisorService  # noqa: E402


GATE = "NF-V2-01"
BASE_COMMIT = "938a32573ba9ebbca56d55404a54747f36d34981"
ROLE = "development_shadow_v2_general_llm_supervisor"
QUESTIONS = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"
QUERY_REQUIREMENTS = ROOT / "artifacts/evaluation/nf-opt-23-r1-query-requirement-serialization/query-requirements.json"
V2_CONTRACT = ROOT / "artifacts/evaluation/nf-v2-00-architecture-contract-freeze/v2-contract-freeze.json"
OUT = ROOT / "artifacts/evaluation/nf-v2-01-general-llm-supervisor"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def discover_model(base_url: str) -> str:
    request = urllib.request.Request(base_url.rstrip("/") + "/models", method="GET")
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("data", []) if isinstance(payload, dict) else []
    if models and isinstance(models[0], dict) and models[0].get("id"):
        return str(models[0]["id"])
    raise RuntimeError("V2_SUPERVISOR_MODEL is not configured and /v1/models returned no model")


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def expected_intent(row: dict[str, Any]) -> str:
    if bool(row.get("requires_calculation")):
        return Intent.CALCULATION.value
    if bool(row.get("requires_multiple_sources")):
        return Intent.MULTI_EVIDENCE.value
    return Intent.DIRECT_FACT.value


def predicted_intent(record: dict[str, Any]) -> str:
    plan = record.get("plan")
    return str(plan.get("intent")) if isinstance(plan, dict) and plan.get("intent") else "INVALID"


def is_answer_leakage(raw: str | None) -> bool:
    if not raw:
        return False
    try:
        payload = json.loads(raw.strip())
        if isinstance(payload, dict) and set(payload) == {"intent", "required_slots", "operation", "next_action"}:
            return False
    except json.JSONDecodeError:
        pass
    return bool(re.search(r"(?:answer\s*:|citation|\$\s*\d|\d+(?:\.\d+)?\s*%)", raw, flags=re.I))


def strict_slot_metrics(records: list[dict[str, Any]], expected_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metric_total = period_total = role_total = 0
    metric_correct = period_correct = role_correct = slot_count_correct = 0
    for record in records:
        expected = expected_by_id[record["question_id"]]
        expected_slots = expected.get("required_slots", [])
        plan = record.get("plan") or {}
        predicted_slots = plan.get("required_slots", []) if isinstance(plan, dict) else []
        if len(expected_slots) == len(predicted_slots):
            slot_count_correct += 1
        for index, slot in enumerate(expected_slots):
            predicted = predicted_slots[index] if index < len(predicted_slots) else {}
            metric_total += 1
            period_total += 1
            role_total += 1
            if norm(predicted.get("metric")) == norm(slot.get("target")):
                metric_correct += 1
            if norm(predicted.get("period")) == norm(slot.get("period")):
                period_correct += 1
            if norm(predicted.get("role")) == norm(slot.get("role")):
                role_correct += 1
    return {
        "metric_correct": metric_correct,
        "metric_total": metric_total,
        "metric_accuracy": metric_correct / metric_total if metric_total else 0.0,
        "period_correct": period_correct,
        "period_total": period_total,
        "period_accuracy": period_correct / period_total if period_total else 0.0,
        "role_correct": role_correct,
        "role_total": role_total,
        "role_accuracy": role_correct / role_total if role_total else 0.0,
        "slot_count_correct": slot_count_correct,
        "slot_count_total": len(records),
        "slot_count_accuracy": slot_count_correct / len(records) if records else 0.0,
    }


def failure_for(record: dict[str, Any], expected: dict[str, Any]) -> str:
    raw = record.get("raw_response")
    if is_answer_leakage(raw):
        return "SP11_answer_leakage"
    if not record.get("schema_valid"):
        return "SP10_schema_invalid"
    if not record.get("plan_valid"):
        error = norm(record.get("error"))
        if "action" in error:
            return "SP9_invalid_action"
        if "operation" in error:
            return "SP8_wrong_operation"
        if "role" in error:
            return "SP7_wrong_operand_role"
        return "SP10_schema_invalid"
    plan = record.get("plan") or {}
    expected_int = expected_intent(expected)
    if plan.get("intent") != expected_int:
        return "SP1_wrong_intent"
    expected_operation = expected.get("operation")
    if expected_int == Intent.CALCULATION.value and plan.get("operation") != expected_operation:
        return "SP8_wrong_operation"
    expected_slots = expected.get("required_slots", [])
    slots = plan.get("required_slots", [])
    if len(slots) != len(expected_slots):
        return "SP6_wrong_slot_count"
    for expected_slot, slot in zip(expected_slots, slots):
        if norm(slot.get("role")) != norm(expected_slot.get("role")):
            return "SP7_wrong_operand_role" if expected_int == Intent.CALCULATION.value else "SP12_other"
        if norm(slot.get("metric")) != norm(expected_slot.get("target")):
            return "SP2_missing_metric_slot" if not slot.get("metric") else "SP3_wrong_metric_slot"
        if norm(slot.get("period")) != norm(expected_slot.get("period")):
            return "SP4_missing_period_slot" if not slot.get("period") else "SP5_wrong_period_slot"
    if plan.get("next_action") not in {Action.RETRIEVE.value, Action.ABSTAIN.value}:
        return "SP9_invalid_action"
    return "SP0_correct"


def evaluate(records: list[dict[str, Any]], rows: list[dict[str, Any]], requirements: dict[str, Any]) -> dict[str, Any]:
    row_by_id = {str(row["case_id"]): row for row in rows}
    expected_by_id = {question_id: requirements[question_id] for question_id in row_by_id}
    calc_ids = {qid for qid, row in row_by_id.items() if bool(row.get("requires_calculation"))}
    expected_labels = {qid: expected_intent(row_by_id[qid]) for qid in row_by_id}
    predicted_labels = {record["question_id"]: predicted_intent(record) for record in records}
    tp = sum(predicted_labels.get(qid) == Intent.CALCULATION.value for qid in calc_ids)
    fp = sum(predicted_labels.get(qid) == Intent.CALCULATION.value for qid in row_by_id if qid not in calc_ids)
    fn = len(calc_ids) - tp
    operation_correct = sum(
        bool(record.get("plan_valid"))
        and record.get("plan", {}).get("intent") == Intent.CALCULATION.value
        and record.get("plan", {}).get("operation") == requirements[record["question_id"]].get("operation")
        for record in records if record["question_id"] in calc_ids
    )
    all_operand = 0
    for record in records:
        qid = record["question_id"]
        if qid not in calc_ids or not record.get("plan_valid"):
            continue
        expected_slots = requirements[qid].get("required_slots", [])
        slots = record.get("plan", {}).get("required_slots", [])
        if record.get("plan", {}).get("operation") != requirements[qid].get("operation") or len(slots) != len(expected_slots):
            continue
        if all(
            norm(a.get("metric")) == norm(b.get("target"))
            and norm(a.get("period")) == norm(b.get("period"))
            and norm(a.get("role")) == norm(b.get("role"))
            for a, b in zip(slots, expected_slots)
        ):
            all_operand += 1
    schema_valid = sum(bool(record.get("schema_valid")) for record in records)
    plan_valid = sum(bool(record.get("plan_valid")) for record in records)
    parse_failures = sum(record.get("error_type") == "parse_failure" for record in records)
    premature_calculate = sum(record.get("raw_next_action") == Action.CALCULATE.value for record in records)
    premature_generate = sum(record.get("raw_next_action") == Action.GENERATE.value for record in records)
    leakage = sum(is_answer_leakage(record.get("raw_response")) for record in records)
    slots = strict_slot_metrics(records, expected_by_id)
    confusion = Counter((expected_labels[qid], predicted_labels.get(qid, "INVALID")) for qid in row_by_id)
    confusion_json = {f"{expected}->{predicted}": count for (expected, predicted), count in sorted(confusion.items())}
    failures = Counter(failure_for(record, row_by_id[record["question_id"]]) for record in records)
    latencies = [float(record.get("latency_ms", 0.0)) for record in records]
    input_tokens = sum(int(record.get("input_tokens") or 0) for record in records)
    output_tokens = sum(int(record.get("output_tokens") or 0) for record in records)
    total_tokens = sum(int(record.get("total_tokens") or 0) for record in records)
    cost_rate = os.getenv("V2_SUPERVISOR_COST_PER_1K_TOKENS")
    estimated_cost = float(cost_rate) * total_tokens / 1000 if cost_rate else None
    return {
        "questions": len(records),
        "calculation": {
            "cohort": len(calc_ids),
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "recall": tp / len(calc_ids) if calc_ids else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "operation_correct": operation_correct,
            "all_operand_slots_correct": all_operand,
        },
        "routing": {"predicted": dict(Counter(predicted_labels.values())), "expected": dict(Counter(expected_labels.values()))},
        "confusion_matrix": confusion_json,
        "plan_validity": {
            "schema_valid": schema_valid,
            "plan_validator_pass": plan_valid,
            "invalid_plan": len(records) - plan_valid,
            "parse_failure": parse_failures,
        },
        "slots": slots,
        "safety": {
            "premature_calculate": premature_calculate,
            "premature_generate": premature_generate,
            "answer_leakage": leakage,
            "retrieval_calls": 0,
            "reranker_calls": 0,
            "binder_calls": 0,
            "calculator_calls": 0,
            "generator_calls": 0,
            "validator_calls": 0,
        },
        "failure_taxonomy": {key: failures.get(key, 0) for key in [f"SP{i}_" + name for i, name in enumerate([
            "correct", "wrong_intent", "missing_metric_slot", "wrong_metric_slot", "missing_period_slot",
            "wrong_period_slot", "wrong_slot_count", "wrong_operand_role", "wrong_operation", "invalid_action",
            "schema_invalid", "answer_leakage", "other",
        ])]},
        "latency_cost": {
            "calls": len(records),
            "average_latency_ms": statistics.mean(latencies) if latencies else None,
            "p50_latency_ms": percentile(latencies, 0.50),
            "p95_latency_ms": percentile(latencies, 0.95),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "cost_rate_source": "V2_SUPERVISOR_COST_PER_1K_TOKENS" if cost_rate else "not_configured",
        },
    }


def decide(metrics: dict[str, Any]) -> dict[str, Any]:
    calc = metrics["calculation"]
    slots = metrics["slots"]
    validity = metrics["plan_validity"]["plan_validator_pass"] / metrics["questions"]
    safety = metrics["safety"]
    strong = (
        calc["recall"] == 1.0 and calc["false_positive"] == 0 and calc["operation_correct"] == 11
        and slots["metric_accuracy"] >= 0.90 and slots["period_accuracy"] >= 0.95
        and validity >= 0.98 and safety["premature_calculate"] == 0
        and safety["premature_generate"] == 0 and safety["answer_leakage"] == 0
    )
    partial = (
        calc["recall"] >= 10 / 11 and calc["false_positive"] == 0
        and slots["metric_accuracy"] >= 0.85 and slots["period_accuracy"] >= 0.90
        and validity >= 0.95 and safety["premature_calculate"] == 0
        and safety["premature_generate"] == 0 and safety["answer_leakage"] == 0
    )
    return {
        "general_llm_supervisor_effective": "true" if strong else "partial" if partial else "false",
        "dominant_failure": "none" if strong else "calculation_routing" if calc["false_positive"] or calc["recall"] < 1 else "slot_extraction" if slots["metric_accuracy"] < 0.85 or slots["period_accuracy"] < 0.90 else "plan_contract_or_safety",
        "next_gate": "v2_02_top20_financial_fact_expansion" if strong or partial else "v2_01_supervisor_failure_review",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None, choices=["api", "local"])
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    base_url = args.base_url or os.getenv("V2_SUPERVISOR_BASE_URL") or os.getenv("LLM_API_BASE_URL") or "http://127.0.0.1:18001/v1"
    model = args.model or os.getenv("V2_SUPERVISOR_MODEL") or os.getenv("LLM_MODEL_NAME")
    if not model:
        model = discover_model(base_url)
    api_key = args.api_key or os.getenv("V2_SUPERVISOR_API_KEY") or os.getenv("LLM_API_KEY") or "not-needed"
    provider_kind = args.provider or os.getenv("V2_SUPERVISOR_PROVIDER") or ("local" if "127.0.0.1" in base_url or "localhost" in base_url else "api")
    provider_cls = LocalProvider if provider_kind == "local" else APIProvider
    provider = provider_cls(base_url=base_url, api_key=api_key, model_name=model, temperature=0.0, max_tokens=512, timeout=args.timeout)
    service = SupervisorService(provider)

    questions = load_question_envelopes(QUESTIONS)
    if len(questions) != 72:
        raise RuntimeError(f"expected 72 question envelopes, got {len(questions)}")
    frozen = {
        "gate": GATE,
        "base_commit": BASE_COMMIT,
        "evaluation_role": ROLE,
        "fresh_blind_evaluation": False,
        "production_default": "V1",
        "production_switch_allowed": False,
        "v2_00_contract_artifact": str(V2_CONTRACT.relative_to(ROOT)),
        "v2_00_contract_sha256": sha256_file(V2_CONTRACT),
        "question_count": len(questions),
        "question_fields_loaded": ["question_id", "question", "document_scope"],
        "gold_fields_loaded_before_seal": False,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "binder_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "validator_calls": 0,
    }
    write_json(OUT / "frozen-v2-contract.json", frozen)
    provider_contract = {
        "gate": GATE,
        "provider": provider_kind,
        "model": model,
        "base_url": base_url,
        "temperature": 0.0,
        "max_tokens": 512,
        "max_supervisor_calls_per_question": 1,
        "retry": False,
        "structured_output": "strict_json_parsing",
        "downstream_execution": False,
    }
    write_json(OUT / "supervisor-provider-contract.json", provider_contract)
    prompt_bytes = SUPERVISOR_SYSTEM_PROMPT_V1.encode("utf-8")
    (OUT / "supervisor-prompt.txt").write_bytes(prompt_bytes)
    (OUT / "supervisor-prompt.sha256").write_text(sha256_bytes(prompt_bytes) + "\n", encoding="ascii")

    records: list[dict[str, Any]] = []
    for envelope in questions:
        started = time.perf_counter()
        run = service.plan(envelope.question)
        metadata = run.metadata
        raw = metadata.raw_response if metadata else None
        plan_dict = run.plan.to_dict() if run.plan is not None else None
        records.append({
            "question_id": envelope.question_id,
            "question": envelope.question,
            "document_scope": list(envelope.document_scope),
            "intent": plan_dict.get("intent") if plan_dict else None,
            "required_slots": plan_dict.get("required_slots", []) if plan_dict else [],
            "operation": plan_dict.get("operation") if plan_dict else None,
            "next_action": plan_dict.get("next_action") if plan_dict else None,
            "raw_next_action": plan_dict.get("next_action") if plan_dict else None,
            "plan": plan_dict,
            "schema_valid": plan_dict is not None,
            "plan_valid": run.plan_valid,
            "error": run.error,
            "error_type": "parse_failure" if metadata and metadata.parse_failure else None,
            "provider": metadata.provider if metadata else provider_kind,
            "model": metadata.model if metadata else model,
            "latency_ms": metadata.latency_ms if metadata else (time.perf_counter() - started) * 1000,
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
        "questions": len(records),
        "model_calls": len(records),
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

    # Evaluation annotations are opened only after the prediction seal.
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    requirements = json.loads(QUERY_REQUIREMENTS.read_text(encoding="utf-8"))
    metrics = evaluate(records, rows, requirements)
    write_json(OUT / "routing-metrics.json", metrics["routing"])
    write_json(OUT / "routing-confusion-matrix.json", metrics["confusion_matrix"])
    write_json(OUT / "slot-metrics.json", metrics["slots"])
    write_json(OUT / "calculation-routing-analysis.json", metrics["calculation"])
    write_json(OUT / "calculation-slot-analysis.json", {"all_operand_slots_correct": metrics["calculation"]["all_operand_slots_correct"], "cohort": 11})
    write_json(OUT / "plan-validity.json", metrics["plan_validity"])
    write_json(OUT / "safety-analysis.json", metrics["safety"])
    write_json(OUT / "latency-cost-analysis.json", metrics["latency_cost"])
    write_json(OUT / "failure-taxonomy.json", metrics["failure_taxonomy"])
    decision = {
        "gate": GATE,
        "evaluation_role": ROLE,
        "base_commit": BASE_COMMIT,
        "production_default": "V1",
        "production_switch_allowed": False,
        "model_calls": len(records),
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "questions": 72,
        "plan_validator_pass": metrics["plan_validity"]["plan_validator_pass"],
        "calculation_recall": metrics["calculation"]["recall"],
        "calculation_precision": metrics["calculation"]["precision"],
        "false_calculation_routing": metrics["calculation"]["false_positive"],
        "operation_accuracy": metrics["calculation"]["operation_correct"],
        "metric_slot_accuracy": metrics["slots"]["metric_accuracy"],
        "period_slot_accuracy": metrics["slots"]["period_accuracy"],
        "all_operand_slots_correct": metrics["calculation"]["all_operand_slots_correct"],
        "premature_calculate": metrics["safety"]["premature_calculate"],
        "premature_generate": metrics["safety"]["premature_generate"],
        "answer_leakage": metrics["safety"]["answer_leakage"],
        **decide(metrics),
    }
    write_json(OUT / "decision.json", decision)
    readme = """# NF-V2-01 — General LLM Supervisor

This is a development-shadow, question-only Supervisor replay.  Each of the
72 frozen questions received at most one temperature-zero model call.  The
result was parsed as a strict `SupervisorPlan` and passed through the
deterministic V2-00 validator.  Retrieval, reranking, binding, calculation,
generation, validation, and production routing were not executed.  Gold and
reference annotations were opened only after the prediction artifact was
sealed.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
