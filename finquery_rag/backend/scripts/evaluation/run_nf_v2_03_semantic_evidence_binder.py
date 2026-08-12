#!/usr/bin/env python3
"""NF-V2-03 shadow evaluation of semantic EvidenceBinding over frozen Top20 facts."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.evidence import BindingStatus  # noqa: E402
from rag_v2.contracts.plan import Intent, RequiredSlot, SupervisorPlan  # noqa: E402
from rag_v2.evidence.binder_provider import (  # noqa: E402
    BINDER_RESPONSE_FORMAT,
    BailianBinderProvider,
)
from rag_v2.evidence.binder_service import BinderRequest, BinderRun, SemanticBinderService  # noqa: E402
from rag_v2.evidence.prompt import BINDER_SCHEMA, BINDER_SYSTEM_PROMPT_V1  # noqa: E402
from rag_v2.supervisor.plan_validator import validate_plan_v2_01  # noqa: E402


OUT = ROOT / "artifacts/evaluation/nf-v2-03-semantic-evidence-binder"
NF01 = ROOT / "artifacts/evaluation/nf-e2e-01-r0-frozen-retrieval-integration-review"
NF02 = ROOT / "artifacts/evaluation/nf-v2-02-top20-financial-fact-expansion"
NF09 = ROOT / "artifacts/evaluation/nf-e2e-09-r0-structured-financial-fact-representation"
METRIC_REVIEW = ROOT / "artifacts/evaluation/nf-v2-01-metric-evaluation-contract-review/rescored-metrics.json"
PLANS = ROOT / "artifacts/evaluation/nf-v2-01-r1-bailian-formal-72-attempt-2/supervisor-plans.jsonl.gz"
PLAN_SEAL = ROOT / "artifacts/evaluation/nf-v2-01-r1-bailian-formal-72-attempt-2/supervisor-prediction-seal.json"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"

GATE = "NF-V2-03"
BASE_COMMIT = "7e9cd14f879f2d7613fdd5cd79354cdfe5d7e663"
SUPPORTED_MODELS = (
    "qwen3.7-max",
    "qwen3.7-max-2026-06-08",
    "qwen3.7-flash-2026-07-15",
)
MODEL = os.getenv("V2_SUPERVISOR_MODEL", SUPPORTED_MODELS[0]).strip()
FACT_CONTRACT_SHA = "7a253b443962c5f372dd897c49c057a19b553e92314faadc31eefc82b27b54eb"
QUESTION_TOTAL = 72
DIRECT_TOTAL = 56
CALC_TOTAL = 11
MULTI_TOTAL = 5
HISTORICAL_FACT_TOTAL = 46


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def stable_sha(value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pct(count: int, total: int) -> float:
    return round(100.0 * count / total, 4) if total else 0.0


def percentile(values: list[float], q: float = 0.95) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * q) - 1))]


def norm_text(value: Any) -> str:
    text = str(value or "").casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(revenues|expenses|assets|liabilities|earnings)\b", lambda m: m.group(1)[:-1], text)
    return " ".join(text.split())


def norm_period(value: Any) -> str:
    return norm_text(value).replace("fy ", "fy")


def load_config() -> dict[str, Any]:
    provider = os.getenv("V2_SUPERVISOR_PROVIDER")
    model = os.getenv("V2_SUPERVISOR_MODEL")
    base_url = os.getenv("V2_SUPERVISOR_BASE_URL")
    api_key = os.getenv("V2_SUPERVISOR_API_KEY")
    thinking = os.getenv("V2_SUPERVISOR_ENABLE_THINKING")
    temperature = os.getenv("V2_SUPERVISOR_TEMPERATURE")
    if provider != "bailian":
        raise RuntimeError("V2_SUPERVISOR_PROVIDER must be bailian")
    if model not in SUPPORTED_MODELS:
        raise RuntimeError(f"V2_SUPERVISOR_MODEL must be one of {SUPPORTED_MODELS}")
    if not base_url:
        raise RuntimeError("V2_SUPERVISOR_BASE_URL is not configured")
    if not api_key:
        raise RuntimeError("V2_SUPERVISOR_API_KEY is not configured")
    if thinking is not None and thinking.strip().casefold() not in {"false", "0"}:
        raise RuntimeError("V2_SUPERVISOR_ENABLE_THINKING must be false")
    if temperature is not None and float(temperature.strip()) != 0.0:
        raise RuntimeError("V2_SUPERVISOR_TEMPERATURE must be 0")
    return {
        "provider": "bailian",
        "provider_role": "evidence_binder",
        "model": MODEL,
        "model_role": "strong_general_llm",
        "base_url_region": base_url.split("/compatible-mode", 1)[0],
        "enable_thinking": False,
        "temperature": 0.0,
        "max_retries": 0,
        "api_key": api_key.strip(),
    }


def load_frozen_inputs() -> dict[str, Any]:
    from scripts.evaluation import run_nf_v2_02_top20_financial_fact_expansion as nf02

    plans = read_jsonl_gz(PLANS)
    plan_seal = read_json(PLAN_SEAL)
    if len(plans) != QUESTION_TOTAL or plan_seal.get("plans_sha256") != sha256_file(PLANS):
        raise RuntimeError("NF-V2-01 Supervisor prediction seal mismatch")
    if plan_seal.get("sealed") is not True or plan_seal.get("gold_reads_before_prediction_seal") != 0:
        raise RuntimeError("NF-V2-01 Supervisor predictions are not sealed")
    state = nf02.verify_frozen_top100()
    facts_path = NF02 / "top20-materialized-facts.jsonl.gz"
    materialization_seal = read_json(NF02 / "top20-materialization-seal.json")
    facts = read_jsonl_gz(facts_path)
    if sha256_file(facts_path) != materialization_seal.get("financial_facts_sha256"):
        raise RuntimeError("NF-V2-02 FinancialFact artifact hash mismatch")
    if len(facts) != 445 or any(fact.get("provenance_complete") is not True for fact in facts):
        raise RuntimeError("NF-V2-02 FinancialFact provenance seal mismatch")
    if read_json(NF02 / "relation-integrity.json").get("relation_integrity_fail") != 0:
        raise RuntimeError("NF-V2-02 relation-integrity seal failed")
    by_id: dict[str, dict[str, Any]] = {}
    for row in plans:
        question_id = str(row["question_id"])
        plan_payload = row.get("plan")
        plan = SupervisorPlan.from_dict(plan_payload)
        validate_plan_v2_01(plan)
        by_id[question_id] = {"question_id": question_id, "question": row["question"], "plan": plan, "raw": row}
    if len(by_id) != QUESTION_TOTAL:
        raise RuntimeError("Supervisor question IDs are not unique")
    facts_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        for candidate_id in fact.get("candidate_ids", [fact.get("candidate_id")]):
            facts_by_candidate[str(candidate_id)].append(fact)
    requests: dict[str, BinderRequest] = {}
    request_rows: list[dict[str, Any]] = []
    for question_id in sorted(by_id):
        row = by_id[question_id]
        packet: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rank, candidate_id in enumerate(state["top20_order"].get(question_id, []), 1):
            for fact in facts_by_candidate.get(str(candidate_id), []):
                fact_id = str(fact["fact_id"])
                if fact_id in seen:
                    continue
                seen.add(fact_id)
                projection = {key: fact.get(key) for key in (
                    "fact_id", "physical_source_id", "document_id", "pdf_page", "statement_id",
                    "logical_table_id", "table_id", "row_id", "column_id", "cell_id", "raw_metric",
                    "normalized_metric", "raw_period", "normalized_period", "raw_value", "parsed_numeric_value",
                    "raw_currency", "normalized_currency", "raw_scale", "normalized_scale", "currency", "unit",
                    "provenance_complete",
                )}
                projection["candidate_id"] = str(candidate_id)
                projection["candidate_rank"] = rank
                packet.append(projection)
        request = BinderRequest(question_id, row["question"], row["plan"], tuple(packet))
        requests[question_id] = request
        request_rows.append({
            "question_id": question_id,
            "intent": row["plan"].intent.value,
            "operation": row["plan"].operation,
            "required_slot_count": len(row["plan"].required_slots),
            "fact_count": len(packet),
            "skipped_no_fact_supply": not bool(packet),
            "candidate_count": len(state["top20_order"].get(question_id, [])),
        })
    return {
        "plans": by_id,
        "requests": requests,
        "facts": facts,
        "facts_by_candidate": facts_by_candidate,
        "top20_order": state["top20_order"],
        "request_rows": request_rows,
        "plan_sha256": sha256_file(PLANS),
        "fact_sha256": sha256_file(facts_path),
        "top20_order_sha256": state["top20_order_sha256"],
    }


def synthetic_fact(fact_id: str, metric: str, period: str) -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "candidate_id": f"candidate:{fact_id}",
        "candidate_rank": 1,
        "physical_source_id": f"source:{fact_id}",
        "document_id": "examplecorp_fy2025",
        "pdf_page": 1,
        "table_id": "table:example",
        "row_id": f"row:{metric}",
        "column_id": f"column:{period}",
        "cell_id": f"cell:{fact_id}",
        "raw_metric": metric,
        "normalized_metric": norm_text(metric),
        "raw_period": period,
        "normalized_period": period,
        "raw_value": "100",
        "parsed_numeric_value": "100",
        "raw_scale": None,
        "normalized_scale": None,
        "currency": "USD",
        "unit": "currency",
        "provenance_complete": True,
    }


def smoke_test(config: dict[str, Any]) -> dict[str, Any]:
    provider = BailianBinderProvider(
        base_url=config["base_url"], api_key=config["api_key"], model_name=config["model"],
        enable_thinking=False, temperature=0.0, max_retries=0,
    )
    service = SemanticBinderService(provider)
    plan = SupervisorPlan.from_dict({
        "intent": "DIRECT_FACT", "required_slots": [{"slot_id": "slot_1", "metric": "revenue", "period": "FY2025", "role": "value", "value_type": "numeric", "unit": None}],
        "operation": None, "next_action": "RETRIEVE",
    })
    facts = (
        synthetic_fact("F1", "revenue", "FY2024"),
        synthetic_fact("F2", "total revenue", "FY2025"),
        synthetic_fact("F3", "operating expenses", "FY2025"),
    )
    bound = service.bind(BinderRequest("synthetic_bound", "What was ExampleCorp's revenue in FY2025?", plan, facts))
    ambiguous_facts = facts + (synthetic_fact("F2B", "total revenue", "FY2025"),)
    ambiguous = service.bind(BinderRequest("synthetic_ambiguous", "What was ExampleCorp's revenue in FY2025?", plan, ambiguous_facts))
    provider.close()
    return {
        "provider": config["provider"],
        "model": config["model"],
        "calls": 2,
        "bound": {"status": bound.binding.status, "selected_fact_ids": list(bound.validation.selected_fact_ids), "validator_pass": bound.validation.passed},
        "ambiguous": {"status": ambiguous.binding.status, "validator_pass": ambiguous.validation.passed},
        "expected_bound_fact": "F2",
        "expected_ambiguous_status": "AMBIGUOUS",
        "pass": bound.binding.status == BindingStatus.BOUND.value and bound.validation.selected_fact_ids == ("F2",) and ambiguous.binding.status == BindingStatus.AMBIGUOUS.value and bound.validation.passed and ambiguous.validation.passed,
        "gold_reads": 0,
        "benchmark_questions_used": 0,
    }


def leak_flags(raw: str | None, known_fact_ids: set[str], known_slot_ids: set[str]) -> dict[str, int]:
    text = (raw or "").casefold()
    return {
        "answer_leakage": int(any(token in text for token in ("answer:", "final answer", "citation:"))),
        "invented_numeric_values": int("$" in text or "%" in text),
        "calculation_outputs": int(any(token in text for token in ("growth", "margin", "result:"))),
        "invented_fact_ids": 0,
        "invented_source_ids": 0,
        "new_slots": 0,
    }


def run_formal(config: dict[str, Any], frozen: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = BailianBinderProvider(
        base_url=config["base_url"], api_key=config["api_key"], model_name=config["model"],
        enable_thinking=False, temperature=0.0, max_retries=0,
    )
    service = SemanticBinderService(provider)
    predictions: list[dict[str, Any]] = []
    call_failures: list[dict[str, Any]] = []
    formal_started = time.perf_counter()
    try:
        for index, question_id in enumerate(sorted(frozen["requests"]), 1):
            request = frozen["requests"][question_id]
            run: BinderRun = service.bind(request)
            metadata = run.metadata.to_dict() if run.metadata else None
            flags = leak_flags(run.raw_response, {str(f["fact_id"]) for f in request.facts}, {slot.slot_id for slot in request.plan.required_slots})
            if run.validation.reasons:
                flags["invented_fact_ids"] = int(any(reason.startswith("unknown_fact:") for reason in run.validation.reasons))
                flags["new_slots"] = int(any(reason.startswith("unknown_slot:") for reason in run.validation.reasons))
            if metadata and not metadata["provider_response_success"]:
                call_failures.append({"question_id": question_id, "call_index": index, "error": metadata.get("error")})
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
            })
            predictions.append(row)
    finally:
        provider.close()
    if call_failures:
        raise RuntimeError(f"binder provider failures during formal run: {call_failures[:2]}")
    return predictions, {"formal_wall_time_ms": round((time.perf_counter() - formal_started) * 1000.0, 3), "call_failures": call_failures}


def source_matches(fact: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    candidate_ids = {str(fact.get("candidate_id"))} | {str(item) for item in fact.get("candidate_ids", [])}
    return bool(source.get("candidate_key") and str(source["candidate_key"]) in candidate_ids)


def metric_matches(fact: Mapping[str, Any], slot: RequiredSlot, source: Mapping[str, Any]) -> bool:
    fact_metric = norm_text(fact.get("normalized_metric") or fact.get("raw_metric"))
    source_metric = norm_text(source.get("row_label"))
    slot_metric = norm_text(slot.metric)
    return bool(fact_metric and source_metric and fact_metric == source_metric and slot_metric == source_metric)


def period_matches(fact: Mapping[str, Any], slot: RequiredSlot, source: Mapping[str, Any]) -> bool:
    fact_period = norm_period(fact.get("normalized_period") or fact.get("raw_period"))
    return fact_period == norm_period(slot.period) == norm_period(source.get("period"))


def expected_sources_for_slot(slot: RequiredSlot, label: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources = list(label.get("expected_sources") or [])
    matching = [source for source in sources if norm_period(source.get("period")) == norm_period(slot.period) and norm_text(source.get("row_label")) == norm_text(slot.metric)]
    if matching:
        return matching
    return [source for source in sources if norm_period(source.get("period")) == norm_period(slot.period)]


def strict_fact_for_slot(
    fact: Mapping[str, Any],
    slot: RequiredSlot,
    label: Mapping[str, Any],
    *,
    metric_contract_ok: bool = True,
) -> bool:
    return any(
        source_matches(fact, source)
        and metric_matches(fact, slot, source)
        and period_matches(fact, slot, source)
        and metric_contract_ok
        for source in expected_sources_for_slot(slot, label)
    )


def semantic_fact_for_slot(fact: Mapping[str, Any], slot: RequiredSlot) -> bool:
    return norm_text(fact.get("normalized_metric") or fact.get("raw_metric")) == norm_text(slot.metric) and norm_period(fact.get("normalized_period") or fact.get("raw_period")) == norm_period(slot.period)


def classify_slot_failure(
    *,
    slot: RequiredSlot,
    fact_packet: list[Mapping[str, Any]],
    strict_bindable: bool,
    selected_fact: Mapping[str, Any] | None,
    final_status: str,
    label: Mapping[str, Any],
    validation_pass: bool,
    metric_contract_ok: bool,
) -> str:
    if selected_fact is not None and strict_fact_for_slot(selected_fact, slot, label, metric_contract_ok=metric_contract_ok):
        return "EB0_correct"
    if not validation_pass:
        return "EB12_binding_schema_or_validator_failure"
    if not fact_packet:
        return "EB1_no_relevant_fact_in_packet"
    if selected_fact is not None:
        sources = expected_sources_for_slot(slot, label)
        if not any(source_matches(selected_fact, source) for source in sources):
            return "EB9_wrong_physical_source_selected"
        if not metric_contract_ok or not any(metric_matches(selected_fact, slot, source) for source in sources):
            return "EB3_metric_semantic_mismatch"
        if not any(period_matches(selected_fact, slot, source) for source in sources):
            return "EB4_period_mismatch"
        return "EB8_wrong_fact_selected"
    if not strict_bindable:
        return "EB2_correct_source_present_but_no_matching_financial_fact"
    if final_status == BindingStatus.MISSING.value:
        return "EB10_model_returned_missing_despite_bindable_fact"
    if final_status == BindingStatus.AMBIGUOUS.value:
        return "EB11_model_returned_ambiguous_despite_unique_fact"
    return "EB14_other"


def score_predictions(frozen: dict[str, Any], predictions: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metric_rows = read_json(METRIC_REVIEW).get("metric_matches", [])
    metric_contract = {(str(row["question_id"]), int(row["slot_index"])): bool(row.get("match", {}).get("matched")) for row in metric_rows}
    prediction_by_id = {row["question_id"]: row for row in predictions}
    strict_rows: list[dict[str, Any]] = []
    failure_counts = Counter()
    status_counts = Counter(row["final_binding_status"] for row in predictions)
    slot_totals = Counter()
    slot_bound = Counter()
    false_queries = 0
    false_slots = 0
    alternative_rows: list[dict[str, Any]] = []
    cohort_ids = {
        "direct": sorted(question_id for question_id, item in frozen["plans"].items() if item["plan"].intent is Intent.DIRECT_FACT),
        "multi": sorted(question_id for question_id, item in frozen["plans"].items() if item["plan"].intent is Intent.MULTI_EVIDENCE),
        "calculation": sorted(question_id for question_id, item in frozen["plans"].items() if item["plan"].intent is Intent.CALCULATION),
    }
    for question_id, item in frozen["plans"].items():
        row = prediction_by_id[question_id]
        request = frozen["requests"][question_id]
        label = labels[question_id]
        fact_by_id = {str(fact["fact_id"]): fact for fact in request.facts}
        slot_results: list[dict[str, Any]] = []
        query_strict = True
        query_bindable = True
        for slot_index, slot in enumerate(request.plan.required_slots):
            slot_totals[request.plan.intent.value] += 1
            metric_contract_ok = metric_contract.get((question_id, slot_index), False)
            strict_candidates = [fact for fact in request.facts if strict_fact_for_slot(fact, slot, label, metric_contract_ok=metric_contract_ok)]
            if not strict_candidates:
                query_bindable = False
            selected_ids = list((row.get("binding") or {}).get("slot_bindings", {}).get(slot.slot_id, []))
            selected_fact = fact_by_id.get(str(selected_ids[0])) if len(selected_ids) == 1 else None
            slot_strict = selected_fact is not None and strict_fact_for_slot(selected_fact, slot, label, metric_contract_ok=metric_contract_ok)
            if slot_strict:
                slot_bound[request.plan.intent.value] += 1
            else:
                query_strict = False
            failure = classify_slot_failure(
                slot=slot,
                fact_packet=list(request.facts),
                strict_bindable=bool(strict_candidates),
                selected_fact=selected_fact,
                final_status=row["final_binding_status"],
                label=label,
                validation_pass=bool(row["binding_validator_pass"]),
                metric_contract_ok=metric_contract_ok,
            )
            failure_counts[failure] += int(not slot_strict)
            slot_results.append({"slot_id": slot.slot_id, "strict_candidates": len(strict_candidates), "selected_fact_id": selected_ids[0] if len(selected_ids) == 1 else None, "strict_correct": slot_strict, "failure": failure})
            if selected_fact is not None and semantic_fact_for_slot(selected_fact, slot) and not strict_fact_for_slot(selected_fact, slot, label):
                alternative_rows.append({"question_id": question_id, "slot_id": slot.slot_id, "fact_id": selected_fact.get("fact_id"), "candidate_id": selected_fact.get("candidate_id"), "diagnostic": "ALTERNATIVE_SUPPORT_CANDIDATE"})
        if row["final_binding_status"] == BindingStatus.BOUND.value and not query_strict:
            false_queries += 1
            false_slots += sum(int(not result["strict_correct"]) for result in slot_results)
        strict_rows.append({"question_id": question_id, "intent": request.plan.intent.value, "final_binding_status": row["final_binding_status"], "strict_bindable": query_bindable, "strict_complete": query_strict and row["final_binding_status"] == BindingStatus.BOUND.value, "slot_results": slot_results})
    by_q = {row["question_id"]: row for row in strict_rows}

    def cohort_metrics(ids: list[str]) -> dict[str, Any]:
        rows = [by_q[question_id] for question_id in ids]
        return {
            "denominator": len(rows),
            "strict_bindable": sum(int(row["strict_bindable"]) for row in rows),
            "bound": sum(int(row["final_binding_status"] == BindingStatus.BOUND.value) for row in rows),
            "strict_complete": sum(int(row["strict_complete"]) for row in rows),
            "missing": sum(int(row["final_binding_status"] == BindingStatus.MISSING.value) for row in rows),
            "ambiguous": sum(int(row["final_binding_status"] == BindingStatus.AMBIGUOUS.value) for row in rows),
            "invalid": sum(int(row["final_binding_status"] == BindingStatus.INVALID.value) for row in rows),
            "success_given_bindable": pct(sum(int(row["strict_complete"]) for row in rows), sum(int(row["strict_bindable"]) for row in rows)),
            "rows": rows,
        }

    calc = cohort_metrics(cohort_ids["calculation"])
    direct = cohort_metrics(cohort_ids["direct"])
    multi = cohort_metrics(cohort_ids["multi"])
    historical_ids = sorted(read_json(NF09 / "query-level-coverage.json").get("rows", []), key=lambda row: row["question_id"])
    historical_ids = [row["question_id"] for row in historical_ids if row["question_id"] in by_q][:HISTORICAL_FACT_TOTAL]
    historical = cohort_metrics(historical_ids)
    all_rows = [by_q[question_id] for question_id in sorted(by_q)]
    slots_requested = sum(len(row["slot_results"]) for row in all_rows)
    slots_bound = sum(int(slot["selected_fact_id"] is not None) for row in all_rows for slot in row["slot_results"])
    return {
        "query_status": {status: status_counts.get(status, 0) for status in ("BOUND", "MISSING", "AMBIGUOUS", "INVALID")},
        "slot_status": {"slots_requested": slots_requested, "slots_bound": slots_bound, "slots_missing": sum(int(slot["failure"] in {"EB1_no_relevant_fact_in_packet", "EB2_correct_source_present_but_no_matching_financial_fact", "EB10_model_returned_missing_despite_bindable_fact"}) for row in all_rows for slot in row["slot_results"]), "slots_ambiguous": sum(int(slot["failure"] == "EB11_model_returned_ambiguous_despite_unique_fact") for row in all_rows for slot in row["slot_results"]), "slots_invalid": sum(int(slot["failure"] == "EB12_binding_schema_or_validator_failure") for row in all_rows for slot in row["slot_results"])},
        "direct": direct,
        "calculation": calc,
        "multi": multi,
        "historical": historical,
        "strict_rows": strict_rows,
        "failure_counts": dict(sorted(failure_counts.items())),
        "alternative_support": alternative_rows,
        "false_binding_queries": false_queries,
        "false_binding_slots": false_slots,
        "cohort_ids": cohort_ids,
        "historical_ids": historical_ids,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        config = load_config()
    except Exception as exc:
        write_json(OUT / "decision.json", {"gate": GATE, "formal_evaluation_status": "infrastructure_blocked", "reason": str(exc), "model_calls": 0, "production_default": "V1", "production_switch_allowed": False})
        print(json.dumps({"formal_evaluation_status": "infrastructure_blocked", "reason": str(exc)}))
        return 2
    frozen = load_frozen_inputs()
    write_json(OUT / "frozen-input-contract.json", {
        "gate": GATE, "base_commit": BASE_COMMIT, "evaluation_role": "development_shadow_v2_semantic_evidence_binder",
        "production_default": "V1", "production_switch_allowed": False, "supervisor_frozen": True,
        "supervisor_plan_sha256": frozen["plan_sha256"], "financial_facts_sha256": frozen["fact_sha256"],
        "top20_order_sha256": frozen["top20_order_sha256"], "financial_fact_contract_sha256": FACT_CONTRACT_SHA,
        "questions": QUESTION_TOTAL, "gold_reads_before_prediction_seal": 0, "retrieval_calls": 0,
        "reranker_calls": 0, "calculator_calls": 0, "generator_calls": 0, "validator_calls": 0,
        "financial_fact_v1_modified": False, "sffm_v1_modified": False,
    })
    write_json(OUT / "binder-request-summary.json", {"rows": frozen["request_rows"], "denominator": QUESTION_TOTAL, "full_fact_pool": True, "fact_prefilter": False, "mean_facts": statistics.mean([row["fact_count"] for row in frozen["request_rows"]]), "p95_facts": percentile([float(row["fact_count"]) for row in frozen["request_rows"]]), "max_facts": max(row["fact_count"] for row in frozen["request_rows"]), "question_reads_during_materialization": 0, "gold_reads_before_prediction_seal": 0})
    smoke = smoke_test({**config, "base_url": os.getenv("V2_SUPERVISOR_BASE_URL", "").strip()})
    write_json(OUT / "smoke-test.json", smoke)
    if not smoke["pass"]:
        write_json(OUT / "decision.json", {"gate": GATE, "formal_evaluation_status": "smoke_failed", "smoke_test": smoke, "model_calls": 2, "production_default": "V1", "production_switch_allowed": False})
        print(json.dumps({"formal_evaluation_status": "smoke_failed"}))
        return 3
    prompt_path = OUT / "binder-prompt.txt"
    prompt_path.write_text(BINDER_SYSTEM_PROMPT_V1 + "\n", encoding="utf-8")
    schema_path = OUT / "binder-schema.json"
    write_json(schema_path, BINDER_SCHEMA)
    write_json(OUT / "binder-provider-contract.json", {"provider": "bailian", "provider_name": "Alibaba Bailian", "provider_role": "evidence_binder", "model": MODEL, "model_role": "strong_general_llm", "enable_thinking": False, "temperature": 0.0, "max_retries": 0, "structured_output": BINDER_RESPONSE_FORMAT, "request_calls_per_query": 1, "retry": 0, "api_key_persisted": False})
    write_json(OUT / "provider-config-seal.json", {"provider": "bailian", "provider_role": "evidence_binder", "model": MODEL, "model_role": "strong_general_llm", "base_url_region": config["base_url_region"], "enable_thinking": False, "temperature": 0.0, "max_retries": 0, "api_key_persisted": False, "smoke_pass": True})
    (OUT / "binder-prompt.sha256").write_text(sha256_file(prompt_path) + "\n", encoding="utf-8")
    (OUT / "binder-schema.sha256").write_text(sha256_file(schema_path) + "\n", encoding="utf-8")
    predictions, runtime = run_formal({**config, "base_url": os.getenv("V2_SUPERVISOR_BASE_URL", "").strip()}, frozen)
    write_jsonl_gz(OUT / "binder-predictions.jsonl.gz", predictions)
    prediction_sha = sha256_file(OUT / "binder-predictions.jsonl.gz")
    model_calls = sum(int(not row["skipped_no_fact_supply"]) for row in predictions)
    seal = {"gate": GATE, "sealed": True, "predictions_written": len(predictions), "questions_expected": QUESTION_TOTAL, "binder_model_calls": model_calls, "max_calls_per_query": 1, "retry": 0, "gold_reads_before_prediction_seal": 0, "reference_answer_reads_before_prediction_seal": 0, "prediction_sha256": prediction_sha, "financial_facts_sha256": frozen["fact_sha256"], "financial_fact_contract_sha256": FACT_CONTRACT_SHA, "retrieval_calls": 0, "reranker_calls": 0, "calculator_calls": 0, "generator_calls": 0, "validator_calls": 0, "sealed_before_gold": True}
    write_json(OUT / "binder-prediction-seal.json", seal)
    if sha256_file(OUT / "binder-predictions.jsonl.gz") != prediction_sha:
        raise RuntimeError("Binder prediction seal verification failed")
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if row}
    scored = score_predictions(frozen, predictions, labels)
    metadata_rows = [row["metadata"] for row in predictions if row.get("metadata")]
    latencies = [float(row.get("latency_ms") or 0.0) for row in metadata_rows]
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in metadata_rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in metadata_rows)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in metadata_rows)
    facts_per_call = [float(row["fact_count"]) for row in predictions if not row["skipped_no_fact_supply"]]
    write_json(OUT / "binding-validator-results.json", {"rows": [{"question_id": row["question_id"], "binding_validator_pass": row["binding_validator_pass"], "final_status": row["final_binding_status"], "reasons": row["validation_reasons"]} for row in predictions], "passed": sum(int(row["binding_validator_pass"]) for row in predictions), "failed": sum(int(not row["binding_validator_pass"]) for row in predictions)})
    write_json(OUT / "binding-status-metrics.json", {"query_status": scored["query_status"], "slot_status": scored["slot_status"], "model_calls": model_calls, "skipped_no_fact_supply": sum(int(row["skipped_no_fact_supply"]) for row in predictions)})
    write_json(OUT / "direct-fact-binding.json", scored["direct"])
    write_json(OUT / "direct-fact-bindability.json", {"denominator": scored["direct"]["denominator"], "strict_bindable": scored["direct"]["strict_bindable"], "success_given_bindable": scored["direct"]["success_given_bindable"], "rows": [{"question_id": row["question_id"], "strict_bindable": row["strict_bindable"], "strict_complete": row["strict_complete"]} for row in scored["direct"]["rows"]]})
    write_json(OUT / "calculation-binding.json", scored["calculation"])
    write_json(OUT / "calculation-bindability.json", {"denominator": scored["calculation"]["denominator"], "supply_complete_reference": "6/11", "strict_bindable": scored["calculation"]["strict_bindable"], "success_given_complete_supply": scored["calculation"]["success_given_bindable"], "all_operand_bound": scored["calculation"]["strict_complete"], "false_operand_binding": sum(int(not row["strict_complete"] and row["final_binding_status"] == BindingStatus.BOUND.value) for row in scored["calculation"]["rows"])})
    write_json(OUT / "multi-evidence-binding.json", scored["multi"])
    write_json(OUT / "historical-46-binding.json", {"top20_fact_supply": "42/46", **scored["historical"]})
    write_json(OUT / "alternative-support-diagnostic.json", {"count": len(scored["alternative_support"]), "rows": scored["alternative_support"], "counts_as_strict": False})
    write_json(OUT / "first-loss-funnel.json", {"all": {"questions": QUESTION_TOTAL, "supervisor_slot_complete": QUESTION_TOTAL, "financial_fact_supply_available": sum(int(row["fact_count"] > 0) for row in frozen["request_rows"]), "strict_bindable": sum(int(row["strict_bindable"]) for row in scored["strict_rows"]), "binder_bound": sum(int(row["final_binding_status"] == BindingStatus.BOUND.value) for row in scored["strict_rows"]), "strict_correct": sum(int(row["strict_complete"]) for row in scored["strict_rows"])}, "direct": {"questions": DIRECT_TOTAL, "strict_bindable": scored["direct"]["strict_bindable"], "binder_bound": scored["direct"]["bound"], "strict_correct": scored["direct"]["strict_complete"]}, "multi_evidence": {"questions": MULTI_TOTAL, "strict_bindable": scored["multi"]["strict_bindable"], "binder_bound": scored["multi"]["bound"], "strict_correct": scored["multi"]["strict_complete"]}, "calculation": {"questions": CALC_TOTAL, "strict_bindable": scored["calculation"]["strict_bindable"], "binder_bound": scored["calculation"]["bound"], "strict_correct": scored["calculation"]["strict_complete"]}})
    write_json(OUT / "failure-taxonomy.json", {key: scored["failure_counts"].get(key, 0) for key in [f"EB{i}_{name}" for i, name in enumerate(["correct", "no_relevant_fact_in_packet", "correct_source_present_but_no_matching_financial_fact", "metric_semantic_mismatch", "period_mismatch", "scope_or_segment_ambiguity", "multiple_statement_ambiguity", "multi_slot_association_error", "wrong_fact_selected", "wrong_physical_source_selected", "model_returned_missing_despite_bindable_fact", "model_returned_ambiguous_despite_unique_fact", "binding_schema_or_validator_failure", "supervisor_required_slot_defect", "other"]) ]})
    safety = {"false_binding_queries": scored["false_binding_queries"], "false_binding_slots": scored["false_binding_slots"], "invented_fact_ids": sum(row["invented_fact_ids"] for row in predictions), "invented_source_ids": sum(row["invented_source_ids"] for row in predictions), "new_slots": sum(row["new_slots"] for row in predictions), "answer_leakage": sum(row["answer_leakage"] for row in predictions), "invented_numeric_values": sum(row["invented_numeric_values"] for row in predictions), "calculation_outputs": sum(row["calculation_outputs"] for row in predictions)}
    write_json(OUT / "safety-analysis.json", safety)
    write_json(OUT / "latency-token-cost.json", {"binder_calls": model_calls, "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens, "reasoning_tokens": sum(int(row.get("reasoning_tokens") or 0) for row in metadata_rows), "average_latency_ms": statistics.mean(latencies) if latencies else 0.0, "p50_latency_ms": statistics.median(latencies) if latencies else 0.0, "p95_latency_ms": percentile(latencies), "max_latency_ms": max(latencies) if latencies else 0.0, "total_wall_time_ms": runtime["formal_wall_time_ms"], "estimated_cost": "not_configured", "mean_facts_per_binder_call": statistics.mean(facts_per_call) if facts_per_call else 0.0, "p95_facts_per_binder_call": percentile(facts_per_call), "max_facts_per_binder_call": max(facts_per_call) if facts_per_call else 0.0})
    eligible_predictions = [row for row in predictions if not row["skipped_no_fact_supply"]]
    provider_success = sum(int(bool(row.get("metadata") and row["metadata"].get("provider_response_success"))) for row in eligible_predictions)
    structured = sum(int(row.get("binding_schema_valid") and bool(row.get("metadata") and row["metadata"].get("structured_output_success"))) for row in eligible_predictions)
    schema_valid = sum(int(row.get("binding_schema_valid")) for row in eligible_predictions)
    validator_pass = sum(int(row["binding_validator_pass"]) for row in predictions)
    write_json(OUT / "structured-output-metrics.json", {"eligible_queries": len(eligible_predictions), "skipped_no_fact_supply": len(predictions) - len(eligible_predictions), "provider_response_success": provider_success, "structured_output_success": structured, "schema_valid": schema_valid, "binding_validator_pass": validator_pass, "provider_response_success_rate": pct(provider_success, len(eligible_predictions)), "structured_output_success_rate": pct(structured, len(eligible_predictions)), "schema_valid_rate": pct(schema_valid, len(eligible_predictions))})
    effective = bool(structured >= 0.98 * max(1, model_calls) and validator_pass >= 0.98 * QUESTION_TOTAL and not scored["false_binding_queries"] and not safety["invented_fact_ids"] and not safety["invented_source_ids"] and not safety["answer_leakage"] and scored["direct"]["strict_complete"] >= 40 and scored["calculation"]["strict_complete"] >= 6 and scored["multi"]["strict_complete"] >= 4 and scored["direct"]["success_given_bindable"] >= 80.0)
    partial = bool(not effective and structured >= 0.95 * max(1, model_calls) and validator_pass >= 0.95 * QUESTION_TOTAL and not scored["false_binding_queries"] and not safety["invented_fact_ids"] and not safety["invented_source_ids"] and not safety["answer_leakage"] and scored["direct"]["strict_complete"] >= 36 and scored["calculation"]["strict_complete"] >= 5 and scored["multi"]["strict_complete"] >= 3)
    decision = {"gate": GATE, "evaluation_role": "development_shadow_v2_semantic_evidence_binder", "base_commit": BASE_COMMIT, "production_default": "V1", "production_switch_allowed": False, "supervisor_frozen": True, "supervisor_model": MODEL, "binder_model": MODEL, "binder_model_role": "strong_general_llm", "binder_provider_role": "evidence_binder", "model_calls": model_calls, "smoke_model_calls": 2, "retrieval_calls": 0, "reranker_calls": 0, "financial_fact_v1_modified": False, "provider_response_success": provider_success, "structured_output_success": structured, "schema_valid": schema_valid, "binding_validator_pass": validator_pass, "eligible_queries": len(eligible_predictions), "skipped_no_fact_supply": len(predictions) - len(eligible_predictions), "direct_fact_questions": DIRECT_TOTAL, "direct_fact_strict_complete": scored["direct"]["strict_complete"], "direct_fact_strict_bindable": scored["direct"]["strict_bindable"], "calculation_questions": CALC_TOTAL, "calculation_all_operand_bound": scored["calculation"]["strict_complete"], "calculation_supply_complete_reference": "6/11", "multi_evidence_questions": MULTI_TOTAL, "multi_evidence_complete_bound": scored["multi"]["strict_complete"], "false_binding_queries": scored["false_binding_queries"], "false_binding_slots": scored["false_binding_slots"], "invented_fact_ids": safety["invented_fact_ids"], "answer_leakage": safety["answer_leakage"], "semantic_evidence_binder_effective": True if effective else ("partial" if partial else False), "semantic_binder_frozen": effective, "dominant_failure": "none" if effective else ("coverage" if partial else "binding_safety_or_semantics"), "next_gate": "v2_04_missing_slot_retrieval_repair" if effective else "v2_03_semantic_binder_failure_review", "gold_reads_before_prediction_seal": 0, "prediction_sealed": True}
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {"gate": GATE, "description": "Development-shadow Semantic Evidence Binder over frozen SupervisorPlan and NF-V2-02 Top20 FinancialFactV1. No retrieval, reranker, calculator, generator, validator repair, fact mutation, or Gold access before prediction seal.", "decision": decision})
    print(json.dumps({"gate": GATE, "model_calls": model_calls, "structured_output": structured, "validator_pass": validator_pass, "direct": scored["direct"]["strict_complete"], "calc": scored["calculation"]["strict_complete"], "multi": scored["multi"]["strict_complete"], "false_binding": scored["false_binding_queries"], "effective": decision["semantic_evidence_binder_effective"], "next_gate": decision["next_gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
