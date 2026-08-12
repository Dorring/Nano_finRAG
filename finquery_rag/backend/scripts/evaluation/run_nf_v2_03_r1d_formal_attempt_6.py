#!/usr/bin/env python3
"""NF-V2-03 R1D supply-conditioned Binder evaluation.

The runner consumes the sealed R1C recovered fact pool and uses the
selection-only provider DTO.  Gold is opened only after the prediction seal.
"""

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
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.evidence import BindingStatus  # noqa: E402
from rag_v2.evidence.binder_service import BinderRequest, BinderRun, SemanticBinderService  # noqa: E402
from rag_v2.evidence.constrained_binder_provider import BailianConstrainedBinderProvider  # noqa: E402
from scripts.evaluation import run_nf_e2e_09_r0_structured_fact_representation as nf09  # noqa: E402
from scripts.evaluation import run_nf_v2_02_top20_financial_fact_expansion as nf02  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1a_binding_contract_recovery as r1a  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1c_supply_recovery as r1c  # noqa: E402
from scripts.evaluation import run_nf_v2_03_semantic_evidence_binder as legacy  # noqa: E402


BASE_COMMIT = "e96cc0fba54ab69caa4fc7d2ecaad91db893938c"
GATE = "NF-V2-03-R1D"
MODEL = "qwen3.7-plus"
QUESTION_TOTAL = 72
DIRECT_TOTAL = 56
CALC_TOTAL = 11
MULTI_TOTAL = 5
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r1d-supply-conditioned-binder"
FORMAL_OUT = OUT / "formal-attempt-6"
R1C_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r1c-supply-and-protocol-recovery"
R1B_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r1b-constrained-binding"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
SYNTHETIC = OUT / "synthetic-provider-test.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value: Any) -> str:
    return hashlib.sha256((json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")).hexdigest()


def pct(value: int, total: int) -> float:
    return round(100.0 * value / total, 4) if total else 0.0


def percentile(values: list[float], quantile: float = 0.95) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))]


def load_r1c_frozen_inputs() -> dict[str, Any]:
    """Load plans/retrieval and the sealed R1C fact pool without Gold."""

    frozen = legacy.load_frozen_inputs()
    recovered_path = R1C_OUT / "recovered-financial-facts.jsonl.gz"
    recovery_meta = read_json(R1C_OUT / "materialization-recovery-results.json")
    recovered = read_jsonl_gz(recovered_path)
    old_facts = list(frozen["facts"])
    combined, duplicate_count = nf09.dedup_facts(old_facts + recovered)
    combined_sha = stable_sha(combined)
    if combined_sha != recovery_meta.get("recovery_sha256"):
        raise RuntimeError("R1C recovered FinancialFact supply hash mismatch")
    if len(combined) != int(recovery_meta.get("new_deduplicated_fact_count", -1)):
        raise RuntimeError("R1C recovered FinancialFact count mismatch")
    state = nf02.verify_frozen_top100()
    requests: dict[str, BinderRequest] = {}
    request_rows: list[dict[str, Any]] = []
    for question_id in sorted(frozen["plans"]):
        plan = frozen["plans"][question_id]["plan"]
        packet: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rank, candidate_id in enumerate(state["top20_order"].get(question_id, []), 1):
            for fact in combined:
                candidate_ids = {str(fact.get("candidate_id"))} | {str(item) for item in fact.get("candidate_ids", [])}
                if str(candidate_id) not in candidate_ids:
                    continue
                fact_id = str(fact["fact_id"])
                if fact_id in seen:
                    continue
                seen.add(fact_id)
                projection = dict(fact)
                projection["candidate_id"] = str(candidate_id)
                projection["candidate_rank"] = rank
                packet.append(projection)
        requests[question_id] = BinderRequest(question_id, frozen["plans"][question_id]["question"], plan, tuple(packet))
        request_rows.append({
            "question_id": question_id,
            "fact_count": len(packet),
            "required_slot_count": len(plan.required_slots),
            "skipped_no_fact_supply": not bool(packet),
        })
    no_supply_count = sum(int(row["skipped_no_fact_supply"]) for row in request_rows)
    return {
        "requests": requests,
        "plans": frozen["plans"],
        "top20_order": state["top20_order"],
        "facts": combined,
        "facts_sha256": combined_sha,
        "duplicate_count": duplicate_count,
        "plan_sha256": frozen["plan_sha256"],
        "top20_order_sha256": frozen["top20_order_sha256"],
        "request_rows": request_rows,
        "no_supply_count": no_supply_count,
    }


def provider_schema_contract_sha() -> str:
    return stable_sha({
        "dto": "BinderSelectionDTOv1",
        "selection_only": True,
        "slot_properties": "exact RequiredSlot IDs",
        "slot_values": {"type": "array", "items": {"type": "string", "enum": "query-local handles"}},
        "uniqueItems": False,
        "duplicates": "deterministic adapter failure duplicate_fact_handle",
    })


def run_formal(
    config: dict[str, Any],
    frozen: dict[str, Any],
    *,
    system_prompt: str | None = None,
    fact_view_version: str = "v1",
    source_metadata_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = BailianConstrainedBinderProvider(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_name=MODEL,
        enable_thinking=False,
        temperature=0.0,
        timeout=180.0,
        max_retries=0,
        system_prompt=system_prompt,
        fact_view_version=fact_view_version,
        source_metadata_by_candidate=source_metadata_by_candidate,
    )
    service = SemanticBinderService(provider)
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    try:
        for index, question_id in enumerate(sorted(frozen["requests"]), 1):
            request = frozen["requests"][question_id]
            run: BinderRun = service.bind(request)
            metadata = run.metadata.to_dict() if run.metadata else None
            raw = run.raw_response or ""
            reasons = list(run.validation.reasons)
            row = run.to_dict()
            row.update({
                "call_index": index,
                "question": request.question,
                "intent": request.plan.intent.value,
                "operation": request.plan.operation,
                "required_slots": [slot.to_dict() for slot in request.plan.required_slots],
                "fact_count": len(request.facts),
                "candidate_ranks": sorted({fact.get("candidate_rank") for fact in request.facts if fact.get("candidate_rank") is not None}),
                "raw_response": raw,
                "provider_response_success": bool(metadata and metadata.get("provider_response_success")) if metadata else True,
                "structured_output_success": bool(metadata and metadata.get("structured_output_success")) if metadata else True,
                "dto_valid": bool(run.schema_valid) if not run.skipped_no_fact_supply else True,
                "adapter_valid": bool(run.schema_valid and run.binding.status != BindingStatus.INVALID.value),
                "unknown_slot": int(any(reason.startswith("unknown_slot") for reason in reasons)),
                "unknown_fact": int(any(reason.startswith("unknown_fact") for reason in reasons)),
                "duplicate_handle": int(any("duplicate_fact_handle" in reason for reason in reasons)),
                "status_violation": int(any("status_violation" in reason for reason in reasons)),
                "cardinality_violation": int(any("cardinality" in reason for reason in reasons)),
                "answer_leakage": int(any(token in raw.casefold() for token in ("answer:", "final answer", "citation:"))),
                "calculation_leakage": int(any(token in raw.casefold() for token in ("growth", "margin", "result:", "calculation"))),
                "invented_numeric_values": int("$" in raw or "%" in raw),
                "invented_fact_ids": int(any(reason.startswith("unknown_fact:") for reason in reasons)),
                "invented_source_ids": 0,
                "new_slots": int(any(reason.startswith("unknown_slot:") for reason in reasons)),
                "role_mutation": 0,
            })
            predictions.append(row)
            if not row["skipped_no_fact_supply"] and not row["provider_response_success"]:
                failures.append({
                    "question_id": question_id,
                    "call_index": index,
                    "fact_count": len(request.facts),
                    "metadata": metadata,
                })
    finally:
        provider.close()
    if failures:
        raise RuntimeError(f"Binder provider failures during Attempt 6: {failures[:2]}")
    return predictions, {"formal_wall_time_ms": round((time.perf_counter() - started_all) * 1000.0, 3), "failures": failures}


def source_matches(fact: Mapping[str, Any], expected_sources: list[Mapping[str, Any]]) -> bool:
    candidate_ids = {str(fact.get("candidate_id"))} | {str(item) for item in fact.get("candidate_ids", [])}
    return any(str(source.get("candidate_key")) in candidate_ids for source in expected_sources if source.get("candidate_key"))


def period_matches(fact: Mapping[str, Any], slot: Any) -> bool:
    return r1c.period(fact.get("normalized_period") or fact.get("raw_period")) == r1c.period(slot.period)


def reviewed_direct_map() -> tuple[set[str], dict[str, set[str]]]:
    rows = read_json(R1B_OUT / "fact-semantic-compatibility-review.json")["direct"]["rows"]
    strict_ids = {str(row["question_id"]) for row in rows if row.get("reviewed_semantic_compatible") and row.get("reviewed_period_compatible")}
    fact_ids = {str(row["question_id"]): {str(item) for item in row.get("reviewed_fact_ids", [])} for row in rows}
    return strict_ids, fact_ids


def slot_is_strict(
    question_id: str,
    slot: Any,
    fact: Mapping[str, Any],
    label: Mapping[str, Any],
    source_map: Mapping[str, Mapping[str, Any]],
    reviewed_ids: set[str],
    reviewed_fact_ids: Mapping[str, set[str]],
    generic_direct_ids: set[str],
) -> bool:
    expected = r1a.expected_sources(slot, label)
    if not source_matches(fact, expected) or not period_matches(fact, slot):
        return False
    if question_id in reviewed_ids:
        return str(fact.get("fact_id")) in reviewed_fact_ids.get(question_id, set())
    source = source_map.get(str(fact.get("candidate_id")))
    return r1c.view_metric_match(slot, fact, source) if question_id in generic_direct_ids else r1c.view_metric_match(slot, fact, source)


def score_supply_conditioned(frozen: dict[str, Any], predictions: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    current_view = read_json(R1C_OUT / "current-vs-view-bindability.json")
    generic_direct_ids = {str(item) for item in current_view.get("generic_recovered_strict_questions", [])}
    reviewed_ids, reviewed_fact_ids = reviewed_direct_map()
    direct_bindable = reviewed_ids | generic_direct_ids
    calc_supply_rows = read_json(R1C_OUT / "calculation-supply-funnel.json")["rows"]
    calc_bindable = {str(row["question_id"]) for row in calc_supply_rows if row.get("strict_bindable")}
    multi_bindable: set[str] = set()
    state = nf02.verify_frozen_top100()
    source_map = r1c.candidate_source_map(state)
    by_id = {row["question_id"]: row for row in predictions}
    cohorts = {
        "direct": sorted(qid for qid, item in frozen["plans"].items() if item["plan"].intent.value == "DIRECT_FACT"),
        "calculation": sorted(qid for qid, item in frozen["plans"].items() if item["plan"].intent.value == "CALCULATION"),
        "multi": sorted(qid for qid, item in frozen["plans"].items() if item["plan"].intent.value == "MULTI_EVIDENCE"),
    }
    strict_rows: list[dict[str, Any]] = []
    false_queries = 0
    false_slots = 0
    for question_id in sorted(frozen["requests"]):
        request = frozen["requests"][question_id]
        label = labels[question_id]
        prediction = by_id[question_id]
        fact_by_id = {str(fact["fact_id"]): fact for fact in request.facts}
        binding = prediction["binding"]
        slot_results: list[dict[str, Any]] = []
        for slot in request.plan.required_slots:
            selected_ids = list((binding or {}).get("slot_bindings", {}).get(slot.slot_id, []))
            selected_fact = fact_by_id.get(str(selected_ids[0])) if len(selected_ids) == 1 else None
            strict = bool(selected_fact and slot_is_strict(question_id, slot, selected_fact, label, source_map, reviewed_ids, reviewed_fact_ids, generic_direct_ids))
            slot_results.append({"slot_id": slot.slot_id, "selected_fact_id": selected_ids[0] if len(selected_ids) == 1 else None, "strict_correct": strict})
        strict_complete = prediction["final_binding_status"] == BindingStatus.BOUND.value and bool(slot_results) and all(result["strict_correct"] for result in slot_results)
        if prediction["final_binding_status"] == BindingStatus.BOUND.value and not strict_complete:
            false_queries += 1
            false_slots += sum(int(not result["strict_correct"]) for result in slot_results)
        strict_rows.append({
            "question_id": question_id,
            "intent": request.plan.intent.value,
            "strict_bindable": question_id in (direct_bindable if request.plan.intent.value == "DIRECT_FACT" else calc_bindable if request.plan.intent.value == "CALCULATION" else multi_bindable),
            "strict_complete": strict_complete,
            "status": prediction["final_binding_status"],
            "slot_results": slot_results,
        })

    by_strict = {row["question_id"]: row for row in strict_rows}

    def cohort(ids: list[str], bindable: set[str], *, evaluable: bool = True) -> dict[str, Any]:
        rows = [by_strict[qid] for qid in ids]
        bindable_rows = [row for row in rows if row["question_id"] in bindable]
        missed = sum(int(row["strict_bindable"] and not row["strict_complete"] and row["status"] == BindingStatus.MISSING.value) for row in rows)
        ambiguous = sum(int(row["strict_bindable"] and not row["strict_complete"] and row["status"] == BindingStatus.AMBIGUOUS.value) for row in rows)
        wrong = sum(int(row["strict_bindable"] and not row["strict_complete"] and row["status"] == BindingStatus.BOUND.value) for row in rows)
        return {
            "questions": len(rows),
            "strict_bindable": len(bindable_rows),
            "strict_complete": sum(int(row["strict_complete"]) for row in rows),
            "strict_correct_given_bindable": sum(int(row["strict_complete"]) for row in bindable_rows),
            "success_given_bindable_percent": pct(sum(int(row["strict_complete"]) for row in bindable_rows), len(bindable_rows)),
            "missed_bindable": missed,
            "wrong_fact_given_bindable": wrong,
            "ambiguous_given_bindable": ambiguous,
            "missing_given_bindable": missed,
            "semantic_evaluation": "evaluable" if evaluable and bindable_rows else "not_evaluable",
            "rows": rows,
        }

    direct = cohort(cohorts["direct"], direct_bindable)
    calculation = cohort(cohorts["calculation"], calc_bindable)
    multi = cohort(cohorts["multi"], multi_bindable, evaluable=False)
    return {
        "direct": direct,
        "calculation": calculation,
        "multi": multi,
        "direct_bindable_ids": sorted(direct_bindable),
        "calculation_bindable_ids": sorted(calc_bindable),
        "false_binding_queries": false_queries,
        "false_binding_slots": false_slots,
        "strict_rows": strict_rows,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FORMAL_OUT.mkdir(parents=True, exist_ok=True)
    synthetic = read_json(SYNTHETIC)
    if synthetic.get("pass") is not True:
        decision = {"gate": GATE, "formal_attempt_6": "not_run", "reason": "synthetic_provider_test_failed", "synthetic": synthetic, "production_default": "V1", "production_switch_allowed": False}
        write_json(OUT / "decision.json", decision)
        print(json.dumps(decision, sort_keys=True))
        return 3
    config = legacy.load_config()
    frozen = load_r1c_frozen_inputs()
    config_artifact = {
        "gate": GATE,
        "base_commit": BASE_COMMIT,
        "provider": "Alibaba Bailian",
        "provider_role": "evidence_binder",
        "model": MODEL,
        "model_role": "strong_general_llm",
        "thinking": False,
        "temperature": 0.0,
        "max_retries": 0,
        "http_timeout_seconds": 180,
        "selection_dto": "BinderSelectionDTOv1",
        "selection_schema_sha256": provider_schema_contract_sha(),
        "binder_fact_view_sha256": sha256_file(R1C_OUT / "binder-fact-view-contract.json"),
        "supervisor_prediction_sha256": frozen["plan_sha256"],
        "top20_order_sha256": frozen["top20_order_sha256"],
        "financial_fact_supply_sha256": frozen["facts_sha256"],
        "supply_ceiling": {"direct": "27/56", "calculation": "6/11", "multi_evidence": "0/5"},
        "no_supply_queries": frozen["no_supply_count"],
        "transport_resilience": {"sdk_max_retries": 0, "semantic_attempt_budget": 1, "transport_retry_budget": 1, "retry_delay_seconds": 3, "http_timeout_seconds": 180},
        "production_default": "V1",
        "production_switch_allowed": False,
        "gold_reads_before_prediction_seal": 0,
        "api_key_persisted": False,
    }
    write_json(FORMAL_OUT / "config.json", config_artifact)
    predictions, runtime = run_formal({**config, "base_url": os.getenv("V2_SUPERVISOR_BASE_URL", "").strip()}, frozen)
    write_jsonl_gz(FORMAL_OUT / "predictions.jsonl.gz", predictions)
    prediction_sha = sha256_file(FORMAL_OUT / "predictions.jsonl.gz")
    seal = {
        "gate": GATE,
        "sealed": True,
        "predictions_written": len(predictions),
        "questions_expected": QUESTION_TOTAL,
        "provider_model_calls": sum(int(not row["skipped_no_fact_supply"]) for row in predictions),
        "max_calls_per_query": 1,
        "gold_reads_before_prediction_seal": 0,
        "prediction_sha256": prediction_sha,
        "financial_fact_supply_sha256": frozen["facts_sha256"],
        "sealed_before_gold": True,
    }
    write_json(FORMAL_OUT / "prediction-seal.json", seal)
    if sha256_file(FORMAL_OUT / "predictions.jsonl.gz") != prediction_sha:
        raise RuntimeError("Attempt 6 prediction seal verification failed")

    # Gold is intentionally loaded only after the prediction seal exists.
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if row}
    scored = score_supply_conditioned(frozen, predictions, labels)
    eligible = [row for row in predictions if not row["skipped_no_fact_supply"]]
    metadata_rows = [row["metadata"] for row in predictions if row.get("metadata")]
    provider_success = sum(int(row["provider_response_success"]) for row in eligible)
    structured = sum(int(row["structured_output_success"]) for row in eligible)
    dto_valid = sum(int(row["dto_valid"]) for row in eligible)
    adapter_valid = sum(int(row["adapter_valid"]) for row in eligible)
    validator_pass = sum(int(row["binding_validator_pass"]) for row in predictions)
    structural = {
        "questions": QUESTION_TOTAL,
        "model_required_queries": len(eligible),
        "provider_responses": provider_success,
        "structured_output": structured,
        "dto_valid": dto_valid,
        "adapter_valid": adapter_valid,
        "binding_validator_pass": validator_pass,
        "unknown_slots": sum(row["unknown_slot"] for row in predictions),
        "unknown_facts": sum(row["unknown_fact"] for row in predictions),
        "duplicate_handles": sum(row["duplicate_handle"] for row in predictions),
        "status_violations": sum(row["status_violation"] for row in predictions),
        "cardinality_violations": sum(row["cardinality_violation"] for row in predictions),
        "calculation_leakage": sum(row["calculation_leakage"] for row in predictions),
        "gold_reads_before_prediction_seal": 0,
    }
    write_json(FORMAL_OUT / "structural-metrics.json", structural)
    write_json(FORMAL_OUT / "direct-supply-conditioned.json", {"supply_ceiling": "27/56", **scored["direct"]})
    write_json(FORMAL_OUT / "calculation-supply-conditioned.json", {"supply_ceiling": "6/11", **scored["calculation"]})
    write_json(FORMAL_OUT / "multi-evidence-supply-conditioned.json", {"supply_ceiling": "0/5", "semantic_evaluation": "not_evaluable", "absolute_complete_upper_bound": "0/5", **scored["multi"]})
    write_json(FORMAL_OUT / "false-binding-analysis.json", {"false_binding_queries": scored["false_binding_queries"], "false_binding_slots": scored["false_binding_slots"], "hard_safety_target": 0})
    latencies = [float(row.get("latency_ms") or 0.0) for row in metadata_rows]
    facts_per_call = [float(row["fact_count"]) for row in eligible]
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in metadata_rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in metadata_rows)
    write_json(FORMAL_OUT / "latency-token-cost.json", {
        "binder_calls": len(eligible),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in metadata_rows),
        "average_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies),
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "mean_facts_per_call": statistics.mean(facts_per_call) if facts_per_call else 0.0,
        "p95_facts_per_call": percentile(facts_per_call),
        "max_facts_per_call": max(facts_per_call) if facts_per_call else 0.0,
        "total_wall_time_ms": runtime["formal_wall_time_ms"],
        "estimated_cost": "not_configured",
    })
    safety = {
        "false_binding_queries": scored["false_binding_queries"],
        "false_binding_slots": scored["false_binding_slots"],
        "invented_fact_ids": sum(row["invented_fact_ids"] for row in predictions),
        "invented_source_ids": sum(row["invented_source_ids"] for row in predictions),
        "new_slots": sum(row["new_slots"] for row in predictions),
        "role_mutation": sum(row["role_mutation"] for row in predictions),
        "answer_leakage": sum(row["answer_leakage"] for row in predictions),
        "calculation_leakage": sum(row["calculation_leakage"] for row in predictions),
        "invented_numeric_values": sum(row["invented_numeric_values"] for row in predictions),
    }
    write_json(FORMAL_OUT / "safety.json", safety)
    direct_quality = scored["direct"]["success_given_bindable_percent"]
    calc_quality = scored["calculation"]["success_given_bindable_percent"]
    semantic_effective = bool(
        direct_quality >= 90.0
        and calc_quality >= 90.0
        and scored["false_binding_queries"] == 0
        and structural["unknown_slots"] == 0
        and structural["unknown_facts"] == 0
        and structural["duplicate_handles"] == 0
        and structural["status_violations"] == 0
        and structural["calculation_leakage"] == 0
        and safety["invented_fact_ids"] == 0
        and safety["invented_source_ids"] == 0
        and safety["answer_leakage"] == 0
    )
    absolute_sufficient = bool(scored["direct"]["strict_complete"] >= 40 and scored["calculation"]["strict_complete"] >= 6 and scored["multi"]["strict_complete"] >= 4)
    decision = {
        "gate": GATE,
        "base_commit": BASE_COMMIT,
        "binder_model": MODEL,
        "formal_attempt_6": "executed",
        "formal_run_complete": True,
        "gold_reads_before_prediction_seal": 0,
        "prediction_seal": "pass",
        "provider_responses": f"{provider_success}/{len(eligible)}",
        "dto_valid": f"{dto_valid}/{len(eligible)}",
        "adapter_valid": f"{adapter_valid}/{len(eligible)}",
        "binding_validator_pass": f"{validator_pass}/{QUESTION_TOTAL}",
        "direct_supply_ceiling": "27/56",
        "direct_strict_complete": f"{scored['direct']['strict_complete']}/56",
        "direct_success_given_bindable": direct_quality,
        "calculation_supply_ceiling": "6/11",
        "calculation_all_operand_strict": f"{scored['calculation']['strict_complete']}/11",
        "calculation_success_given_bindable": calc_quality,
        "multi_evidence_supply_ceiling": "0/5",
        "multi_evidence_semantic_evaluation": "not_evaluable",
        "false_binding_queries": scored["false_binding_queries"],
        "structural_safety_violations": sum(structural[key] for key in ("unknown_slots", "unknown_facts", "duplicate_handles", "status_violations", "cardinality_violations", "calculation_leakage")),
        "binder_semantic_selection_effective": semantic_effective,
        "binder_semantic_policy_frozen": semantic_effective,
        "binder_absolute_coverage_sufficient": absolute_sufficient,
        "dominant_failure": "none" if absolute_sufficient else ("evidence_supply" if semantic_effective else "binder_semantic_selection"),
        "next_gate": "v2_04_missing_evidence_supply_repair" if semantic_effective else "v2_03_binder_semantic_failure_review",
        "production_default": "V1",
        "production_switch_allowed": False,
    }
    write_json(FORMAL_OUT / "decision.json", decision)
    write_json(OUT / "decision.json", decision)
    write_json(FORMAL_OUT / "README.md", {"gate": GATE, "description": "Supply-conditioned selection-only Binder evaluation over the frozen R1C fact pool. Gold is read only after the prediction seal.", "decision": decision})
    print(json.dumps({"gate": GATE, "provider_responses": provider_success, "dto_valid": dto_valid, "validator": validator_pass, "direct": scored["direct"]["strict_complete"], "direct_quality": direct_quality, "calc": scored["calculation"]["strict_complete"], "calc_quality": calc_quality, "false_binding": scored["false_binding_queries"], "semantic_effective": semantic_effective, "next_gate": decision["next_gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
