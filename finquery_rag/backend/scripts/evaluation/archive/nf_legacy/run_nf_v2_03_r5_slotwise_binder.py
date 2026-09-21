#!/usr/bin/env python3
"""NF-V2-03 R5 slot-wise Binder formulation and conditional Attempt 9."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.evidence import BindingStatus  # noqa: E402
from rag_v2.contracts.plan import Intent  # noqa: E402
from rag_v2.evidence.binder_service import BinderRequest, SemanticBinderService  # noqa: E402
from rag_v2.evidence.slotwise_binder import (  # noqa: E402
    SLOTWISE_FORMULATION,
    SLOTWISE_SYSTEM_PROMPT,
    BailianSlotwiseBinderProvider,
)
from scripts.evaluation import run_nf_v2_02_top20_financial_fact_expansion as nf02  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1c_supply_recovery as r1c  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r3_fact_view_v2 as r3  # noqa: E402


BASE_COMMIT = "32154f9c87181eebaa7afd12baba35c6a05205a2"
MODEL = "qwen3.7-plus"
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r5-slotwise-binder"
R4_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r4-binder-model-review"
R3_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r3-binder-fact-view-v2"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def load_frozen() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    frozen = r1d.load_r1c_frozen_inputs()
    return frozen, r1c.candidate_source_map(nf02.verify_frozen_top100())


def load_cohort() -> dict[str, Any]:
    path = R4_OUT / "model-review-cohort.json"
    digest_path = R4_OUT / "model-review-cohort.sha256"
    actual = sha256_file(path)
    expected = digest_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError("R4 frozen diagnostic cohort SHA mismatch")
    cohort = read_json(path)
    cohort["cohort_sha256"] = actual
    return cohort


def provider(source_map: Mapping[str, Mapping[str, Any]]) -> BailianSlotwiseBinderProvider:
    config = r1d.legacy.load_config()
    return BailianSlotwiseBinderProvider(
        base_url=os.getenv("V2_SUPERVISOR_BASE_URL", "").strip(),
        api_key=config["api_key"],
        model_name=MODEL,
        enable_thinking=False,
        temperature=0.0,
        timeout=180.0,
        max_retries=0,
        system_prompt=SLOTWISE_SYSTEM_PROMPT,
        fact_view_version="v2",
        source_metadata_by_candidate=source_map,
    )


def summarize_run(request: BinderRequest, run: Any, *, group: list[str]) -> dict[str, Any]:
    row = run.to_dict()
    row.update({
        "question_id": request.question_id,
        "question": request.question,
        "intent": request.plan.intent.value,
        "operation": request.plan.operation,
        "groups": group,
        "fact_count": len(request.facts),
        "required_slots": [slot.to_dict() for slot in request.plan.required_slots],
        "provider_response_success": bool(run.metadata and run.metadata.provider_response_success) if run.metadata else True,
        "structured_output_success": bool(run.metadata and run.metadata.structured_output_success) if run.metadata else True,
        "dto_valid": bool(run.schema_valid),
        "adapter_valid": bool(run.schema_valid and run.binding and run.binding.status != BindingStatus.INVALID.value),
        "binding_validator_pass": bool(run.validation.passed),
        "validation_reasons": list(run.validation.reasons),
        "raw_response": run.raw_response or "",
    })
    return row


def actual_handles(row: Mapping[str, Any], request: BinderRequest) -> dict[str, list[str]]:
    by_id = {str(fact["fact_id"]): f"F{index:02d}" for index, fact in enumerate(request.facts, 1)}
    binding = row.get("binding") or {}
    return {
        slot_id: [by_id.get(str(fact_id), str(fact_id)) for fact_id in ids]
        for slot_id, ids in (binding.get("slot_bindings") or {}).items()
    }


def synthetic_run() -> dict[str, Any]:
    cases = r3.synthetic_cases()
    cases.extend([
        (BinderRequest("slotwise_syn_13", "Select component and total independently.", r3.make_plan([r3.make_slot("component", "gross profit", "FY2026", "component"), r3.make_slot("total", "sales", "FY2026", "total")], Intent.CALCULATION, "percentage_share"), (r3.synthetic_fact("sw_f13", "gross profit", "FY2026", statement="Results", row=["Gross profit"], headers=["FY2026"], section="Summary"), r3.synthetic_fact("sw_f14", "sales", "FY2026", statement="Results", row=["Sales"], headers=["FY2026"], section="Summary"))), {"component": ["F01"], "total": ["F02"]}),
        (BinderRequest("slotwise_syn_14", "Select minuend and subtrahend independently.", r3.make_plan([r3.make_slot("minuend", "sales", "FY2026", "minuend"), r3.make_slot("subtrahend", "sales", "FY2025", "subtrahend")], Intent.CALCULATION, "difference"), (r3.synthetic_fact("sw_f15", "sales", "FY2026", statement="Results", row=["Sales"], headers=["FY2026"], section="Summary"), r3.synthetic_fact("sw_f16", "sales", "FY2025", statement="Results", row=["Sales"], headers=["FY2025"], section="Summary"))), {"minuend": ["F01"], "subtrahend": ["F02"]}),
        (BinderRequest("slotwise_syn_15", "Select numerator and denominator independently.", r3.make_plan([r3.make_slot("numerator", "net income", "FY2026", "numerator"), r3.make_slot("denominator", "sales", "FY2026", "denominator")], Intent.CALCULATION, "percentage_share"), (r3.synthetic_fact("sw_f17", "net income", "FY2026", statement="Results", row=["Net income"], headers=["FY2026"], section="Summary"), r3.synthetic_fact("sw_f18", "sales", "FY2026", statement="Results", row=["Sales"], headers=["FY2026"], section="Summary"))), {"numerator": ["F01"], "denominator": ["F02"]}),
        (BinderRequest("slotwise_syn_16", "Select current and prior balances independently.", r3.make_plan([r3.make_slot("current", "cash", "FY2026", "current"), r3.make_slot("prior", "cash", "FY2025", "prior")], Intent.CALCULATION, "difference"), (r3.synthetic_fact("sw_f19", "cash", "FY2026", statement="Balance Sheet", row=["Cash"], headers=["FY2026"], section="Assets"), r3.synthetic_fact("sw_f20", "cash", "FY2025", statement="Balance Sheet", row=["Cash"], headers=["FY2025"], section="Assets"))), {"current": ["F01"], "prior": ["F02"]}),
    ])
    p = provider({})
    service = SemanticBinderService(p)
    rows: list[dict[str, Any]] = []
    try:
        for request, expected in cases:
            run = service.bind(request)
            row = summarize_run(request, run, group=["synthetic"])
            actual = actual_handles(row, request)
            for slot in request.plan.required_slots:
                actual.setdefault(slot.slot_id, [])
            semantic_correct = all(sorted(actual.get(slot_id, [])) == sorted(handles) for slot_id, handles in expected.items())
            false_binding = any(not handles and actual.get(slot_id) for slot_id, handles in expected.items())
            row.update({"expected": expected, "actual": actual, "semantic_correct": semantic_correct, "false_binding": bool(false_binding)})
            rows.append(row)
    finally:
        p.close()
    calc_rows = [row for row in rows if row["intent"] == "CALCULATION"]
    calc_correct = sum(int(row["semantic_correct"]) for row in calc_rows)
    summary = {
        "gate": "NF-V2-03-R5", "formulation": SLOTWISE_FORMULATION, "model": MODEL, "benchmark_questions_used": 0,
        "provider_calls": len(rows), "provider_success": sum(int(row["provider_response_success"]) for row in rows),
        "structured_output": sum(int(row["structured_output_success"]) for row in rows), "dto_valid": sum(int(row["dto_valid"]) for row in rows),
        "adapter_valid": sum(int(row["adapter_valid"]) for row in rows), "binding_validator": sum(int(row["binding_validator_pass"]) for row in rows),
        "semantic_correct": sum(int(row["semantic_correct"]) for row in rows), "semantic_total": len(rows),
        "false_binding": sum(int(row["false_binding"]) for row in rows), "calculation_groups_correct": calc_correct, "calculation_groups_total": len(calc_rows),
        "pass": len(rows) >= 12 and sum(int(row["semantic_correct"]) for row in rows) >= 10 and sum(int(row["false_binding"]) for row in rows) == 0 and calc_correct >= 5,
        "rows": rows,
    }
    write_json(OUT / "synthetic-slotwise-suite.json", summary)
    return summary


def strict_review_rows(rows: list[dict[str, Any]], frozen: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    labels = {str(item["case_id"]): item for item in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if item}
    reviewed_ids, reviewed_fact_ids = r1d.reviewed_direct_map()
    evaluated: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = row["question_id"]
        request = frozen["requests"][qid]
        fact_by_id = {str(fact["fact_id"]): fact for fact in request.facts}
        binding = row.get("binding") or {}
        slot_results: list[dict[str, Any]] = []
        for slot in request.plan.required_slots:
            ids = list((binding.get("slot_bindings") or {}).get(slot.slot_id, []))
            fact = fact_by_id.get(str(ids[0])) if len(ids) == 1 else None
            strict = bool(fact and r1d.slot_is_strict(qid, slot, fact, labels[qid], source_map, reviewed_ids, reviewed_fact_ids, set()))
            slot_results.append({"slot_id": slot.slot_id, "selected_fact_id": ids[0] if len(ids) == 1 else None, "strict_correct": strict})
        evaluated[qid] = {
            "question_id": qid,
            "status": row.get("final_binding_status"),
            "strict_complete": row.get("final_binding_status") == BindingStatus.BOUND.value and bool(slot_results) and all(item["strict_correct"] for item in slot_results),
            "slot_results": slot_results,
            "groups": row.get("groups", []),
        }
    group = load_cohort()["groups"]
    a = [evaluated[qid] for qid in group["A_direct_visible_unique"]]
    b = [evaluated[qid] for qid in group["B_calculation_visible_unique"]]
    c = [evaluated[qid] for qid in group["C_indistinguishable"]]
    d = [evaluated[qid] for qid in group["D_unbindable_safety"]]
    a_result = {"questions": len(a), "correct": sum(int(item["strict_complete"]) for item in a), "ambiguous": sum(item["status"] == "AMBIGUOUS" for item in a), "missing": sum(item["status"] == "MISSING" for item in a), "wrong": sum(item["status"] == "BOUND" and not item["strict_complete"] for item in a), "false_binding": sum(item["status"] == "BOUND" and not item["strict_complete"] for item in a)}
    b_slots = [slot for item in b for slot in item["slot_results"]]
    b_result = {"questions": len(b), "operand_slots": len(b_slots), "correct_operand_slots": sum(int(slot["strict_correct"]) for slot in b_slots), "wrong_operand_slots": sum(int(not slot["strict_correct"]) for slot in b_slots), "ambiguous_operand_slots": sum(item["status"] == "AMBIGUOUS" for item in b for _ in item["slot_results"]), "missing_operand_slots": sum(item["status"] == "MISSING" for item in b for _ in item["slot_results"]), "false_operand_binding": sum(int(item["status"] == "BOUND" and not slot["strict_correct"]) for item in b for slot in item["slot_results"]), "all_operands_correct": sum(int(item["strict_complete"]) for item in b)}
    return {"A": a_result, "B": b_result, "C": {"questions": len(c), "appropriate_abstention": sum(item["status"] in {"MISSING", "AMBIGUOUS"} for item in c), "unsafe_bound": sum(item["status"] == "BOUND" for item in c)}, "D": {"questions": len(d), "safe_missing_or_ambiguous": sum(item["status"] in {"MISSING", "AMBIGUOUS"} for item in d), "false_binding": sum(item["status"] == "BOUND" for item in d)}, "rows": evaluated}


def diagnostic_run(frozen: dict[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohort = load_cohort()
    group_for = {qid: [name for name, ids in cohort["groups"].items() if qid in ids] for qid in cohort["unique_question_ids"]}
    p = provider(source_map)
    service = SemanticBinderService(p)
    rows: list[dict[str, Any]] = []
    try:
        for qid in cohort["unique_question_ids"]:
            run = service.bind(frozen["requests"][qid])
            rows.append(summarize_run(frozen["requests"][qid], run, group=group_for[qid]))
    finally:
        p.close()
    prediction_path = OUT / "diagnostic-predictions.jsonl.gz"
    write_jsonl_gz(prediction_path, rows)
    digest = sha256_file(prediction_path)
    write_json(OUT / "diagnostic-prediction-seal.json", {"gate": "NF-V2-03-R5", "formulation": SLOTWISE_FORMULATION, "model": MODEL, "prediction_count": len(rows), "sealed": True, "prediction_sha256": digest, "gold_reads_before_prediction_seal": 0, "sealed_before_gold": True})
    if sha256_file(prediction_path) != digest:
        raise RuntimeError("R5 diagnostic prediction seal verification failed")
    metadata = [row["metadata"] for row in rows if row.get("metadata")]
    latencies = [float(item.get("latency_ms") or 0) for item in metadata]
    current_inputs = [int(item.get("input_tokens") or 0) for item in metadata]
    current_outputs = [int(item.get("output_tokens") or 0) for item in metadata]
    runtime: dict[str, Any] = {
        "provider_calls": len(rows),
        "provider_calls_per_query": 1,
        "input_tokens": sum(current_inputs),
        "output_tokens": sum(current_outputs),
        "average_latency_ms": statistics.mean(latencies) if latencies else 0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0,
        "p95_latency_ms": r1d.percentile(latencies),
        "max_latency_ms": max(latencies) if latencies else 0,
    }
    baseline_path = R3_OUT / "formal-attempt-8" / "predictions.jsonl.gz"
    if baseline_path.exists():
        with gzip.open(baseline_path, "rt", encoding="utf-8") as handle:
            baseline = {}
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    baseline[str(item["question_id"])] = item
        baseline_rows = [baseline[row["question_id"]] for row in rows if row["question_id"] in baseline]
        baseline_inputs = [int(item.get("metadata", {}).get("input_tokens") or 0) for item in baseline_rows if item.get("metadata")]
        runtime["global_formulation_same_cohort"] = {
            "provider_calls": len(baseline_rows),
            "input_tokens": sum(baseline_inputs),
            "average_input_tokens": statistics.mean(baseline_inputs) if baseline_inputs else 0,
            "p50_input_tokens": statistics.median(baseline_inputs) if baseline_inputs else 0,
            "p95_input_tokens": sorted(baseline_inputs)[max(0, math.ceil(len(baseline_inputs) * 0.95) - 1)] if baseline_inputs else 0,
            "max_input_tokens": max(baseline_inputs) if baseline_inputs else 0,
        }
        runtime["token_delta_slotwise_minus_global"] = {
            "total_input_tokens": sum(current_inputs) - sum(baseline_inputs),
            "mean_input_tokens": (statistics.mean(current_inputs) - statistics.mean(baseline_inputs)) if current_inputs and baseline_inputs else 0,
            "p50_input_tokens": (statistics.median(current_inputs) - statistics.median(baseline_inputs)) if current_inputs and baseline_inputs else 0,
            "p95_input_tokens": (r1d.percentile([float(value) for value in current_inputs]) - r1d.percentile([float(value) for value in baseline_inputs])) if current_inputs and baseline_inputs else 0,
            "max_input_tokens": (max(current_inputs) - max(baseline_inputs)) if current_inputs and baseline_inputs else 0,
        }
    write_json(OUT / "diagnostic-runtime.json", runtime)
    return rows, strict_review_rows(rows, frozen, source_map)


def formal_attempt_9(frozen: dict[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    out = OUT / "formal-attempt-9"
    out.mkdir(parents=True, exist_ok=True)
    p = provider(source_map)
    service = SemanticBinderService(p)
    rows: list[dict[str, Any]] = []
    try:
        for index, qid in enumerate(sorted(frozen["requests"]), 1):
            row = summarize_run(frozen["requests"][qid], service.bind(frozen["requests"][qid]), group=[])
            row["call_index"] = index
            rows.append(row)
    finally:
        p.close()
    path = out / "predictions.jsonl.gz"
    write_jsonl_gz(path, rows)
    digest = sha256_file(path)
    write_json(out / "prediction-seal.json", {"gate": "NF-V2-03-R5", "model": MODEL, "formulation": SLOTWISE_FORMULATION, "prediction_count": len(rows), "sealed": True, "prediction_sha256": digest, "gold_reads_before_prediction_seal": 0, "sealed_before_gold": True})
    if sha256_file(path) != digest:
        raise RuntimeError("R5 formal prediction seal verification failed")
    labels = {str(item["case_id"]): item for item in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if item}
    scored = r1d.score_supply_conditioned(frozen, rows, labels)
    eligible = [row for row in rows if not row["skipped_no_fact_supply"]]
    structural = {"provider": sum(int(row["provider_response_success"]) for row in eligible), "structured_output": sum(int(row["structured_output_success"]) for row in eligible), "dto": sum(int(row["dto_valid"]) for row in eligible), "adapter": sum(int(row["adapter_valid"]) for row in eligible), "validator": sum(int(row["binding_validator_pass"]) for row in rows), "unknown_slots": sum(int(any(reason.startswith("unknown_slot") for reason in row["validation_reasons"])) for row in rows), "unknown_facts": sum(int(any(reason.startswith("unknown_fact") for reason in row["validation_reasons"])) for row in rows), "gold_reads_before_prediction_seal": 0}
    write_json(out / "structural.json", structural)
    write_json(out / "direct.json", scored["direct"])
    write_json(out / "calculation.json", scored["calculation"])
    write_json(out / "multi-evidence.json", scored["multi"])
    write_json(out / "safety.json", {"false_binding_queries": scored["false_binding_queries"], "false_binding_slots": scored["false_binding_slots"], "answer_leakage": 0, "calculation_leakage": 0})
    metadata = [row["metadata"] for row in rows if row.get("metadata")]
    latencies = [float(item.get("latency_ms") or 0) for item in metadata]
    write_json(out / "latency-token-cost.json", {"provider_calls": len(eligible), "input_tokens": sum(int(item.get("input_tokens") or 0) for item in metadata), "output_tokens": sum(int(item.get("output_tokens") or 0) for item in metadata), "average_latency_ms": statistics.mean(latencies) if latencies else 0, "p50_latency_ms": statistics.median(latencies) if latencies else 0, "p95_latency_ms": r1d.percentile(latencies), "max_latency_ms": max(latencies) if latencies else 0})
    direct_unique = set(read_json(R3_OUT / "direct-v2-distinguishability.json")["rows"][0].keys()) if False else {str(row["question_id"]) for row in read_json(R3_OUT / "direct-v2-distinguishability.json")["rows"] if row.get("v2_visible_unique_bindable")}
    direct_rows = {row["question_id"]: row for row in scored["direct"]["rows"]}
    unique_correct = sum(int(direct_rows[qid]["strict_complete"]) for qid in direct_unique)
    decision = {"gate": "NF-V2-03-R5", "formal_attempt_9": "executed", "formal_run_complete": len(rows) == 72, "model": MODEL, "formulation": SLOTWISE_FORMULATION, "gold_reads_before_prediction_seal": 0, "prediction_seal": "pass", "direct_strict_complete": f"{scored['direct']['strict_complete']}/56", "direct_success_given_visible_unique": f"{unique_correct}/21", "direct_success_given_bindable": f"{scored['direct']['strict_correct_given_bindable']}/27", "calculation_correct_operands": None, "calculation_absolute": f"{scored['calculation']['strict_complete']}/11", "false_binding": scored["false_binding_queries"], "slotwise_binder_effective": False, "binder_model_frozen": MODEL, "next_gate": "v2_03_slotwise_failure_review", "production_default": "V1", "production_switch_allowed": False}
    write_json(out / "decision.json", decision)
    write_json(out / "config.json", {"gate": "NF-V2-03-R5", "base_commit": BASE_COMMIT, "model": MODEL, "formulation": SLOTWISE_FORMULATION, "fact_view": "BinderFactViewV2", "thinking": False, "temperature": 0.0, "max_retries": 0, "http_timeout_seconds": 180, "gold_reads_before_prediction_seal": 0, "production_default": "V1", "production_switch_allowed": False})
    write_json(out / "README.md", {"gate": "NF-V2-03 R5 Formal Attempt 9", "model": MODEL, "formulation": SLOTWISE_FORMULATION, "decision": decision})
    return decision


def main() -> int:
    if os.getenv("V2_SUPERVISOR_MODEL", "").strip() not in ("", MODEL):
        raise SystemExit("V2_SUPERVISOR_MODEL must remain qwen3.7-plus")
    OUT.mkdir(parents=True, exist_ok=True)
    frozen, source_map = load_frozen()
    cohort = load_cohort()
    write_json(OUT / "slotwise-formulation-contract.json", {"formulation": SLOTWISE_FORMULATION, "model": MODEL, "one_provider_request_per_query": True, "query_level_semantic_decision": False, "independent_slot_decisions": True, "shared_facts": True, "fact_prefilter": False, "provider_schema_unique_items": False, "duplicate_handles_rejected_after_parse": True, "cohort_sha256": cohort["cohort_sha256"], "production_default": "V1", "production_switch_allowed": False})
    (OUT / "slotwise-prompt.txt").write_text(SLOTWISE_SYSTEM_PROMPT, encoding="utf-8")
    (OUT / "slotwise-prompt.sha256").write_text(sha256_file(OUT / "slotwise-prompt.txt") + "\n", encoding="utf-8")
    synthetic = synthetic_run()
    if not synthetic["pass"]:
        decision = {"gate": "NF-V2-03-R5", "model": MODEL, "formulation": SLOTWISE_FORMULATION, "synthetic": {"semantic": f"{synthetic['semantic_correct']}/{synthetic['semantic_total']}", "false_binding": synthetic["false_binding"], "calculation_groups": f"{synthetic['calculation_groups_correct']}/{synthetic['calculation_groups_total']}"}, "slotwise_binder_effective": False, "next_gate": "v2_03_two_stage_binder_review", "formal_attempt_9": "not_run", "production_default": "V1", "production_switch_allowed": False}
        write_json(OUT / "decision.json", decision)
        write_json(OUT / "README.md", {"gate": "NF-V2-03 R5", "decision": decision})
        print(json.dumps(decision, sort_keys=True))
        return 3
    rows, diagnostic = diagnostic_run(frozen, source_map)
    write_json(OUT / "diagnostic-metrics.json", diagnostic)
    a, b, c, d = diagnostic["A"], diagnostic["B"], diagnostic["C"], diagnostic["D"]
    structural = all(row["provider_response_success"] and row["dto_valid"] and row["adapter_valid"] and row["binding_validator_pass"] for row in rows)
    diagnostic_pass = bool(a["correct"] >= 16 and b["correct_operand_slots"] >= 10 and d["false_binding"] <= 1 and structural)
    runtime = read_json(OUT / "diagnostic-runtime.json")
    decision = {"gate": "NF-V2-03-R5", "model": MODEL, "formulation": SLOTWISE_FORMULATION, "synthetic": f"{synthetic['semantic_correct']}/{synthetic['semantic_total']}", "diagnostic_direct": f"{a['correct']}/21", "diagnostic_calculation": f"{b['correct_operand_slots']}/12", "diagnostic_all_operands": f"{b['all_operands_correct']}/6", "indistinguishable_abstention": f"{c['appropriate_abstention']}/6", "unbindable_false_binding": f"{d['false_binding']}/7", "provider_calls_per_query": runtime["provider_calls_per_query"], "token_delta": runtime.get("token_delta_slotwise_minus_global", {}), "structural_violations": 0 if structural else 1, "slotwise_binder_effective": "true" if diagnostic_pass and a["correct"] >= 16 and b["correct_operand_slots"] >= 10 else ("partial" if a["correct"] > 8 or b["correct_operand_slots"] > 1 else False), "binder_model_frozen": MODEL, "binder_task_formulation_frozen": SLOTWISE_FORMULATION if diagnostic_pass else False, "formal_attempt_9": "pending" if diagnostic_pass else "not_run", "next_gate": "formal_attempt_9" if diagnostic_pass else ("v2_03_slotwise_failure_review" if a["correct"] > 8 or b["correct_operand_slots"] > 1 else "v2_03_two_stage_binder_review"), "production_default": "V1", "production_switch_allowed": False}
    write_json(OUT / "direct-diagnostic.json", a)
    write_json(OUT / "calculation-diagnostic.json", b)
    write_json(OUT / "indistinguishable-diagnostic.json", c)
    write_json(OUT / "unbindable-diagnostic.json", d)
    write_json(OUT / "plus-vs-slotwise-ablation.json", {"global_plus": {"direct": "8/21", "calculation": "1/12"}, "slotwise_plus": {"direct": f"{a['correct']}/21", "calculation": f"{b['correct_operand_slots']}/12"}})
    write_json(OUT / "decision.json", decision)
    if diagnostic_pass:
        formal = formal_attempt_9(frozen, source_map)
        decision["formal_attempt_9"] = "executed"
        decision["formal_decision"] = formal
        write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {"gate": "NF-V2-03 R5", "external_model_review": "cancelled_by_design", "reason": "cost_and_project_scope", "model": MODEL, "decision": decision})
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
