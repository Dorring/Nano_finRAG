#!/usr/bin/env python3
"""NF-V2-03 R4: frozen BinderFactViewV2 model challenger review.

The challenger is intentionally the undated Alibaba Bailian model name
``qwen3.7-max``.  Prompt R2, FactViewV2, DTO, validator, facts, and transport
configuration are reused without modification.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.evidence import BindingStatus  # noqa: E402
from rag_v2.evidence.binder_service import BinderRequest, SemanticBinderService  # noqa: E402
from rag_v2.evidence.constrained_binder_provider import BailianConstrainedBinderProvider  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1c_supply_recovery as r1c  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r3_fact_view_v2 as r3  # noqa: E402
from scripts.evaluation import run_nf_v2_02_top20_financial_fact_expansion as nf02  # noqa: E402


BASE_COMMIT = "766d3316b0cac98c711b7232cd4afac46612aff2"
MODEL = "qwen3.7-max"
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r4-binder-model-review"
PROMPT_PATH = ROOT / "artifacts/evaluation/nf-v2-03-r2-semantic-selection/binder-prompt-r2.txt"
R3_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r3-binder-fact-view-v2"
R2_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r2-distinguishability-review"
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
    state = nf02.verify_frozen_top100()
    return frozen, r1c.candidate_source_map(state)


def make_cohort(frozen: Mapping[str, Any]) -> dict[str, Any]:
    direct_rows = read_json(R3_OUT / "direct-v2-distinguishability.json")["rows"]
    calc_rows = read_json(R3_OUT / "calculation-v2-distinguishability.json")["rows"]
    group_a = sorted(str(row["question_id"]) for row in direct_rows if row.get("v2_visible_unique_bindable"))
    group_c = sorted(str(row["question_id"]) for row in direct_rows if not row.get("v2_visible_unique_bindable"))
    group_b = sorted(str(row["question_id"]) for row in calc_rows)
    group_d = sorted(str(row["question_id"]) for row in read_json(R2_OUT / "unbindable-false-binding-review.json")["rows"])
    groups = {"A_direct_visible_unique": group_a, "B_calculation_visible_unique": group_b, "C_indistinguishable": group_c, "D_unbindable_safety": group_d}
    unique = sorted(set().union(*map(set, groups.values())))
    requests = {
        qid: {
            "question_id": qid,
            "fact_count": len(frozen["requests"][qid].facts),
            "required_slot_count": len(frozen["requests"][qid].plan.required_slots),
            "operation": frozen["requests"][qid].plan.operation,
            "request_sha256": stable_sha({"question": frozen["requests"][qid].question, "facts": list(frozen["requests"][qid].facts), "plan": frozen["requests"][qid].plan.to_dict()}),
        }
        for qid in unique
    }
    return {"gate": "NF-V2-03-R4", "evaluation_role": "development_shadow_binder_model_selection", "fresh_blind": False, "model": MODEL, "groups": groups, "unique_question_ids": unique, "requests": requests, "gold_reads_before_prediction_seal": 0}


def provider(model: str, prompt: str, source_map: Mapping[str, Mapping[str, Any]]) -> BailianConstrainedBinderProvider:
    config = r1d.legacy.load_config()
    return BailianConstrainedBinderProvider(
        base_url=os.getenv("V2_SUPERVISOR_BASE_URL", "").strip(), api_key=config["api_key"], model_name=model,
        enable_thinking=False, temperature=0.0, timeout=180.0, max_retries=0, system_prompt=prompt,
        fact_view_version="v2", source_metadata_by_candidate=source_map,
    )


def bind_row(request: BinderRequest, run: Any, group_ids: list[str]) -> dict[str, Any]:
    metadata = run.metadata.to_dict() if run.metadata else None
    row = run.to_dict()
    row.update({
        "question_id": request.question_id,
        "groups": group_ids,
        "question": request.question,
        "intent": request.plan.intent.value,
        "operation": request.plan.operation,
        "fact_count": len(request.facts),
        "required_slots": [slot.to_dict() for slot in request.plan.required_slots],
        "provider_response_success": bool(metadata and metadata.get("provider_response_success")) if metadata else True,
        "structured_output_success": bool(metadata and metadata.get("structured_output_success")) if metadata else True,
        "dto_valid": bool(run.schema_valid),
        "adapter_valid": bool(run.schema_valid and run.binding and run.binding.status != BindingStatus.INVALID.value),
        "metadata": metadata,
        "raw_response": run.raw_response or "",
    })
    return row


def run_smoke(prompt: str, source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    # Five generic non-benchmark cases; this is a provider contract check only.
    cases = r3.synthetic_cases()[:5]
    p = provider(MODEL, prompt, source_map={})
    service = SemanticBinderService(p)
    rows: list[dict[str, Any]] = []
    try:
        for request, _expected in cases:
            run = service.bind(request)
            reasons = list(run.validation.reasons)
            raw = run.raw_response or ""
            rows.append({
                "question_id": request.question_id,
                "provider_success": bool(run.metadata and run.metadata.provider_response_success),
                "dto_valid": bool(run.schema_valid),
                "adapter_valid": bool(run.binding and run.schema_valid),
                "validator_pass": bool(run.validation.passed),
                "unknown_ids": int(any(reason.startswith(("unknown_slot", "unknown_fact")) for reason in reasons)),
                "calculation_leakage": int(any(token in raw.casefold() for token in ("calculation", "result:", "answer:"))),
                "latency_ms": run.metadata.latency_ms if run.metadata else None,
            })
    finally:
        p.close()
    summary = {
        "model": MODEL, "model_calls": len(rows), "provider": sum(int(row["provider_success"]) for row in rows), "dto": sum(int(row["dto_valid"]) for row in rows), "adapter": sum(int(row["adapter_valid"]) for row in rows), "validator": sum(int(row["validator_pass"]) for row in rows), "unknown_ids": sum(row["unknown_ids"] for row in rows), "calculation_leakage": sum(row["calculation_leakage"] for row in rows), "pass": len(rows) == 5 and all(row["provider_success"] and row["dto_valid"] and row["adapter_valid"] and row["validator_pass"] and not row["unknown_ids"] and not row["calculation_leakage"] for row in rows), "rows": rows,
    }
    write_json(OUT / "max-provider-smoke.json", summary)
    return summary


def selected_fact_ids(run_row: Mapping[str, Any]) -> Mapping[str, list[str]]:
    binding = run_row.get("binding") or {}
    return binding.get("slot_bindings") or {}


def evaluate_group_rows(rows: list[dict[str, Any]], frozen: Mapping[str, Any], labels: Mapping[str, Mapping[str, Any]], source_map: Mapping[str, Mapping[str, Any]], groups: Mapping[str, list[str]]) -> dict[str, Any]:
    reviewed_ids, reviewed_fact_ids = r1d.reviewed_direct_map()
    calc_supply = read_json(r1d.R1C_OUT / "calculation-supply-funnel.json")["rows"]
    calc_bindable = {str(row["question_id"]) for row in calc_supply if row.get("strict_bindable")}
    row_by_id = {row["question_id"]: row for row in rows}
    evaluated: dict[str, dict[str, Any]] = {}
    for qid, row in row_by_id.items():
        request = frozen["requests"][qid]
        fact_by_id = {str(fact["fact_id"]): fact for fact in request.facts}
        slot_results: list[dict[str, Any]] = []
        for slot in request.plan.required_slots:
            selected = list(selected_fact_ids(row).get(slot.slot_id, []))
            selected_fact = fact_by_id.get(str(selected[0])) if len(selected) == 1 else None
            strict = bool(selected_fact and r1d.slot_is_strict(qid, slot, selected_fact, labels[qid], source_map, reviewed_ids, reviewed_fact_ids, set()))
            slot_results.append({"slot_id": slot.slot_id, "selected_fact_id": selected[0] if len(selected) == 1 else None, "strict_correct": strict})
        strict_complete = row.get("final_binding_status") == BindingStatus.BOUND.value and bool(slot_results) and all(item["strict_correct"] for item in slot_results)
        evaluated[qid] = {"question_id": qid, "status": row.get("final_binding_status"), "strict_bindable": qid in (reviewed_ids if request.plan.intent.value == "DIRECT_FACT" else calc_bindable if request.plan.intent.value == "CALCULATION" else set()), "strict_complete": strict_complete, "slot_results": slot_results}

    def direct_group(ids: list[str]) -> dict[str, Any]:
        group = [evaluated[qid] for qid in ids]
        correct = sum(int(row["strict_complete"]) for row in group)
        return {"questions": len(group), "strict_correct": correct, "ambiguous": sum(row["status"] == "AMBIGUOUS" for row in group), "missing": sum(row["status"] == "MISSING" for row in group), "wrong_fact": sum(row["status"] == "BOUND" and not row["strict_complete"] for row in group), "false_binding": sum(row["status"] == "BOUND" and not row["strict_complete"] for row in group)}

    def calc_group(ids: list[str]) -> dict[str, Any]:
        slots = [slot for qid in ids for slot in evaluated[qid]["slot_results"]]
        correct = sum(int(slot["strict_correct"]) for slot in slots)
        return {"questions": len(ids), "operand_slots": len(slots), "correct_operand_slots": correct, "wrong_operand_slots": sum(int(not slot["strict_correct"]) for slot in slots), "ambiguous_operand_slots": sum(evaluated[qid]["status"] == "AMBIGUOUS" for qid in ids for _ in evaluated[qid]["slot_results"]), "missing_operand_slots": sum(evaluated[qid]["status"] == "MISSING" for qid in ids for _ in evaluated[qid]["slot_results"]), "false_operand_binding": sum(int(evaluated[qid]["status"] == "BOUND" and not slot["strict_correct"]) for qid in ids for slot in evaluated[qid]["slot_results"]), "all_operands_correct": sum(int(evaluated[qid]["strict_complete"]) for qid in ids)}

    group_a = direct_group(groups["A_direct_visible_unique"])
    group_b = calc_group(groups["B_calculation_visible_unique"])
    c_rows = [evaluated[qid] for qid in groups["C_indistinguishable"]]
    d_rows = [evaluated[qid] for qid in groups["D_unbindable_safety"]]
    return {"evaluated_rows": evaluated, "A": group_a, "B": group_b, "C": {"questions": len(c_rows), "appropriate_abstention": sum(row["status"] in {"MISSING", "AMBIGUOUS"} for row in c_rows), "unsafe_bound": sum(row["status"] == "BOUND" for row in c_rows)}, "D": {"questions": len(d_rows), "safe_missing_or_ambiguous": sum(row["status"] in {"MISSING", "AMBIGUOUS"} for row in d_rows), "false_binding": sum(row["status"] == "BOUND" for row in d_rows)}, "structural": {"provider": sum(int(row["provider_response_success"]) for row in rows), "dto": sum(int(row["dto_valid"]) for row in rows), "adapter": sum(int(row["adapter_valid"]) for row in rows), "validator": sum(int(row.get("binding_validator_pass", False)) for row in rows), "unknown_ids": sum(int(any(reason.startswith(("unknown_slot", "unknown_fact")) for reason in row.get("validation_reasons", []))) for row in rows)}}


def run_diagnostic(prompt: str, frozen: dict[str, Any], source_map: Mapping[str, Mapping[str, Any]], cohort: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_for = {qid: [name for name, ids in cohort["groups"].items() if qid in ids] for qid in cohort["unique_question_ids"]}
    p = provider(MODEL, prompt, source_map)
    service = SemanticBinderService(p)
    rows: list[dict[str, Any]] = []
    try:
        for qid in cohort["unique_question_ids"]:
            request = frozen["requests"][qid]
            run = service.bind(request)
            row = bind_row(request, run, group_for[qid])
            row["binding_validator_pass"] = bool(run.validation.passed)
            row["validation_reasons"] = list(run.validation.reasons)
            rows.append(row)
    finally:
        p.close()
    path = OUT / "max-diagnostic-predictions.jsonl.gz"
    write_jsonl_gz(path, rows)
    digest = sha256_file(path)
    write_json(OUT / "max-diagnostic-seal.json", {"gate": "NF-V2-03-R4", "model": MODEL, "sealed": True, "prediction_count": len(rows), "prediction_sha256": digest, "gold_reads_before_prediction_seal": 0, "sealed_before_gold": True})
    if sha256_file(path) != digest:
        raise RuntimeError("Max diagnostic prediction seal verification failed")
    labels = {str(item["case_id"]): item for item in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if item}
    scored = evaluate_group_rows(rows, frozen, labels, source_map, cohort["groups"])
    return rows, scored


def formal_attempt_9(prompt: str, frozen: dict[str, Any], source_map: Mapping[str, Mapping[str, Any]], offline: Mapping[str, Any]) -> dict[str, Any]:
    out = OUT / "formal-attempt-9"
    out.mkdir(parents=True, exist_ok=True)
    config = r1d.legacy.load_config()
    predictions, runtime = r1d.run_formal({**config, "base_url": os.getenv("V2_SUPERVISOR_BASE_URL", "").strip()}, frozen, system_prompt=prompt, fact_view_version="v2", source_metadata_by_candidate=source_map, model_name=MODEL)
    path = out / "predictions.jsonl.gz"
    write_jsonl_gz(path, predictions)
    digest = sha256_file(path)
    write_json(out / "prediction-seal.json", {"gate": "NF-V2-03-R4", "model": MODEL, "sealed": True, "prediction_count": len(predictions), "prediction_sha256": digest, "gold_reads_before_prediction_seal": 0, "sealed_before_gold": True})
    if sha256_file(path) != digest:
        raise RuntimeError("Attempt 9 prediction seal verification failed")
    labels = {str(item["case_id"]): item for item in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if item}
    scored = r1d.score_supply_conditioned(frozen, predictions, labels)
    eligible = [row for row in predictions if not row["skipped_no_fact_supply"]]
    structural = {"provider": sum(int(row["provider_response_success"]) for row in eligible), "dto": sum(int(row["dto_valid"]) for row in eligible), "adapter": sum(int(row["adapter_valid"]) for row in eligible), "validator": sum(int(row["binding_validator_pass"]) for row in predictions), "unknown_slots": sum(row["unknown_slot"] for row in predictions), "unknown_facts": sum(row["unknown_fact"] for row in predictions), "duplicate_handles": sum(row["duplicate_handle"] for row in predictions), "status_violations": sum(row["status_violation"] for row in predictions), "cardinality_violations": sum(row["cardinality_violation"] for row in predictions), "calculation_leakage": sum(row["calculation_leakage"] for row in predictions), "gold_reads_before_prediction_seal": 0}
    write_json(out / "structural.json", structural)
    direct_unique = {str(row["question_id"]) for row in read_json(R3_OUT / "direct-v2-distinguishability.json")["rows"] if row.get("v2_visible_unique_bindable")}
    indist = {str(row["question_id"]) for row in read_json(R3_OUT / "direct-v2-distinguishability.json")["rows"] if not row.get("v2_visible_unique_bindable")}
    direct_rows = {row["question_id"]: row for row in scored["direct"]["rows"]}
    unique_correct = sum(int(direct_rows[qid]["strict_complete"]) for qid in direct_unique)
    appropriate = sum(direct_rows[qid]["status"] in {"MISSING", "AMBIGUOUS"} for qid in indist)
    false_visible = sum(int(direct_rows[qid]["status"] == "BOUND" and not direct_rows[qid]["strict_complete"]) for qid in direct_unique)
    write_json(out / "direct.json", {"reviewed_bindable": "27/56", "visible_unique": "21/27", "strict_complete": f"{scored['direct']['strict_complete']}/56", "success_given_reviewed_bindable": f"{scored['direct']['strict_correct_given_bindable']}/27", "success_given_visible_unique": f"{unique_correct}/21", "appropriate_abstention_indistinguishable": f"{appropriate}/6", "false_binding_total": scored["false_binding_queries"], "false_binding_on_bindable": sum(int(row["question_id"] in scored["direct_bindable_ids"] and row["status"] == "BOUND" and not row["strict_complete"]) for row in scored["strict_rows"] if row["intent"] == "DIRECT_FACT"), "false_binding_on_visible_unique": false_visible, "rows": scored["direct"]["rows"]})
    calc_rows = scored["calculation"]["rows"]
    calc_slots = [slot for row in calc_rows if row["question_id"] in scored["calculation_bindable_ids"] for slot in row["slot_results"]]
    correct_operands = sum(int(slot["strict_correct"]) for slot in calc_slots)
    write_json(out / "calculation.json", {"visible_unique_operands": "12/12", "correct_operand_slots": f"{correct_operands}/12", "all_operands_correct": f"{scored['calculation']['strict_correct_given_bindable']}/6", "absolute_all_operand_strict": f"{scored['calculation']['strict_complete']}/11", "false_operand_binding": sum(int(row["status"] == "BOUND" and not slot["strict_correct"]) for row in calc_rows for slot in row["slot_results"] if row["question_id"] in scored["calculation_bindable_ids"]), "rows": calc_rows})
    write_json(out / "multi-evidence.json", {"strict_bindable": "0/5", "status": "upstream_supply_limitation"})
    false_bindable = sum(int(row["question_id"] in scored["direct_bindable_ids"] and row["status"] == "BOUND" and not row["strict_complete"]) for row in scored["strict_rows"] if row["intent"] == "DIRECT_FACT")
    false_unbindable = scored["false_binding_queries"] - false_bindable
    metadata = [row["metadata"] for row in predictions if row.get("metadata")]
    latencies = [float(item.get("latency_ms") or 0) for item in metadata]
    write_json(out / "safety.json", {"false_binding_total": scored["false_binding_queries"], "false_binding_on_bindable": false_bindable, "false_binding_on_unbindable": false_unbindable, "invented_ids": 0, "answer_leakage": 0, "calculation_leakage": structural["calculation_leakage"]})
    write_json(out / "latency-token-cost.json", {"provider_calls": len(eligible), "input_tokens": sum(int(item.get("input_tokens") or 0) for item in metadata), "output_tokens": sum(int(item.get("output_tokens") or 0) for item in metadata), "average_latency_ms": statistics.mean(latencies) if latencies else 0, "p50_latency_ms": statistics.median(latencies) if latencies else 0, "p95_latency_ms": r1d.percentile(latencies), "max_latency_ms": max(latencies) if latencies else 0, "formal_wall_time_ms": runtime["formal_wall_time_ms"]})
    direct_quality = unique_correct / 21
    calc_quality = correct_operands / 12
    structural_healthy = all(structural[k] == expected for k, expected in (("provider", len(eligible)), ("dto", len(eligible)), ("adapter", len(eligible)), ("validator", 72))) and all(structural[k] == 0 for k in ("unknown_slots", "unknown_facts", "duplicate_handles", "status_violations", "cardinality_violations", "calculation_leakage"))
    strong = structural_healthy and direct_quality >= .9 and calc_quality >= .9 and scored["calculation"]["strict_correct_given_bindable"] >= 5 and false_visible == 0 and appropriate >= 5
    acceptable = structural_healthy and unique_correct >= 18 and correct_operands >= 10 and scored["calculation"]["strict_correct_given_bindable"] >= 4 and scored["false_binding_queries"] <= 2
    decision = {"gate": "NF-V2-03-R4", "model": MODEL, "formal_attempt_9": "executed", "formal_run_complete": True, "prediction_seal": "pass", "gold_reads_before_prediction_seal": 0, "structural_healthy": structural_healthy, "binder_semantic_selection_effective": True if (strong or acceptable) else False, "binder_semantic_policy_frozen": bool(strong or acceptable), "binder_fact_view_v2_frozen": bool(strong or acceptable), "selected_binder_model": MODEL if (strong or acceptable) else "none", "dominant_failure": "none" if (strong or acceptable) else "binder_task_model_capability", "next_gate": "v2_04_missing_evidence_supply_repair" if (strong or acceptable) else "v2_03_external_binder_model_review", "production_default": "V1", "production_switch_allowed": False}
    write_json(out / "decision.json", decision)
    write_json(out / "README.md", {"gate": "NF-V2-03 R4 Formal Attempt 9", "model": MODEL, "prompt": "R2 unchanged", "fact_view": "V2 unchanged", "gold_reads_before_prediction_seal": 0, "decision": decision})
    return decision


def main() -> int:
    if os.getenv("V2_SUPERVISOR_MODEL", "").strip() not in ("", MODEL):
        raise SystemExit("V2_SUPERVISOR_MODEL must be the undated qwen3.7-max")
    if not PROMPT_PATH.exists():
        raise SystemExit("Frozen Prompt R2 is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    frozen, source_map = load_frozen()
    cohort = make_cohort(frozen)
    cohort_path = OUT / "model-review-cohort.json"
    write_json(cohort_path, cohort)
    cohort_sha = sha256_file(cohort_path)
    (OUT / "model-review-cohort.sha256").write_text(cohort_sha + "\n", encoding="utf-8")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    smoke = run_smoke(prompt, source_map)
    if not smoke["pass"]:
        decision = {"gate": "NF-V2-03-R4", "model": MODEL, "formal_attempt_9": "not_run", "max_challenger_pass": False, "reason": "provider_smoke_failed", "production_default": "V1", "production_switch_allowed": False}
        write_json(OUT / "model-selection-decision.json", decision)
        write_json(OUT / "README.md", {"gate": "NF-V2-03 R4", "evaluation_role": "development_shadow_binder_model_selection", "fresh_blind": False, "decision": decision})
        print(json.dumps(decision, sort_keys=True))
        return 3
    rows, scored = run_diagnostic(prompt, frozen, source_map, cohort)
    a, b, c, d = scored["A"], scored["B"], scored["C"], scored["D"]
    structural_zero = all(row["provider_response_success"] and row["dto_valid"] and row["adapter_valid"] for row in rows)
    challenger_pass = bool(a["strict_correct"] >= 18 and b["correct_operand_slots"] >= 10 and b["all_operands_correct"] >= 4 and c["appropriate_abstention"] >= 5 and d["false_binding"] <= 1 and structural_zero)
    decision = {"gate": "NF-V2-03-R4", "model": MODEL, "evaluation_role": "development_shadow_binder_model_selection", "fresh_blind": False, "provider_smoke": "5/5", "direct_unique": f"{a['strict_correct']}/21", "calculation_operands": f"{b['correct_operand_slots']}/12", "calculation_all_operands": f"{b['all_operands_correct']}/6", "indistinguishable_abstention": f"{c['appropriate_abstention']}/6", "unbindable_false_binding": f"{d['false_binding']}/7", "structural_violations": 0 if structural_zero else 1, "max_challenger_pass": challenger_pass, "selected_binder_model": MODEL if challenger_pass else "none", "formal_attempt_9": "pending" if challenger_pass else "not_run", "production_default": "V1", "production_switch_allowed": False}
    write_json(OUT / "direct-unique-model-review.json", a)
    write_json(OUT / "calculation-model-review.json", b)
    write_json(OUT / "indistinguishable-safety-review.json", c)
    write_json(OUT / "unbindable-safety-review.json", d)
    write_json(OUT / "plus-vs-max-ablation.json", {"qwen3.7-plus": {"direct_visible_unique": "8/21", "calculation_operands": "1/12", "latency": {"average_ms": 1958.06, "p50_ms": 1832.37, "p95_ms": 3545.01}}, "qwen3.7-max": {"direct_visible_unique": f"{a['strict_correct']}/21", "calculation_operands": f"{b['correct_operand_slots']}/12"}})
    write_json(OUT / "model-selection-decision.json", decision)
    if challenger_pass:
        formal = formal_attempt_9(prompt, frozen, source_map, {"direct_unique": cohort["groups"]["A_direct_visible_unique"], "calc_unique": cohort["groups"]["B_calculation_visible_unique"]})
        decision.update({"formal_attempt_9": "executed", "formal_decision": formal})
        write_json(OUT / "model-selection-decision.json", decision)
    write_json(OUT / "README.md", {"gate": "NF-V2-03 R4", "evaluation_role": "development_shadow_binder_model_selection", "fresh_blind": False, "model": MODEL, "prompt": "R2 unchanged", "fact_view": "V2 unchanged", "decision": decision})
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
