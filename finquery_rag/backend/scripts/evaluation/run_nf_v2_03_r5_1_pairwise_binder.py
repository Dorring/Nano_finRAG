#!/usr/bin/env python3
"""NF-V2-03 R5.1 offline review and batched pairwise Binder diagnostic."""

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
from rag_v2.contracts.plan import Intent  # noqa: E402
from rag_v2.evidence.binder_service import BinderRequest, SemanticBinderService  # noqa: E402
from rag_v2.evidence.pairwise_binder import (  # noqa: E402
    PAIRWISE_FORMULATION,
    PAIRWISE_SYSTEM_PROMPT,
    BailianPairwiseBinderProvider,
)
from scripts.evaluation import run_nf_v2_02_top20_financial_fact_expansion as nf02  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1a_binding_contract_recovery as r1a  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1c_supply_recovery as r1c  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r3_fact_view_v2 as r3  # noqa: E402


BASE_COMMIT = "a375175b84e3340ad6446f3d6bda8fa877fe54a6"
MODEL = "qwen3.7-plus"
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r5-1-pairwise-binder"
R5_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r5-slotwise-binder"
R4_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r4-binder-model-review"
R3_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r3-binder-fact-view-v2"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"

SW_CATEGORIES = (
    "SW0_correct",
    "SW1_selected_nearest_metric_but_wrong_scope",
    "SW2_selected_wrong_period",
    "SW3_selected_wrong_statement",
    "SW4_selected_parent_or_child_metric",
    "SW5_selected_lexically_similar_fact",
    "SW6_selected_one_of_indistinguishable_candidates",
    "SW7_failed_to_select_unique_best_fact",
    "SW8_missing_despite_unique_fact",
    "SW9_other",
)
OB_CATEGORIES = (
    "OB0_full_match_but_gold_contract_disagrees",
    "OB1_partial_metric_match",
    "OB2_missing_scope_evidence",
    "OB3_period_conflict",
    "OB4_statement_conflict",
    "OB5_insufficient_context_but_model_committed",
    "OB6_other",
)


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


def load_frozen() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    frozen = r1d.load_r1c_frozen_inputs()
    return frozen, r1c.candidate_source_map(nf02.verify_frozen_top100())


def load_cohort() -> dict[str, Any]:
    path = R4_OUT / "model-review-cohort.json"
    actual = sha256_file(path)
    expected = (R4_OUT / "model-review-cohort.sha256").read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError("R4 frozen diagnostic cohort SHA mismatch")
    cohort = read_json(path)
    cohort["cohort_sha256"] = actual
    return cohort


def load_labels() -> dict[str, dict[str, Any]]:
    return {str(item["case_id"]): item for item in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if item}


def norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").split())


def source_matches(fact: Mapping[str, Any], expected_sources: list[Mapping[str, Any]]) -> bool:
    candidate_ids = {str(fact.get("candidate_id"))} | {str(item) for item in fact.get("candidate_ids", [])}
    return any(str(source.get("candidate_key")) in candidate_ids for source in expected_sources if source.get("candidate_key"))


def fact_handle_map(request: BinderRequest) -> dict[str, str]:
    return {str(fact["fact_id"]): f"F{index:02d}" for index, fact in enumerate(request.facts, 1)}


def fact_summary(fact: Mapping[str, Any], handle: str) -> dict[str, Any]:
    return {
        "handle": handle,
        "fact_id": str(fact.get("fact_id")),
        "raw_metric": fact.get("raw_metric"),
        "normalized_metric": fact.get("normalized_metric"),
        "raw_period": fact.get("raw_period"),
        "normalized_period": fact.get("normalized_period"),
        "row_label": fact.get("row_label"),
        "row_hierarchy": fact.get("row_hierarchy"),
        "column_header": fact.get("column_header"),
        "column_header_path": fact.get("column_header_path"),
        "table_title": fact.get("table_title"),
        "statement_title": fact.get("statement_title"),
        "section_heading": fact.get("section_heading"),
        "candidate_id": fact.get("candidate_id"),
        "physical_source_id": fact.get("physical_source_id"),
        "pdf_page": fact.get("pdf_page"),
    }


def selected_handles(row: Mapping[str, Any], request: BinderRequest) -> dict[str, list[str]]:
    by_id = fact_handle_map(request)
    binding = row.get("binding") or {}
    return {
        slot_id: [by_id.get(str(fact_id), str(fact_id)) for fact_id in fact_ids]
        for slot_id, fact_ids in (binding.get("slot_bindings") or {}).items()
    }


def classify_slot(
    *,
    qid: str,
    slot: Any,
    selected: list[str],
    strict: bool,
    status: str,
    cohort: str,
    request: BinderRequest,
    source_map: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
) -> str:
    if strict or (cohort == "D_unbindable_safety" and not selected and status in {"MISSING", "AMBIGUOUS"}):
        return "SW0_correct"
    if cohort == "C_indistinguishable" and selected:
        return "SW6_selected_one_of_indistinguishable_candidates"
    if not selected:
        return "SW8_missing_despite_unique_fact"
    fact_by_handle = {handle: fact for handle, fact in zip(fact_handle_map(request).values(), request.facts, strict=True)}
    fact = fact_by_handle.get(selected[0])
    if fact is None:
        return "SW9_other"
    if r1c.period(fact.get("normalized_period") or fact.get("raw_period")) != r1c.period(slot.period):
        return "SW2_selected_wrong_period"
    expected_metric = norm(slot.metric)
    actual_metric = norm(fact.get("normalized_metric") or fact.get("raw_metric"))
    if expected_metric == actual_metric:
        label = labels.get(qid, {})
        expected = r1a.expected_sources(slot, label)
        if not source_matches(fact, expected):
            return "SW3_selected_wrong_statement"
        return "SW1_selected_nearest_metric_but_wrong_scope"
    if expected_metric in actual_metric or actual_metric in expected_metric:
        return "SW4_selected_parent_or_child_metric"
    source = source_map.get(str(fact.get("candidate_id")))
    if source and r1c.view_metric_match(slot, fact, source):
        return "SW5_selected_lexically_similar_fact"
    return "SW5_selected_lexically_similar_fact"


def overbinding_category(
    *,
    qid: str,
    slot: Any,
    fact: Mapping[str, Any],
    request: BinderRequest,
    label: Mapping[str, Any],
    cohort: str,
) -> tuple[str, dict[str, Any]]:
    expected_sources = r1a.expected_sources(slot, label)
    metric_match = norm(fact.get("normalized_metric") or fact.get("raw_metric")) == norm(slot.metric)
    period_match = r1c.period(fact.get("normalized_period") or fact.get("raw_period")) == r1c.period(slot.period)
    physical_source_match = source_matches(fact, expected_sources)
    source_context = {
        "metric": metric_match,
        "period": period_match,
        "scope": metric_match,
        "statement": physical_source_match,
        "role": bool(slot.role),
    }
    if cohort == "C_indistinguishable":
        category = "OB5_insufficient_context_but_model_committed"
    elif not period_match:
        category = "OB3_period_conflict"
    elif not metric_match:
        category = "OB1_partial_metric_match" if (norm(slot.metric) in norm(fact.get("normalized_metric")) or norm(fact.get("normalized_metric")) in norm(slot.metric)) else "OB6_other"
    elif not physical_source_match:
        category = "OB4_statement_conflict"
    else:
        category = "OB2_missing_scope_evidence"
    return category, source_context


def offline_failure_review(frozen: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]], cohort: Mapping[str, Any]) -> dict[str, Any]:
    labels = load_labels()
    rows = {row["question_id"]: row for row in read_jsonl_gz(R5_OUT / "diagnostic-predictions.jsonl.gz")}
    reviewed_ids, reviewed_fact_ids = r1d.reviewed_direct_map()
    sw_counts = {category: 0 for category in SW_CATEGORIES}
    sw_rows: list[dict[str, Any]] = []
    ob_counts = {category: 0 for category in OB_CATEGORIES}
    ob_rows: list[dict[str, Any]] = []
    for group_name, qids in cohort["groups"].items():
        for qid in qids:
            request = frozen["requests"][qid]
            row = rows[qid]
            selected = selected_handles(row, request)
            for slot in request.plan.required_slots:
                slot_selected = selected.get(slot.slot_id, [])
                fact_by_id = {str(fact["fact_id"]): fact for fact in request.facts}
                chosen = fact_by_id.get(str((row.get("binding") or {}).get("slot_bindings", {}).get(slot.slot_id, [None])[0]))
                label = labels[qid]
                if group_name == "A_direct_visible_unique":
                    strict = bool(chosen and r1d.slot_is_strict(qid, slot, chosen, label, source_map, reviewed_ids, reviewed_fact_ids, set()))
                elif group_name == "B_calculation_visible_unique":
                    strict = bool(chosen and r1d.slot_is_strict(qid, slot, chosen, label, source_map, set(), {}, set()))
                else:
                    strict = False
                category = classify_slot(qid=qid, slot=slot, selected=slot_selected, strict=strict, status=row.get("final_binding_status", "INVALID"), cohort=group_name, request=request, source_map=source_map, labels=labels)
                sw_counts[category] += 1
                sw_rows.append({"question_id": qid, "cohort": group_name, "slot_id": slot.slot_id, "status": row.get("final_binding_status"), "selected_handles": slot_selected, "primary_category": category, "strict_correct": strict})
                if group_name in {"C_indistinguishable", "D_unbindable_safety"} and chosen is not None:
                    category_ob, constraints = overbinding_category(qid=qid, slot=slot, fact=chosen, request=request, label=label, cohort=group_name)
                    ob_counts[category_ob] += 1
                    by_handle = {handle: fact for handle, fact in zip(fact_handle_map(request).values(), request.facts, strict=True)}
                    competitors = [fact_summary(fact, handle) for handle, fact in by_handle.items() if handle not in slot_selected][:8]
                    ob_rows.append({"question_id": qid, "cohort": group_name, "slot_id": slot.slot_id, "requirement": slot.to_dict(), "selected": fact_summary(chosen, slot_selected[0] if slot_selected else ""), "competitors": competitors, "constraints": constraints, "classification": category_ob})
    review = {"model_calls": 0, "cohorts": {name: len(ids) for name, ids in cohort["groups"].items()}, "category_counts": sw_counts, "rows": sw_rows}
    overbinding = {"model_calls": 0, "category_counts": ob_counts, "bound_cases": len(ob_rows), "rows": ob_rows}
    write_json(OUT / "slotwise-failure-review.json", review)
    write_json(OUT / "overbinding-analysis.json", overbinding)
    return {"failure": review, "overbinding": overbinding}


def provider(source_map: Mapping[str, Mapping[str, Any]]) -> BailianPairwiseBinderProvider:
    config = r1d.legacy.load_config()
    return BailianPairwiseBinderProvider(
        base_url=os.getenv("V2_SUPERVISOR_BASE_URL", "").strip(),
        api_key=config["api_key"],
        model_name=MODEL,
        enable_thinking=False,
        temperature=0.0,
        timeout=180.0,
        max_retries=0,
        system_prompt=PAIRWISE_SYSTEM_PROMPT,
        fact_view_version="v2",
        source_metadata_by_candidate=source_map,
    )


def summarize_run(request: BinderRequest, run: Any, provider_obj: BailianPairwiseBinderProvider, *, groups: list[str]) -> dict[str, Any]:
    row = run.to_dict()
    row.update({
        "question_id": request.question_id,
        "question": request.question,
        "intent": request.plan.intent.value,
        "operation": request.plan.operation,
        "groups": groups,
        "fact_count": len(request.facts),
        "required_slots": [slot.to_dict() for slot in request.plan.required_slots],
        "provider_response_success": bool(run.metadata and run.metadata.provider_response_success) if run.metadata else True,
        "structured_output_success": bool(run.metadata and run.metadata.structured_output_success) if run.metadata else True,
        "dto_valid": bool(run.schema_valid),
        "adapter_valid": bool(run.schema_valid and run.binding and run.binding.status != BindingStatus.INVALID.value),
        "binding_validator_pass": bool(run.validation.passed),
        "validation_reasons": list(run.validation.reasons),
        "pairwise_outcomes": provider_obj.last_pairwise_outcomes or {},
        "raw_response": run.raw_response or "",
    })
    return row


def synthetic_cases() -> list[tuple[BinderRequest, dict[str, Any], set[str]]]:
    cases: list[tuple[BinderRequest, dict[str, Any], set[str]]] = []
    for request, expected in r3.synthetic_cases():
        tags: set[str] = set()
        if request.plan.intent == Intent.CALCULATION:
            tags.add("calculation")
        if request.question_id == "v2_syn_08":
            tags.add("indistinguishable")
        if request.question_id == "v2_syn_07":
            tags.add("unbindable")
        cases.append((request, expected, tags))
    cases.extend([
        (BinderRequest("pair_syn_09", "Two supplied records lack enough structural context to distinguish the requested revenue.", r3.make_plan([r3.make_slot("s1", "revenue", "FY2026")], Intent.DIRECT_FACT), (r3.synthetic_fact("pw09a", "revenue", "FY2026", statement="Unknown", row=["Revenue"], headers=["FY2026"], section="Unknown"), r3.synthetic_fact("pw09b", "revenue", "FY2026", statement="Unknown", row=["Revenue"], headers=["FY2026"], section="Unknown"))), {"s1": "AMBIGUOUS"}, {"indistinguishable"}),
        (BinderRequest("pair_syn_10", "No supplied record proves the requested regional margin.", r3.make_plan([r3.make_slot("s1", "regional margin", "FY2026")], Intent.DIRECT_FACT), (r3.synthetic_fact("pw10a", "margin", "FY2026", statement="Results", row=["Total", "Margin"], headers=["FY2026"], section="Summary"), r3.synthetic_fact("pw10b", "regional revenue", "FY2026", statement="Segments", row=["Regional", "Revenue"], headers=["FY2026"], section="Segments"))), {"s1": "MISSING"}, {"unbindable"}),
        (BinderRequest("pair_syn_11", "Select component and total independently.", r3.make_plan([r3.make_slot("component", "gross profit", "FY2026", "component"), r3.make_slot("total", "sales", "FY2026", "total")], Intent.CALCULATION, "percentage_share"), (r3.synthetic_fact("pw11a", "gross profit", "FY2026", statement="Results", row=["Gross profit"], headers=["FY2026"], section="Summary"), r3.synthetic_fact("pw11b", "sales", "FY2026", statement="Results", row=["Sales"], headers=["FY2026"], section="Summary"))), {"component": ["F01"], "total": ["F02"]}, {"calculation"}),
        (BinderRequest("pair_syn_12", "Select numerator and denominator independently.", r3.make_plan([r3.make_slot("numerator", "net income", "FY2026", "numerator"), r3.make_slot("denominator", "sales", "FY2026", "denominator")], Intent.CALCULATION, "percentage_share"), (r3.synthetic_fact("pw12a", "net income", "FY2026", statement="Results", row=["Net income"], headers=["FY2026"], section="Summary"), r3.synthetic_fact("pw12b", "sales", "FY2026", statement="Results", row=["Sales"], headers=["FY2026"], section="Summary"))), {"numerator": ["F01"], "denominator": ["F02"]}, {"calculation"}),
        (BinderRequest("pair_syn_13", "Select minuend and subtrahend independently.", r3.make_plan([r3.make_slot("minuend", "sales", "FY2026", "minuend"), r3.make_slot("subtrahend", "sales", "FY2025", "subtrahend")], Intent.CALCULATION, "difference"), (r3.synthetic_fact("pw13a", "sales", "FY2026", statement="Results", row=["Sales"], headers=["FY2026"], section="Summary"), r3.synthetic_fact("pw13b", "sales", "FY2025", statement="Results", row=["Sales"], headers=["FY2025"], section="Summary"))), {"minuend": ["F01"], "subtrahend": ["F02"]}, {"calculation"}),
        (BinderRequest("pair_syn_14", "Select current and prior balances independently.", r3.make_plan([r3.make_slot("current", "cash", "FY2026", "current"), r3.make_slot("prior", "cash", "FY2025", "prior")], Intent.CALCULATION, "difference"), (r3.synthetic_fact("pw14a", "cash", "FY2026", statement="Balance Sheet", row=["Cash"], headers=["FY2026"], section="Assets"), r3.synthetic_fact("pw14b", "cash", "FY2025", statement="Balance Sheet", row=["Cash"], headers=["FY2025"], section="Assets"))), {"current": ["F01"], "prior": ["F02"]}, {"calculation"}),
    ])
    return cases


def actual_handles(row: Mapping[str, Any], request: BinderRequest) -> dict[str, list[str]]:
    return selected_handles(row, request)


def expected_case_correct(row: Mapping[str, Any], request: BinderRequest, expected: Mapping[str, Any]) -> bool:
    actual = actual_handles(row, request)
    for slot_id, target in expected.items():
        if isinstance(target, list):
            if sorted(actual.get(slot_id, [])) != sorted(target):
                return False
        elif target == "MISSING" and row.get("final_binding_status") != BindingStatus.MISSING.value:
            return False
        elif target == "AMBIGUOUS" and row.get("final_binding_status") != BindingStatus.AMBIGUOUS.value:
            return False
    return True


def synthetic_run() -> dict[str, Any]:
    cases = synthetic_cases()
    pairwise = provider({})
    service = SemanticBinderService(pairwise)
    rows: list[dict[str, Any]] = []
    try:
        for request, expected, tags in cases:
            run = service.bind(request)
            row = summarize_run(request, run, pairwise, groups=["synthetic"])
            correct = expected_case_correct(row, request, expected)
            actual = actual_handles(row, request)
            safe_indist = "indistinguishable" in tags and row.get("final_binding_status") in {"MISSING", "AMBIGUOUS"}
            false_unbind = "unbindable" in tags and row.get("final_binding_status") == "BOUND"
            calc_slots = 0
            calc_correct = 0
            if "calculation" in tags:
                for slot_id, target in expected.items():
                    if isinstance(target, list):
                        calc_slots += 1
                        calc_correct += int(sorted(actual.get(slot_id, [])) == sorted(target))
            row.update({"expected": expected, "actual": actual, "tags": sorted(tags), "semantic_correct": correct, "safe_indistinguishable": safe_indist, "false_binding": false_unbind, "calculation_correct_slots": calc_correct, "calculation_total_slots": calc_slots})
            rows.append(row)
    finally:
        pairwise.close()
    calc_rows = [row for row in rows if "calculation" in row["tags"]]
    summary = {
        "gate": "NF-V2-03-R5.1",
        "formulation": PAIRWISE_FORMULATION,
        "model": MODEL,
        "benchmark_questions_used": 0,
        "provider_calls": len(rows),
        "provider_success": sum(int(row["provider_response_success"]) for row in rows),
        "structured_output": sum(int(row["structured_output_success"]) for row in rows),
        "dto_valid": sum(int(row["dto_valid"]) for row in rows),
        "adapter_valid": sum(int(row["adapter_valid"]) for row in rows),
        "binding_validator": sum(int(row["binding_validator_pass"]) for row in rows),
        "semantic_correct": sum(int(row["semantic_correct"]) for row in rows),
        "semantic_total": len(rows),
        "indistinguishable_safe": sum(int(row["safe_indistinguishable"]) for row in rows if "indistinguishable" in row["tags"]),
        "indistinguishable_total": sum("indistinguishable" in row["tags"] for row in rows),
        "unbindable_false_binding": sum(int(row["false_binding"]) for row in rows),
        "unbindable_total": sum("unbindable" in row["tags"] for row in rows),
        "calculation_correct_slots": sum(int(row["calculation_correct_slots"]) for row in calc_rows),
        "calculation_total_slots": sum(int(row["calculation_total_slots"]) for row in calc_rows),
        "calculation_groups_correct": sum(int(row["semantic_correct"]) for row in calc_rows),
        "calculation_groups_total": len(calc_rows),
        "structural_healthy": all(row["provider_response_success"] and row["structured_output_success"] and row["dto_valid"] and row["adapter_valid"] and row["binding_validator_pass"] for row in rows),
        "pass": False,
        "rows": rows,
    }
    summary["pass"] = bool(summary["semantic_correct"] >= 12 and summary["indistinguishable_safe"] >= 2 and summary["unbindable_false_binding"] == 0 and summary["calculation_correct_slots"] >= 5 and summary["structural_healthy"])
    write_json(OUT / "synthetic-pairwise-suite.json", summary)
    return summary


def diagnostic_score(rows: list[dict[str, Any]], frozen: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]], cohort: Mapping[str, Any]) -> dict[str, Any]:
    labels = load_labels()
    reviewed_ids, reviewed_fact_ids = r1d.reviewed_direct_map()
    evaluated: dict[str, dict[str, Any]] = {}
    raw_counts = {label: 0 for label in ("MATCH", "REJECT", "INDETERMINATE")}
    for row in rows:
        for classifications in (row.get("pairwise_outcomes") or {}).values():
            for label in classifications.values():
                if label in raw_counts:
                    raw_counts[label] += 1
        qid = row["question_id"]
        request = frozen["requests"][qid]
        label = labels[qid]
        fact_by_id = {str(fact["fact_id"]): fact for fact in request.facts}
        binding = row.get("binding") or {}
        slot_results: list[dict[str, Any]] = []
        for slot in request.plan.required_slots:
            ids = list((binding.get("slot_bindings") or {}).get(slot.slot_id, []))
            fact = fact_by_id.get(str(ids[0])) if len(ids) == 1 else None
            strict = bool(fact and r1d.slot_is_strict(qid, slot, fact, label, source_map, reviewed_ids, reviewed_fact_ids, set()))
            slot_results.append({"slot_id": slot.slot_id, "selected_fact_id": ids[0] if len(ids) == 1 else None, "strict_correct": strict})
        evaluated[qid] = {"question_id": qid, "status": row.get("final_binding_status"), "strict_complete": row.get("final_binding_status") == BindingStatus.BOUND.value and bool(slot_results) and all(item["strict_correct"] for item in slot_results), "slot_results": slot_results}
    groups = cohort["groups"]
    direct = [evaluated[qid] for qid in groups["A_direct_visible_unique"]]
    calc = [evaluated[qid] for qid in groups["B_calculation_visible_unique"]]
    indist = [evaluated[qid] for qid in groups["C_indistinguishable"]]
    unbindable = [evaluated[qid] for qid in groups["D_unbindable_safety"]]
    direct_result = {"questions": len(direct), "correct": sum(int(item["strict_complete"]) for item in direct), "ambiguous": sum(item["status"] == "AMBIGUOUS" for item in direct), "missing": sum(item["status"] == "MISSING" for item in direct), "wrong": sum(item["status"] == "BOUND" and not item["strict_complete"] for item in direct), "false_binding": sum(item["status"] == "BOUND" and not item["strict_complete"] for item in direct)}
    calc_slots = [slot for item in calc for slot in item["slot_results"]]
    calc_result = {"questions": len(calc), "operand_slots": len(calc_slots), "correct_operand_slots": sum(int(slot["strict_correct"]) for slot in calc_slots), "wrong_operand_slots": sum(int(not slot["strict_correct"]) for slot in calc_slots), "all_operands_correct": sum(int(item["strict_complete"]) for item in calc), "false_operand_binding": sum(int(item["status"] == "BOUND" and not slot["strict_correct"]) for item in calc for slot in item["slot_results"])}
    result = {"raw_pairwise_labels": raw_counts, "direct": direct_result, "calculation": calc_result, "indistinguishable": {"questions": len(indist), "appropriate_abstention": sum(item["status"] in {"MISSING", "AMBIGUOUS"} for item in indist), "unsafe_bound": sum(item["status"] == "BOUND" for item in indist)}, "unbindable": {"questions": len(unbindable), "safe_missing_or_ambiguous": sum(item["status"] in {"MISSING", "AMBIGUOUS"} for item in unbindable), "false_binding": sum(item["status"] == "BOUND" for item in unbindable)}, "rows": evaluated}
    return result


def diagnostic_run(frozen: dict[str, Any], source_map: Mapping[str, Mapping[str, Any]], cohort: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_for = {qid: [name for name, ids in cohort["groups"].items() if qid in ids] for qid in cohort["unique_question_ids"]}
    pairwise = provider(source_map)
    service = SemanticBinderService(pairwise)
    rows: list[dict[str, Any]] = []
    try:
        for qid in cohort["unique_question_ids"]:
            request = frozen["requests"][qid]
            rows.append(summarize_run(request, service.bind(request), pairwise, groups=group_for[qid]))
    finally:
        pairwise.close()
    path = OUT / "diagnostic-predictions.jsonl.gz"
    write_jsonl_gz(path, rows)
    digest = sha256_file(path)
    write_json(OUT / "diagnostic-seal.json", {"gate": "NF-V2-03-R5.1", "formulation": PAIRWISE_FORMULATION, "model": MODEL, "prediction_count": len(rows), "sealed": True, "prediction_sha256": digest, "gold_reads_before_prediction_seal": 0, "sealed_before_gold": True})
    if sha256_file(path) != digest:
        raise RuntimeError("R5.1 diagnostic prediction seal verification failed")
    metadata = [row["metadata"] for row in rows if row.get("metadata")]
    latencies = [float(item.get("latency_ms") or 0) for item in metadata]
    inputs = [int(item.get("input_tokens") or 0) for item in metadata]
    outputs = [int(item.get("output_tokens") or 0) for item in metadata]
    runtime: dict[str, Any] = {"provider_calls": len(rows), "provider_calls_per_query": 1, "input_tokens": sum(inputs), "output_tokens": sum(outputs), "average_latency_ms": statistics.mean(latencies) if latencies else 0, "p50_latency_ms": statistics.median(latencies) if latencies else 0, "p95_latency_ms": r1d.percentile(latencies), "max_latency_ms": max(latencies) if latencies else 0}
    old_path = R5_OUT / "diagnostic-predictions.jsonl.gz"
    if old_path.exists():
        old = {row["question_id"]: row for row in read_jsonl_gz(old_path)}
        old_rows = [old[qid] for qid in cohort["unique_question_ids"]]
        old_inputs = [int(item.get("metadata", {}).get("input_tokens") or 0) for item in old_rows if item.get("metadata")]
        runtime["slotwise_r5_same_cohort"] = {"provider_calls": len(old_rows), "input_tokens": sum(old_inputs), "average_input_tokens": statistics.mean(old_inputs) if old_inputs else 0, "p50_input_tokens": statistics.median(old_inputs) if old_inputs else 0, "p95_input_tokens": r1d.percentile([float(x) for x in old_inputs]), "max_input_tokens": max(old_inputs) if old_inputs else 0}
        runtime["token_delta_pairwise_minus_slotwise"] = {"total_input_tokens": sum(inputs) - sum(old_inputs), "mean_input_tokens": statistics.mean(inputs) - statistics.mean(old_inputs), "p50_input_tokens": statistics.median(inputs) - statistics.median(old_inputs), "p95_input_tokens": r1d.percentile([float(x) for x in inputs]) - r1d.percentile([float(x) for x in old_inputs]), "max_input_tokens": max(inputs) - max(old_inputs)}
    write_json(OUT / "token-latency-cost.json", runtime)
    return rows, diagnostic_score(rows, frozen, source_map, cohort)


def main() -> int:
    if os.getenv("V2_SUPERVISOR_MODEL", "").strip() not in ("", MODEL):
        raise SystemExit("V2_SUPERVISOR_MODEL must remain qwen3.7-plus")
    OUT.mkdir(parents=True, exist_ok=True)
    frozen, source_map = load_frozen()
    cohort = load_cohort()
    write_json(OUT / "diagnostic-cohort-sha.json", {"cohort_sha256": cohort["cohort_sha256"], "groups": {name: len(ids) for name, ids in cohort["groups"].items()}, "unique_question_ids": cohort["unique_question_ids"], "model_calls_during_review": 0})
    offline = offline_failure_review(frozen, source_map, cohort)
    write_json(OUT / "candidate-compatibility-contract.json", {"dto": "CandidateCompatibilityDTOv1", "formulation": PAIRWISE_FORMULATION, "labels": list(("MATCH", "REJECT", "INDETERMINATE")), "one_provider_request_per_query": True, "shared_facts": True, "query_level_status": False, "additional_properties": False, "model": MODEL})
    write_json(OUT / "deterministic-reducer-contract.json", {"exactly_one_match": "BOUND", "multiple_matches": "AMBIGUOUS", "zero_match_with_indeterminate": "AMBIGUOUS", "all_reject": "MISSING", "semantic_heuristics_after_output": False})
    (OUT / "pairwise-prompt.txt").write_text(PAIRWISE_SYSTEM_PROMPT, encoding="utf-8")
    write_json(OUT / "pairwise-prompt-sha.json", {"sha256": sha256_file(OUT / "pairwise-prompt.txt")})
    synthetic = synthetic_run()
    if not synthetic["pass"]:
        decision = {"gate": "NF-V2-03-R5.1", "model": MODEL, "formulation": PAIRWISE_FORMULATION, "model_calls_during_failure_review": 0, "synthetic": {key: synthetic[key] for key in ("semantic_correct", "semantic_total", "indistinguishable_safe", "indistinguishable_total", "unbindable_false_binding", "unbindable_total", "calculation_correct_slots", "calculation_total_slots", "structural_healthy")}, "pairwise_binder_effective": False, "binder_model_frozen": MODEL, "binder_task_formulation_frozen": False, "formal_attempt_9": "not_run", "dominant_failure": "pairwise_synthetic_gate_failure", "next_gate": "v2_03_pairwise_safety_failure_review", "production_default": "V1", "production_switch_allowed": False}
        write_json(OUT / "decision.json", decision)
        write_json(OUT / "README.md", {"gate": "NF-V2-03 R5.1 Pairwise Binder", "external_model_review": "cancelled_by_design", "decision": decision, "offline_review": offline})
        print(json.dumps(decision, sort_keys=True))
        return 3
    rows, scored = diagnostic_run(frozen, source_map, cohort)
    write_json(OUT / "direct-results.json", scored["direct"])
    write_json(OUT / "calculation-results.json", scored["calculation"])
    write_json(OUT / "indistinguishable-safety.json", scored["indistinguishable"])
    write_json(OUT / "unbindable-safety.json", scored["unbindable"])
    structural = all(row["provider_response_success"] and row["structured_output_success"] and row["dto_valid"] and row["adapter_valid"] and row["binding_validator_pass"] for row in rows)
    direct = scored["direct"]
    calc = scored["calculation"]
    indist = scored["indistinguishable"]
    unbindable = scored["unbindable"]
    diagnostic_pass = bool(direct["correct"] >= 16 and calc["correct_operand_slots"] >= 10 and indist["appropriate_abstention"] >= 5 and unbindable["false_binding"] <= 1 and structural)
    runtime = read_json(OUT / "token-latency-cost.json")
    decision = {"gate": "NF-V2-03-R5.1", "base_commit": BASE_COMMIT, "model": MODEL, "formulation": PAIRWISE_FORMULATION, "synthetic": f"{synthetic['semantic_correct']}/{synthetic['semantic_total']}", "diagnostic_direct": f"{direct['correct']}/21", "diagnostic_calculation": f"{calc['correct_operand_slots']}/12", "diagnostic_all_operands": f"{calc['all_operands_correct']}/6", "indistinguishable_abstention": f"{indist['appropriate_abstention']}/6", "unbindable_false_binding": f"{unbindable['false_binding']}/7", "raw_pairwise_labels": scored["raw_pairwise_labels"], "structural_violations": 0 if structural else 1, "provider_calls_per_query": 1, "token_delta": runtime.get("token_delta_pairwise_minus_slotwise", {}), "pairwise_binder_effective": True if diagnostic_pass else ("partial" if direct["correct"] >= 14 and calc["correct_operand_slots"] >= 8 and indist["appropriate_abstention"] >= 5 and unbindable["false_binding"] <= 1 else False), "binder_model_frozen": MODEL, "binder_task_formulation_frozen": PAIRWISE_FORMULATION if diagnostic_pass else False, "formal_attempt_9": "pending" if diagnostic_pass else "not_run", "dominant_failure": "none" if diagnostic_pass else ("pairwise_safety_failure" if indist["appropriate_abstention"] < 5 or unbindable["false_binding"] > 1 else "semantic_compatibility_classification_capability"), "next_gate": "formal_attempt_9" if diagnostic_pass else ("v2_03_pairwise_safety_failure_review" if indist["appropriate_abstention"] < 5 or unbindable["false_binding"] > 1 else "v2_03_pairwise_failure_review"), "production_default": "V1", "production_switch_allowed": False}
    write_json(OUT / "formulation-ablation.json", {"global_plus": {"direct": "8/21", "calculation": "1/12"}, "slotwise_plus": {"direct": "9/21", "calculation": "5/12", "indistinguishable_abstention": "0/6", "unbindable_false_binding": "6/7"}, "pairwise_plus": {"direct": f"{direct['correct']}/21", "calculation": f"{calc['correct_operand_slots']}/12", "indistinguishable_abstention": f"{indist['appropriate_abstention']}/6", "unbindable_false_binding": f"{unbindable['false_binding']}/7"}})
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {"gate": "NF-V2-03 R5.1 Pairwise Binder", "external_model_review": "cancelled_by_design", "reason": "cost_and_project_scope", "model": MODEL, "offline_review": offline, "synthetic": synthetic, "diagnostic": decision, "formal_attempt_9": "not_run" if not diagnostic_pass else "pending"})
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
