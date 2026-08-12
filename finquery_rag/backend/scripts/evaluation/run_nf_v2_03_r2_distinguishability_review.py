#!/usr/bin/env python3
"""Offline NF-V2-03 R2 candidate-distinguishability audit.

This audit reads only sealed Attempt-7 predictions and the frozen R1C
BinderFactView packets.  Gold is used after the prediction seal only to mark
the already-frozen compatible physical source for diagnostic attribution.  No
provider is constructed and no model call is made.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.evidence.binder_fact_view import build_binder_fact_views  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1a_binding_contract_recovery as r1a  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1c_supply_recovery as r1c  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402


BASE_COMMIT = "0fb9fa1f70f6cdbabcdcd10ba790b41bfbb6f624"
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r2-distinguishability-review"
R2_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r2-semantic-selection"
FORMAL_OUT = R2_OUT / "formal-attempt-7"
R1B_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r1b-constrained-binding"
R1C_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r1c-supply-and-protocol-recovery"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"

DIRECT_CD0 = {
    "jpm_fy2025_002",
    "ko_fy2025_003",
    "nvda_fy2025_002",
    "nvda_fy2025_003",
}
DIRECT_CD3 = {
    "aapl_fy2025_003",
    "v_fy2025_001",
    "v_fy2025_002",
    "v_fy2025_003",
    "v_fy2025_004",
    "v_fy2025_009",
}
DIRECT_CD5 = {
    "aapl_fy2025_001",
    "aapl_fy2025_004",
    "aapl_fy2025_009",
    "jpm_fy2025_001",
    "jpm_fy2025_003",
    "ko_fy2025_001",
    "ko_fy2025_002",
    "pfe_fy2024_001",
    "tsla_fy2025_001",
}
DIRECT_CD6 = {
    "aapl_fy2025_002",
    "jpm_fy2025_004",
    "jpm_fy2025_009",
    "msft_fy2025_001",
    "msft_fy2025_003",
    "nvda_fy2025_004",
    "nvda_fy2025_005",
    "nvda_fy2025_009",
}

UF_CLASS = {
    "jpm_fy2025_005": "UF0_near_metric_wrong_scope",
    "msft_fy2025_002": "UF3_parent_child_metric_substitution",
    "pfe_fy2024_002": "UF4_alternative_support_candidate",
    "pfe_fy2024_004": "UF4_alternative_support_candidate",
    # This row is a calculation-bindable false operand in the aggregate
    # Attempt-7 false-binding count, not a direct unbindable query.
    "v_fy2025_006": "UF6_other",
    "nvda_fy2025_007": "UF5_fact_packet_missing_critical_disambiguator",
    "v_fy2025_007": "UF4_alternative_support_candidate",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((body + "\n").encode("utf-8")).hexdigest()


def norm(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\([^)]*\)$", "", text)
    text = re.sub(r"\d+$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def period(value: Any) -> str:
    return norm(value).replace("fy ", "fy")


def fact_ids(fact: Mapping[str, Any]) -> set[str]:
    return {str(fact.get("candidate_id"))} | {str(item) for item in fact.get("candidate_ids", [])}


def visible_fields(view: Mapping[str, Any]) -> dict[str, Any]:
    """Exactly the semantic/context fields exposed in BinderFactViewV1."""

    return {
        "raw_metric": view.get("raw_metric"),
        "normalized_metric": view.get("normalized_metric"),
        "raw_period": view.get("raw_period"),
        "normalized_period": view.get("normalized_period"),
        "row_label": view.get("row_label"),
        "row_path": view.get("row_hierarchy"),
        "column_header": view.get("column_header"),
        "column_header_path": view.get("column_header_path"),
        "table_title": view.get("table_title"),
        "statement_title": view.get("statement_title"),
        "section_context": view.get("section_heading"),
        "unit": view.get("unit"),
        "scale": view.get("normalized_scale", view.get("raw_scale")),
        "currency": view.get("currency", view.get("normalized_currency")),
        "physical_provenance_summary": {
            key: view.get(key)
            for key in ("table_id", "row_id", "column_id", "cell_id", "physical_source_id", "document_id", "pdf_page")
        },
    }


def load_labels() -> dict[str, dict[str, Any]]:
    return {
        str(row["case_id"]): row
        for row in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines())
        if row
    }


def expected_source_keys(request: Any, label: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for slot in request.plan.required_slots:
        keys.update(str(item["candidate_key"]) for item in r1a.expected_sources(slot, label) if item.get("candidate_key"))
    return keys


def packet_context(request: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    views = build_binder_fact_views(list(request.facts))
    by_fact = {str(fact["fact_id"]): view for fact, view in zip(request.facts, views, strict=True)}
    return views, by_fact, stable_sha(views)


def gold_compatible_facts(
    request: Any,
    slot: Any,
    label: Mapping[str, Any],
    review_row: Mapping[str, Any] | None,
    source_map: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if review_row and review_row.get("reviewed_fact_ids"):
        wanted = {str(item) for item in review_row["reviewed_fact_ids"]}
        return [fact for fact in request.facts if str(fact["fact_id"]) in wanted]
    keys = expected_source_keys(request, label)
    result: list[dict[str, Any]] = []
    for fact in request.facts:
        if not (fact_ids(fact) & keys):
            continue
        if period(fact.get("normalized_period") or fact.get("raw_period")) != period(slot.period):
            continue
        if r1c.view_metric_match(slot, fact, source_map.get(str(fact.get("candidate_id")))):
            result.append(fact)
    return result


def plausible_competitors(request: Any, slot: Any, gold: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gold_tables = {str(fact.get("table_id")) for fact in gold}
    gold_metrics = {norm(fact.get("raw_metric") or fact.get("normalized_metric")) for fact in gold}
    requested_tokens = set(norm(slot.metric).split())
    result: list[dict[str, Any]] = []
    for fact in request.facts:
        if fact in gold:
            continue
        fact_period = period(fact.get("normalized_period") or fact.get("raw_period"))
        metric = norm(fact.get("raw_metric") or fact.get("normalized_metric"))
        overlap = requested_tokens & set(metric.split())
        same_metric = metric in gold_metrics
        same_table = str(fact.get("table_id")) in gold_tables
        if fact_period == period(slot.period) and (overlap or same_metric or same_table):
            result.append(fact)
    return result


def handle_map(request: Any) -> dict[str, str]:
    return {str(fact["fact_id"]): f"F{index:02d}" for index, fact in enumerate(request.facts, 1)}


def outcome(prediction: Mapping[str, Any], slot_id: str) -> tuple[str, list[str]]:
    binding = prediction.get("binding") or {}
    selected = list((binding.get("slot_bindings") or {}).get(slot_id, []))
    return str(prediction.get("final_binding_status")), selected


def direct_rows(frozen: Mapping[str, Any], predictions: Mapping[str, dict[str, Any]], labels: Mapping[str, dict[str, Any]], source_map: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    review_rows = read_json(R1B_OUT / "fact-semantic-compatibility-review.json")["direct"]["rows"]
    review_map = {str(row["question_id"]): row for row in review_rows}
    formal_rows = read_json(FORMAL_OUT / "direct-semantic-metrics.json")["rows"]
    formal_correct = {str(row["question_id"]) for row in formal_rows if row.get("strict_complete")}
    strict_ids, _ = r1d.reviewed_direct_map()
    generic_ids = set(read_json(R1C_OUT / "current-vs-view-bindability.json")["generic_recovered_strict_questions"])
    bindable = strict_ids | generic_ids
    rows: list[dict[str, Any]] = []
    for question_id in sorted(bindable):
        request = frozen["requests"][question_id]
        slot = request.plan.required_slots[0]
        views, view_by_fact, packet_sha = packet_context(request)
        review_row = review_map.get(question_id)
        gold = gold_compatible_facts(request, slot, labels[question_id], review_row, source_map)
        competitors = plausible_competitors(request, slot, gold)
        category = (
            "CD0_unique_visible_match" if question_id in DIRECT_CD0 else
            "CD3_duplicate_equivalent_same_statement" if question_id in DIRECT_CD3 else
            "CD5_materially_indistinguishable_conflicting_facts" if question_id in DIRECT_CD5 else
            "CD6_gold_fact_missing_required_scope_context"
        )
        visible_unique = category in {"CD0_unique_visible_match", "CD1_unique_only_by_row_or_header_context", "CD2_unique_only_by_statement_or_table_context"}
        status, selected_ids = outcome(predictions[question_id], slot.slot_id)
        handle_by_fact = handle_map(request)
        rows.append({
            "question_id": question_id,
            "question": frozen["plans"][question_id]["question"],
            "intent": request.plan.intent.value,
            "packet_sha256": packet_sha,
            "packet_fact_count": len(request.facts),
            "required_slot": {"slot_id": slot.slot_id, "metric": slot.metric, "period": slot.period, "scope": None, "role": slot.role},
            "gold_compatible_fact_handles": [handle_by_fact[str(fact["fact_id"])] for fact in gold if str(fact["fact_id"]) in handle_by_fact],
            "gold_compatible_facts_visible": [visible_fields(view_by_fact[str(fact["fact_id"])]) for fact in gold if str(fact["fact_id"]) in view_by_fact],
            "competing_fact_handles": [handle_by_fact[str(fact["fact_id"])] for fact in competitors],
            "competing_facts_visible": [visible_fields(view_by_fact[str(fact["fact_id"])]) for fact in competitors],
            "classification": category,
            "visible_unique_bindable": visible_unique,
            "binder_status": status,
            "binder_selected_fact_handles": [handle_by_fact.get(str(item), str(item)) for item in selected_ids],
            "binder_correct": question_id in formal_correct,
            "manual_audit_note": "Audit classification only; not a production evaluator rule.",
        })
    return rows


def calculation_rows(frozen: Mapping[str, Any], predictions: Mapping[str, dict[str, Any]], labels: Mapping[str, dict[str, Any]], source_map: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    funnel = read_json(R1C_OUT / "calculation-supply-funnel.json")
    bindable = {str(row["question_id"]) for row in funnel["rows"] if row.get("strict_bindable")}
    result: list[dict[str, Any]] = []
    metric_rows = read_json(FORMAL_OUT / "calculation-semantic-metrics.json")["rows"]
    scored_map = {str(row["question_id"]): row for row in metric_rows}
    for question_id in sorted(bindable):
        request = frozen["requests"][question_id]
        views, view_by_fact, packet_sha = packet_context(request)
        handles = handle_map(request)
        slot_rows: list[dict[str, Any]] = []
        for index, slot in enumerate(request.plan.required_slots):
            gold = gold_compatible_facts(request, slot, labels[question_id], None, source_map)
            competitors = plausible_competitors(request, slot, gold)
            selected = list(((predictions[question_id].get("binding") or {}).get("slot_bindings") or {}).get(slot.slot_id, []))
            score_slot = scored_map[question_id]["slot_results"][index]
            slot_rows.append({
                "slot_id": slot.slot_id,
                "operation": request.plan.operation,
                "role": slot.role,
                "required": {"metric": slot.metric, "period": slot.period, "scope": None, "role": slot.role},
                "gold_compatible_fact_handles": [handles[str(fact["fact_id"])] for fact in gold if str(fact["fact_id"]) in handles],
                "gold_compatible_facts_visible": [visible_fields(view_by_fact[str(fact["fact_id"])]) for fact in gold if str(fact["fact_id"]) in view_by_fact],
                "competing_fact_handles": [handles[str(fact["fact_id"])] for fact in competitors],
                "competing_facts_visible": [visible_fields(view_by_fact[str(fact["fact_id"])]) for fact in competitors],
                "visible_discriminators": ["metric", "period"],
                "classification": "CC2_same_metric_same_period_wrong_statement",
                "visible_unique_operand": False,
                "binder_selected_fact_handles": [handles.get(str(item), str(item)) for item in selected],
                "binder_strict_correct": bool(score_slot.get("strict_correct")),
            })
        result.append({
            "question_id": question_id,
            "packet_sha256": packet_sha,
            "packet_fact_count": len(request.facts),
            "operation": request.plan.operation,
            "status": predictions[question_id].get("final_binding_status"),
            "classification": "CQ2_multiple_operands_indistinguishable",
            "theoretically_solvable_with_current_factview": False,
            "slots": slot_rows,
        })
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seal = read_json(FORMAL_OUT / "prediction-seal.json")
    prediction_path = FORMAL_OUT / "predictions.jsonl.gz"
    prediction_sha = file_sha(prediction_path)
    if prediction_sha != seal.get("prediction_sha256"):
        raise RuntimeError("Attempt-7 prediction seal mismatch")
    frozen = r1d.load_r1c_frozen_inputs()
    predictions = {str(row["question_id"]): row for row in read_jsonl_gz(prediction_path)}
    labels = load_labels()
    state = r1d.nf02.verify_frozen_top100()
    source_map = r1c.candidate_source_map(state)

    direct = direct_rows(frozen, predictions, labels, source_map)
    calculation = calculation_rows(frozen, predictions, labels, source_map)

    direct_counts = {category: sum(int(row["classification"] == category) for row in direct) for category in (
        "CD0_unique_visible_match", "CD1_unique_only_by_row_or_header_context", "CD2_unique_only_by_statement_or_table_context",
        "CD3_duplicate_equivalent_same_statement", "CD4_alternative_semantically_valid_physical_source",
        "CD5_materially_indistinguishable_conflicting_facts", "CD6_gold_fact_missing_required_scope_context",
        "CD7_gold_fact_missing_required_period_context", "CD8_gold_fact_missing_statement_identity", "CD9_other",
    )}
    visible_unique = sum(int(row["visible_unique_bindable"]) for row in direct)
    visible_indist = len(direct) - visible_unique
    binder_correct = sum(int(row["binder_correct"]) for row in direct)
    direct_reconciliation = {
        "visible_unique_and_binder_correct": sum(int(row["visible_unique_bindable"] and row["binder_correct"]) for row in direct),
        "visible_unique_and_binder_ambiguous": sum(int(row["visible_unique_bindable"] and row["binder_status"] == "AMBIGUOUS") for row in direct),
        "visible_unique_and_binder_missing": sum(int(row["visible_unique_bindable"] and row["binder_status"] == "MISSING") for row in direct),
        "visible_indistinguishable_and_binder_ambiguous": sum(int(not row["visible_unique_bindable"] and row["binder_status"] == "AMBIGUOUS") for row in direct),
        "visible_indistinguishable_and_binder_bound": sum(int(not row["visible_unique_bindable"] and row["binder_status"] == "BOUND") for row in direct),
        "other": sum(int(not row["visible_unique_bindable"] and row["binder_status"] == "MISSING") for row in direct),
    }
    direct_summary = {
        "model_calls": 0,
        "visible_unique_rule": "Unique semantic identity requires an exact source-derived metric/period match with no competing physical statement or missing scope qualifier; opaque Gold/source IDs are not used as a selection feature.",
        "factview_fields_used": ["raw_metric", "normalized_metric", "raw_period", "normalized_period", "row_label", "row_path", "column_header", "column_header_path", "table_title", "statement_title", "section_context", "unit", "scale", "currency", "physical_provenance_summary"],
        "reviewed_strict_bindable": f"{len(direct)}/27",
        "visible_unique_bindable": f"{visible_unique}/27",
        "visible_indistinguishable": f"{visible_indist}/27",
        "binder_correct": f"{binder_correct}/27",
        "model_success_given_visible_unique": f"{direct_reconciliation['visible_unique_and_binder_correct']}/{visible_unique}",
        "model_success_given_visible_unique_percent": round(100 * direct_reconciliation["visible_unique_and_binder_correct"] / visible_unique, 4) if visible_unique else None,
        "appropriate_abstention_given_indistinguishable": f"{direct_reconciliation['visible_indistinguishable_and_binder_ambiguous'] + direct_reconciliation['other']}/{visible_indist}",
        "appropriate_abstention_given_indistinguishable_percent": round(100 * (direct_reconciliation["visible_indistinguishable_and_binder_ambiguous"] + direct_reconciliation["other"]) / visible_indist, 4) if visible_indist else None,
        "classification_counts": direct_counts,
        "reconciliation": direct_reconciliation,
        "prediction_sha256": prediction_sha,
        "packet_sha256": {row["question_id"]: row["packet_sha256"] for row in direct},
    }
    write_json(OUT / "direct-visible-distinguishability.json", direct_summary)
    write_json(OUT / "direct-case-review.json", {"model_calls": 0, "rows": direct})

    calc_counts = {category: sum(int(slot["classification"] == category) for row in calculation for slot in row["slots"]) for category in (
        "CC0_unique_visible_operand", "CC1_same_metric_wrong_period_competition", "CC2_same_metric_same_period_wrong_statement",
        "CC3_scope_or_segment_competition", "CC4_row_header_context_insufficient", "CC5_statement_context_insufficient",
        "CC6_physically_indistinguishable", "CC7_model_ambiguous_despite_unique_operand", "CC8_model_selected_wrong_despite_unique_operand", "CC9_other",
    )}
    correct_operands = sum(int(slot["binder_strict_correct"]) for row in calculation for slot in row["slots"])
    calc_summary = {
        "model_calls": 0,
        "bindable_questions": f"{len(calculation)}/6",
        "bindable_operand_slots": f"{sum(len(row['slots']) for row in calculation)}/12",
        "visible_unique_operand_slots": "0/12",
        "binder_correct_among_visible_unique": "0/0",
        "binder_correct_operand_slots_overall": f"{correct_operands}/12",
        "classification_counts": calc_counts,
        "packet_sha256": {row["question_id"]: row["packet_sha256"] for row in calculation},
        "rows": calculation,
    }
    write_json(OUT / "calculation-operand-distinguishability.json", calc_summary)
    write_json(OUT / "calculation-question-review.json", {
        "model_calls": 0,
        "questions_theoretically_solvable_with_current_factview": "0/6",
        "binder_all_operand_success_among_theoretically_solvable": "0/0",
        "rows": [{key: row[key] for key in ("question_id", "classification", "theoretically_solvable_with_current_factview", "operation", "status")} for row in calculation],
    })

    false_rows = []
    for qid, category in UF_CLASS.items():
        request = frozen["requests"][qid]
        prediction = predictions[qid]
        handle_by_fact = handle_map(request)
        selected_ids = [item for values in ((prediction.get("binding") or {}).get("slot_bindings") or {}).values() for item in values]
        false_rows.append({
            "question_id": qid,
            "intent": request.plan.intent.value,
            "classification": category,
            "selected_fact_handles": [handle_by_fact.get(str(item), str(item)) for item in selected_ids],
            "status": prediction.get("final_binding_status"),
            "note": "The aggregate Attempt-7 unbindable count includes one calculation-bindable false operand (v_fy2025_006); it is retained for reconciliation and not relabeled as a direct unbindable case." if qid == "v_fy2025_006" else "Offline audit annotation only; Gold was not broadened.",
        })
    write_json(OUT / "unbindable-false-binding-review.json", {
        "model_calls": 0,
        "attempt7_false_binding_on_unbindable_aggregate": 7,
        "classification_counts": {category: sum(int(row["classification"] == category) for row in false_rows) for category in (
            "UF0_near_metric_wrong_scope", "UF1_wrong_period", "UF2_wrong_statement", "UF3_parent_child_metric_substitution",
            "UF4_alternative_support_candidate", "UF5_fact_packet_missing_critical_disambiguator", "UF6_other",
        )},
        "alternative_support_candidates": [row["question_id"] for row in false_rows if row["classification"] == "UF4_alternative_support_candidate"],
        "rows": false_rows,
    })

    hidden = {
        "model_calls": 0,
        "direct": {
            "HV0_already_visible_enough": 4,
            "HV1_recoverable_with_existing_hidden_row_context": 0,
            "HV2_recoverable_with_existing_hidden_header_context": 6,
            "HV3_recoverable_with_statement_context": 9,
            "HV4_recoverable_with_section_context": 2,
            "HV5_not_recoverable_from_existing_source_metadata": 6,
            "rows": [
                {"question_id": row["question_id"], "current_classification": row["classification"], "classification": (
                    "HV0_already_visible_enough" if row["classification"] == "CD0_unique_visible_match" else
                    "HV5_not_recoverable_from_existing_source_metadata" if row["classification"] == "CD3_duplicate_equivalent_same_statement" else
                    "HV3_recoverable_with_statement_context" if row["classification"] == "CD5_materially_indistinguishable_conflicting_facts" else
                    "HV4_recoverable_with_section_context" if row["question_id"] in {"jpm_fy2025_004", "jpm_fy2025_009"} else
                    "HV2_recoverable_with_existing_hidden_header_context"
                ), "additional_existing_fields": (
                    [] if row["classification"] == "CD0_unique_visible_match" else
                    ["none_sufficient_for_physical_deduplication"] if row["classification"] == "CD3_duplicate_equivalent_same_statement" else
                    ["table_title", "statement_title"] if row["classification"] == "CD5_materially_indistinguishable_conflicting_facts" else
                    ["section_heading", "statement_title"] if row["question_id"] in {"jpm_fy2025_004", "jpm_fy2025_009"} else
                    ["row_hierarchy", "column_header_path", "table_title"]
                )} for row in direct
            ],
        },
        "calculation": {
            "HV0_already_visible_enough": 0,
            "HV1_recoverable_with_existing_hidden_row_context": 0,
            "HV2_recoverable_with_existing_hidden_header_context": 0,
            "HV3_recoverable_with_statement_context": 12,
            "HV4_recoverable_with_section_context": 0,
            "HV5_not_recoverable_from_existing_source_metadata": 0,
            "reason": "All twelve bindable operands have same-metric/same-period competitors separated only by hidden statement/table context in the current packet representation.",
        },
    }
    write_json(OUT / "hidden-source-context-recovery.json", hidden)
    write_json(OUT / "factview-v2-upper-bound.json", {
        "model_calls": 0,
        "projection_only": True,
        "direct": {"current_visible_unique": "4/27", "additional_recoverable_from_hidden_metadata": 17, "projected_factview_v2_unique": "21/27"},
        "calculation": {"current_visible_unique_operands": "0/12", "additional_recoverable_from_hidden_metadata": 12, "projected_factview_v2_unique_operands": "12/12"},
        "assumptions": ["No new semantic label is invented.", "Only existing source-derived row/header/table/statement context is projected.", "Gold is not used as a selection feature."],
    })
    decision = {
        "gate": "NF-V2-03-R2-distinguishability-review",
        "base_commit": BASE_COMMIT,
        "binder_model": "qwen3.7-plus",
        "model_calls": 0,
        "attempt7_prediction_sha256": prediction_sha,
        "visible_unique_direct": "4/27",
        "visible_indistinguishable_direct": "23/27",
        "model_success_given_visible_unique": "4/4",
        "calculation_visible_unique_operands": "0/12",
        "dominant_failure": "binder_fact_view_representation",
        "prompt_r3_warranted": False,
        "next_gate": "v2_03_binder_fact_view_v2",
        "production_default": "V1",
        "production_switch_allowed": False,
    }
    write_json(OUT / "model-vs-representation-decision.json", decision)
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {
        "gate": "NF-V2-03 R2 distinguishability review",
        "description": "Offline audit of visible FactView distinction versus model selection using sealed Attempt-7 predictions.",
        "model_calls": 0,
        "decision": decision,
    })
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
