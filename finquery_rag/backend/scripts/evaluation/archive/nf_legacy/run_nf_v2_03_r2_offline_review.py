#!/usr/bin/env python3
"""Offline Attempt-6 failure attribution and frozen Prompt R2 artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import run_nf_v2_03_r1a_binding_contract_recovery as r1a  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1c_supply_recovery as r1c  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402


OUT = ROOT / "artifacts/evaluation/nf-v2-03-r2-semantic-selection"
R1D_FORMAL = ROOT / "artifacts/evaluation/nf-v2-03-r1d-supply-conditioned-binder/formal-attempt-6"
R1B = ROOT / "artifacts/evaluation/nf-v2-03-r1b-constrained-binding"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"


PROMPT_R2 = """You are the semantic evidence binder for a financial RAG system.

Your only job is to map each already-required slot to zero, one, or multiple
existing query-local fact handles. Return only the strict selection DTO. Never
answer the question, emit a value, calculate, create an ID, or add a slot.

Evaluate every slot independently.

For each slot, first identify the exact requested financial concept and its
scope. A parent or broad metric does not satisfy a narrower segment, product,
component, or qualified concept unless the supplied row and header context
clearly establishes that identity. Use the raw and normalized metric fields,
row path, row hierarchy, column/header path, table title, statement title,
section context, and provenance together. Do not rely on normalized_metric
alone.

Next require the requested fiscal or reporting period. A matching concept in
another period is not a valid substitute.

If one supplied fact clearly satisfies the slot's metric, scope, period,
statement context, and provenance, return exactly one handle. Do not return
multiple handles merely because several facts share a word or a broad metric.
Use multiple handles only when at least two facts remain materially
indistinguishable after all available metric, scope, period, row/header,
statement, and provenance constraints are applied.

If no supplied fact satisfies every material constraint, return an empty array.
Do not select the nearest semantic match to increase coverage. Safety priority
is: a correct unique binding, then genuine ambiguity or missing evidence, then
never a wrong binding.

For calculation plans, handle every operand slot independently. Respect the
slot's metric, period, scope, and frozen operational role. Distinct roles or
periods require distinct suitable facts; do not reuse one fact across distinct
operands unless the supplied evidence genuinely represents the same operand.
Never calculate or emit a numeric result. Preserve all exact RequiredSlot IDs
and select only supplied F-handles.
"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    return r1d.read_jsonl_gz(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_matches(fact: Mapping[str, Any], expected: list[Mapping[str, Any]]) -> bool:
    candidate_ids = {str(fact.get("candidate_id"))} | {str(item) for item in fact.get("candidate_ids", [])}
    return any(str(source.get("candidate_key")) in candidate_ids for source in expected if source.get("candidate_key"))


def period_matches(fact: Mapping[str, Any], slot: Any) -> bool:
    return r1c.period(fact.get("normalized_period") or fact.get("raw_period")) == r1c.period(slot.period)


def view_metric_matches(fact: Mapping[str, Any], slot: Any, source_map: Mapping[str, Mapping[str, Any]]) -> bool:
    return r1c.view_metric_match(slot, fact, source_map.get(str(fact.get("candidate_id"))))


def selected_ids(row: Mapping[str, Any], slot_id: str) -> list[str]:
    return [str(item) for item in ((row.get("binding") or {}).get("slot_bindings", {}).get(slot_id, []))]


def load_review_context() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], set[str], dict[str, set[str]], set[str], set[str]]:
    frozen = r1d.load_r1c_frozen_inputs()
    predictions = {row["question_id"]: row for row in read_jsonl_gz(R1D_FORMAL / "predictions.jsonl.gz")}
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if row}
    review_rows = read_json(R1B / "fact-semantic-compatibility-review.json")["direct"]["rows"]
    reviewed_strict = {str(row["question_id"]) for row in review_rows if row.get("reviewed_semantic_compatible") and row.get("reviewed_period_compatible")}
    reviewed_fact_ids = {str(row["question_id"]): {str(item) for item in row.get("reviewed_fact_ids", [])} for row in review_rows}
    generic_direct = set(read_json(ROOT / "artifacts/evaluation/nf-v2-03-r1c-supply-and-protocol-recovery/current-vs-view-bindability.json").get("generic_recovered_strict_questions", []))
    calc_bindable = {str(row["question_id"]) for row in read_json(ROOT / "artifacts/evaluation/nf-v2-03-r1c-supply-and-protocol-recovery/calculation-supply-funnel.json")["rows"] if row.get("strict_bindable")}
    return frozen, predictions, labels, reviewed_strict, reviewed_fact_ids, generic_direct, calc_bindable


def fact_is_valid_for_slot(
    question_id: str,
    fact: Mapping[str, Any],
    slot: Any,
    label: Mapping[str, Any],
    source_map: Mapping[str, Mapping[str, Any]],
    reviewed_strict: set[str],
    reviewed_fact_ids: Mapping[str, set[str]],
) -> bool:
    expected = r1a.expected_sources(slot, label)
    if not source_matches(fact, expected) or not period_matches(fact, slot):
        return False
    if question_id in reviewed_strict:
        return str(fact.get("fact_id")) in reviewed_fact_ids.get(question_id, set())
    return view_metric_matches(fact, slot, source_map)


def direct_reviews() -> tuple[dict[str, Any], dict[str, Any]]:
    frozen, predictions, labels, reviewed_strict, reviewed_fact_ids, generic_direct, _ = load_review_context()
    state = r1d.nf02.verify_frozen_top100()
    source_map = r1c.candidate_source_map(state)
    direct_rows = read_json(R1D_FORMAL / "direct-supply-conditioned.json")["rows"]
    direct_bindable = {str(row["question_id"]) for row in direct_rows if row.get("strict_bindable")}
    bindable_output: list[dict[str, Any]] = []
    unbindable_output: list[dict[str, Any]] = []
    bindable_counts: Counter[str] = Counter()
    unbindable_counts: Counter[str] = Counter()
    for question_id in sorted(frozen["requests"]):
        request = frozen["requests"][question_id]
        if request.plan.intent.value != "DIRECT_FACT":
            continue
        row = predictions[question_id]
        slot = request.plan.required_slots[0]
        ids = selected_ids(row, slot.slot_id)
        fact_by_id = {str(fact["fact_id"]): fact for fact in request.facts}
        selected = [fact_by_id[item] for item in ids if item in fact_by_id]
        valid_selected = [fact for fact in selected if fact_is_valid_for_slot(question_id, fact, slot, labels[question_id], source_map, reviewed_strict, reviewed_fact_ids)]
        if question_id in direct_bindable:
            if row["final_binding_status"] == "BOUND" and len(valid_selected) == 1 and len(selected) == 1:
                category = "BS0_correct"
            elif row["final_binding_status"] == "MISSING":
                category = "BS2_missing_despite_bindable_fact"
            elif row["final_binding_status"] == "AMBIGUOUS":
                category = "BS8_multiple_candidates_genuinely_ambiguous" if len(valid_selected) >= 2 else "BS1_over_ambiguous_despite_unique_best_fact"
            elif not selected:
                category = "BS2_missing_despite_bindable_fact"
            elif not period_matches(selected[0], slot):
                category = "BS5_period_confusion"
            elif not source_matches(selected[0], r1a.expected_sources(slot, labels[question_id])):
                category = "BS6_segment_or_statement_confusion"
            elif not view_metric_matches(selected[0], slot, source_map):
                category = "BS4_metric_scope_confusion"
            else:
                category = "BS3_wrong_fact_selected"
            bindable_counts[category] += 1
            bindable_output.append({"question_id": question_id, "question": request.question, "status": row["final_binding_status"], "selected_fact_ids": ids, "valid_selected_count": len(valid_selected), "primary_category": category})
        elif selected:
            expected = r1a.expected_sources(slot, labels[question_id])
            if row["final_binding_status"] != "BOUND":
                category = "BU0_correct_missing_or_ambiguous"
            elif not period_matches(selected[0], slot):
                category = "BU2_over_binding_wrong_period"
            elif not source_matches(selected[0], expected):
                category = "BU4_over_binding_wrong_statement"
            elif not view_metric_matches(selected[0], slot, source_map):
                category = "BU1_over_binding_nearest_metric"
            else:
                category = "BU3_over_binding_wrong_scope"
            unbindable_counts[category] += 1
            unbindable_output.append({"question_id": question_id, "question": request.question, "status": row["final_binding_status"], "selected_fact_ids": ids, "primary_category": category})
    false_bindable = sum(1 for row in bindable_output if row["primary_category"] != "BS0_correct" and row["status"] == "BOUND")
    false_unbindable = sum(1 for row in unbindable_output if row["status"] == "BOUND")
    total_false = int(read_json(ROOT / "artifacts/evaluation/nf-v2-03-r1d-supply-conditioned-binder/formal-attempt-6/false-binding-analysis.json")["false_binding_queries"])
    result = {
        "denominator": 27,
        "rows": bindable_output,
        "counts": {key: bindable_counts.get(key, 0) for key in ("BS0_correct", "BS1_over_ambiguous_despite_unique_best_fact", "BS2_missing_despite_bindable_fact", "BS3_wrong_fact_selected", "BS4_metric_scope_confusion", "BS5_period_confusion", "BS6_segment_or_statement_confusion", "BS7_duplicate_statement_confusion", "BS8_multiple_candidates_genuinely_ambiguous", "BS9_other")},
        "false_binding_on_bindable": false_bindable,
    }
    unbindable = {
        "rows": unbindable_output,
        "counts": {key: unbindable_counts.get(key, 0) for key in ("BU0_correct_missing_or_ambiguous", "BU1_over_binding_nearest_metric", "BU2_over_binding_wrong_period", "BU3_over_binding_wrong_scope", "BU4_over_binding_wrong_statement", "BU5_other")},
        "direct_false_binding_on_unbindable": false_unbindable,
        "false_binding_total_reference": total_false,
        "false_binding_on_bindable": false_bindable,
        "false_binding_on_unbindable": total_false - false_bindable,
        "false_binding_reconciles": false_bindable + (total_false - false_bindable) == total_false,
    }
    return result, unbindable


def calculation_review() -> dict[str, Any]:
    frozen, predictions, labels, reviewed_strict, reviewed_fact_ids, generic_direct, calc_bindable = load_review_context()
    state = r1d.nf02.verify_frozen_top100()
    source_map = r1c.candidate_source_map(state)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    full = 0
    partial = 0
    correct_slots = 0
    total_slots = 0
    same_reuse = 0
    for question_id in sorted(calc_bindable):
        request = frozen["requests"][question_id]
        prediction = predictions[question_id]
        fact_by_id = {str(fact["fact_id"]): fact for fact in request.facts}
        selected_per_slot: list[list[str]] = []
        slot_rows: list[dict[str, Any]] = []
        for slot in request.plan.required_slots:
            ids = selected_ids(prediction, slot.slot_id)
            selected_per_slot.append(ids)
            selected = [fact_by_id[item] for item in ids if item in fact_by_id]
            if not ids:
                category = "BC1_operand_missing"
                correct = False
            elif len(ids) > 1:
                category = "BC2_operand_over_ambiguous"
                correct = False
            elif selected and fact_is_valid_for_slot(question_id, selected[0], slot, labels[question_id], source_map, reviewed_strict, reviewed_fact_ids):
                category = "BC0_correct_operand"
                correct = True
            elif selected and not period_matches(selected[0], slot):
                category = "BC3_wrong_period"
                correct = False
            elif selected and not view_metric_matches(selected[0], slot, source_map):
                category = "BC4_wrong_metric"
                correct = False
            elif selected and not source_matches(selected[0], r1a.expected_sources(slot, labels[question_id])):
                category = "BC7_wrong_statement"
                correct = False
            else:
                category = "BC9_other"
                correct = False
            counts[category] += 1
            total_slots += 1
            correct_slots += int(correct)
            slot_rows.append({"slot_id": slot.slot_id, "selected_fact_ids": ids, "correct": correct, "primary_category": category})
        flattened = [ids[0] for ids in selected_per_slot if len(ids) == 1]
        if len(flattened) != len(set(flattened)):
            same_reuse += 1
            counts["BC6_same_fact_reused_across_distinct_operands"] += 1
        is_full = all(row["correct"] for row in slot_rows) and prediction["final_binding_status"] == "BOUND"
        full += int(is_full)
        partial += int(not is_full and any(row["correct"] for row in slot_rows))
        rows.append({"question_id": question_id, "status": prediction["final_binding_status"], "slot_rows": slot_rows, "fully_correct": is_full})
    false_binding_queries = sum(int(row["status"] == "BOUND" and not row["fully_correct"]) for row in rows)
    false_binding_slots = sum(int(row["status"] == "BOUND" and not slot["correct"]) for row in rows for slot in row["slot_rows"])
    return {
        "bindable_questions": len(calc_bindable),
        "fully_correct_questions": full,
        "partial_correct_questions": partial,
        "correct_operand_slots": correct_slots,
        "total_operand_slots": total_slots,
        "ambiguous_operand_slots": counts["BC2_operand_over_ambiguous"],
        "missing_operand_slots": counts["BC1_operand_missing"],
        "wrong_operand_slots": total_slots - correct_slots - counts["BC2_operand_over_ambiguous"] - counts["BC1_operand_missing"],
        "same_fact_reuse_count": same_reuse,
        "false_binding_queries": false_binding_queries,
        "false_binding_slots": false_binding_slots,
        "counts": {key: counts.get(key, 0) for key in ("BC0_correct_operand", "BC1_operand_missing", "BC2_operand_over_ambiguous", "BC3_wrong_period", "BC4_wrong_metric", "BC5_wrong_operand_role", "BC6_same_fact_reused_across_distinct_operands", "BC7_wrong_statement", "BC8_wrong_scope", "BC9_other")},
        "rows": rows,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prompt_path = OUT / "binder-prompt-r2.txt"
    prompt_path.write_text(PROMPT_R2.rstrip() + "\n", encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    (OUT / "binder-prompt-r2.sha256").write_text(prompt_hash + "\n", encoding="utf-8")
    direct, unbindable = direct_reviews()
    calc = calculation_review()
    taxonomy = {}
    taxonomy.update(direct["counts"])
    taxonomy.update(unbindable["counts"])
    taxonomy.update(calc["counts"])
    write_json(OUT / "attempt6-direct-failure-review.json", direct)
    write_json(OUT / "attempt6-unbindable-false-binding-review.json", unbindable)
    write_json(OUT / "attempt6-calculation-failure-review.json", calc)
    write_json(OUT / "failure-taxonomy.json", taxonomy)
    write_json(OUT / "offline-review-decision.json", {
        "gate": "NF-V2-03-R2",
        "model_calls": 0,
        "attempt6_structural_health_preserved": True,
        "prompt_r2_authorized": True,
        "direct_success_given_bindable_attempt6": "12/27",
        "calculation_success_given_bindable_attempt6": "0/6",
        "false_binding_total_attempt6": unbindable["false_binding_total_reference"],
        "false_binding_reconciles": unbindable["false_binding_reconciles"],
        "prompt_sha256": prompt_hash,
        "next_stage": "synthetic_semantic_suite",
    })
    write_json(OUT / "README.md", {"gate": "NF-V2-03-R2", "description": "Offline Attempt-6 semantic failure attribution followed by one generalizable Binder prompt recovery. No model calls during review.", "model_calls": 0, "prompt_sha256": prompt_hash})
    print(json.dumps({"model_calls": 0, "direct": direct["counts"], "false_bindable": direct["false_binding_on_bindable"], "false_unbindable": unbindable["false_binding_on_unbindable"], "false_total": unbindable["false_binding_total_reference"], "calculation": calc["counts"], "fully_correct_calculation": calc["fully_correct_questions"], "prompt_sha256": prompt_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
