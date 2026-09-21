#!/usr/bin/env python3
"""Audit-only review of the sealed NF-V2-01 R1 slot metrics.

The review reads the immutable Attempt 2 prediction artifact and the frozen
evaluation annotations.  It never calls a provider and never writes back to
the prediction artifact or to any evaluator contract.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PREDICTION_DIR = ROOT / "artifacts/evaluation/nf-v2-01-r1-bailian-formal-72-attempt-2"
OUT = ROOT / "artifacts/evaluation/nf-v2-01-r1-slot-semantic-failure-review"
QUESTIONS = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"
REQUIREMENTS = ROOT / "artifacts/evaluation/nf-opt-23-r1-query-requirement-serialization/query-requirements.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def intent_for(row: dict[str, Any]) -> str:
    if row.get("requires_calculation"):
        return "CALCULATION"
    if row.get("requires_multiple_sources"):
        return "MULTI_EVIDENCE"
    return "DIRECT_FACT"


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    with gzip.open(PREDICTION_DIR / "supervisor-plans.jsonl.gz", "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    rows = {str(row["case_id"]): row for row in map(json.loads, QUESTIONS.read_text(encoding="utf-8").splitlines()) if row}
    requirements = json.loads(REQUIREMENTS.read_text(encoding="utf-8"))
    return records, rows, requirements


# These are audit judgments, not normalization rules.  They are deliberately
# explicit so that the review remains reproducible without changing runtime.
METRIC_JUDGMENTS: dict[tuple[str, int], tuple[str, bool, bool, bool, bool, str]] = {
    ("aapl_fy2025_003", 0): ("SM4_extra_metric_qualifier", True, True, False, False, "percentage is an explicit value-type qualifier for gross margin"),
    ("aapl_fy2025_008", 0): ("SM0_evaluator_format_only", True, True, False, False, "disclose predicate omitted; requested fact semantics unchanged"),
    ("jpm_fy2025_008", 0): ("SM0_evaluator_format_only", True, True, False, False, "disclose predicate omitted; requested fact semantics unchanged"),
    ("ko_fy2025_003", 0): ("SM0_evaluator_format_only", True, True, False, False, "trailing of is a surface annotation difference"),
    ("ko_fy2025_008", 0): ("SM0_evaluator_format_only", True, True, False, False, "annotation contains a malformed leading s; requested disclosure fact is preserved"),
    ("msft_fy2025_007", 0): ("SM4_extra_metric_qualifier", True, True, False, False, "Intelligent Cloud revenue names the segment revenue numerator"),
    ("msft_fy2025_008", 0): ("SM0_evaluator_format_only", True, True, False, False, "disclose predicate omitted; requested fact semantics unchanged"),
    ("nvda_fy2025_003", 0): ("SM4_extra_metric_qualifier", True, True, False, False, "percentage is an explicit value-type qualifier for GAAP gross margin"),
    ("nvda_fy2025_007", 1): ("SM4_extra_metric_qualifier", True, True, False, False, "percentage is an explicit value-type qualifier for GAAP gross margin"),
    ("nvda_fy2025_008", 0): ("SM0_evaluator_format_only", True, True, False, False, "disclose predicate omitted; requested fact semantics unchanged"),
    ("pfe_fy2024_008", 0): ("SM4_extra_metric_qualifier", True, True, False, False, "FY2026 qualifies the same guaranteed-result metric; second period slot is separately missing"),
    ("pfe_fy2024_008", 1): ("SM11_multi_metric_slot_alignment_error", False, False, True, False, "the second required period slot was not emitted"),
    ("tsla_fy2025_008", 0): ("SM4_extra_metric_qualifier", True, True, False, False, "FY2026 qualifies the same guaranteed-price metric; second period slot is separately missing"),
    ("tsla_fy2025_008", 1): ("SM11_multi_metric_slot_alignment_error", False, False, True, False, "the second required period slot was not emitted"),
    ("v_fy2025_003", 0): ("SM4_extra_metric_qualifier", True, True, False, False, "Visa's is an explicit company scope qualifier"),
    ("v_fy2025_007", 1): ("SM4_extra_metric_qualifier", True, True, False, False, "Visa's is an explicit company scope qualifier"),
    ("v_fy2025_008", 0): ("SM0_evaluator_format_only", True, True, False, False, "disclose predicate omitted; requested fact semantics unchanged"),
}


ROLE_JUDGMENTS: dict[tuple[str, int], tuple[str, str]] = {
    ("aapl_fy2025_007", 0): ("SR1_surface_role_name_mismatch", "multi-evidence positional left role emitted as generic value"),
    ("aapl_fy2025_007", 1): ("SR1_surface_role_name_mismatch", "multi-evidence positional right role emitted as generic value"),
    ("jpm_fy2025_007", 0): ("SR1_surface_role_name_mismatch", "multi-evidence positional left role emitted as generic value"),
    ("jpm_fy2025_007", 1): ("SR1_surface_role_name_mismatch", "multi-evidence positional right role emitted as generic value"),
    ("nvda_fy2025_007", 0): ("SR1_surface_role_name_mismatch", "multi-evidence positional left role emitted as generic value"),
    ("nvda_fy2025_007", 1): ("SR1_surface_role_name_mismatch", "multi-evidence positional right role emitted as generic value"),
    ("pfe_fy2024_007", 0): ("SR1_surface_role_name_mismatch", "multi-evidence positional left role emitted as generic value"),
    ("pfe_fy2024_007", 1): ("SR1_surface_role_name_mismatch", "multi-evidence positional right role emitted as generic value"),
    ("pfe_fy2024_008", 0): ("SR1_surface_role_name_mismatch", "period_1 role was collapsed to generic value"),
    ("pfe_fy2024_008", 1): ("SR6_role_missing", "period_2 slot and role were omitted"),
    ("tsla_fy2025_008", 0): ("SR1_surface_role_name_mismatch", "period_1 role was collapsed to generic value"),
    ("tsla_fy2025_008", 1): ("SR6_role_missing", "period_2 slot and role were omitted"),
    ("v_fy2025_007", 0): ("SR1_surface_role_name_mismatch", "multi-evidence positional left role emitted as generic value"),
    ("v_fy2025_007", 1): ("SR1_surface_role_name_mismatch", "multi-evidence positional right role emitted as generic value"),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records, rows, requirements = load_inputs()
    by_id = {record["question_id"]: record for record in records}
    metric_errors: list[dict[str, Any]] = []
    period_errors: list[dict[str, Any]] = []
    role_errors: list[dict[str, Any]] = []
    slot_count_errors: list[dict[str, Any]] = []
    metric_correct = period_correct = role_correct = slot_count_correct = 0
    metric_total = period_total = role_total = 0

    for qid, record in by_id.items():
        requirement = requirements[qid]
        expected_slots = requirement.get("required_slots", [])
        predicted_slots = (record.get("plan") or {}).get("required_slots", [])
        intent = intent_for(rows[qid])
        if len(expected_slots) == len(predicted_slots):
            slot_count_correct += 1
        else:
            slot_count_errors.append({
                "question_id": qid,
                "question": record["question"],
                "intent": intent,
                "gold_slot_count": len(expected_slots),
                "predicted_slot_count": len(predicted_slots),
                "gold_slots": expected_slots,
                "predicted_slots": predicted_slots,
                "primary_taxonomy": "SSC0_missing_slot" if len(predicted_slots) < len(expected_slots) else "SSC1_extra_slot",
                "review_note": "two-period disclosure question was under-decomposed into one slot",
            })
        for index, expected in enumerate(expected_slots):
            predicted = predicted_slots[index] if index < len(predicted_slots) else {}
            metric_total += 1
            period_total += 1
            role_total += 1
            if norm(expected.get("target")) == norm(predicted.get("metric")):
                metric_correct += 1
            else:
                judgment = METRIC_JUDGMENTS[(qid, index)]
                taxonomy, semantic_equivalent, canonical, prompt_only, ambiguous, note = judgment
                metric_errors.append({
                    "question_id": qid,
                    "question": record["question"],
                    "intent": intent,
                    "slot_index": index,
                    "gold_metric": expected.get("target"),
                    "predicted_metric": predicted.get("metric"),
                    "gold_slots": expected_slots,
                    "predicted_slots": predicted_slots,
                    "primary_taxonomy": taxonomy,
                    "semantic_correct_evaluator_mismatch": semantic_equivalent,
                    "contract_normalization_recoverable": canonical,
                    "prompt_extractable_semantic_error": prompt_only,
                    "intrinsically_ambiguous": ambiguous,
                    "review_note": note,
                })
            if norm(expected.get("period")) == norm(predicted.get("period")):
                period_correct += 1
            else:
                period_errors.append({
                    "question_id": qid,
                    "question": record["question"],
                    "intent": intent,
                    "slot_index": index,
                    "gold_period": expected.get("period"),
                    "predicted_period": predicted.get("period"),
                    "primary_taxonomy": "SPER4_missing_period" if predicted.get("period") is None else "SPER5_other",
                    "review_note": "missing second period is caused by slot under-decomposition, not a fiscal/calendar normalization error",
                })
            if norm(expected.get("role")) == norm(predicted.get("role")):
                role_correct += 1
            else:
                taxonomy, note = ROLE_JUDGMENTS[(qid, index)]
                role_errors.append({
                    "question_id": qid,
                    "question": record["question"],
                    "intent": intent,
                    "slot_index": index,
                    "gold_role": expected.get("role"),
                    "predicted_role": predicted.get("role"),
                    "gold_slots": expected_slots,
                    "predicted_slots": predicted_slots,
                    "primary_taxonomy": taxonomy,
                    "review_note": note,
                })

    metric_taxonomy = Counter(error["primary_taxonomy"] for error in metric_errors)
    role_taxonomy = Counter(error["primary_taxonomy"] for error in role_errors)
    semantic_equivalent = sum(error["semantic_correct_evaluator_mismatch"] for error in metric_errors)
    canonical_only = sum(error["contract_normalization_recoverable"] for error in metric_errors)
    prompt_only = sum(error["prompt_extractable_semantic_error"] for error in metric_errors)
    true_semantic_errors = sum(not error["semantic_correct_evaluator_mismatch"] and not error["prompt_extractable_semantic_error"] for error in metric_errors)
    semantic_numerator = metric_correct + semantic_equivalent
    raw_metric_accuracy = metric_correct / metric_total
    semantic_metric_accuracy = semantic_numerator / metric_total
    canonical_projected = (metric_correct + canonical_only) / metric_total
    prompt_combined_projected = (metric_correct + canonical_only + prompt_only) / metric_total

    calc_role_total = calc_role_correct = noncalc_role_total = noncalc_role_correct = 0
    for qid, record in by_id.items():
        is_calc = intent_for(rows[qid]) == "CALCULATION"
        expected_slots = requirements[qid].get("required_slots", [])
        predicted_slots = (record.get("plan") or {}).get("required_slots", [])
        for index, expected in enumerate(expected_slots):
            predicted = predicted_slots[index] if index < len(predicted_slots) else {}
            if is_calc:
                calc_role_total += 1
                calc_role_correct += norm(expected.get("role")) == norm(predicted.get("role"))
            else:
                noncalc_role_total += 1
                noncalc_role_correct += norm(expected.get("role")) == norm(predicted.get("role"))

    write_json(OUT / "metric-error-review.json", {
        "raw_metric_correct": metric_correct,
        "raw_metric_total": metric_total,
        "errors": metric_errors,
    })
    write_json(OUT / "metric-taxonomy.json", {
        "counts": {f"SM{i}_{name}": metric_taxonomy.get(f"SM{i}_{name}", 0) for i, name in enumerate([
            "evaluator_format_only", "singular_plural_or_surface_form", "financial_synonym_same_semantics", "missing_metric_qualifier", "extra_metric_qualifier", "hierarchical_metric_too_broad", "hierarchical_metric_too_narrow", "wrong_financial_metric", "row_header_composition_loss", "segment_or_scope_loss", "derived_metric_misidentified", "multi_metric_slot_alignment_error", "other"
        ])},
        "semantic_correct_evaluator_mismatch": semantic_equivalent,
        "contract_normalization_recoverable": canonical_only,
        "prompt_extractable_semantic_error": prompt_only,
        "intrinsically_ambiguous": sum(error["intrinsically_ambiguous"] for error in metric_errors),
        "true_semantic_errors": true_semantic_errors,
    })
    write_json(OUT / "role-error-review.json", {
        "calculation": {"correct": calc_role_correct, "total": calc_role_total, "accuracy": calc_role_correct / calc_role_total if calc_role_total else 0.0},
        "non_calculation": {"correct": noncalc_role_correct, "total": noncalc_role_total, "accuracy": noncalc_role_correct / noncalc_role_total if noncalc_role_total else 0.0},
        "errors": role_errors,
        "reconciliation": "Overall role errors are entirely outside calculation roles; All Operand Slots 10/11 loses one calculation case on metric lexical mismatch (Intelligent Cloud vs Intelligent Cloud revenue), not on role.",
    })
    write_json(OUT / "role-taxonomy.json", {"counts": {f"SR{i}_{name}": role_taxonomy.get(f"SR{i}_{name}", 0) for i, name in enumerate(["role_not_applicable_but_scored", "surface_role_name_mismatch", "current_prior_reversed", "numerator_denominator_reversed", "minuend_subtrahend_reversed", "component_total_misassigned", "role_missing", "extra_role", "other"])}})
    write_json(OUT / "period-error-review.json", {"errors": period_errors, "period_correct": period_correct, "period_total": period_total})
    write_json(OUT / "slot-count-review.json", {"errors": slot_count_errors, "slot_count_correct": slot_count_correct, "slot_count_total": len(records)})
    write_json(OUT / "semantic-metric-analysis.json", {
        "raw_metric_accuracy": {"correct": metric_correct, "total": metric_total, "accuracy": raw_metric_accuracy},
        "semantic_metric_accuracy": {"correct": semantic_numerator, "total": metric_total, "accuracy": semantic_metric_accuracy},
        "semantic_correct_error_slots": semantic_equivalent,
        "true_semantic_error_slots": true_semantic_errors,
        "canonical_only_projected_accuracy": {"correct": metric_correct + canonical_only, "total": metric_total, "accuracy": canonical_projected},
        "semantic_review_scope": "All 15 non-missing raw metric mismatches were manually judged financially equivalent; the two missing slots are structural decomposition failures rather than wrong metric semantics.",
    })
    write_json(OUT / "projected-recoverability.json", {
        "raw_metric_accuracy": {"correct": metric_correct, "total": metric_total, "accuracy": raw_metric_accuracy},
        "semantic_metric_accuracy": {"correct": semantic_numerator, "total": metric_total, "accuracy": semantic_metric_accuracy},
        "canonical_only_projected_accuracy": {"correct": metric_correct + canonical_only, "total": metric_total, "accuracy": canonical_projected},
        "prompt_recoverable_projected_accuracy": {"correct": metric_correct + canonical_only + prompt_only, "total": metric_total, "accuracy": prompt_combined_projected},
        "recovery_buckets_no_double_count": {"raw_strict": metric_correct, "canonical_only_increment": canonical_only, "prompt_only_increment": prompt_only},
    })
    decision = {
        "gate": "NF-V2-01-R1-slot-semantic-failure-review",
        "evaluation_role": "development_shadow_v2_supervisor_slot_semantic_review",
        "model_calls": 0,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "frozen_predictions_preserved": True,
        "raw_metric_accuracy": raw_metric_accuracy,
        "semantic_metric_accuracy": semantic_metric_accuracy,
        "canonical_only_projected_accuracy": canonical_projected,
        "prompt_recoverable_projected_accuracy": prompt_combined_projected,
        "supervisor_semantic_capability_sufficient": semantic_metric_accuracy >= 0.90 and true_semantic_errors <= 0.10 * metric_total,
        "prompt_r2_warranted": true_semantic_errors >= 5,
        "model_switch_review_warranted": prompt_combined_projected < 0.85,
        "dominant_failure": "metric_evaluation_contract",
        "next_gate": "v2_01_metric_evaluation_contract_review",
    }
    write_json(OUT / "decision.json", decision)
    (OUT / "README.md").write_text("""# NF-V2-01 R1 Supervisor Slot Semantic Failure Review\n\nAudit-only review of the sealed Attempt 2 predictions. No model, retrieval, reranker, evaluator, prompt, contract, or prediction was changed. The 17 raw metric mismatches contain 15 semantically equivalent surface/qualifier differences and two missing-slot decomposition errors; no true wrong-financial-metric case was found. Role errors are non-calculation positional-role contract mismatches, while the one calculation operand miss is a metric qualifier mismatch.\n""", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
