"""Focused tests for deterministic Metric Evaluation Contract V2."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from scripts.evaluation.metric_match_contract_v2 import MetricMatchType, match_metric, match_slots


def test_surface_normalization_is_deterministic() -> None:
    first = match_metric("Transactions", "transaction")
    second = match_metric("Transactions", "transaction")
    assert first == second
    assert first.matched
    assert first.match_type == MetricMatchType.SURFACE_NORMALIZED.value
    assert match_metric("accuracy of a proprietary model", "accuracy of proprietary model").match_type == MetricMatchType.SURFACE_NORMALIZED.value


def test_metric_rules_have_no_question_id_or_gold_dependency() -> None:
    source = inspect.getsource(match_metric)
    assert "question_id" not in source
    assert "gold" not in source.casefold()


def test_metric_contract_module_does_not_read_benchmark_or_gold_values() -> None:
    import scripts.evaluation.metric_match_contract_v2 as contract

    source = inspect.getsource(contract)
    assert "query-requirements" not in source
    assert "questions.golden" not in source
    assert "expected_value" not in source


def test_non_conflicting_qualifiers_are_deterministic() -> None:
    assert match_metric("gross margin percentage", "gross margin", predicted_value_type="percentage").match_type == MetricMatchType.NON_CONFLICTING_QUALIFIER_EQUIVALENT.value
    assert match_metric("Intelligent Cloud revenue", "Intelligent Cloud").match_type == MetricMatchType.NON_CONFLICTING_QUALIFIER_EQUIVALENT.value
    assert match_metric("transactions processed on Visa's networks", "transactions processed on networks").match_type == MetricMatchType.NON_CONFLICTING_QUALIFIER_EQUIVALENT.value


def test_scope_changing_qualifier_is_rejected() -> None:
    assert not match_metric("operating revenue", "revenue").matched
    assert not match_metric("operating income", "net income").matched


def test_multislot_matching_is_permutation_invariant_for_non_operational_roles() -> None:
    reference = [
        {"target": "iPhone net sales", "period": "FY2025", "role": "left"},
        {"target": "Services net sales", "period": "FY2025", "role": "right"},
    ]
    predicted = [
        {"metric": "Services net sales", "period": "FY2025", "role": "value"},
        {"metric": "iPhone net sales", "period": "FY2025", "role": "value"},
    ]
    result = match_slots(predicted, reference)
    assert result.complete
    assert not result.unmatched_reference


def test_calculation_operand_roles_remain_strict() -> None:
    reference = [
        {"target": "revenue", "period": "FY2025", "role": "numerator"},
        {"target": "revenue", "period": "FY2024", "role": "denominator"},
    ]
    predicted = [
        {"metric": "revenue", "period": "FY2025", "role": "denominator"},
        {"metric": "revenue", "period": "FY2024", "role": "numerator"},
    ]
    result = match_slots(predicted, reference)
    assert not result.complete


def test_missing_period_and_slot_count_remain_failures() -> None:
    reference = [
        {"target": "guaranteed financial result", "period": "FY2024", "role": "period_1"},
        {"target": "guaranteed financial result", "period": "FY2026", "role": "period_2"},
    ]
    predicted = [{"metric": "guaranteed FY2026 financial result", "period": "FY2024", "role": "value"}]
    result = match_slots(predicted, reference)
    assert not result.complete
    assert result.unmatched_reference == (1,)


def test_sealed_prediction_sha_is_unchanged() -> None:
    path = Path("artifacts/evaluation/nf-v2-01-r1-bailian-formal-72-attempt-2/supervisor-plans.jsonl.gz")
    seal_path = Path("artifacts/evaluation/nf-v2-01-r1-bailian-formal-72-attempt-2/supervisor-prediction-seal.json")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == __import__("json").loads(seal_path.read_text(encoding="utf-8"))["plans_sha256"]
