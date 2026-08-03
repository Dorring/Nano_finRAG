"""Pure metrics and attribution helpers for NF-OPT-05."""

from __future__ import annotations


def classify_first_failure(
    *,
    route_correct: bool,
    evidence_sufficient: bool,
    operands_correct: bool,
    execution_success: bool,
    result_correct: bool,
) -> str:
    if not route_correct:
        return "calculation_intent_not_detected"
    if not evidence_sufficient:
        return "production_evidence_missing"
    if not operands_correct:
        return "operand_extraction_failed"
    if not execution_success:
        return "calculation_execution_failed"
    if not result_correct:
        return "calculation_result_wrong"
    return "passed"


def calculation_routing_gate(
    *,
    route_recall: float,
    route_precision: float,
    operation_accuracy: float,
    false_positive_count: int,
    no_answer_false_positive_count: int,
    oracle_operand_accuracy: float,
    oracle_result_accuracy: float,
) -> dict[str, bool]:
    router = (
        route_recall >= 10 / 11
        and route_precision >= 0.95
        and operation_accuracy >= 10 / 11
    )
    safety = false_positive_count <= 1 and no_answer_false_positive_count == 0
    oracle = oracle_operand_accuracy >= 9 / 11 and oracle_result_accuracy >= 9 / 11
    return {
        "passed": router and safety and oracle,
        "router_passed": router,
        "negative_safety_passed": safety,
        "oracle_passed": oracle,
    }
