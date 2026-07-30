"""Pure, side-effect-free NF40 answer-pipeline attribution helpers.

These functions intentionally evaluate observations made by the production
pipeline; they do not alter retrieval, prompting, validation, or release
behaviour.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable


class ContextCoverage(StrEnum):
    NO_GOLD_IN_FINAL = "no_gold_in_final"
    PARTIAL_GOLD_IN_FINAL = "partial_gold_in_final"
    ALL_GOLD_IN_FINAL = "all_gold_in_final"
    NO_ANSWER_CASE = "no_answer_case"


class ValidationOutcome(StrEnum):
    TRUE_ACCEPT = "true_accept"
    TRUE_REJECT = "true_reject"
    FALSE_REJECT = "false_reject"
    FALSE_ACCEPT = "false_accept"


class AnswerFailureStage(StrEnum):
    RETRIEVAL_NO_GOLD = "retrieval_no_gold"
    RETRIEVAL_PARTIAL_GOLD = "retrieval_partial_gold"
    GENERATION_MISSING_ANSWER = "generation_missing_answer"
    GENERATION_WRONG_FACT = "generation_wrong_fact"
    GENERATION_WRONG_NUMBER = "generation_wrong_number"
    GENERATION_WRONG_UNIT = "generation_wrong_unit"
    GENERATION_WRONG_PERIOD = "generation_wrong_period"
    GENERATION_UNSUPPORTED_CLAIM = "generation_unsupported_claim"
    CALCULATION_WRONG_OPERATION = "calculation_wrong_operation"
    CALCULATION_WRONG_OPERAND = "calculation_wrong_operand"
    CALCULATION_FAILED = "calculation_failed"
    CITATION_MISSING = "citation_missing"
    CITATION_WRONG_DOCUMENT = "citation_wrong_document"
    CITATION_WRONG_PAGE = "citation_wrong_page"
    CITATION_INCOMPLETE = "citation_incomplete"
    VALIDATOR_FALSE_REJECT_NUMERIC = "validator_false_reject_numeric"
    VALIDATOR_FALSE_REJECT_UNIT = "validator_false_reject_unit"
    VALIDATOR_FALSE_REJECT_PERIOD = "validator_false_reject_period"
    VALIDATOR_FALSE_REJECT_CITATION = "validator_false_reject_citation"
    VALIDATOR_FALSE_ACCEPT = "validator_false_accept"
    REPAIR_FAILED = "repair_failed"
    CORRECT = "correct"
    NO_ANSWER_CORRECT = "no_answer_correct"
    NO_ANSWER_FALSE_ACCEPT = "no_answer_false_accept"
    NO_ANSWER_FALSE_REJECT = "no_answer_false_reject"


@dataclass(frozen=True)
class StageEvaluation:
    case_id: str
    context_coverage: ContextCoverage
    raw_answer_present: bool = False
    raw_fact_correct: bool = False
    raw_numeric_correct: bool = False
    raw_unit_correct: bool = False
    raw_period_correct: bool = False
    raw_citation_correct: bool = False
    raw_answer_correct: bool = False
    released_answer_correct: bool = False
    released: bool = False
    calculation_attempted: bool = False
    calculation_failed: bool = False
    calculation_wrong_operation: bool = False
    calculation_wrong_operand: bool = False
    validation_failures: tuple[str, ...] = ()
    repair_attempted: bool = False
    repair_succeeded: bool = False
    no_answer_correct: bool | None = None
    latency_ms: float | None = None
    secondary_failures: tuple[AnswerFailureStage, ...] = field(default_factory=tuple)


def classify_context_coverage(
    *,
    expected_no_answer: bool,
    expected_source_count: int,
    matched_gold_source_count: int,
) -> ContextCoverage:
    """Classify final-context completeness without inspecting answer text."""
    if expected_no_answer:
        return ContextCoverage.NO_ANSWER_CASE
    if expected_source_count <= 0:
        raise ValueError("Answerable case requires at least one expected source")
    if matched_gold_source_count <= 0:
        return ContextCoverage.NO_GOLD_IN_FINAL
    if matched_gold_source_count < expected_source_count:
        return ContextCoverage.PARTIAL_GOLD_IN_FINAL
    return ContextCoverage.ALL_GOLD_IN_FINAL


def classify_validation_outcome(*, raw_answer_correct: bool, released: bool) -> ValidationOutcome:
    if raw_answer_correct and released:
        return ValidationOutcome.TRUE_ACCEPT
    if not raw_answer_correct and not released:
        return ValidationOutcome.TRUE_REJECT
    if raw_answer_correct:
        return ValidationOutcome.FALSE_REJECT
    return ValidationOutcome.FALSE_ACCEPT


def map_false_reject_reason(failures: Iterable[str]) -> AnswerFailureStage:
    lowered = " ".join(failures).lower()
    if "unit" in lowered or "currency" in lowered:
        return AnswerFailureStage.VALIDATOR_FALSE_REJECT_UNIT
    if "period" in lowered or "date" in lowered or "year" in lowered:
        return AnswerFailureStage.VALIDATOR_FALSE_REJECT_PERIOD
    if "citation" in lowered or "source" in lowered or "page" in lowered:
        return AnswerFailureStage.VALIDATOR_FALSE_REJECT_CITATION
    return AnswerFailureStage.VALIDATOR_FALSE_REJECT_NUMERIC


def determine_primary_failure(evaluation: StageEvaluation) -> AnswerFailureStage:
    """Attribute one primary failure using the fixed NF40 precedence order."""
    if evaluation.context_coverage is ContextCoverage.NO_ANSWER_CASE:
        return (
            AnswerFailureStage.NO_ANSWER_CORRECT
            if evaluation.no_answer_correct
            else AnswerFailureStage.NO_ANSWER_FALSE_ACCEPT
        )
    if evaluation.context_coverage is ContextCoverage.NO_GOLD_IN_FINAL:
        return AnswerFailureStage.RETRIEVAL_NO_GOLD
    if evaluation.context_coverage is ContextCoverage.PARTIAL_GOLD_IN_FINAL:
        return AnswerFailureStage.RETRIEVAL_PARTIAL_GOLD
    if not evaluation.raw_answer_present:
        return AnswerFailureStage.GENERATION_MISSING_ANSWER
    if not evaluation.raw_fact_correct:
        return AnswerFailureStage.GENERATION_WRONG_FACT
    if not evaluation.raw_numeric_correct:
        return AnswerFailureStage.GENERATION_WRONG_NUMBER
    if not evaluation.raw_unit_correct:
        return AnswerFailureStage.GENERATION_WRONG_UNIT
    if not evaluation.raw_period_correct:
        return AnswerFailureStage.GENERATION_WRONG_PERIOD
    if evaluation.calculation_wrong_operation:
        return AnswerFailureStage.CALCULATION_WRONG_OPERATION
    if evaluation.calculation_wrong_operand:
        return AnswerFailureStage.CALCULATION_WRONG_OPERAND
    if evaluation.calculation_failed:
        return AnswerFailureStage.CALCULATION_FAILED
    if not evaluation.raw_citation_correct:
        return AnswerFailureStage.CITATION_INCOMPLETE
    validation = classify_validation_outcome(
        raw_answer_correct=evaluation.raw_answer_correct,
        released=evaluation.released,
    )
    if validation is ValidationOutcome.FALSE_REJECT:
        return map_false_reject_reason(evaluation.validation_failures)
    if validation is ValidationOutcome.FALSE_ACCEPT:
        return AnswerFailureStage.VALIDATOR_FALSE_ACCEPT
    if evaluation.repair_attempted and not evaluation.repair_succeeded:
        return AnswerFailureStage.REPAIR_FAILED
    return AnswerFailureStage.CORRECT


def _rate(numerator: int, denominator: int) -> dict[str, int | float]:
    return {"count": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else 1.0}


def compute_nf40_metrics(evaluations: Iterable[StageEvaluation]) -> dict:
    """Build auditable count-plus-rate metrics from immutable observations."""
    rows = list(evaluations)
    with_any_gold = [row for row in rows if row.context_coverage in {ContextCoverage.PARTIAL_GOLD_IN_FINAL, ContextCoverage.ALL_GOLD_IN_FINAL}]
    with_all_gold = [row for row in rows if row.context_coverage is ContextCoverage.ALL_GOLD_IN_FINAL]
    no_answer = [row for row in rows if row.context_coverage is ContextCoverage.NO_ANSWER_CASE]
    raw_correct = [row for row in rows if row.raw_answer_correct]
    incorrect_raw = [row for row in rows if not row.raw_answer_correct and row.context_coverage is not ContextCoverage.NO_ANSWER_CASE]
    released_correct = [row for row in rows if row.released_answer_correct]
    validation = Counter(
        classify_validation_outcome(raw_answer_correct=row.raw_answer_correct, released=row.released).value
        for row in rows if row.context_coverage is not ContextCoverage.NO_ANSWER_CASE
    )
    primary = Counter(determine_primary_failure(row).value for row in rows)
    no_answer_correct = sum(bool(row.no_answer_correct) for row in no_answer)
    blocked_incorrect = sum(not row.released for row in incorrect_raw)
    raw_with_gold = sum(row.raw_answer_correct for row in with_any_gold)
    released_with_gold = sum(row.released_answer_correct for row in with_any_gold)
    raw_with_all_gold = sum(row.raw_answer_correct for row in with_all_gold)
    released_with_all_gold = sum(row.released_answer_correct for row in with_all_gold)
    correct_raw_released = sum(row.raw_answer_correct and row.released for row in rows)
    sufficient_gold = len(with_all_gold)
    return {
        "case_count": len(rows),
        "context_coverage": dict(Counter(row.context_coverage.value for row in rows)),
        "golden_pass": _rate(sum(row.released_answer_correct for row in rows) + no_answer_correct, len(rows)),
        "released_answer_accuracy": _rate(len(released_correct), len(rows) - len(no_answer)),
        "no_answer_accuracy": _rate(no_answer_correct, len(no_answer)),
        "conditional_any_gold": {"raw_accuracy": _rate(raw_with_gold, len(with_any_gold)), "released_accuracy": _rate(released_with_gold, len(with_any_gold))},
        "conditional_all_gold": {"raw_accuracy": _rate(raw_with_all_gold, len(with_all_gold)), "released_accuracy": _rate(released_with_all_gold, len(with_all_gold))},
        "validator_confusion": dict(validation),
        "context_utilization_rate": _rate(raw_with_gold, len(with_any_gold)),
        "validation_retention_rate": _rate(correct_raw_released, len(raw_correct)),
        "validator_catch_rate": _rate(blocked_incorrect, len(incorrect_raw)),
        "retrieval_ceiling": {"any_gold": _rate(len(with_any_gold) + no_answer_correct, len(rows)), "all_gold": _rate(sufficient_gold + no_answer_correct, len(rows))},
        "primary_failures": dict(primary),
    }
