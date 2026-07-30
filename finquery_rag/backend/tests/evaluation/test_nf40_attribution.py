from src.evaluation.nf40_attribution import (
    AnswerFailureStage,
    ContextCoverage,
    StageEvaluation,
    ValidationOutcome,
    classify_context_coverage,
    classify_validation_outcome,
    compute_nf40_metrics,
    determine_primary_failure,
)


def test_partial_multi_source_context_is_not_complete():
    assert classify_context_coverage(expected_no_answer=False, expected_source_count=2, matched_gold_source_count=1) is ContextCoverage.PARTIAL_GOLD_IN_FINAL


def test_no_gold_context_precedes_generation_failure():
    result = StageEvaluation(case_id="a", context_coverage=ContextCoverage.NO_GOLD_IN_FINAL)
    assert determine_primary_failure(result) is AnswerFailureStage.RETRIEVAL_NO_GOLD


def test_validation_outcomes_keep_raw_and_release_distinct():
    assert classify_validation_outcome(raw_answer_correct=True, released=False) is ValidationOutcome.FALSE_REJECT
    assert classify_validation_outcome(raw_answer_correct=False, released=True) is ValidationOutcome.FALSE_ACCEPT


def test_correct_raw_blocked_is_false_reject():
    result = StageEvaluation(case_id="a", context_coverage=ContextCoverage.ALL_GOLD_IN_FINAL, raw_answer_present=True, raw_fact_correct=True, raw_numeric_correct=True, raw_unit_correct=True, raw_period_correct=True, raw_citation_correct=True, raw_answer_correct=True, released=False, validation_failures=("unit mismatch",))
    assert determine_primary_failure(result) is AnswerFailureStage.VALIDATOR_FALSE_REJECT_UNIT


def test_metrics_report_counts_and_retrieval_ceiling():
    rows = [
        StageEvaluation(case_id="a", context_coverage=ContextCoverage.ALL_GOLD_IN_FINAL, raw_answer_present=True, raw_fact_correct=True, raw_numeric_correct=True, raw_unit_correct=True, raw_period_correct=True, raw_citation_correct=True, raw_answer_correct=True, released=True, released_answer_correct=True),
        StageEvaluation(case_id="b", context_coverage=ContextCoverage.NO_GOLD_IN_FINAL),
        StageEvaluation(case_id="c", context_coverage=ContextCoverage.NO_ANSWER_CASE, no_answer_correct=True),
    ]
    report = compute_nf40_metrics(rows)
    assert report["context_utilization_rate"] == {"count": 1, "denominator": 1, "rate": 1.0}
    assert report["retrieval_ceiling"]["all_gold"] == {"count": 2, "denominator": 3, "rate": 2 / 3}
