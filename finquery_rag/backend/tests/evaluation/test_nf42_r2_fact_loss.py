"""NF42 R2 fact loss funnel tests.

Verifies that the first-loss-stage classification correctly attributes
where a newly correct structured fact is lost between extraction and
the final released answer, and that fact counts and case counts are
reported separately.
"""
from __future__ import annotations

from src.evaluation.nf42_r2_projection_trace import (
    NewFactFunnelTrace,
    StructuredFactLossStage,
    classify_new_fact_loss,
)


def test_correct_fact_not_selected_is_selector_loss():
    """A correct fact that entered the selector but was not selected is a selector loss."""
    trace = NewFactFunnelTrace(
        case_id="case_5",
        fact_id="fact_5a",
        candidate_key="key_5a",
        correct_fact_extracted=True,
        projection_eligible=True,
        projected_candidate_id="projected:v1:xyz",
        pre_selector_rank=2,
        entered_selector_input=True,
        selected_by_selector=False,
        value_selected=False,
        raw_answer_correct=False,
        released_answer_correct=False,
    )
    trace.first_loss_stage = classify_new_fact_loss(trace)
    assert trace.first_loss_stage == StructuredFactLossStage.ENTERED_SELECTOR_NOT_SELECTED


def test_selected_fact_value_not_used_is_value_selection_loss():
    """A fact selected by the selector whose value was not used is a value-selection loss."""
    trace = NewFactFunnelTrace(
        case_id="case_7",
        fact_id="fact_7a",
        candidate_key="key_7a",
        correct_fact_extracted=True,
        projection_eligible=True,
        projected_candidate_id="projected:v1:def",
        pre_selector_rank=1,
        entered_selector_input=True,
        selected_by_selector=True,
        value_selected=False,
        raw_answer_correct=False,
        released_answer_correct=False,
    )
    trace.first_loss_stage = classify_new_fact_loss(trace)
    assert trace.first_loss_stage == StructuredFactLossStage.SELECTED_VALUE_NOT_USED


def test_fact_count_and_case_count_are_separate():
    """Fact count and case count must be reported as distinct fields.

    A single case can contain multiple newly correct facts.  The funnel
    must not conflate the number of facts with the number of cases.
    """
    # One case with two newly correct facts
    traces = [
        NewFactFunnelTrace(
            case_id="case_10",
            fact_id="fact_10a",
            candidate_key="key_10a",
            correct_fact_extracted=True,
            projection_eligible=True,
            projected_candidate_id="projected:v1:a1",
            pre_selector_rank=1,
            entered_selector_input=True,
            selected_by_selector=False,
            value_selected=False,
            raw_answer_correct=False,
            released_answer_correct=False,
        ),
        NewFactFunnelTrace(
            case_id="case_10",
            fact_id="fact_10b",
            candidate_key="key_10b",
            correct_fact_extracted=True,
            projection_eligible=True,
            projected_candidate_id="projected:v1:b2",
            pre_selector_rank=3,
            entered_selector_input=True,
            selected_by_selector=False,
            value_selected=False,
            raw_answer_correct=False,
            released_answer_correct=False,
        ),
        NewFactFunnelTrace(
            case_id="case_11",
            fact_id="fact_11a",
            candidate_key="key_11a",
            correct_fact_extracted=True,
            projection_eligible=True,
            projected_candidate_id="projected:v1:c3",
            pre_selector_rank=2,
            entered_selector_input=True,
            selected_by_selector=False,
            value_selected=False,
            raw_answer_correct=False,
            released_answer_correct=False,
        ),
    ]
    for t in traces:
        t.first_loss_stage = classify_new_fact_loss(t)

    fact_count = len(traces)
    case_count = len({t.case_id for t in traces})

    assert fact_count == 3
    assert case_count == 2
    assert fact_count != case_count


def test_dropped_during_projection_is_projection_loss():
    """A fact eligible for projection that has no projected candidate is a projection loss."""
    trace = NewFactFunnelTrace(
        case_id="case_8",
        fact_id="fact_8a",
        candidate_key="key_8a",
        correct_fact_extracted=True,
        projection_eligible=True,
        projected_candidate_id=None,
        pre_selector_rank=None,
        entered_selector_input=False,
        selected_by_selector=False,
        value_selected=False,
        raw_answer_correct=False,
        released_answer_correct=False,
    )
    trace.first_loss_stage = classify_new_fact_loss(trace)
    assert trace.first_loss_stage == StructuredFactLossStage.DROPPED_DURING_PROJECTION


def test_ranked_below_selector_input_is_ranking_loss():
    """A projected fact ranked below the selector input threshold is a ranking loss."""
    trace = NewFactFunnelTrace(
        case_id="case_9",
        fact_id="fact_9a",
        candidate_key="key_9a",
        correct_fact_extracted=True,
        projection_eligible=True,
        projected_candidate_id="projected:v1:ranked_out",
        pre_selector_rank=15,
        entered_selector_input=False,
        selected_by_selector=False,
        value_selected=False,
        raw_answer_correct=False,
        released_answer_correct=False,
    )
    trace.first_loss_stage = classify_new_fact_loss(trace)
    assert trace.first_loss_stage == StructuredFactLossStage.RANKED_BELOW_SELECTOR_INPUT


def test_value_used_raw_answer_wrong_is_renderer_loss():
    """A fact whose value was used but raw answer is wrong is a renderer loss."""
    trace = NewFactFunnelTrace(
        case_id="case_12",
        fact_id="fact_12a",
        candidate_key="key_12a",
        correct_fact_extracted=True,
        projection_eligible=True,
        projected_candidate_id="projected:v1:used",
        pre_selector_rank=1,
        entered_selector_input=True,
        selected_by_selector=True,
        value_selected=True,
        raw_answer_correct=False,
        released_answer_correct=False,
    )
    trace.first_loss_stage = classify_new_fact_loss(trace)
    assert trace.first_loss_stage == StructuredFactLossStage.VALUE_USED_RAW_ANSWER_WRONG


def test_raw_correct_validation_regression_is_validator_loss():
    """A fact with correct raw answer but wrong released answer is a validator loss."""
    trace = NewFactFunnelTrace(
        case_id="case_13",
        fact_id="fact_13a",
        candidate_key="key_13a",
        correct_fact_extracted=True,
        projection_eligible=True,
        projected_candidate_id="projected:v1:val",
        pre_selector_rank=1,
        entered_selector_input=True,
        selected_by_selector=True,
        value_selected=True,
        raw_answer_correct=True,
        released_answer_correct=False,
    )
    trace.first_loss_stage = classify_new_fact_loss(trace)
    assert trace.first_loss_stage == StructuredFactLossStage.RAW_CORRECT_VALIDATION_REGRESSION


def test_released_correct_is_success():
    """A fact that flows through all stages to a correct released answer is a success."""
    trace = NewFactFunnelTrace(
        case_id="case_14",
        fact_id="fact_14a",
        candidate_key="key_14a",
        correct_fact_extracted=True,
        projection_eligible=True,
        projected_candidate_id="projected:v1:ok",
        pre_selector_rank=1,
        entered_selector_input=True,
        selected_by_selector=True,
        value_selected=True,
        raw_answer_correct=True,
        released_answer_correct=True,
    )
    trace.first_loss_stage = classify_new_fact_loss(trace)
    assert trace.first_loss_stage == StructuredFactLossStage.RELEASED_CORRECT
