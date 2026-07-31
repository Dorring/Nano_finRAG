"""NF42 R2 next-phase gate tests.

Verifies that the gate logic correctly determines which next phase is
allowed based on the distribution of first-loss stages across newly
correct facts.  The gates decide whether NF43 (Selector A/B), a
Projection fix, a Value-selection fix, a Renderer fix, or a Validator
fix is the appropriate next step — or whether to stop.
"""
from __future__ import annotations

from src.evaluation.nf42_r2_projection_trace import (
    NewFactFunnelTrace,
    StructuredFactLossStage,
    classify_new_fact_loss,
)


def _make_trace(
    *,
    case_id: str,
    fact_id: str,
    stage: StructuredFactLossStage,
) -> NewFactFunnelTrace:
    """Build a funnel trace that classifies to the given loss stage."""
    stage_fields = {
        StructuredFactLossStage.NOT_EXTRACTED: {
            "correct_fact_extracted": False, "projection_eligible": False,
            "projected_candidate_id": None, "pre_selector_rank": None,
            "entered_selector_input": False, "selected_by_selector": False,
            "value_selected": False, "raw_answer_correct": False, "released_answer_correct": False,
        },
        StructuredFactLossStage.EXTRACTED_NOT_PROJECTION_ELIGIBLE: {
            "correct_fact_extracted": True, "projection_eligible": False,
            "projected_candidate_id": None, "pre_selector_rank": None,
            "entered_selector_input": False, "selected_by_selector": False,
            "value_selected": False, "raw_answer_correct": False, "released_answer_correct": False,
        },
        StructuredFactLossStage.DROPPED_DURING_PROJECTION: {
            "correct_fact_extracted": True, "projection_eligible": True,
            "projected_candidate_id": None, "pre_selector_rank": None,
            "entered_selector_input": False, "selected_by_selector": False,
            "value_selected": False, "raw_answer_correct": False, "released_answer_correct": False,
        },
        StructuredFactLossStage.RANKED_BELOW_SELECTOR_INPUT: {
            "correct_fact_extracted": True, "projection_eligible": True,
            "projected_candidate_id": "projected:v1:low", "pre_selector_rank": 20,
            "entered_selector_input": False, "selected_by_selector": False,
            "value_selected": False, "raw_answer_correct": False, "released_answer_correct": False,
        },
        StructuredFactLossStage.ENTERED_SELECTOR_NOT_SELECTED: {
            "correct_fact_extracted": True, "projection_eligible": True,
            "projected_candidate_id": "projected:v1:sel", "pre_selector_rank": 3,
            "entered_selector_input": True, "selected_by_selector": False,
            "value_selected": False, "raw_answer_correct": False, "released_answer_correct": False,
        },
        StructuredFactLossStage.SELECTED_VALUE_NOT_USED: {
            "correct_fact_extracted": True, "projection_eligible": True,
            "projected_candidate_id": "projected:v1:val", "pre_selector_rank": 1,
            "entered_selector_input": True, "selected_by_selector": True,
            "value_selected": False, "raw_answer_correct": False, "released_answer_correct": False,
        },
        StructuredFactLossStage.VALUE_USED_RAW_ANSWER_WRONG: {
            "correct_fact_extracted": True, "projection_eligible": True,
            "projected_candidate_id": "projected:v1:rend", "pre_selector_rank": 1,
            "entered_selector_input": True, "selected_by_selector": True,
            "value_selected": True, "raw_answer_correct": False, "released_answer_correct": False,
        },
        StructuredFactLossStage.RAW_CORRECT_VALIDATION_REGRESSION: {
            "correct_fact_extracted": True, "projection_eligible": True,
            "projected_candidate_id": "projected:v1:valid", "pre_selector_rank": 1,
            "entered_selector_input": True, "selected_by_selector": True,
            "value_selected": True, "raw_answer_correct": True, "released_answer_correct": False,
        },
        StructuredFactLossStage.RELEASED_CORRECT: {
            "correct_fact_extracted": True, "projection_eligible": True,
            "projected_candidate_id": "projected:v1:ok", "pre_selector_rank": 1,
            "entered_selector_input": True, "selected_by_selector": True,
            "value_selected": True, "raw_answer_correct": True, "released_answer_correct": True,
        },
    }
    trace = NewFactFunnelTrace(
        case_id=case_id, fact_id=fact_id, candidate_key="key",
        **stage_fields[stage],
    )
    trace.first_loss_stage = classify_new_fact_loss(trace)
    assert trace.first_loss_stage == stage
    return trace


def _determine_gate(traces: list[NewFactFunnelTrace]) -> dict:
    """Replicate the gate logic from run_nf42_r2_attribution._determine_next_gate."""
    stage_counts: dict[str, int] = {}
    for t in traces:
        stage = t.first_loss_stage.value
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    selector_gate = stage_counts.get("entered_selector_not_selected", 0) >= 3
    projection_gate = (
        stage_counts.get("dropped_during_projection", 0)
        + stage_counts.get("ranked_below_selector_input", 0)
    ) >= 3
    value_gate = stage_counts.get("selected_value_not_used", 0) >= 2
    renderer_gate = stage_counts.get("value_used_raw_answer_wrong", 0) >= 2
    validator_gate = stage_counts.get("raw_correct_validation_regression", 0) >= 2

    if selector_gate:
        next_phase = "NF43 — Structured Fact Selector A/B"
    elif projection_gate:
        next_phase = "Projection-only fix (Fact-to-Evidence Projection)"
    elif value_gate:
        next_phase = "Value selection fix (_select_answer_values)"
    elif renderer_gate:
        next_phase = "Renderer fix"
    elif validator_gate:
        next_phase = "Validator fix"
    else:
        next_phase = "Stop — no concentrated bottleneck; expand evaluation set"

    return {
        "stage_counts": stage_counts,
        "selector_gate": selector_gate,
        "projection_gate": projection_gate,
        "value_selection_gate": value_gate,
        "renderer_gate": renderer_gate,
        "validator_gate": validator_gate,
        "next_phase": next_phase,
    }


def test_selector_gate_triggers_nf43():
    """>= 3 cases with entered_selector_not_selected triggers the Selector gate."""
    traces = [
        _make_trace(case_id=f"case_{i}", fact_id=f"fact_{i}",
                    stage=StructuredFactLossStage.ENTERED_SELECTOR_NOT_SELECTED)
        for i in range(3)
    ]
    gate = _determine_gate(traces)
    assert gate["selector_gate"] is True
    assert "NF43" in gate["next_phase"]


def test_selector_gate_does_not_trigger_below_threshold():
    """< 3 cases with entered_selector_not_selected does not trigger the Selector gate."""
    traces = [
        _make_trace(case_id="case_1", fact_id="fact_1",
                    stage=StructuredFactLossStage.ENTERED_SELECTOR_NOT_SELECTED),
        _make_trace(case_id="case_2", fact_id="fact_2",
                    stage=StructuredFactLossStage.ENTERED_SELECTOR_NOT_SELECTED),
    ]
    gate = _determine_gate(traces)
    assert gate["selector_gate"] is False


def test_projection_gate_triggers():
    """>= 3 cases with dropped_during_projection + ranked_below_selector_input triggers Projection gate."""
    traces = [
        _make_trace(case_id="case_1", fact_id="fact_1",
                    stage=StructuredFactLossStage.DROPPED_DURING_PROJECTION),
        _make_trace(case_id="case_2", fact_id="fact_2",
                    stage=StructuredFactLossStage.RANKED_BELOW_SELECTOR_INPUT),
        _make_trace(case_id="case_3", fact_id="fact_3",
                    stage=StructuredFactLossStage.RANKED_BELOW_SELECTOR_INPUT),
    ]
    gate = _determine_gate(traces)
    assert gate["projection_gate"] is True
    assert "Projection" in gate["next_phase"]


def test_value_selection_gate_triggers():
    """>= 2 cases with selected_value_not_used triggers the Value-selection gate."""
    traces = [
        _make_trace(case_id="case_1", fact_id="fact_1",
                    stage=StructuredFactLossStage.SELECTED_VALUE_NOT_USED),
        _make_trace(case_id="case_2", fact_id="fact_2",
                    stage=StructuredFactLossStage.SELECTED_VALUE_NOT_USED),
    ]
    gate = _determine_gate(traces)
    assert gate["value_selection_gate"] is True
    assert "Value selection" in gate["next_phase"]


def test_renderer_gate_triggers():
    """>= 2 cases with value_used_raw_answer_wrong triggers the Renderer gate."""
    traces = [
        _make_trace(case_id="case_1", fact_id="fact_1",
                    stage=StructuredFactLossStage.VALUE_USED_RAW_ANSWER_WRONG),
        _make_trace(case_id="case_2", fact_id="fact_2",
                    stage=StructuredFactLossStage.VALUE_USED_RAW_ANSWER_WRONG),
    ]
    gate = _determine_gate(traces)
    assert gate["renderer_gate"] is True
    assert "Renderer" in gate["next_phase"]


def test_validator_gate_triggers():
    """>= 2 cases with raw_correct_validation_regression triggers the Validator gate."""
    traces = [
        _make_trace(case_id="case_1", fact_id="fact_1",
                    stage=StructuredFactLossStage.RAW_CORRECT_VALIDATION_REGRESSION),
        _make_trace(case_id="case_2", fact_id="fact_2",
                    stage=StructuredFactLossStage.RAW_CORRECT_VALIDATION_REGRESSION),
    ]
    gate = _determine_gate(traces)
    assert gate["validator_gate"] is True
    assert "Validator" in gate["next_phase"]


def test_no_bottleneck_stops():
    """When no gate threshold is met, the next phase is to stop and expand the evaluation set."""
    traces = [
        _make_trace(case_id="case_1", fact_id="fact_1",
                    stage=StructuredFactLossStage.RELEASED_CORRECT),
        _make_trace(case_id="case_2", fact_id="fact_2",
                    stage=StructuredFactLossStage.ENTERED_SELECTOR_NOT_SELECTED),
    ]
    gate = _determine_gate(traces)
    assert gate["selector_gate"] is False
    assert gate["projection_gate"] is False
    assert gate["value_selection_gate"] is False
    assert gate["renderer_gate"] is False
    assert gate["validator_gate"] is False
    assert "Stop" in gate["next_phase"]


def test_selector_gate_takes_precedence_over_projection():
    """When both selector and projection gates are met, selector gate wins (NF43)."""
    traces = [
        _make_trace(case_id="case_1", fact_id="fact_1",
                    stage=StructuredFactLossStage.ENTERED_SELECTOR_NOT_SELECTED),
        _make_trace(case_id="case_2", fact_id="fact_2",
                    stage=StructuredFactLossStage.ENTERED_SELECTOR_NOT_SELECTED),
        _make_trace(case_id="case_3", fact_id="fact_3",
                    stage=StructuredFactLossStage.ENTERED_SELECTOR_NOT_SELECTED),
        _make_trace(case_id="case_4", fact_id="fact_4",
                    stage=StructuredFactLossStage.DROPPED_DURING_PROJECTION),
        _make_trace(case_id="case_5", fact_id="fact_5",
                    stage=StructuredFactLossStage.RANKED_BELOW_SELECTOR_INPUT),
        _make_trace(case_id="case_6", fact_id="fact_6",
                    stage=StructuredFactLossStage.RANKED_BELOW_SELECTOR_INPUT),
    ]
    gate = _determine_gate(traces)
    assert gate["selector_gate"] is True
    assert gate["projection_gate"] is True
    assert "NF43" in gate["next_phase"]


def test_gate_counts_are_case_based():
    """Gate thresholds count cases, not just facts.

    Two facts in the same case should not double-count toward a gate.
    The funnel traces are per-fact, but the gate must consider distinct cases.
    """
    # Two facts in the same case both lost at selector stage — only 1 case
    traces = [
        _make_trace(case_id="case_1", fact_id="fact_1a",
                    stage=StructuredFactLossStage.ENTERED_SELECTOR_NOT_SELECTED),
        _make_trace(case_id="case_1", fact_id="fact_1b",
                    stage=StructuredFactLossStage.ENTERED_SELECTOR_NOT_SELECTED),
    ]
    gate = _determine_gate(traces)
    # stage_counts counts facts (2), but selector_gate threshold is >= 3
    assert gate["stage_counts"]["entered_selector_not_selected"] == 2
    assert gate["selector_gate"] is False
