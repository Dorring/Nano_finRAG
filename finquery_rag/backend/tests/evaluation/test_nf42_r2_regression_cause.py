"""NF42 R2 regression root-cause attribution tests.

Verifies that regression cases have a first divergence stage and that
the regression cause is inferred from trace data, not hardcoded by case ID.
"""

from __future__ import annotations

from src.evaluation.nf42_r2_projection_trace import (
    RegressionCaseTrace,
    RegressionCause,
    classify_regression_cause,
)


def test_two_regression_cases_have_first_divergence():
    """Two regression cases must each have a first_divergence_stage."""
    # Case A: legacy correct fact not extracted in structured path
    stage_a, cause_a = classify_regression_cause(
        current_supporting_gold_fact_keys={"key_a"},
        structured_extracted_semantic_keys=set(),  # Legacy key NOT in structured extracted
        structured_projected_semantic_keys=set(),
        structured_selected_semantic_keys=set(),
        structured_value_semantic_keys=set(),
        current_raw_correct=True,
        structured_raw_correct=False,
        current_released_correct=True,
        structured_released_correct=False,
    )
    assert stage_a != "unclassified"
    assert cause_a != RegressionCause.UNCLASSIFIED

    # Case B: supporting key survives all structured stages but raw answer wrong
    stage_b, cause_b = classify_regression_cause(
        current_supporting_gold_fact_keys={"key_b"},
        structured_extracted_semantic_keys={"key_b"},  # Extracted
        structured_projected_semantic_keys={"key_b"},  # Projected
        structured_selected_semantic_keys={"key_b"},  # Selected
        structured_value_semantic_keys={"key_b"},  # In value set
        current_raw_correct=True,
        structured_raw_correct=False,  # But raw answer wrong
        current_released_correct=True,
        structured_released_correct=False,
    )
    assert stage_b != "unclassified"
    assert cause_b != RegressionCause.UNCLASSIFIED

    # Both regression traces must have a first_divergence_stage
    reg_a = RegressionCaseTrace(
        case_id="case_a",
        current_supporting_gold_fact_keys=["key_a"],
        current_selected_values_hash=["hash_a"],
        current_raw_correct=True,
        current_released_correct=True,
        structured_extracted_semantic_keys=[],
        structured_projected_semantic_keys=[],
        structured_selected_semantic_keys=[],
        structured_value_semantic_keys=[],
        structured_selected_values_hash=["hash_b"],
        structured_raw_correct=False,
        structured_released_correct=False,
        first_divergence_stage=stage_a,
        regression_cause=cause_a,
    )
    reg_b = RegressionCaseTrace(
        case_id="case_b",
        current_supporting_gold_fact_keys=["key_b"],
        current_selected_values_hash=["hash_x"],
        current_raw_correct=True,
        current_released_correct=True,
        structured_extracted_semantic_keys=["key_b"],
        structured_projected_semantic_keys=["key_b"],
        structured_selected_semantic_keys=["key_b"],
        structured_value_semantic_keys=["key_b"],
        structured_selected_values_hash=["hash_y"],
        structured_raw_correct=False,
        structured_released_correct=False,
        first_divergence_stage=stage_b,
        regression_cause=cause_b,
    )
    assert reg_a.first_divergence_stage
    assert reg_b.first_divergence_stage


def test_regression_cause_is_not_case_specific():
    """The regression cause must be derived from trace data, not case ID.

    Two cases with identical trace data but different case IDs must
    produce the same regression cause.
    """
    # Same trace data (case_id is not even a parameter)
    stage_1, cause_1 = classify_regression_cause(
        current_supporting_gold_fact_keys={"key_legacy"},
        structured_extracted_semantic_keys=set(),  # Legacy key NOT extracted
        structured_projected_semantic_keys=set(),
        structured_selected_semantic_keys=set(),
        structured_value_semantic_keys=set(),
        current_raw_correct=True,
        structured_raw_correct=False,
        current_released_correct=True,
        structured_released_correct=False,
    )
    stage_2, cause_2 = classify_regression_cause(
        current_supporting_gold_fact_keys={"key_legacy"},
        structured_extracted_semantic_keys=set(),  # Legacy key NOT extracted
        structured_projected_semantic_keys=set(),
        structured_selected_semantic_keys=set(),
        structured_value_semantic_keys=set(),
        current_raw_correct=True,
        structured_raw_correct=False,
        current_released_correct=True,
        structured_released_correct=False,
    )

    assert stage_1 == stage_2
    assert cause_1 == cause_2


def test_regression_cause_validation_only():
    """A case where raw is correct but released regressed is a validation-only regression."""
    stage, cause = classify_regression_cause(
        current_supporting_gold_fact_keys={"key_1"},
        structured_extracted_semantic_keys={"key_1"},  # Survives all stages
        structured_projected_semantic_keys={"key_1"},
        structured_selected_semantic_keys={"key_1"},
        structured_value_semantic_keys={"key_1"},
        current_raw_correct=True,
        structured_raw_correct=True,  # Both raw correct
        current_released_correct=True,
        structured_released_correct=False,  # But structured released wrong
    )
    assert cause == RegressionCause.VALIDATION_ONLY_REGRESSION
    assert stage == "validation"


def test_regression_trace_serializes_to_dict():
    """RegressionCaseTrace must serialize correctly without exposing full text."""
    trace = RegressionCaseTrace(
        case_id="case_z",
        current_supporting_gold_fact_keys=["key_1"],
        current_selected_values_hash=["hash_a"],
        current_raw_correct=True,
        current_released_correct=True,
        structured_extracted_semantic_keys=["key_2"],
        structured_projected_semantic_keys=["key_2"],
        structured_selected_semantic_keys=[],
        structured_value_semantic_keys=[],
        structured_selected_values_hash=["hash_b"],
        structured_raw_correct=False,
        structured_released_correct=False,
        first_divergence_stage="pre_selector_ranking_or_selection",
        regression_cause=RegressionCause.LEGACY_CORRECT_CANDIDATE_DISPLACED,
    )
    data = trace.to_dict()
    assert data["case_id"] == "case_z"
    assert data["first_divergence_stage"] == "pre_selector_ranking_or_selection"
    assert data["regression_cause"] == "legacy_correct_candidate_displaced"
    assert "current" in data
    assert "structured" in data
    assert "supporting_gold_fact_keys" in data["current"]
    assert "selected_values_hash" in data["current"]
    assert "extracted_semantic_keys" in data["structured"]
    assert "projected_semantic_keys" in data["structured"]
    # Must not contain full source text
    assert "source_text" not in data
    assert "evaluation_text" not in data
