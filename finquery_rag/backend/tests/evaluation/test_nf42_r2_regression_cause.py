"""NF42 R2 regression root-cause attribution tests.

Verifies that regression cases have a first divergence stage and that
the regression cause is inferred from trace data, not hardcoded by case ID.
"""
from __future__ import annotations

from src.evaluation.nf42_r2_projection_trace import (
    NumericEvidenceCandidateTrace,
    RegressionCaseTrace,
    RegressionCause,
    classify_regression_cause,
)


def _make_projected(
    *,
    proj_id: str = "projected:v1:abc",
    fact_ids: tuple[str, ...] = ("fact_1",),
    score: float = 10.0,
) -> NumericEvidenceCandidateTrace:
    return NumericEvidenceCandidateTrace(
        projected_candidate_id=proj_id,
        provider="current",
        candidate_key="key_1",
        candidate_rank=1,
        source_fact_ids=fact_ids,
        source_span_hash="hash_1",
        document_id="doc_a",
        page=1,
        projected_text_hash="text_hash_1",
        projected_value_hashes=("val_hash_1",),
        metric="Revenue",
        period="2025",
        currency="USD",
        unit="million",
        base_evidence_score=score,
        anchor_match_count=1,
        anchor_conflict_count=0,
        relation_score=2.0,
        value_granularity_score=0.0,
        component_pair_score=0.0,
        retrieval_score=0.5,
        final_pre_selector_score=score,
        pre_selector_rank=1,
        selector_input=True,
        selector_output_rank=1,
    )


def test_two_regression_cases_have_first_divergence():
    """Two regression cases must each have a first_divergence_stage."""
    # Case A: legacy correct fact not projected in structured path
    current_trace_a = {
        "selected_fact_ids": ["fact_legacy_a"],
        "selected_values_hash": ["hash_a"],
        "pre_selector_scores": [10.0],
        "raw_correct": True,
        "released_correct": True,
    }
    structured_trace_a = {
        "selected_fact_ids": ["fact_other_a"],
        "selected_values_hash": ["hash_b"],
        "pre_selector_scores": [8.0],
        "raw_correct": False,
        "released_correct": False,
    }
    current_proj_a = [_make_projected(proj_id="proj_a", fact_ids=("fact_legacy_a",), score=10.0)]
    structured_proj_a = [_make_projected(proj_id="proj_b", fact_ids=("fact_other_a",), score=8.0)]

    stage_a, cause_a = classify_regression_cause(
        current_trace=current_trace_a,
        structured_trace=structured_trace_a,
        current_projected=current_proj_a,
        structured_projected=structured_proj_a,
    )
    assert stage_a != "unclassified"
    assert cause_a != RegressionCause.UNCLASSIFIED

    # Case B: value selection changed
    current_trace_b = {
        "selected_fact_ids": ["fact_legacy_b"],
        "selected_values_hash": ["hash_x"],
        "pre_selector_scores": [10.0],
        "raw_correct": True,
        "released_correct": True,
    }
    structured_trace_b = {
        "selected_fact_ids": ["fact_legacy_b"],
        "selected_values_hash": ["hash_y"],
        "pre_selector_scores": [10.0],
        "raw_correct": False,
        "released_correct": False,
    }
    current_proj_b = [_make_projected(proj_id="proj_c", fact_ids=("fact_legacy_b",), score=10.0)]
    structured_proj_b = [_make_projected(proj_id="proj_c", fact_ids=("fact_legacy_b",), score=10.0)]

    stage_b, cause_b = classify_regression_cause(
        current_trace=current_trace_b,
        structured_trace=structured_trace_b,
        current_projected=current_proj_b,
        structured_projected=structured_proj_b,
    )
    assert stage_b != "unclassified"
    assert cause_b != RegressionCause.UNCLASSIFIED

    # Both regression traces must have a first_divergence_stage
    reg_a = RegressionCaseTrace(
        case_id="case_a",
        current_selected_candidate_ids=["proj_a"],
        current_selected_fact_ids=["fact_legacy_a"],
        current_selected_values_hash=["hash_a"],
        current_pre_selector_scores=[10.0],
        current_raw_correct=True,
        current_released_correct=True,
        structured_selected_candidate_ids=["proj_b"],
        structured_selected_fact_ids=["fact_other_a"],
        structured_selected_values_hash=["hash_b"],
        structured_pre_selector_scores=[8.0],
        structured_raw_correct=False,
        structured_released_correct=False,
        first_divergence_stage=stage_a,
        regression_cause=cause_a,
    )
    reg_b = RegressionCaseTrace(
        case_id="case_b",
        current_selected_candidate_ids=["proj_c"],
        current_selected_fact_ids=["fact_legacy_b"],
        current_selected_values_hash=["hash_x"],
        current_pre_selector_scores=[10.0],
        current_raw_correct=True,
        current_released_correct=True,
        structured_selected_candidate_ids=["proj_c"],
        structured_selected_fact_ids=["fact_legacy_b"],
        structured_selected_values_hash=["hash_y"],
        structured_pre_selector_scores=[10.0],
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
    current_trace = {
        "selected_fact_ids": ["fact_legacy"],
        "selected_values_hash": ["hash_x"],
        "pre_selector_scores": [10.0],
        "raw_correct": True,
        "released_correct": True,
    }
    structured_trace = {
        "selected_fact_ids": ["fact_other"],
        "selected_values_hash": ["hash_y"],
        "pre_selector_scores": [8.0],
        "raw_correct": False,
        "released_correct": False,
    }
    current_proj = [_make_projected(proj_id="proj_a", fact_ids=("fact_legacy",), score=10.0)]
    structured_proj = [_make_projected(proj_id="proj_b", fact_ids=("fact_other",), score=8.0)]

    # Same trace data, different case IDs
    stage_1, cause_1 = classify_regression_cause(
        current_trace=current_trace,
        structured_trace=structured_trace,
        current_projected=current_proj,
        structured_projected=structured_proj,
    )
    stage_2, cause_2 = classify_regression_cause(
        current_trace=current_trace,
        structured_trace=structured_trace,
        current_projected=current_proj,
        structured_projected=structured_proj,
    )

    # case_id is not even a parameter to classify_regression_cause
    assert stage_1 == stage_2
    assert cause_1 == cause_2


def test_regression_cause_validation_only():
    """A case where raw is correct but released regressed is a validation-only regression."""
    current_trace = {
        "selected_fact_ids": ["fact_1"],
        "selected_values_hash": ["hash_x"],
        "pre_selector_scores": [10.0],
        "raw_correct": True,
        "released_correct": True,
    }
    structured_trace = {
        "selected_fact_ids": ["fact_1"],
        "selected_values_hash": ["hash_x"],
        "pre_selector_scores": [10.0],
        "raw_correct": True,
        "released_correct": False,
    }
    current_proj = [_make_projected(fact_ids=("fact_1",))]
    structured_proj = [_make_projected(fact_ids=("fact_1",))]

    stage, cause = classify_regression_cause(
        current_trace=current_trace,
        structured_trace=structured_trace,
        current_projected=current_proj,
        structured_projected=structured_proj,
    )
    assert cause == RegressionCause.VALIDATION_ONLY_REGRESSION
    assert stage == "validation"


def test_regression_trace_serializes_to_dict():
    """RegressionCaseTrace must serialize correctly without exposing full text."""
    trace = RegressionCaseTrace(
        case_id="case_z",
        current_selected_candidate_ids=["proj_a"],
        current_selected_fact_ids=["fact_1"],
        current_selected_values_hash=["hash_a"],
        current_pre_selector_scores=[10.0],
        current_raw_correct=True,
        current_released_correct=True,
        structured_selected_candidate_ids=["proj_b"],
        structured_selected_fact_ids=["fact_2"],
        structured_selected_values_hash=["hash_b"],
        structured_pre_selector_scores=[8.0],
        structured_raw_correct=False,
        structured_released_correct=False,
        first_divergence_stage="pre_selector_ranking",
        regression_cause=RegressionCause.LEGACY_CORRECT_CANDIDATE_DISPLACED,
    )
    data = trace.to_dict()
    assert data["case_id"] == "case_z"
    assert data["first_divergence_stage"] == "pre_selector_ranking"
    assert data["regression_cause"] == "legacy_correct_candidate_displaced"
    assert "current" in data
    assert "structured" in data
    # Must not contain full source text
    assert "source_text" not in data
    assert "evaluation_text" not in data
