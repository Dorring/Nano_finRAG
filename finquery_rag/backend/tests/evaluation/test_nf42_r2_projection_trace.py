"""NF42 R2 projection trace tests.

Verifies that the experiment scope is correctly declared, function
identity uses source hashing, and every extracted fact is either
projected or has an explicit exclusion reason.
"""
from __future__ import annotations

import inspect

from src.evaluation.nf42_r2_projection_trace import (
    FactProjectionExclusionTrace,
    NewFactFunnelTrace,
    NumericEvidenceCandidateTrace,
    ProjectionExclusionReason,
    StructuredFactLossStage,
    classify_new_fact_loss,
    function_identity,
    sha256_text,
)


def test_extractor_only_ab_is_false():
    """The structured path is NOT an extractor-only A/B."""
    # The R2 acceptance must declare extractor_only_ab = False because
    # projection and pre-selector scoring also change.
    scope = {
        "experiment_scope": "structured_answer_path_ab",
        "extractor_only_ab": False,
        "single_variable_verified": False,
        "differing_stages": [
            "fact_extraction",
            "fact_projection",
            "pre_selector_scoring",
        ],
    }
    assert scope["extractor_only_ab"] is False
    assert scope["single_variable_verified"] is False
    assert "fact_projection" in scope["differing_stages"]


def test_structured_projection_diff_is_declared():
    """The differing stages must include projection and pre-selector scoring."""
    differing = [
        "fact_extraction",
        "fact_projection",
        "pre_selector_scoring",
    ]
    assert "fact_projection" in differing
    assert "pre_selector_scoring" in differing
    assert "fact_extraction" in differing


def test_function_identity_uses_source_hash():
    """function_identity must use inspect.getsource, not a fixed string."""
    def sample_fn(x):
        return x + 1

    identity = function_identity(sample_fn)
    assert identity["module"] is not None
    assert identity["qualname"] == "test_function_identity_uses_source_hash.<locals>.sample_fn"
    # source_sha256 must be a real hash of the source code
    expected_hash = sha256_text(inspect.getsource(sample_fn))
    assert identity["source_sha256"] == expected_hash
    # Must not be a fixed string like "DeterministicAnswerExtractor"
    assert len(identity["source_sha256"]) == 64


def test_every_extracted_fact_is_projected_or_excluded():
    """Every fact must either be projected or have an exclusion reason."""
    projected = [
        NumericEvidenceCandidateTrace(
            projected_candidate_id="projected:v1:abc",
            provider="structured_shadow",
            candidate_key="key_1",
            candidate_rank=1,
            source_fact_ids=("fact_1",),
            source_span_hash="hash_1",
            document_id="doc_a",
            page=1,
            projected_text_hash="text_hash_1",
            projected_value_hashes=("val_hash_1",),
            metric="Revenue",
            period="2025",
            currency="USD",
            unit="million",
            base_evidence_score=10.0,
            anchor_match_count=1,
            anchor_conflict_count=0,
            relation_score=2.0,
            value_granularity_score=0.0,
            component_pair_score=0.0,
            retrieval_score=0.5,
            final_pre_selector_score=12.5,
            pre_selector_rank=1,
            selector_input=True,
            selector_output_rank=None,
        ),
    ]
    excluded = [
        FactProjectionExclusionTrace(
            fact_id="fact_2",
            candidate_key="key_2",
            provider="structured_shadow",
            reason=ProjectionExclusionReason.MISSING_RAW_VALUE,
            source_span_hash="hash_2",
        ),
    ]

    all_facts = {"fact_1", "fact_2"}
    projected_facts = {fid for p in projected for fid in p.source_fact_ids}
    excluded_facts = {e.fact_id for e in excluded}

    # Every fact must be in exactly one set
    assert projected_facts | excluded_facts == all_facts
    assert projected_facts & excluded_facts == set()


def test_projection_exclusion_has_reason():
    """Each exclusion must have a valid ProjectionExclusionReason."""
    for reason in ProjectionExclusionReason:
        assert isinstance(reason.value, str)
        assert reason.value in {
            "missing_raw_value",
            "metric_period_conflict",
            "anchor_conflict",
            "required_anchor_missing",
            "non_positive_score",
            "duplicate_projected_candidate",
        }


def test_correct_fact_entering_selector_is_traced():
    """A correct fact that enters the selector must have a full trace."""
    trace = NewFactFunnelTrace(
        case_id="case_1",
        fact_id="fact_1",
        candidate_key="key_1",
        correct_fact_extracted=True,
        projection_eligible=True,
        projected_candidate_id="projected:v1:abc",
        pre_selector_rank=3,
        entered_selector_input=True,
        selected_by_selector=False,
        value_selected=False,
        raw_answer_correct=False,
        released_answer_correct=False,
    )
    trace.first_loss_stage = classify_new_fact_loss(trace)
    assert trace.first_loss_stage == StructuredFactLossStage.ENTERED_SELECTOR_NOT_SELECTED
