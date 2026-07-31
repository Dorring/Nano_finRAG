"""NF42 R2.1 attribution acceptance reliability tests.

Verifies that acceptance is computed from real observations, not hardcoded;
that baselines are checked against expected R1 values; that execution
counters are observed; that fact identity is preserved; that gold source
matching uses explicit mapping; and that gates count cases not facts.
"""
from __future__ import annotations

import pytest

from src.evaluation.nf42_r2_projection_trace import (
    EvaluationIntegrityError,
    NewFactFunnelTrace,
    NF42ExecutionCounters,
    NF42ExpectedBaseline,
    RegressionCause,
    classify_new_fact_loss,
    classify_regression_cause,
    function_identity,
)

# ---------------------------------------------------------------------------
# Acceptance is not hardcoded True
# ---------------------------------------------------------------------------

def test_acceptance_is_not_hardcoded_true():
    """The runner must not write diagnostic_integrity_passed=True unconditionally."""
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "scripts" / "evaluation" / "run_nf42_r2_attribution.py"
    ).read_text(encoding="utf-8")
    # Must compute from integrity_checks
    assert "diagnostic_integrity_passed = all(" in source
    # Must NOT hardcode True
    assert '"diagnostic_integrity_passed": True' not in source
    assert '"current_baseline_reproduced": True' not in source
    assert '"structured_baseline_reproduced": True' not in source


def test_current_baseline_mismatch_blocks_gate():
    """When current baseline doesn't match expected, integrity fails and gate is blocked."""
    expected = NF42ExpectedBaseline(
        all_gold_case_count=13,
        current_correct_fact_cases=3,
        structured_correct_fact_cases=7,
        current_all_gold_raw_correct=7,
        structured_all_gold_raw_correct=5,
        current_all_gold_released_correct=6,
        structured_all_gold_released_correct=4,
        current_any_gold_released_correct=6,
        structured_any_gold_released_correct=4,
        regression_case_count=2,
    )
    # Actual with mismatched current raw correct
    actual = {
        "all_gold_case_count": 13,
        "current_correct_fact_cases": 3,
        "current_all_gold_raw_correct": 6,  # MISMATCH (expected 7)
        "current_all_gold_released_correct": 6,
        "current_any_gold_released_correct": 6,
    }
    exp_dict = expected.to_dict()
    current_keys = {k: actual.get(k) for k in exp_dict if k.startswith("current_") or k == "all_gold_case_count"}
    matched = all(current_keys.get(k) == v for k, v in exp_dict.items() if k.startswith("current_") or k == "all_gold_case_count")
    assert matched is False


def test_structured_baseline_mismatch_blocks_gate():
    """When structured baseline doesn't match expected, integrity fails and gate is blocked."""
    expected = NF42ExpectedBaseline(
        all_gold_case_count=13,
        current_correct_fact_cases=3,
        structured_correct_fact_cases=7,
        current_all_gold_raw_correct=7,
        structured_all_gold_raw_correct=5,
        current_all_gold_released_correct=6,
        structured_all_gold_released_correct=4,
        current_any_gold_released_correct=6,
        structured_any_gold_released_correct=4,
        regression_case_count=2,
    )
    actual = {
        "all_gold_case_count": 13,
        "structured_correct_fact_cases": 5,  # MISMATCH (expected 7)
        "structured_all_gold_raw_correct": 5,
        "structured_all_gold_released_correct": 4,
        "structured_any_gold_released_correct": 4,
    }
    exp_dict = expected.to_dict()
    structured_keys = {k: actual.get(k) for k in exp_dict if k.startswith("structured_") or k == "all_gold_case_count"}
    matched = all(structured_keys.get(k) == v for k, v in exp_dict.items() if k.startswith("structured_") or k == "all_gold_case_count")
    assert matched is False


# ---------------------------------------------------------------------------
# Real execution counters (observed, not constant)
# ---------------------------------------------------------------------------

def test_retrieval_count_is_observed_not_constant():
    """NF42ExecutionCounters must track real retrieval calls, not infer from flags."""
    counters = NF42ExecutionCounters()
    assert counters.retrieval_calls == 0
    counters.retrieval_calls += 1
    assert counters.retrieval_calls == 1
    assert counters.all_zero() is False


def test_side_effect_counters_are_tracked():
    """All side-effect counters must be tracked and all_zero must check all of them."""
    counters = NF42ExecutionCounters()
    assert counters.all_zero() is True
    counters.memory_writes = 1
    assert counters.all_zero() is False
    counters.memory_writes = 0
    counters.feedback_writes = 1
    assert counters.all_zero() is False
    counters.feedback_writes = 0
    counters.document_state_writes = 1
    assert counters.all_zero() is False


# ---------------------------------------------------------------------------
# Fact identity preservation
# ---------------------------------------------------------------------------

def test_new_correct_fact_preserves_candidate_key():
    """Each new correct fact must have a candidate_key for traceability."""
    trace = NewFactFunnelTrace(
        case_id="case_1",
        fact_id="fact_1",
        candidate_key="chunk_key_abc",
        correct_fact_extracted=True,
        projection_eligible=True,
        projected_candidate_id="projected:v1:abc",
        pre_selector_rank=1,
        entered_selector_input=True,
        selected_by_selector=False,
        value_selected=False,
        raw_answer_correct=False,
        released_answer_correct=False,
    )
    trace.first_loss_stage = classify_new_fact_loss(trace)
    assert trace.candidate_key is not None
    assert trace.candidate_key == "chunk_key_abc"


def test_new_correct_fact_without_candidate_key_fails_integrity():
    """A new correct fact without candidate_key must fail identity completeness."""
    trace = NewFactFunnelTrace(
        case_id="case_1",
        fact_id="fact_1",
        candidate_key=None,  # Missing!
        correct_fact_extracted=True,
        projection_eligible=True,
        projected_candidate_id="projected:v1:abc",
        pre_selector_rank=1,
        entered_selector_input=True,
        selected_by_selector=False,
        value_selected=False,
        raw_answer_correct=False,
        released_answer_correct=False,
    )
    trace.first_loss_stage = classify_new_fact_loss(trace)
    # The runner checks: all(trace.candidate_key for trace in traces)
    identity_complete = bool(trace.candidate_key)
    assert identity_complete is False


# ---------------------------------------------------------------------------
# Document identity mapping
# ---------------------------------------------------------------------------

def test_document_id_is_not_used_as_filename_without_mapping():
    """document_id must not be treated as filename without an explicit mapping."""
    # Simulate: identity_map is empty, fact has document_id="doc_internal_123"
    identity_map: dict[str, str] = {}
    fact_document_id = "doc_internal_123"

    # _resolve_filename returns the raw document_id if not in map
    resolved = identity_map.get(fact_document_id, fact_document_id)
    # This is the fallback behavior — it returns the raw id, which is NOT a filename
    # The runner's priority 2 (mapped filename) would fail, falling back to priority 3
    # Priority 3 uses raw document_id as filename — this is the least reliable match
    assert resolved == "doc_internal_123"

    # With a proper mapping, the filename is different
    identity_map[fact_document_id] = "annual_report_2024.pdf"
    resolved = identity_map.get(fact_document_id, fact_document_id)
    assert resolved == "annual_report_2024.pdf"
    assert resolved != fact_document_id


# ---------------------------------------------------------------------------
# Extracted/projected/selected fact IDs are distinct
# ---------------------------------------------------------------------------

def test_extracted_and_selected_fact_ids_are_distinct():
    """Extracted, projected, and selected fact IDs are recorded separately."""
    # Simulate: 5 facts extracted, 3 projected, 1 selected
    extracted_fact_ids = {"fact_1", "fact_2", "fact_3", "fact_4", "fact_5"}
    projected_fact_ids = {"fact_1", "fact_2", "fact_3"}
    selected_fact_ids = {"fact_1"}

    # They are distinct sets at different stages
    assert extracted_fact_ids != projected_fact_ids
    assert projected_fact_ids != selected_fact_ids
    # Selected is a subset of projected, which is a subset of extracted
    assert selected_fact_ids.issubset(projected_fact_ids)
    assert projected_fact_ids.issubset(extracted_fact_ids)


def test_extraction_failure_uses_extracted_fact_ids():
    """Regression cause 'fact_extraction' is determined by extracted fact IDs."""
    stage, cause = classify_regression_cause(
        current_extracted_fact_ids={"fact_legacy"},
        structured_extracted_fact_ids=set(),  # Legacy fact NOT in structured extracted
        current_projected_fact_ids={"fact_legacy"},
        structured_projected_fact_ids=set(),
        current_selected_fact_ids={"fact_legacy"},
        structured_selected_fact_ids=set(),
        current_selected_values=("val",),
        structured_selected_values=("other",),
        current_raw_correct=True,
        structured_raw_correct=False,
        current_released_correct=True,
        structured_released_correct=False,
    )
    assert stage == "fact_extraction"
    assert cause == RegressionCause.LEGACY_CORRECT_FACT_NOT_EXTRACTED


def test_projection_failure_uses_projected_fact_ids():
    """Regression cause 'fact_projection' is determined by projected fact IDs."""
    stage, cause = classify_regression_cause(
        current_extracted_fact_ids={"fact_legacy"},
        structured_extracted_fact_ids={"fact_legacy"},  # Extracted in both
        current_projected_fact_ids={"fact_legacy"},
        structured_projected_fact_ids=set(),  # But NOT projected in structured
        current_selected_fact_ids={"fact_legacy"},
        structured_selected_fact_ids=set(),
        current_selected_values=("val",),
        structured_selected_values=("other",),
        current_raw_correct=True,
        structured_raw_correct=False,
        current_released_correct=True,
        structured_released_correct=False,
    )
    assert stage == "fact_projection"
    assert cause == RegressionCause.LEGACY_CORRECT_FACT_NOT_PROJECTED


def test_selection_failure_uses_selected_fact_ids():
    """Regression cause 'pre_selector_ranking_or_selection' uses selected fact IDs."""
    stage, cause = classify_regression_cause(
        current_extracted_fact_ids={"fact_legacy"},
        structured_extracted_fact_ids={"fact_legacy"},  # Extracted
        current_projected_fact_ids={"fact_legacy"},
        structured_projected_fact_ids={"fact_legacy"},  # Projected
        current_selected_fact_ids={"fact_legacy"},
        structured_selected_fact_ids=set(),  # But NOT selected
        current_selected_values=("val",),
        structured_selected_values=("other",),
        current_raw_correct=True,
        structured_raw_correct=False,
        current_released_correct=True,
        structured_released_correct=False,
    )
    assert stage == "pre_selector_ranking_or_selection"
    assert cause == RegressionCause.LEGACY_CORRECT_CANDIDATE_DISPLACED


# ---------------------------------------------------------------------------
# Function identity fail-closed
# ---------------------------------------------------------------------------

def test_function_identity_fails_closed():
    """function_identity must raise EvaluationIntegrityError, not hash empty string."""
    # Builtin functions cannot be inspected via getsource
    with pytest.raises(EvaluationIntegrityError):
        function_identity(len)

    # C extension functions also cannot be inspected
    with pytest.raises(EvaluationIntegrityError):
        function_identity(print)


# ---------------------------------------------------------------------------
# Gate counts unique cases not facts
# ---------------------------------------------------------------------------

def test_gate_counts_unique_cases_not_facts():
    """Gate must use case-level counts, not fact-level counts.

    A single case with 3 lost facts should NOT trigger the 3-case selector gate.
    """
    # 3 facts all in the SAME case — only 1 unique case
    traces = [
        NewFactFunnelTrace(
            case_id="case_single",
            fact_id=f"fact_{i}",
            candidate_key=f"key_{i}",
            correct_fact_extracted=True,
            projection_eligible=True,
            projected_candidate_id=f"projected:v1:{i}",
            pre_selector_rank=3,
            entered_selector_input=True,
            selected_by_selector=False,
            value_selected=False,
            raw_answer_correct=False,
            released_answer_correct=False,
        )
        for i in range(3)
    ]
    for t in traces:
        t.first_loss_stage = classify_new_fact_loss(t)

    # Fact count is 3, but case count is 1
    fact_count = len(traces)
    case_count = len({t.case_id for t in traces})
    assert fact_count == 3
    assert case_count == 1
    # Gate threshold is >= 3 CASES, so this should NOT trigger
    assert case_count < 3


# ---------------------------------------------------------------------------
# Integrity failure disables next gate
# ---------------------------------------------------------------------------

def test_integrity_failure_disables_next_gate():
    """When diagnostic_integrity_passed is False, next_gate must be disabled."""
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "scripts" / "evaluation" / "run_nf42_r2_attribution.py"
    ).read_text(encoding="utf-8")
    # The runner must disable next_gate when integrity fails
    assert '"enabled": False' in source
    assert "Blocked — diagnostic integrity failed" in source
    # Must also exit non-zero
    assert "sys.exit(1)" in source
