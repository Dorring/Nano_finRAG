"""Tests for the NF38 Ranking Gate decision logic."""
from __future__ import annotations

from src.evaluation.nf38_evaluator import compute_case_diff, evaluate_ranking_gate


def _make_hybrid_metrics(
    case_hit_40: float = 0.5,
    source_recall_40: float = 0.4,
    coverage_40: float = 0.3,
    final_hit_5: float = 0.3,
    final_mrr: float = 0.25,
    dense_hit_40: float = 0.5,
    dense_recall_40: float = 0.4,
) -> dict:
    return {
        "case_hit_rate_at_40": dense_hit_40,
        "source_recall_at_40": dense_recall_40,
        "rrf": {
            "case_hit_rate_at_40": case_hit_40,
            "source_recall_at_40": source_recall_40,
            "all_source_coverage_at_40": coverage_40,
        },
        "final": {
            "case_hit_rate_at_5": final_hit_5,
            "mrr": final_mrr,
        },
    }


def test_gate_passes_when_all_criteria_met():
    baseline = _make_hybrid_metrics(
        case_hit_40=0.5,
        source_recall_40=0.4,
        final_hit_5=0.3,
        final_mrr=0.25,
    )
    variant = _make_hybrid_metrics(
        case_hit_40=0.6,
        source_recall_40=0.5,
        final_hit_5=0.4,
        final_mrr=0.30,
    )
    result = evaluate_ranking_gate(baseline, variant)
    assert result["passed"] is True
    assert result["candidate_improved"] is True
    assert result["no_regression"] is True
    assert result["final_improved"] is True


def test_gate_fails_when_candidate_not_improved():
    baseline = _make_hybrid_metrics(case_hit_40=0.6, source_recall_40=0.5)
    variant = _make_hybrid_metrics(case_hit_40=0.6, source_recall_40=0.5, final_hit_5=0.4)
    result = evaluate_ranking_gate(baseline, variant)
    assert result["passed"] is False
    assert result["candidate_improved"] is False


def test_gate_fails_when_regression_in_source_recall():
    baseline = _make_hybrid_metrics(source_recall_40=0.5, coverage_40=0.4)
    variant = _make_hybrid_metrics(
        source_recall_40=0.4,
        coverage_40=0.4,
        final_hit_5=0.5,
    )
    result = evaluate_ranking_gate(baseline, variant)
    assert result["passed"] is False
    assert result["no_regression"] is False


def test_gate_fails_when_regression_in_coverage():
    baseline = _make_hybrid_metrics(coverage_40=0.5, source_recall_40=0.5)
    variant = _make_hybrid_metrics(
        coverage_40=0.4,
        source_recall_40=0.5,
        final_hit_5=0.5,
    )
    result = evaluate_ranking_gate(baseline, variant)
    assert result["passed"] is False
    assert result["no_regression"] is False


def test_gate_fails_when_final_not_improved():
    baseline = _make_hybrid_metrics(
        case_hit_40=0.5,
        source_recall_40=0.4,
        final_hit_5=0.4,
        final_mrr=0.30,
    )
    variant = _make_hybrid_metrics(
        case_hit_40=0.6,
        source_recall_40=0.5,
        final_hit_5=0.4,
        final_mrr=0.30,
    )
    result = evaluate_ranking_gate(baseline, variant)
    assert result["passed"] is False
    assert result["candidate_improved"] is True
    assert result["no_regression"] is True
    assert result["final_improved"] is False


def test_gate_passes_with_mrr_improvement_only():
    """Final MRR improvement >= 0.02 without Case Hit increase."""
    baseline = _make_hybrid_metrics(
        case_hit_40=0.5,
        source_recall_40=0.4,
        final_hit_5=0.3,
        final_mrr=0.25,
    )
    variant = _make_hybrid_metrics(
        case_hit_40=0.6,
        source_recall_40=0.5,
        final_hit_5=0.3,
        final_mrr=0.28,
    )
    result = evaluate_ranking_gate(baseline, variant)
    assert result["passed"] is True
    assert result["final_improved"] is True


def test_gate_fails_when_mrr_improvement_below_threshold():
    """MRR improvement < 0.02 should not pass."""
    baseline = _make_hybrid_metrics(
        case_hit_40=0.5,
        source_recall_40=0.4,
        final_hit_5=0.3,
        final_mrr=0.25,
    )
    variant = _make_hybrid_metrics(
        case_hit_40=0.6,
        source_recall_40=0.5,
        final_hit_5=0.3,
        final_mrr=0.26,
    )
    result = evaluate_ranking_gate(baseline, variant)
    assert result["passed"] is False
    assert result["final_improved"] is False


def test_gate_dense_recall_improvement_counts():
    """Dense Source Recall@40 improvement counts as candidate improvement."""
    baseline = _make_hybrid_metrics(
        dense_recall_40=0.3,
        case_hit_40=0.5,
        source_recall_40=0.4,
    )
    variant = _make_hybrid_metrics(
        dense_recall_40=0.5,
        case_hit_40=0.5,
        source_recall_40=0.4,
        final_hit_5=0.4,
    )
    result = evaluate_ranking_gate(baseline, variant)
    assert result["candidate_improved"] is True


def test_compute_case_diff_identifies_improvements():
    from src.evaluation.evaluation import EvaluationCase, ExpectedSource

    cases = [EvaluationCase(case_id="c1", question="q", expected_sources=(ExpectedSource(filename="a.pdf", page=1),))]
    baseline = {"c1": [{"candidate_id": "wrong", "evidence_id": "wrong", "document_id": "b.pdf", "page": 2, "block_type": "text", "score": 0.9, "rank": 0}]}
    variant = {"c1": [{"candidate_id": "r1", "evidence_id": "r1", "document_id": "a.pdf", "page": 1, "block_type": "text", "score": 0.9, "rank": 0}]}
    diffs = compute_case_diff(cases, baseline, variant)
    assert len(diffs) == 1
    assert diffs[0]["improved"] is True
    assert diffs[0]["regressed"] is False


def test_compute_case_diff_identifies_regressions():
    from src.evaluation.evaluation import EvaluationCase, ExpectedSource

    cases = [EvaluationCase(case_id="c1", question="q", expected_sources=(ExpectedSource(filename="a.pdf", page=1),))]
    baseline = {"c1": [{"candidate_id": "r1", "evidence_id": "r1", "document_id": "a.pdf", "page": 1, "block_type": "text", "score": 0.9, "rank": 0}]}
    variant = {"c1": [{"candidate_id": "wrong", "evidence_id": "wrong", "document_id": "b.pdf", "page": 2, "block_type": "text", "score": 0.9, "rank": 0}]}
    diffs = compute_case_diff(cases, baseline, variant)
    assert diffs[0]["improved"] is False
    assert diffs[0]["regressed"] is True


def test_compute_case_diff_handles_no_answer_cases():
    from src.evaluation.evaluation import EvaluationCase

    cases = [EvaluationCase(case_id="c1", question="q", expected_no_answer=True)]
    baseline = {"c1": []}
    variant = {"c1": []}
    diffs = compute_case_diff(cases, baseline, variant)
    assert len(diffs) == 0


def test_gate_result_includes_metric_comparison():
    baseline = _make_hybrid_metrics(
        case_hit_40=0.5,
        source_recall_40=0.4,
        final_hit_5=0.3,
        final_mrr=0.25,
        dense_hit_40=0.4,
        dense_recall_40=0.3,
    )
    variant = _make_hybrid_metrics(
        case_hit_40=0.6,
        source_recall_40=0.5,
        final_hit_5=0.4,
        final_mrr=0.30,
        dense_hit_40=0.5,
        dense_recall_40=0.4,
    )
    result = evaluate_ranking_gate(baseline, variant)
    assert result["dense_case_hit_40"]["baseline"] == 0.4
    assert result["dense_case_hit_40"]["variant"] == 0.5
    assert result["rrf_case_hit_40"]["baseline"] == 0.5
    assert result["rrf_case_hit_40"]["variant"] == 0.6
    assert result["final_case_hit_5"]["baseline"] == 0.3
    assert result["final_case_hit_5"]["variant"] == 0.4
    assert result["final_mrr"]["baseline"] == 0.25
    assert result["final_mrr"]["variant"] == 0.30
