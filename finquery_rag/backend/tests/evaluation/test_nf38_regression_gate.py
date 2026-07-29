"""Tests for NF38's stage-separated ranking gate and case diff artifacts."""
from __future__ import annotations

from src.evaluation.evaluation import EvaluationCase, ExpectedSource
from src.evaluation.nf38_evaluator import compute_case_diff, evaluate_ranking_gate


def _dense(*, hit40: float = 0.92, recall40: float = 0.857) -> dict:
    return {"case_hit_rate_at_40": hit40, "source_recall_at_40": recall40}


def _hybrid(
    *,
    rrf_hit40: float = 0.96,
    rrf_recall40: float = 0.857,
    coverage40: float = 0.555,
    final_hit5: float = 0.60,
    final_mrr: float = 0.360,
) -> dict:
    return {
        "rrf": {
            "case_hit_rate_at_40": rrf_hit40,
            "source_recall_at_40": rrf_recall40,
            "all_source_coverage_at_40": coverage40,
        },
        "final": {
            "case_hit_rate_at_5": final_hit5,
            "source_recall_at_5": final_hit5,
            "all_source_coverage_at_5": 0.0,
            "mrr": final_mrr,
        },
    }


def test_nf38_real_shape_fails_gate_when_only_final_ranking_improves():
    result = evaluate_ranking_gate(
        baseline_dense=_dense(hit40=0.92, recall40=0.857),
        variant_dense=_dense(hit40=0.88, recall40=0.771),
        baseline_hybrid=_hybrid(final_hit5=0.60, final_mrr=0.360),
        variant_hybrid=_hybrid(final_hit5=0.64, final_mrr=0.485),
    )
    assert result["dense_candidate_improved"] is False
    assert result["rrf_candidate_improved"] is False
    assert result["candidate_improved"] is False
    assert result["no_regression"] is True
    assert result["final_improved"] is True
    assert result["passed"] is False


def test_gate_passes_when_candidate_and_final_metrics_improve_without_regression():
    result = evaluate_ranking_gate(
        baseline_dense=_dense(hit40=0.80, recall40=0.70),
        variant_dense=_dense(hit40=0.84, recall40=0.71),
        baseline_hybrid=_hybrid(rrf_recall40=0.70, coverage40=0.40, final_hit5=0.40),
        variant_hybrid=_hybrid(rrf_recall40=0.70, coverage40=0.40, final_hit5=0.44),
    )
    assert result["passed"] is True


def test_gate_fails_when_rrf_source_coverage_regresses():
    result = evaluate_ranking_gate(
        baseline_dense=_dense(hit40=0.80, recall40=0.70),
        variant_dense=_dense(hit40=0.84, recall40=0.71),
        baseline_hybrid=_hybrid(rrf_recall40=0.70, coverage40=0.40, final_hit5=0.40),
        variant_hybrid=_hybrid(rrf_recall40=0.69, coverage40=0.40, final_hit5=0.44),
    )
    assert result["no_regression"] is False
    assert result["passed"] is False


def test_case_diff_uses_supplied_final_rankings_and_retrieval_hit_naming():
    case = EvaluationCase(
        case_id="c1",
        question="q",
        expected_sources=(ExpectedSource(filename="a.pdf", page=1),),
    )
    baseline = {
        "c1": [{"candidate_id": "wrong", "document_id": "b.pdf", "page": 2,
                "block_type": "text", "score": 1.0, "rank": 0}],
    }
    variant = {
        "c1": [{"candidate_id": "right", "document_id": "a.pdf", "page": 1,
                "block_type": "text", "score": 1.0, "rank": 0}],
    }
    result = compute_case_diff([case], baseline, variant, stage="final")
    assert result[0]["stage"] == "final"
    assert result[0]["retrieval_hit_baseline"] is False
    assert result[0]["retrieval_hit_variant"] is True
    assert result[0]["improved"] is True
    assert "golden_pass_baseline" not in result[0]


def test_case_diff_skips_no_answer_cases():
    case = EvaluationCase(case_id="na", question="q", expected_no_answer=True)
    assert compute_case_diff([case], {"na": []}, {"na": []}, stage="final") == []
