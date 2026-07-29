"""Tests for NF39 regression gate, trigger threshold, and production safety."""
from __future__ import annotations

import inspect

from src.evaluation.evaluation import EvaluationCase, ExpectedSource
from src.evaluation.nf39_attribution import FinalLossStage
from src.evaluation.nf39_gate import (
    compute_fusion_case_diff,
    evaluate_ranking_gate,
    should_trigger_fusion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_expected(
    chunk_id: str = "ev_1",
    filename: str = "doc_a",
    page: int | None = None,
) -> ExpectedSource:
    return ExpectedSource(filename=filename, page=page, chunk_id=chunk_id)


def _make_case(
    case_id: str = "case_1",
    expected_sources: tuple[ExpectedSource, ...] = (),
    expected_no_answer: bool = False,
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question=f"Question for {case_id}",
        expected_sources=expected_sources,
        expected_answer_contains=(),
        expected_numbers=(),
        expected_no_answer=expected_no_answer,
        expected_calculations=(),
        expected_intent=None,
        document_names=("doc_a",),
        tags=(),
        metadata={},
    )


def _make_candidate(eid: str, doc: str = "doc_a", page: int = 1) -> dict:
    return {
        "evidence_id": eid,
        "candidate_id": eid,
        "document_id": doc,
        "page": page,
        "block_type": "text",
        "parent_id": None,
        "table_id": None,
        "score": 0.5,
    }


def _metrics(
    case_hit_5: float = 0.5,
    source_recall_5: float = 0.5,
    coverage_5: float = 0.5,
    mrr: float = 0.4,
    case_count: int = 10,
) -> dict:
    return {
        "case_hit_rate_at_5": case_hit_5,
        "source_recall_at_5": source_recall_5,
        "all_source_coverage_at_5": coverage_5,
        "mrr": mrr,
        "case_count": case_count,
    }


# ---------------------------------------------------------------------------
# Trigger threshold tests
# ---------------------------------------------------------------------------


def test_trigger_when_reranker_source_recall_below_rrf():
    """Trigger when Reranker Source Recall@5 < RRF Source Recall@5."""
    rrf_metrics = _metrics(source_recall_5=0.6)
    reranker_metrics = _metrics(source_recall_5=0.4)
    attribution = {stage.value: 0 for stage in FinalLossStage}

    trigger, reasons = should_trigger_fusion(
        rrf_metrics=rrf_metrics,
        reranker_metrics=reranker_metrics,
        attribution_summary=attribution,
    )
    assert trigger is True
    assert any("source_recall" in r for r in reasons)


def test_trigger_when_reranker_coverage_below_rrf():
    """Trigger when Reranker All-source Coverage@5 < RRF All-source Coverage@5."""
    rrf_metrics = _metrics(coverage_5=0.6)
    reranker_metrics = _metrics(coverage_5=0.3)
    attribution = {stage.value: 0 for stage in FinalLossStage}

    trigger, reasons = should_trigger_fusion(
        rrf_metrics=rrf_metrics,
        reranker_metrics=reranker_metrics,
        attribution_summary=attribution,
    )
    assert trigger is True
    assert any("coverage" in r for r in reasons)


def test_trigger_when_demoted_by_reranker_ge_2():
    """Trigger when demoted_by_reranker case count >= 2."""
    rrf_metrics = _metrics()
    reranker_metrics = _metrics()
    attribution = {stage.value: 0 for stage in FinalLossStage}
    attribution[FinalLossStage.DEMOTED_BY_RERANKER.value] = 3

    trigger, reasons = should_trigger_fusion(
        rrf_metrics=rrf_metrics,
        reranker_metrics=reranker_metrics,
        attribution_summary=attribution,
    )
    assert trigger is True
    assert any("demoted" in r for r in reasons)


def test_no_trigger_when_all_conditions_met():
    """Do not trigger when reranker is not worse than RRF and demoted < 2."""
    rrf_metrics = _metrics(source_recall_5=0.5, coverage_5=0.5)
    reranker_metrics = _metrics(source_recall_5=0.5, coverage_5=0.5)
    attribution = {stage.value: 0 for stage in FinalLossStage}
    attribution[FinalLossStage.DEMOTED_BY_RERANKER.value] = 1

    trigger, reasons = should_trigger_fusion(
        rrf_metrics=rrf_metrics,
        reranker_metrics=reranker_metrics,
        attribution_summary=attribution,
    )
    assert trigger is False
    assert reasons == []


# ---------------------------------------------------------------------------
# Ranking gate tests
# ---------------------------------------------------------------------------


def test_gate_passes_with_improvement_and_no_regression():
    """Gate passes when metrics improve and no regression."""
    baseline = _metrics(case_hit_5=0.5, mrr=0.4, case_count=10)
    fusion = _metrics(case_hit_5=0.6, mrr=0.5, case_count=10)
    case_diff = {
        "improved_count": 1,
        "regressed_count": 0,
        "unchanged_count": 9,
        "gold_source_promoted": 1,
        "gold_source_demoted": 0,
    }

    gate = evaluate_ranking_gate(
        baseline_metrics=baseline,
        fusion_metrics=fusion,
        case_diff=case_diff,
    )
    assert gate["passed"] is True


def test_gate_fails_with_regression():
    """Gate fails when any case regresses."""
    baseline = _metrics(case_hit_5=0.5, mrr=0.4, case_count=10)
    fusion = _metrics(case_hit_5=0.6, mrr=0.5, case_count=10)
    case_diff = {
        "improved_count": 1,
        "regressed_count": 1,
        "unchanged_count": 8,
        "gold_source_promoted": 1,
        "gold_source_demoted": 1,
    }

    gate = evaluate_ranking_gate(
        baseline_metrics=baseline,
        fusion_metrics=fusion,
        case_diff=case_diff,
    )
    assert gate["passed"] is False
    assert gate["no_regression"] is False


def test_gate_fails_with_no_improvement():
    """Gate fails when metrics don't decline but don't improve either."""
    baseline = _metrics(case_hit_5=0.5, mrr=0.4, case_count=10)
    fusion = _metrics(case_hit_5=0.5, mrr=0.4, case_count=10)
    case_diff = {
        "improved_count": 0,
        "regressed_count": 0,
        "unchanged_count": 10,
        "gold_source_promoted": 0,
        "gold_source_demoted": 0,
    }

    gate = evaluate_ranking_gate(
        baseline_metrics=baseline,
        fusion_metrics=fusion,
        case_diff=case_diff,
    )
    assert gate["passed"] is False
    assert gate["at_least_one_gain"] is False


def test_gate_fails_when_mrr_declines():
    """Gate fails when MRR decreases."""
    baseline = _metrics(case_hit_5=0.5, mrr=0.5, case_count=10)
    fusion = _metrics(case_hit_5=0.6, mrr=0.4, case_count=10)
    case_diff = {
        "improved_count": 1,
        "regressed_count": 0,
        "unchanged_count": 9,
        "gold_source_promoted": 1,
        "gold_source_demoted": 0,
    }

    gate = evaluate_ranking_gate(
        baseline_metrics=baseline,
        fusion_metrics=fusion,
        case_diff=case_diff,
    )
    assert gate["passed"] is False
    assert gate["mrr_ok"] is False


# ---------------------------------------------------------------------------
# Case diff tests
# ---------------------------------------------------------------------------


def test_case_diff_counts_improved_and_regressed():
    """Case diff must correctly count improved and regressed cases."""
    cases = [
        _make_case("case_1", expected_sources=(_make_expected("ev_1"),)),
        _make_case("case_2", expected_sources=(_make_expected("ev_2"),)),
    ]
    baseline = {
        "case_1": [_make_candidate("ev_1")],  # hit
        "case_2": [_make_candidate("ev_2")],  # hit
    }
    fusion = {
        "case_1": [_make_candidate("ev_1")],  # still hit
        "case_2": [_make_candidate("ev_99")],  # miss
    }

    diff = compute_fusion_case_diff(
        cases=cases,
        baseline_rankings=baseline,
        fusion_rankings=fusion,
        top_k=5,
    )
    assert diff["regressed_count"] == 1
    assert diff["improved_count"] == 0
    assert "case_2" in diff["regressed_case_ids"]


def test_case_diff_skips_no_answer_cases():
    """No-answer cases must be skipped in case diff."""
    cases = [
        _make_case("case_1", expected_sources=(_make_expected("ev_1"),)),
        _make_case("case_2", expected_no_answer=True),
    ]
    baseline = {"case_1": [_make_candidate("ev_1")], "case_2": []}
    fusion = {"case_1": [_make_candidate("ev_1")], "case_2": []}

    diff = compute_fusion_case_diff(
        cases=cases,
        baseline_rankings=baseline,
        fusion_rankings=fusion,
        top_k=5,
    )
    # Only case_1 is eligible
    assert diff["improved_count"] + diff["regressed_count"] + diff["unchanged_count"] == 1


# ---------------------------------------------------------------------------
# Production safety tests
# ---------------------------------------------------------------------------


def test_question_and_label_hashes_unchanged():
    """NF39 must use the same fingerprint functions as NF38.

    The question_hash and label_hash must be computed via
    ``case_fingerprints.question_fingerprint`` and
    ``case_fingerprints.label_fingerprint``, not ad-hoc hashing.
    """
    import scripts.evaluation.export_nf39_rrf_pool as export_script

    source = inspect.getsource(export_script)
    assert "question_fingerprint" in source, (
        "NF39 must use case_fingerprints.question_fingerprint"
    )
    assert "label_fingerprint" in source, (
        "NF39 must use case_fingerprints.label_fingerprint"
    )


def test_production_default_is_unchanged_without_gate():
    """The production retrieval pipeline must not import or use rank_preserving_fusion.

    The fusion is an offline experiment only.  Production code must not
    reference it.
    """
    import src.retrieval.retrieval_pipeline as pipeline_mod

    pipeline_source = inspect.getsource(pipeline_mod)
    assert "rank_preserving_fusion" not in pipeline_source, (
        "Production retrieval_pipeline must not import rank_preserving_fusion"
    )
    assert "FINAL_RANKING_MODE" not in pipeline_source, (
        "Production retrieval_pipeline must not reference FINAL_RANKING_MODE"
    )

    import src.services.rag_engine as engine_mod

    engine_source = inspect.getsource(engine_mod)
    assert "rank_preserving_fusion" not in engine_source, (
        "Production rag_engine must not import rank_preserving_fusion"
    )


def test_no_case_specific_logic():
    """NF39 attribution and gate modules must not contain case-specific logic.

    No hardcoded case IDs, question text, or document-specific rules.
    """
    import src.evaluation.nf39_attribution as attr_mod
    import src.evaluation.nf39_gate as gate_mod

    for module in (attr_mod, gate_mod):
        source = inspect.getsource(module)
        # Must not reference specific case IDs
        for case_id in ("case_1", "case_2", "case_3", "nf39_case"):
            assert case_id not in source, (
                f"{module.__name__} must not reference case ID '{case_id}'"
            )
        # Must not reference specific document names
        for doc in ("annual_report", "balance_sheet", "income_statement"):
            assert doc not in source, (
                f"{module.__name__} must not reference document '{doc}'"
            )


def test_no_document_specific_logic():
    """NF39 rank_preserving_fusion must not contain document-specific logic."""
    import src.retrieval.rank_preserving_fusion as fusion_mod

    source = inspect.getsource(fusion_mod)
    for doc in ("annual_report", "balance_sheet", "income_statement", "doc_a"):
        assert doc not in source, (
            f"rank_preserving_fusion must not reference document '{doc}'"
        )


def test_fusion_script_does_not_modify_production_config():
    """The NF39 evaluation script must not modify production configuration."""
    import scripts.evaluation.run_nf39_evaluation as eval_script

    source = inspect.getsource(eval_script)
    # Must not set environment variables
    assert "os.environ" not in source, (
        "Evaluation script must not modify environment variables"
    )
    # Must not write to production config files
    assert "online.env" not in source, (
        "Evaluation script must not modify production config"
    )
