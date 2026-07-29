from src.evaluation.evaluation import EvaluationCase, ExpectedSource
from src.evaluation.nf39_r1_integrity import stage_metrics_same_k


def _case(case_id="one", sources=(), no_answer=False):
    return EvaluationCase(
        case_id=case_id,
        question="q",
        expected_sources=tuple(sources),
        expected_no_answer=no_answer,
    )


def _row(identifier, page=1):
    return {
        "evidence_id": identifier,
        "document_id": "report.pdf",
        "page": page,
        "block_type": "text",
    }


def test_rrf_and_reranker_metrics_use_same_k_and_denominator():
    source = ExpectedSource(filename="report.pdf", page=1, chunk_id="gold")
    cases = [_case(sources=(source,)), _case("na", no_answer=True)]
    rrf = {"one": [_row("gold")], "na": []}
    reranker = {"one": [_row("gold")], "na": []}
    left = stage_metrics_same_k(cases=cases, rankings=rrf, ks=(5, 20))
    right = stage_metrics_same_k(cases=cases, rankings=reranker, ks=(5, 20))
    assert left["denominators"] == right["denominators"]
    assert left["denominators"]["retrieval_case_count"] == 1
    assert left["case_hit_rate_at_5"] == right["case_hit_rate_at_5"] == 1.0
    assert left["mrr_at_20"] == 1.0


def test_source_level_metrics_handle_multi_source_case():
    sources = (
        ExpectedSource(filename="report.pdf", page=1, chunk_id="one"),
        ExpectedSource(filename="report.pdf", page=2, chunk_id="two"),
    )
    metric = stage_metrics_same_k(
        cases=[_case(sources=sources)],
        rankings={"one": [_row("one"), _row("two", 2)]},
        ks=(5,),
    )
    assert metric["source_recall_at_5"] == 1.0
    assert metric["all_source_coverage_at_5"] == 1.0

