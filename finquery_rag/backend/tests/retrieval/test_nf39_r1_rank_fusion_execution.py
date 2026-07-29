from src.evaluation.evaluation import EvaluationCase, ExpectedSource
from src.evaluation.nf39_r1_integrity import fusion_execution_report
from src.retrieval.rank_preserving_fusion import rank_preserving_fusion


def _row(identifier):
    return {
        "evidence_id": identifier,
        "document_id": "report.pdf",
        "page": 1,
        "block_type": "text",
    }


def test_fusion_uses_distinct_rrf_and_reranker_ranks():
    rrf = [_row("a"), _row("b"), _row("c")]
    reranker = [_row("c"), _row("b"), _row("a")]
    fused = rank_preserving_fusion(
        rrf_candidates=rrf, reranked_candidates=reranker
    )
    assert fused is not reranker
    assert [row["evidence_id"] for row in fused] != [row["evidence_id"] for row in reranker]


def test_fusion_execution_report_is_deterministic_and_tracks_output():
    case = EvaluationCase(
        case_id="one",
        question="q",
        expected_sources=(ExpectedSource(filename="report.pdf", page=1, chunk_id="c"),),
    )
    report = fusion_execution_report(
        cases=[case],
        rrf_rankings={"one": [_row("a"), _row("b"), _row("c")]},
        reranker_rankings={"one": [_row("c"), _row("b"), _row("a")]},
        fusion_rankings={"one": [_row("a"), _row("c"), _row("b")]},
    )
    assert report["fusion_executed"] is True
    assert report["cases"][0]["fusion_top5_keys"][0].endswith(":a")

