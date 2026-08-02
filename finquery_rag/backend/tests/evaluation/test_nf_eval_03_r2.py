from src.evaluation.nf_eval_03_r2 import (
    BaselineFailureStage,
    canonical_stage_identity,
    classify_final_coverage,
    first_failure_stage,
    infer_reranker_input_source,
    validate_candidate_lineage,
)


def candidate(key: str, document: str = "doc") -> dict:
    return {
        "candidate_key": key,
        "canonical_document_id": document,
        "content_hash": f"hash-{key}",
        "evidence_id": key,
        "page": 1,
    }


def test_candidate_identity_requires_all_stable_fields():
    assert canonical_stage_identity(candidate("a")) is not None
    broken = candidate("a")
    broken["content_hash"] = ""
    assert canonical_stage_identity(broken) is None


def test_rrf_all_is_inferred_from_same_full_order():
    assert infer_reranker_input_source(rrf_keys=["a", "b"], input_keys=["a", "b"], input_limit=20) == "rrf_all"
    assert infer_reranker_input_source(rrf_keys=["a", "b"], input_keys=["a"], input_limit=1) == "rrf_top_n"
    assert infer_reranker_input_source(rrf_keys=["a", "b"], input_keys=["a", "c"], input_limit=20) == "normalized_union"


def test_conservation_allows_only_reranker_reorder_and_final_subset():
    rrf = [candidate("a"), candidate("b"), candidate("c")]
    reranker_input = [candidate("a"), candidate("b"), candidate("c")]
    reranker_output = [candidate("c"), candidate("a")]
    final = [candidate("c")]
    result = validate_candidate_lineage(
        rrf_candidates=rrf,
        reranker_input=reranker_input,
        reranker_output=reranker_output,
        final_candidates=final,
        reranker_input_source="rrf_all",
        reranker_input_limit=20,
    )
    assert result["lineage_integrity_passed"] is True


def test_candidate_injection_fails_lineage():
    result = validate_candidate_lineage(
        rrf_candidates=[candidate("a")],
        reranker_input=[candidate("a"), candidate("b")],
        reranker_output=[candidate("a")],
        final_candidates=[candidate("a")],
        reranker_input_source="rrf_all",
        reranker_input_limit=20,
    )
    assert result["reranker_input_not_in_rrf_count"] == 1
    assert result["lineage_integrity_passed"] is False


def test_partial_context_is_distinct_from_complete_context():
    expected = [{"candidate_key": "a"}, {"candidate_key": "b"}]
    def match(source, item):
        return source["candidate_key"] == item.get("candidate_key")

    assert classify_final_coverage(expected_sources=expected, final_candidates=[candidate("a")], source_matches=match) == "partial_gold_in_final"
    assert classify_final_coverage(expected_sources=expected, final_candidates=[candidate("a"), candidate("b")], source_matches=match) == "all_gold_in_final"


def test_first_failure_stage_is_unique_and_retrieval_precedes_answer():
    assert first_failure_stage(
        gold_identity_valid=True,
        gold_in_rrf=True,
        gold_in_reranker_input=True,
        gold_in_reranker_output=False,
        gold_in_final=False,
        final_partial=False,
        raw_contract_correct=False,
        released_contract_correct=False,
        raw_value_correct=False,
        released_value_correct=False,
        raw_unit_correct=False,
        released_unit_correct=False,
        raw_period_correct=False,
        released_period_correct=False,
        citation_full_recall=False,
        execution_mode="llm_generation",
        requires_calculation=False,
        calculation_route_hit=True,
    ) == BaselineFailureStage.GOLD_DROPPED_BY_RERANKER
