from src.evaluation.nf_eval_03_r2_1 import (
    GoldStageCoverage,
    R2FailureStage,
    classify_first_failure,
    classify_no_answer_case,
    classify_stage_coverage,
    compare_identity_stability,
)


def candidate(key="a", document="doc", content_hash="hash-a"):
    return {
        "candidate_key": key,
        "canonical_document_id": document,
        "content_hash": content_hash,
        "parent_candidate_key": "parent-a",
        "evidence_id": "evidence-a",
        "page": 1,
    }


def test_partial_rrf_is_not_classified_as_zero_rrf():
    assert classify_stage_coverage(
        expected_source_keys=["a", "b"],
        stage_candidate_keys=["a"],
    ) is GoldStageCoverage.PARTIAL


def test_all_rrf_sources_required_for_all_coverage():
    assert classify_stage_coverage(
        expected_source_keys=["a", "b"],
        stage_candidate_keys=["a", "b", "c"],
    ) is GoldStageCoverage.ALL


def test_partial_final_reaches_gold_partial_in_final():
    result = classify_first_failure(
        gold_identity_valid=True,
        rrf_coverage=GoldStageCoverage.ALL,
        reranker_input_coverage=GoldStageCoverage.ALL,
        reranker_output_coverage=GoldStageCoverage.ALL,
        final_coverage=GoldStageCoverage.PARTIAL,
        raw_contract_correct=False,
        released_contract_correct=False,
        raw_value_correct=False,
        released_value_correct=False,
        raw_unit_correct=False,
        raw_period_correct=False,
        citation_full_recall=False,
        execution_mode="deterministic_fact",
        requires_calculation=False,
        calculation_route_hit=True,
    )
    assert result is R2FailureStage.GOLD_PARTIAL_IN_FINAL


def test_gold_partial_in_final_is_not_unreachable():
    result = classify_first_failure(
        gold_identity_valid=True,
        rrf_coverage="all",
        reranker_input_coverage="all",
        reranker_output_coverage="all",
        final_coverage="partial",
        raw_contract_correct=False,
        released_contract_correct=False,
        raw_value_correct=False,
        released_value_correct=False,
        raw_unit_correct=False,
        raw_period_correct=False,
        citation_full_recall=False,
        execution_mode="llm_generation",
        requires_calculation=False,
        calculation_route_hit=True,
    )
    assert result.value == "gold_partial_in_final"


def test_same_key_changed_content_hash_fails_identity_stability():
    result = compare_identity_stability(
        stages={
            "rrf": [candidate(content_hash="one")],
            "reranker_input": [candidate(content_hash="two")],
            "reranker": [candidate(content_hash="two")],
            "final": [candidate(content_hash="two")],
        }
    )
    assert result["candidate_identity_changed_between_stages"] == 1
    assert result["candidate_identity_stability_passed"] is False


def test_same_key_changed_document_fails_identity_stability():
    result = compare_identity_stability(
        stages={
            "rrf": [candidate(document="doc-1")],
            "reranker_input": [candidate(document="doc-2")],
            "reranker": [candidate(document="doc-2")],
            "final": [candidate(document="doc-2")],
        }
    )
    assert result["candidate_identity_changed_between_stages"] == 1


def test_unchanged_full_identity_passes():
    item = candidate()
    result = compare_identity_stability(
        stages={
            "rrf": [item],
            "reranker_input": [item],
            "reranker": [item],
            "final": [item],
        }
    )
    assert result["candidate_identity_stability_passed"] is True


def test_calculation_route_total_and_first_failure_are_distinct():
    result = classify_first_failure(
        gold_identity_valid=True,
        rrf_coverage="all",
        reranker_input_coverage="all",
        reranker_output_coverage="all",
        final_coverage="all",
        raw_contract_correct=False,
        released_contract_correct=False,
        raw_value_correct=False,
        released_value_correct=False,
        raw_unit_correct=False,
        raw_period_correct=False,
        citation_full_recall=False,
        execution_mode="deterministic_fact",
        requires_calculation=True,
        calculation_route_hit=False,
    )
    assert result is R2FailureStage.CALCULATION_ROUTE_MISSED


def test_no_answer_origin_and_validator_result_are_separate():
    result = classify_no_answer_case(
        {
            "case_id": "x",
            "expected_no_answer": True,
            "answer_execution_mode": "llm_generation",
            "released_answer_contract_correct": False,
            "validation_status": "passed",
        }
    )
    assert result["answer_origin"] == "llm"
    assert result["answerability_result"] == "false_positive"
    assert result["primary_failure"] == "false_answer_llm"
    assert result["validator_release_result"] == "unknown"
