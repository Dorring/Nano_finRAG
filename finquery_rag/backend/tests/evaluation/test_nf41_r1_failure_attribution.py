from types import SimpleNamespace

from src.evaluation.nf41_production_attribution import (
    ProductionFactFailure,
    classify_observed_fact_failure,
)


def _fact(identifier: str, correct: bool):
    return SimpleNamespace(fact_id=identifier, correct=correct)


def test_correct_fact_absent_is_extractor_failure():
    result = classify_observed_fact_failure(
        facts=[_fact("wrong", False)], selected_fact_ids=("wrong",), raw_answer_correct=False,
        fact_matches_gold=lambda fact: fact.correct,
    )
    assert result is ProductionFactFailure.PRODUCTION_FACT_NOT_EXTRACTED


def test_correct_fact_present_wrong_selected_is_selector_failure():
    result = classify_observed_fact_failure(
        facts=[_fact("right", True), _fact("wrong", False)], selected_fact_ids=("wrong",), raw_answer_correct=False,
        fact_matches_gold=lambda fact: fact.correct,
    )
    assert result is ProductionFactFailure.PRODUCTION_FACT_AVAILABLE_NOT_SELECTED


def test_correct_fact_selected_wrong_answer_is_renderer_failure():
    result = classify_observed_fact_failure(
        facts=[_fact("right", True)], selected_fact_ids=("right",), raw_answer_correct=False,
        fact_matches_gold=lambda fact: fact.correct,
    )
    assert result is ProductionFactFailure.PRODUCTION_FACT_SELECTED_RENDER_WRONG


def test_missing_trace_is_not_guessed_from_raw_answer():
    result = classify_observed_fact_failure(
        facts=[], selected_fact_ids=(), raw_answer_correct=False, fact_matches_gold=lambda _fact: True,
    )
    assert result is ProductionFactFailure.PRODUCTION_TRACE_INSUFFICIENT
