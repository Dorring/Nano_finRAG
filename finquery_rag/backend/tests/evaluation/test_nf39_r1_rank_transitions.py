import pytest

from src.evaluation.nf39_attribution import EvaluationIntegrityError
from src.evaluation.nf39_r1_integrity import (
    RankTransition,
    classify_rank_transition,
)


def test_rank_transition_definitions_are_same_k_correct():
    assert classify_rank_transition(
        rrf_rank=3, reranker_input_top_n=20, reranker_rank=11, final_rank=None
    ) is RankTransition.DEMOTED_OUT_OF_TOP5
    assert classify_rank_transition(
        rrf_rank=30, reranker_input_top_n=20, reranker_rank=None, final_rank=None
    ) is RankTransition.TRUNCATED_BEFORE_RERANKER
    assert classify_rank_transition(
        rrf_rank=10, reranker_input_top_n=20, reranker_rank=4, final_rank=4
    ) is RankTransition.PROMOTED_INTO_TOP5
    assert classify_rank_transition(
        rrf_rank=30, reranker_input_top_n=20, reranker_rank=10, final_rank=None
    ) is RankTransition.TRUNCATED_BEFORE_RERANKER


def test_reranker_input_without_rank_is_integrity_error():
    with pytest.raises(EvaluationIntegrityError):
        classify_rank_transition(
            rrf_rank=5, reranker_input_top_n=20, reranker_rank=None, final_rank=None
        )

