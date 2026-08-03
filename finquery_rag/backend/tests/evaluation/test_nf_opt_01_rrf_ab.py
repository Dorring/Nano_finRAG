from src.evaluation.nf_opt_01 import coverage_state


def test_bm25_and_rrf_stage_inputs_can_be_compared_without_reranker():
    assert coverage_state(["a", "b"], ["a"]) == "partial"
    assert coverage_state(["a", "b"], ["a", "b"]) == "all"


def test_shadow_ab_does_not_require_answer_generation():
    assert coverage_state(["gold"], []) == "none"
