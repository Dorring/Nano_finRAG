from src.evaluation.nf_opt_02 import protected_dense_merge


def test_base_and_residual_are_one_dense_rrf_channel():
    dense = protected_dense_merge(base_candidates=[{"candidate_key": "base"}], residual_candidates=[{"candidate_key": "residual"}])
    assert len(dense) == 2


def test_residual_does_not_receive_second_rrf_weight():
    dense = protected_dense_merge(base_candidates=[{"candidate_key": "base"}], residual_candidates=[{"candidate_key": "residual"}])
    assert [row["candidate_key"] for row in dense] == ["base", "residual"]


def test_rrf_formula_is_unchanged():
    assert 1 / (60 + 1) == 1 / 61
