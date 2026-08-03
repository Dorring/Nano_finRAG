from src.evaluation.nf_opt_02 import base_retention, protected_dense_merge


def test_base_top40_is_preserved_exactly():
    base = [{"candidate_key": "a"}, {"candidate_key": "b"}]
    merged = protected_dense_merge(base_candidates=base, residual_candidates=[{"candidate_key": "c"}])
    assert merged[:2] == base


def test_base_candidate_order_is_unchanged():
    base = [{"candidate_key": "a"}, {"candidate_key": "b"}]
    audit = base_retention(base_candidates=base, protected_candidates=protected_dense_merge(base_candidates=base, residual_candidates=[]))
    assert audit["base_candidate_order_changed_count"] == 0


def test_residual_candidates_are_appended():
    merged = protected_dense_merge(base_candidates=[{"candidate_key": "a"}], residual_candidates=[{"candidate_key": "b"}])
    assert [row["candidate_key"] for row in merged] == ["a", "b"]


def test_duplicate_candidate_is_removed():
    merged = protected_dense_merge(base_candidates=[{"candidate_key": "a"}], residual_candidates=[{"candidate_key": "a"}, {"candidate_key": "b"}])
    assert [row["candidate_key"] for row in merged] == ["a", "b"]
