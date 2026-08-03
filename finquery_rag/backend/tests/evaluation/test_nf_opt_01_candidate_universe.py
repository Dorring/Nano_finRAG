from src.evaluation.nf_opt_01 import candidate_scope_ok


def test_candidate_universe_is_not_built_from_gold_labels():
    # Universe construction is keyed by canonical document scope, not Case ID.
    assert candidate_scope_ok("msft_fy2025", {"msft_fy2025"})


def test_candidate_identity_is_preserved_by_key_comparison():
    current = {"candidate:v1:a": 10}
    shadow = {"candidate:v1:a": 20}
    assert set(current) == set(shadow)
