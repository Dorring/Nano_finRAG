from src.evaluation.nf_opt_02_r1 import coverage_counts, hit_set, promotion_demotion

def test_transfer_metrics_are_separate_from_rrf_metrics():
    rows = [{"case_id": "c", "candidate_key": "a", "rrf_rank": 30, "reranker_rank": 2}, {"case_id": "c", "candidate_key": "b", "rrf_rank": 2, "reranker_rank": 30}]
    assert coverage_counts(rows, "reranker")["@5"]["source_hit_count"] == 1
    assert hit_set(rows, "reranker", 5) != hit_set(rows, "rrf", 5)

def test_gold_promotion_and_demotion_are_recorded():
    result = promotion_demotion([{"rrf_rank": 10, "reranker_rank": 2}, {"rrf_rank": 2, "reranker_rank": 10}, {"rrf_rank": 3, "reranker_rank": 3}], "reranker")
    assert result == {"gold_promotion_count": 1, "gold_demotion_count": 1, "gold_unchanged_count": 1}

def test_variant_candidate_identity_is_preserved():
    assert hit_set([{"case_id": "c", "candidate_key": "stable", "reranker_rank": 1}], "reranker", 20) == {"c:stable"}
