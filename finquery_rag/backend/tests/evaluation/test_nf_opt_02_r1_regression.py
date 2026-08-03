from src.evaluation.nf_opt_02_r1 import hit_set

def test_total_gain_cannot_hide_regression():
    before = hit_set([{"case_id": "a", "candidate_key": "old", "reranker_rank": 1}, {"case_id": "b", "candidate_key": "keep", "reranker_rank": 1}], "reranker", 20)
    after = hit_set([{"case_id": "b", "candidate_key": "keep", "reranker_rank": 1}, {"case_id": "c", "candidate_key": "new", "reranker_rank": 1}], "reranker", 20)
    assert len(after) == len(before)
    assert before - after == {"a:old"}
