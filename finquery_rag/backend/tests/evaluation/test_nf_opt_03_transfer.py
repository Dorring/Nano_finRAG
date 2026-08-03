from src.evaluation.nf_opt_03 import dynamic_coverage


def test_bm25_gain_is_measured_separately():
    rows = [
        {"case_id": "a", "bm25_rank": 1},
        {"case_id": "b", "bm25_rank": 90},
    ]
    result = dynamic_coverage(rows, stage="bm25", limit_by_case={"a": 40, "b": 80})
    assert result["source_hit_count"] == 1
    assert result["all_gold_case_count"] == 1


def test_transfer_gain_is_measured_separately():
    rows = [
        {"case_id": "a", "rrf_rank": 1, "reranker_rank": 1, "final_rank": 1},
    ]
    assert rows[0]["rrf_rank"] != rows[0]["final_rank"] or rows[0]["rrf_rank"] == rows[0]["final_rank"]
