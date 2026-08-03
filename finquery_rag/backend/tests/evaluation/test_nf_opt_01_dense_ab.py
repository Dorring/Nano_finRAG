from src.evaluation.nf_opt_01 import compare_rank_maps, rank_metrics


def test_current_and_shadow_ranks_are_separate():
    current = [{"case_id": "c1", "candidate_key": "a", "rank": 80}]
    shadow = [{"case_id": "c1", "candidate_key": "a", "rank": 37}]
    assert rank_metrics(current, "rank")["@100"]["source_hit_count"] == 1
    assert rank_metrics(shadow, "rank")["@40"]["source_hit_count"] == 1


def test_new_dense_hit_and_regression_are_recorded():
    assert compare_rank_maps({"a": 5}, {"a": 4, "b": 10}, cutoff=20) == {
        "current_hit_count": 1,
        "shadow_hit_count": 2,
        "new_hit_count": 1,
        "regressed_hit_count": 0,
        "both_hit_count": 1,
        "both_missed_count": 0,
    }
    assert compare_rank_maps({"a": 5}, {}, cutoff=20)["regressed_hit_count"] == 1


def test_gold_metrics_keep_missing_sources_in_fixed_denominator():
    rows = [
        {"case_id": "c1", "candidate_key": "a", "rank": 7},
        {"case_id": "c1", "candidate_key": "b", "rank": None},
    ]
    metrics = rank_metrics(rows, "rank")
    assert metrics["source_count"] == 2
    assert metrics["@20"]["source_hit_count"] == 1
