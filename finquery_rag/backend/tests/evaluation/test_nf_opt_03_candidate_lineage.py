from src.evaluation.nf_opt_03 import lineage_subset_report


def _candidate(key):
    return {"candidate_key": key}


def test_reranker_receives_full_rrf_list():
    items = [_candidate("a"), _candidate("b")]
    result = lineage_subset_report(items, items[:1], items[:1])
    assert result["reranker_input_count"] == 2
    assert result["passed"] is True


def test_reranker_output_is_subset_of_input():
    items = [_candidate("a"), _candidate("b")]
    result = lineage_subset_report(items, [_candidate("z")], [_candidate("z")])
    assert result["reranker_candidate_injection_count"] == 1
    assert result["passed"] is False


def test_final_output_is_subset_of_reranker():
    items = [_candidate("a"), _candidate("b")]
    result = lineage_subset_report(items, items, [_candidate("z")])
    assert result["final_candidate_injection_count"] == 1
    assert result["passed"] is False
