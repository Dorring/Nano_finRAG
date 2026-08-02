from src.evaluation.nf_eval_04 import source_coverage


def test_gold_source_has_unique_coverage_state():
    expected = ["a", "b"]
    assert source_coverage(expected, []) == "none"
    assert source_coverage(expected, ["a"]) == "partial"
    assert source_coverage(expected, ["a", "b"]) == "all"
