from src.retrieval_v3.query_features import extract_metric_phrases, extract_periods


def test_preserves_metric_qualifiers() -> None:
    periods, _ = extract_periods("What was segment operating income in FY2025?")
    metrics = extract_metric_phrases("What was segment operating income in FY2025?", periods)
    assert metrics[0].normalized_text == "segment operating income"


def test_splits_explicit_both_comparison_only() -> None:
    periods, _ = extract_periods("Report both revenue and operating income in FY2025")
    metrics = extract_metric_phrases("Report both revenue and operating income in FY2025", periods)
    assert len(metrics) == 2
