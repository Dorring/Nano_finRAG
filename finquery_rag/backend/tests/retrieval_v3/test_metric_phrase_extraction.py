from src.retrieval_v3.query_features import extract_metric_phrases, extract_periods


def test_preserves_metric_qualifiers() -> None:
    periods, _ = extract_periods("What was segment operating income in FY2025?")
    metrics = extract_metric_phrases("What was segment operating income in FY2025?", periods)
    assert metrics[0].normalized_text == "segment operating income"


def test_splits_explicit_both_comparison_only() -> None:
    periods, _ = extract_periods("Report both revenue and operating income in FY2025")
    metrics = extract_metric_phrases("Report both revenue and operating income in FY2025", periods)
    assert len(metrics) == 2


def test_consumes_iso_filing_date_and_source_wrappers() -> None:
    question = "What does the AMZN filing for 2023-12-31 report for the 'Uncertain Tax Positions' row?"
    periods, _ = extract_periods(question)
    metrics = extract_metric_phrases(question, periods)

    assert [(item.raw_text, item.normalized_period) for item in periods] == [("2023-12-31", "FY2023")]
    assert [item.normalized_text for item in metrics] == ["uncertain tax positions"]


def test_comparison_keeps_fy_periods_and_drops_query_noise() -> None:
    question = "Compare Apple FY2024 vs FY2023 Revenue."
    periods, _ = extract_periods(question)
    metrics = extract_metric_phrases(question, periods)

    assert [item.normalized_period for item in periods] == ["FY2024", "FY2023"]
    assert [item.normalized_text for item in metrics] == ["revenue"]


def test_quoted_date_row_is_metric_not_query_period() -> None:
    question = "What does the AMZN filing for 2025-12-31 report for the 'December 31, 2025' row?"
    periods, _ = extract_periods(question)
    metrics = extract_metric_phrases(question, periods)

    assert [item.normalized_period for item in periods] == ["FY2025"]
    assert [item.normalized_text for item in metrics] == ["december 31 2025"]


def test_quoted_multi_metric_labels_override_wrapper_prose() -> None:
    question = (
        "Using the reported values for 'Deferred tax assets' and 'Operating leases' "
        "in the filing for 2025-12-31, what is their sum?"
    )
    periods, _ = extract_periods(question)
    metrics = extract_metric_phrases(question, periods)

    assert [item.normalized_text for item in metrics] == ["deferred tax assets", "operating leases"]


def test_possessive_apostrophe_is_not_treated_as_quote() -> None:
    periods, _ = extract_periods("What was the company's revenue in FY2025?")
    metrics = extract_metric_phrases("What was the company's revenue in FY2025?", periods)

    assert [item.normalized_text for item in metrics] == ["revenue"]


def test_metric_prepositions_are_preserved() -> None:
    periods, _ = extract_periods("What was the cost of sales from operations in FY2025?")
    metrics = extract_metric_phrases("What was the cost of sales from operations in FY2025?", periods)

    assert [item.normalized_text for item in metrics] == ["cost of sales from operations"]


def test_issuer_scope_preposition_does_not_leak_into_metric() -> None:
    question = "What was annual revenue for GOOGL in 2019?"
    periods, _ = extract_periods(question)
    metrics = extract_metric_phrases(question, periods)

    assert [item.normalized_text for item in metrics] == ["annual revenue"]


def test_quarter_marker_is_kept_without_in_wrapper() -> None:
    question = "What was standalone quarterly revenue for GOOGL in 2025 Q4?"
    periods, _ = extract_periods(question)
    metrics = extract_metric_phrases(question, periods)

    assert [item.normalized_text for item in metrics] == ["standalone quarterly revenue q4"]
