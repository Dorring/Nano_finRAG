from src.retrieval_v3.query_features import extract_periods


def test_extracts_fiscal_year_and_range() -> None:
    periods, unresolved = extract_periods("Compare fiscal 2024 with FY2025")
    assert [item.normalized_period for item in periods] == ["FY2024", "FY2025"]
    assert unresolved == ()


def test_relative_period_is_fail_closed() -> None:
    periods, unresolved = extract_periods("What was revenue in the current year?")
    assert periods == ()
    assert unresolved == ("relative_period_without_filing_context",)
