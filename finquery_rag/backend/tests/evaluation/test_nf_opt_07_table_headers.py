from src.evaluation.nf_opt_07 import AuditInput, has_headers


def test_year_header_is_not_numeric_value():
    item = AuditInput(
        "key",
        "doc",
        1,
        "hash",
        "| 2025 | 2024 |\n| Revenue | 100 | 90 |",
        {"type": "table"},
        None,
    )
    assert has_headers(item)
