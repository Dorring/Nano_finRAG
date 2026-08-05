from __future__ import annotations

from decimal import Decimal

from src.evaluation.pdf_source_representation_v2 import (
    extract_scale,
    parse_number,
    resolve_period_headers,
    resolve_lineage,
    resolve_period_headers_v2,
    row_label,
    stable_identity,
    statement_from_lines,
)


def test_multilevel_year_rows_are_not_resolved_by_baseline() -> None:
    matrix = [
        ["Years ended January 31", "", ""],
        ["", "2026", ""],
        ["", "", "2025"],
    ]

    assert resolve_period_headers(matrix, 3) == [None, None, None]


def test_single_period_single_value_remains_explicit_r1_gap() -> None:
    matrix = [["Year ended 2025", ""], ["Revenue", "100"]]

    assert resolve_period_headers(matrix, 2) == [None, None]


def test_v2_resolves_years_split_across_header_rows() -> None:
    matrix = [
        ["Years ended January 31", "", ""],
        ["", "2026", ""],
        ["", "", "2025"],
        ["Revenue", "100", "90"],
    ]

    resolved = resolve_period_headers_v2(matrix, 3)

    assert resolved[1]["normalized_period"] == "FY2026"
    assert resolved[2]["normalized_period"] == "FY2025"
    assert resolved[1]["period_kind"] == "duration"


def test_v2_allows_single_period_only_for_one_numeric_column() -> None:
    one_value = [["Year ended 2025", ""], ["Revenue", "100"]]
    multiple_values = [["Year ended 2025", "", ""], ["Revenue", "100", "90"]]

    assert resolve_period_headers_v2(one_value, 2)[1]["normalized_period"] == "FY2025"
    assert resolve_period_headers_v2(multiple_values, 3) == [None, None, None]


def test_lineage_classifies_statement_and_note_without_company_rules() -> None:
    statement = resolve_lineage(["Consolidated Statements of Comprehensive Income"])
    note = resolve_lineage(["Note 8 — Revenue"])

    assert statement["lineage_type"] == "primary_financial_statement"
    assert statement["statement_type"] == "income_statement"
    assert note["lineage_type"] == "note_section"


def test_v2_period_keeps_duration_semantics() -> None:
    matrix = [
        ["Three Months Ended", "2025", "2024"],
        ["Revenue", "100", "90"],
    ]

    resolved = resolve_period_headers_v2(matrix, 3)

    assert resolved[1]["normalized_period"] == "FY2025"
    assert resolved[1]["period_kind"] == "duration"


def test_header_grid_resolves_explicit_period_columns() -> None:
    matrix = [["", "2025", "2024", "2023"], ["Revenue", "100", "90", "80"]]
    assert resolve_period_headers(matrix, 4) == [None, "FY2025", "FY2024", "FY2023"]


def test_single_year_does_not_propagate_across_business_columns() -> None:
    assert resolve_period_headers([["Year ended 2025", "", "", ""]], 4) == [None, None, None, None]


def test_row_and_scale_and_statement_are_deterministic() -> None:
    assert row_label(["Total net sales", "$", "100", "$", "90"]) == "Total net sales"
    assert extract_scale("Consolidated Statements ($ in millions)") == ("$ in millions", "million")
    assert statement_from_lines(["Note 1", "Consolidated Statements of Operations"]) == "Consolidated Statements of Operations"


def test_numeric_parser_and_identity_fail_closed() -> None:
    assert parse_number("(1,234)") == Decimal("-1234")
    assert parse_number("1,234 estimated") is None
    assert stable_identity("row", "doc", 1) == stable_identity("row", "doc", 1)
