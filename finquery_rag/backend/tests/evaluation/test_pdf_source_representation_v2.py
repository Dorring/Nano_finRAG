from __future__ import annotations

from decimal import Decimal

from src.evaluation.pdf_source_representation_v2 import (
    extract_scale,
    parse_number,
    resolve_period_headers,
    row_label,
    stable_identity,
    statement_from_lines,
)


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
