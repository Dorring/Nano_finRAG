"""Focused contracts for the NF-V2-18A-R2 HTML physical adapter."""
from src.pdf_retrieval_v4.html_semantic_adapter import (
    _adapt_table,
    _physical_header_grid,
    html_period_semantics,
)


def test_sec_period_vocabulary_is_conservative():
    assert html_period_semantics("Three Months Ended June 30, 2025") == "QUARTER"
    assert html_period_semantics("Six Months Ended June 30, 2025") == "YTD"
    assert html_period_semantics("Nine Months Ended September 30, 2025") == "YTD"
    assert html_period_semantics("Year Ended December 31, 2025") == "ANNUAL"
    assert html_period_semantics("As of June 30, 2025") == "INSTANT"
    assert html_period_semantics("reported period") == "UNKNOWN"


def test_physical_header_grid_expands_spans_before_shared_resolution():
    rows = [
        [{"text": "Duration", "colspan": 2}, {"text": "Point", "rowspan": 2}],
        ["Q1", "Q2"],
    ]
    assert _physical_header_grid(rows) == [
        ["Duration", "Duration", "Point"],
        ["Q1", "Q2", "Point"],
    ]


def test_html_adapter_preserves_a4_ids_and_provenance():
    table = {
        "table_id": "table_demo",
        "table_title": "Revenue",
        "header_rows": [["Three Months Ended", "Three Months Ended"]],
        "column_headers": ["June 30, 2025", "June 30, 2024"],
        "period_columns": [
            {"header_text": "Three Months Ended June 30, 2025", "period_end": "2025-06-30"},
            {"header_text": "Three Months Ended June 30, 2024", "period_end": "2024-06-30"},
        ],
        "rows": [{"row_id": "row_demo", "row_label": "Revenue"}],
        "cells": [
            {
                "cell_id": "cell_demo",
                "row_id": "row_demo",
                "column_index": 0,
                "raw_value": "1,234",
                "normalized_value": 1234,
                "period_end": "2025-06-30",
            }
        ],
        "section_type": "INCOME_STATEMENT",
    }
    out = _adapt_table(
        table,
        {
            "document_id": "SEC_DEMO",
            "accession_number": "0000000000-00-000000",
            "source_raw_sha256": "raw-sha",
            "fiscal_year": 2025,
        },
    )
    cell = out["cells"][0]
    assert out["table_fragment_id"] == "table_demo"
    assert out["rows"][0]["row_id"] == "row_demo"
    assert cell["cell_id"] == "cell_demo"
    assert cell["period_semantics"] == "QUARTER"
    assert cell["source_provenance"]["source_type"] == "SEC_HTML"
    assert cell["source_provenance"]["document_id"] == "SEC_DEMO"
