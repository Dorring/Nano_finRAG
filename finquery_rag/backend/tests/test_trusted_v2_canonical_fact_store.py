from src.runtime.trusted_v2_canonical_fact_store import build_canonical_fact_store


def _document() -> dict:
    return {
        "document": {"document_id": "acme-fy2024", "company": "Acme Corp", "ticker": "ACME", "fiscal_year": 2024},
        "ixbrl_facts": [],
        "tables": [{
            "table_id": "income", "table_title": "Consolidated Statements of Operations", "section_type": "INCOME_STATEMENT", "scale": "millions", "currency": "USD",
            "header_rows": [["", "Year ended 2024"]], "column_headers": ["Metric", "Year ended 2024"],
            "period_columns": [{}, {"header_text": "Year ended 2024", "period_semantics": "ANNUAL", "period_end": "2024-12-31"}],
            "cells": [
                {"cell_id": "income:r1:c0", "row_id": "r1", "column_index": 0, "raw_value": "Revenue"},
                {"cell_id": "income:r1:c1", "row_id": "r1", "column_index": 1, "raw_value": "100", "normalized_value": "100", "period_end": "2024-12-31", "period_semantics": "ANNUAL"},
            ],
            "rows": [{"row_id": "r1", "row_label": "Revenue"}],
        }],
    }


def test_build_canonical_fact_store_uses_only_source_bound_atomic_facts() -> None:
    records, summary = build_canonical_fact_store([_document()])
    assert summary.documents_seen == 1
    assert summary.emitted_facts == 1
    assert summary.annual_fy_periods == 1
    record = records[0]
    assert record["entity"] == "Acme Corp"
    assert record["metric"].casefold().endswith("revenue")
    assert record["period"] == "FY2024"
    assert record["normalized_period"] == "2024-12-31"
    assert record["value"] == "100"
    assert record["currency"] == "USD"
    assert record["provenance_complete"] is True
    assert record["source_traceback"]["cell_id"] == "income:r1:c1"


def test_build_canonical_fact_store_excludes_cell_without_explicit_period() -> None:
    document = _document()
    document["tables"][0]["cells"][1]["period_end"] = None
    document["tables"][0]["cells"][1]["period_semantics"] = "UNKNOWN"
    document["tables"][0]["period_columns"][1] = {}
    document["tables"][0]["header_rows"] = [["", ""]]
    document["tables"][0]["column_headers"] = ["Metric", ""]
    records, summary = build_canonical_fact_store([document])
    assert records == []
    assert summary.atomic_facts_seen == 0
    assert summary.skipped_missing_period == 0
