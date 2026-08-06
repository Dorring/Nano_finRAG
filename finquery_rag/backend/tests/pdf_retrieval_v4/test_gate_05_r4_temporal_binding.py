"""Deterministic, Oracle-blind contracts for V4 Gate 05 R4 temporal binding."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "temporal_binding", ROOT / "scripts" / "evaluation" / "temporal_binding.py"
)
assert SPEC and SPEC.loader
temporal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(temporal)


def _table(*, headers, rows, caption="", title="Revenue"):
    cells = []
    facts = []
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row.get("values", [])):
            cell_id = f"cell-{row_index}-{column_index}"
            fact_id = f"fact-{row_index}-{column_index}"
            cell = {
                "cell_id": cell_id,
                "row_id": row["row_id"],
                "column_index": column_index,
                "header_path": headers.get(str(column_index), []),
                "metric_path": row.get("metric_path", []),
                "parsed_value": value,
                "value_kind": row.get("value_kind", "currency"),
                "scale": row.get("scale", "million"),
            }
            fact = {
                "fact_id": fact_id,
                "cell_id": cell_id,
                "row_id": row["row_id"],
                "raw_value": str(value),
                "parsed_value": value,
                "metric_path": row.get("metric_path", []),
                "value_kind": row.get("value_kind", "currency"),
                "scale": row.get("scale", "million"),
                "currency": row.get("currency", "USD"),
            }
            cells.append(cell)
            facts.append(fact)
    return {
        "table_fragment_id": "fragment-1",
        "document_id": "doc-1",
        "pdf_page": 1,
        "table_context": {
            "title": title,
            "caption": caption,
            "statement": "Income Statement",
            "table_scale": "million",
            "table_currency": "USD",
        },
        "column_header_paths": headers,
        "rows": rows,
        "cells": cells,
        "facts": facts,
    }


def _bind(table, fact_index=0, classification=None):
    fact = table["facts"][fact_index]
    cell = next(cell for cell in table["cells"] if cell["cell_id"] == fact["cell_id"])
    row = next(row for row in table["rows"] if row["row_id"] == fact["row_id"])
    schema = temporal.classify_schema(table)
    return schema, temporal.bind_fact(table, schema, fact, cell, row, classification or {})


def test_period_on_columns_and_atomic_binding():
    table = _table(
        headers={
            "0": ["Metric"],
            "1": ["Years ended June 30", "2025"],
            "2": ["Years ended June 30", "2024"],
        },
        rows=[{"row_id": "row-1", "metric_path": ["Revenue"], "values": [None, 10, 9]}],
    )
    schema, record = _bind(table, 1)
    assert schema["schema_type"] == "period_on_columns"
    assert record["temporal_binding"] == {
        "kind": "duration",
        "period": "FY2025",
        "period_type": "annual_duration",
    }
    assert record["fact_semantic_type"] == "atomic_fact"


def test_period_on_rows_uses_row_header():
    table = _table(
        headers={"0": ["Period"], "1": ["Revenue"]},
        rows=[
            {"row_id": "row-1", "raw_label": "FY2025", "metric_path": ["Revenue"], "values": [None, 10]},
            {"row_id": "row-2", "raw_label": "FY2024", "metric_path": ["Revenue"], "values": [None, 9]},
        ],
    )
    schema, record = _bind(table, 1)
    assert schema["schema_type"] == "period_on_rows"
    assert record["temporal_source"] == "row_header_path"
    assert record["temporal_binding"]["period"] == "FY2025"


def test_single_period_snapshot_is_point_period():
    table = _table(
        headers={"0": ["Metric"], "1": ["Segment A"], "2": ["Total"]},
        caption="As of December 31, 2025",
        rows=[{"row_id": "row-1", "metric_path": ["Assets"], "values": [None, 10, 20]}],
    )
    schema, record = _bind(table, 1)
    assert schema["schema_type"] == "single_period_snapshot"
    assert record["temporal_binding"]["kind"] == "point"
    assert record["temporal_binding"]["period"] == "FY2025"


def test_comparison_column_creates_comparison_window():
    table = _table(
        headers={
            "0": ["Metric"],
            "1": ["FY2025"],
            "2": ["FY2024"],
            "3": ["Change"],
        },
        rows=[{"row_id": "row-1", "metric_path": ["Revenue"], "values": [None, 10, 9, 1]}],
    )
    schema, record = _bind(table, 3)
    assert schema["schema_type"] == "comparison_change_table"
    assert record["fact_semantic_type"] == "comparison_fact"
    assert record["temporal_binding"] == {
        "kind": "comparison",
        "current_period": "FY2025",
        "base_period": "FY2024",
        "measure": "absolute_change",
    }


def test_roll_forward_preserves_opening_role():
    table = _table(
        headers={"0": ["Metric"], "1": ["FY2025"]},
        rows=[
            {"row_id": "row-1", "raw_label": "Balance at January 1, 2025", "metric_path": ["Assets"], "values": [None, 10]},
            {"row_id": "row-2", "raw_label": "Additions", "metric_path": ["Assets"], "values": [None, 2]},
        ],
    )
    schema, record = _bind(table, 1)
    assert schema["schema_type"] == "roll_forward"
    assert record["temporal_binding"]["role"] == "opening_instant"
    assert record["temporal_binding"]["kind"] == "point"


def test_bucket_binding_keeps_bucket_label():
    table = _table(
        headers={"0": ["Maturity Bucket"], "1": ["Less than 1 year"]},
        caption="As of December 31, 2025",
        rows=[{"row_id": "row-1", "metric_path": ["Debt"], "values": [None, 10]}],
    )
    schema, record = _bind(table, 1)
    assert schema["schema_type"] == "maturity_or_bucket_table"
    assert record["fact_semantic_type"] == "bucket_fact"
    assert record["temporal_binding"]["kind"] == "bucket"
    assert "Less than 1 year" in record["temporal_binding"]["bucket_label"]


def test_metric_by_segment_is_row_matrix_not_atomic():
    table = _table(
        headers={"0": ["Metric"], "1": ["Automotive"], "2": ["Energy"]},
        rows=[{"row_id": "row-1", "metric_path": ["Revenue"], "values": [None, 10, 9]}],
    )
    schema, record = _bind(table, 1)
    assert schema["schema_type"] == "metric_by_segment_matrix"
    assert record["fact_semantic_type"] == "row_matrix_evidence"
    assert record["temporal_binding"]["kind"] == "not_applicable"


def test_percentage_does_not_require_currency_scale():
    table = _table(
        headers={"0": ["Metric"], "1": ["FY2025"]},
        rows=[{"row_id": "row-1", "metric_path": ["Margin"], "value_kind": "percentage", "values": [None, 68]}],
    )
    schema, record = _bind(table, 1)
    assert record["value_kind"] == "percentage"
    assert record["scale"] == "percent"
    assert record["currency"] is None
    assert record["fact_semantic_type"] == "atomic_fact"


def test_soft_continuation_is_not_a_temporal_source():
    table = _table(
        headers={"0": ["Metric"], "1": ["Segment A"]},
        rows=[{"row_id": "row-1", "metric_path": ["Revenue"], "values": [None, 10]}],
    )
    table["soft_continuation_group_id"] = "cg-1"
    _, record = _bind(table, 1)
    assert record["temporal_source"] != "soft_continuation"
    assert record["temporal_source"] != "inherited_from_soft_continuation"
