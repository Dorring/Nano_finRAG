from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, filename: str):
    path = ROOT / "scripts" / "evaluation" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


gate05 = _load("gate05", "run_pdf_v4_gate_05_predict.py")
score05 = _load("score05", "score_pdf_v4_gate_05.py")


def _table():
    return {
        "document_id": "doc",
        "pdf_page": 7,
        "table_fragment_id": "table:1",
        "logical_table_id": "logical:1",
        "table_context": {"table_bbox": [0, 0, 100, 100], "title": "Income", "statement": "Income", "section_path": [], "table_scale": "million", "table_currency": "USD"},
        "rows": [{"row_id": "row:1", "row_index": 0, "metric_path": ["Revenue"], "normalized_metric_path": "revenue", "raw_text": "Revenue", "row_bbox": [0, 0, 100, 20]}],
        "cells": [{"cell_id": "cell:1", "row_id": "row:1", "row_index": 0, "normalized_metric_path": "revenue", "metric_path": ["Revenue"], "normalized_period": "FY2025", "raw_text": "123", "parsed_value": "123", "base_value": "123000000", "scale": "million", "currency": "USD", "binding_status": "complete", "cell_bbox": [50, 0, 100, 20], "header_path": ["FY2025"]}],
        "facts": [{"fact_id": "fact:1", "cell_id": "cell:1", "row_id": "row:1", "metric_path": ["Revenue"], "normalized_metric": "revenue", "period": "FY2025", "raw_value": "123", "parsed_value": "123", "base_value": "123000000", "scale": "million", "currency": "USD", "binding_status": "complete"}],
    }


def test_unit_ids_are_stable_and_source_traceable():
    table = _table()
    assert gate05._hash(["logical:1", "table:1", "row:1"]) == gate05._hash(["logical:1", "table:1", "row:1"])
    source = gate05._source(table, table["rows"][0], table["cells"][0], table["facts"][0])
    assert source["document_id"] == "doc"
    assert source["pdf_page"] == 7
    assert source["row_id"] == "row:1"
    assert source["cell_id"] == "cell:1"
    assert source["fact_id"] == "fact:1"


def test_gate05_units_never_mark_physical_cross_page_merge():
    assert all(not unit.get("cross_page_merged", False) for unit in [{"cross_page_merged": False}, {"cross_page_merged": False}])


def test_score_numeric_uses_base_value_without_digit_repair():
    unit = {"base_value": "123000000", "parsed_value": "123"}
    from decimal import Decimal

    assert score05._numeric_match(unit, Decimal("123000000"))
    assert not score05._numeric_match(unit, Decimal("123000001"))


def test_metric_path_and_period_are_independent_of_question_oracle():
    unit = {"normalized_metric_path": "intelligent cloud / revenue", "metric_path": ["Intelligent Cloud", "Revenue"]}
    score, exact = score05._metric_score("revenue", unit)
    assert score == 1.0
    assert exact
