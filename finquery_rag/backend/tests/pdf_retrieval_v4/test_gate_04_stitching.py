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


candidate = _load("gate04a", "run_pdf_v4_gate_04a.py")
predict = _load("gate04b", "run_pdf_v4_gate_04b_predict.py")


def _table(fragment_id: str, document_id: str, page: int, bbox: list[float], rows: list[str], periods: list[str] | None = None):
    periods = periods or ["FY2025", "FY2024"]
    cells = []
    for column, period in enumerate(periods, start=1):
        cells.append({"cell_id": f"cell:{fragment_id}:{column}", "row_id": f"row:{fragment_id}:0", "row_index": 0, "column_index": column, "cell_bbox": [100 * column, bbox[1], 100 * column + 40, bbox[1] + 10], "normalized_period": period, "period_type": "annual_duration", "value_kind": "currency", "parsed_numeric": [{"normalized": "1"}]})
    return {"table_fragment_id": fragment_id, "document_id": document_id, "pdf_page": page, "table_context": {"table_bbox": bbox, "table_currency": "USD", "table_scale": "million", "title": "Statements of Operations", "statement": "Income"}, "column_header_paths": {str(i): [period] for i, period in enumerate(periods, start=1)}, "rows": [{"row_id": f"row:{fragment_id}:0", "raw_label": rows[0], "row_role": "metric", "indent_level": 0, "metric_path": [rows[0]]}], "cells": cells}


def test_adjacent_page_candidate_generation():
    left = _table("left", "doc", 1, [0, 700, 500, 790], ["Revenue"])
    right = _table("right", "doc", 2, [0, 10, 500, 100], ["Expenses"])
    candidates = candidate.build_candidates({"pages": [{"document_id": "doc", "pdf_page": 1, "tables": [left]}, {"document_id": "doc", "pdf_page": 2, "tables": [right]}]})
    assert len(candidates) == 1
    assert candidates[0]["page_gap"] == 1


def test_no_cross_document_candidate():
    left = _table("left", "doc-a", 1, [0, 700, 500, 790], ["Revenue"])
    right = _table("right", "doc-b", 2, [0, 10, 500, 100], ["Revenue"])
    assert candidate.build_candidates({"pages": [{"document_id": "doc-a", "pdf_page": 1, "tables": [left]}, {"document_id": "doc-b", "pdf_page": 2, "tables": [right]}]}) == []


def test_conflicting_columns_block():
    left = _table("left", "doc", 1, [0, 700, 500, 790], ["Revenue"])
    right = _table("right", "doc", 2, [0, 10, 500, 100], ["Revenue"], ["FY2025", "FY2024", "FY2023"])
    item = candidate.build_candidates({"pages": [{"document_id": "doc", "pdf_page": 1, "tables": [left]}, {"document_id": "doc", "pdf_page": 2, "tables": [right]}]})[0]
    assert "column_count_conflict" in item["hard_blockers"]
    state, _ = predict._automatic_state(item, 0.90)
    assert state == "do_not_merge"


def test_three_state_fail_closed():
    state, reasons = predict._automatic_state({"features": {"same_or_compatible_title": True, "column_count_compatible": True, "same_statement": True, "column_band_similarity": 0.5}, "hard_blockers": []}, 0.90)
    assert state == "blocked_ambiguous"
    assert reasons


def test_logical_identity_is_fragment_based():
    table = _table("fragment", "doc", 1, [0, 700, 500, 790], ["Revenue"])
    assert "question" not in predict._payload_hash([table["document_id"], [table["table_fragment_id"]], "", predict._header_fingerprint(table)])
