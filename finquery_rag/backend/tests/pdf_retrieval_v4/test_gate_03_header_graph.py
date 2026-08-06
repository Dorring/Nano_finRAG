"""Contract tests for the Oracle-blind V4 Gate 03 graph builder."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluation.run_pdf_v4_gate_03_predict import (
    _period_type,
    build_table_graph,
)


GRAPH_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluation/run_pdf_v4_gate_03_predict.py"


def _cell(cell_id: str, row: int, column: int, text: str, numeric: str | None = None, period: str | None = None, header: list[str] | None = None) -> dict:
    return {
        "cell_id": cell_id,
        "row_id": f"row:{row}",
        "row_index": row,
        "column_index": column,
        "raw_text": text,
        "resolved_text": text,
        "native_text": text,
        "parsed_numeric": ([{"normalized": numeric, "raw": text, "percent": False}] if numeric is not None else []),
        "normalized_period": period,
        "period_kind": None,
        "header_path": header or [],
        "cell_bbox": [20.0 + column * 40.0, 20.0 + row * 20.0, 50.0 + column * 40.0, 35.0 + row * 20.0],
    }


def _table() -> dict:
    return {
        "table_fragment_id": "table:test",
        "document_id": "doc:test",
        "pdf_page": 1,
        "table_bbox": [20.0, 20.0, 200.0, 140.0],
        "parser_backend": "mineru_hybrid_high",
        "header_texts": ["2025", "2024"],
        "scale_candidates": ["in millions"],
        "rows": [
            {"row_id": "row:0", "row_index": 0, "metric_text": "Intelligent Cloud", "raw_text": "Intelligent Cloud", "row_bbox": [20, 20, 200, 35], "cell_ids": ["c0"]},
            {"row_id": "row:1", "row_index": 1, "metric_text": "Revenue", "raw_text": "Revenue | 106265", "row_bbox": [20, 40, 200, 55], "cell_ids": ["c1", "c2"]},
            {"row_id": "row:2", "row_index": 2, "metric_text": "Operating income", "raw_text": "Operating income | 50000", "row_bbox": [20, 60, 200, 75], "cell_ids": ["c3", "c4"]},
            {"row_id": "row:3", "row_index": 3, "metric_text": "", "raw_text": "—", "row_bbox": [20, 80, 200, 95], "cell_ids": ["c5"]},
            {"row_id": "row:4", "row_index": 4, "metric_text": "Revenue", "raw_text": "Revenue | 12", "row_bbox": [20, 100, 200, 115], "cell_ids": ["c6", "c7"]},
        ],
        "cells": [
            _cell("c0", 0, 0, "Intelligent Cloud"),
            _cell("c1", 1, 0, "Revenue"),
            _cell("c2", 1, 1, "106265", "106265", "FY2025", ["Years ended", "2025"]),
            _cell("c3", 2, 0, "Operating income"),
            _cell("c4", 2, 1, "50000", "50000", "FY2025", ["Years ended", "2025"]),
            _cell("c5", 3, 0, "—"),
            _cell("c6", 4, 0, "Revenue"),
            _cell("c7", 4, 1, "12", "12", "FY2025", ["Years ended", "2025"]),
        ],
    }


def test_metric_parent_from_category_and_reset_on_separator() -> None:
    graph = build_table_graph(_table())
    rows = {row["row_index"]: row for row in graph["rows"]}

    assert rows[1]["metric_path"] == ["Intelligent Cloud", "Revenue"]
    assert rows[2]["normalized_metric_path"] == "intelligent cloud / operating income"
    assert rows[4]["metric_path"] == ["Total", "Revenue"]


def test_metric_parent_does_not_cross_table() -> None:
    first = build_table_graph(_table())
    second_input = _table()
    second_input["table_fragment_id"] = "table:other"
    second_input["rows"][0]["metric_text"] = "Other category"
    second = build_table_graph(second_input)

    assert first["rows"][1]["parent_row_id"] == "row:0"
    assert second["rows"][1]["parent_row_id"] == "row:0"
    assert first["table_fragment_id"] != second["table_fragment_id"]


def test_header_path_and_fact_binding_are_composite() -> None:
    graph = build_table_graph(_table())
    cell = next(cell for cell in graph["cells"] if cell["cell_id"] == "c2")
    fact = next(fact for fact in graph["facts"] if fact["cell_id"] == "c2")

    assert cell["header_path"] == ["Years ended", "2025"]
    assert cell["normalized_metric_path"] == "intelligent cloud / revenue"
    assert cell["binding_status"] == "complete"
    assert fact["normalized_metric"] == "intelligent cloud / revenue"
    assert fact["period"] == "FY2025"


def test_period_instant_and_duration_are_distinct() -> None:
    assert _period_type(["As of June 30, 2025"], "FY2025", "instant") == "instant"
    assert _period_type(["Year ended June 30, 2025"], "FY2025", "duration") == "annual_duration"
    assert _period_type(["Three months ended June 30, 2025"], "FY2025", "duration") == "quarter_duration"


def test_graph_replay_is_deterministic() -> None:
    first = build_table_graph(_table())
    second = build_table_graph(_table())
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_prediction_script_is_oracle_blind_and_non_production() -> None:
    source = GRAPH_SCRIPT.read_text(encoding="utf-8")
    assert "manual-mapping-review-package" not in source
    assert "labels.golden" not in source
    assert '"runtime_oracle_reads": 0' in source
    assert '"retrieval_runs": 0' in source
    assert '"production_index_writes": 0' in source
