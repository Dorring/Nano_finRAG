"""Oracle-blind contracts for V4 Gate 05 Fact integrity recovery."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, filename: str):
    path = ROOT / "scripts" / "evaluation" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


recovery = _load("fact_recovery", "fact_recovery.py")


def _classification(path: Path, fact_id: str, category: str = "eligible_row_only") -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(json.dumps({"stream": "header", "record_count": 1}).encode())
            compressed.write(b"\n")
            compressed.write(json.dumps({"fact_id": fact_id, "eligibility_class": category}).encode())
            compressed.write(b"\n")


def _source(*, header_path=None, column_header_paths=None, period=None, metric_path=None, value_kind="currency"):
    cell = {
        "cell_id": "cell-1",
        "row_id": "row-1",
        "column_index": 2,
        "header_path": header_path or [],
        "metric_path": metric_path or [],
        "parsed_value": "123",
        "value_kind": value_kind,
        "scale": "million",
    }
    row = {
        "row_id": "row-1",
        "metric_path": metric_path or ["Revenue"],
    }
    fact = {
        "fact_id": "fact-1",
        "cell_id": "cell-1",
        "row_id": "row-1",
        "raw_value": "123",
        "period": period,
        "metric_path": metric_path or [],
        "parsed_value": "123",
        "value_kind": value_kind,
        "scale": "million",
    }
    table = {
        "table_fragment_id": "fragment-1",
        "table_context": {"title": "Revenue", "caption": "", "table_scale": "million", "table_currency": "USD"},
        "column_header_paths": column_header_paths or {},
        "rows": [row],
        "cells": [cell],
        "facts": [fact],
    }
    return {"pages": [{"tables": [table]}]}


def test_period_from_cell_header_path(tmp_path):
    path = tmp_path / "classification.jsonl.gz"
    _classification(path, "fact-1")
    recovered, summary = recovery.recover_graph(
        _source(header_path=["Years ended June 30", "2025"]), path
    )
    fact = recovered["pages"][0]["tables"][0]["facts"][0]
    assert fact["period"] == "FY2025"
    assert fact["period_source"] == "cell_header_path"
    assert fact["fact_eligible"] is True
    assert summary["soft_continuation_period_inheritance_count"] == 0


def test_period_from_stable_column_schema(tmp_path):
    path = tmp_path / "classification.jsonl.gz"
    _classification(path, "fact-1")
    source = _source(column_header_paths={"2": ["Years ended", "2025"]})
    recovered, _ = recovery.recover_graph(source, path)
    fact = recovered["pages"][0]["tables"][0]["facts"][0]
    assert fact["period"] == "FY2025"
    assert fact["period_source"] == "stable_column_schema"


def test_conflicting_column_period_fails_closed(tmp_path):
    path = tmp_path / "classification.jsonl.gz"
    _classification(path, "fact-1")
    source = _source(column_header_paths={"2": ["2025", "2024"]})
    recovered, _ = recovery.recover_graph(source, path)
    fact = recovered["pages"][0]["tables"][0]["facts"][0]
    assert fact.get("period") is None
    assert fact["fact_eligible"] is False
    assert fact["binding_status"] == "row_only"


def test_metric_path_comes_from_row_hierarchy_without_question(tmp_path):
    path = tmp_path / "classification.jsonl.gz"
    _classification(path, "fact-1")
    source = _source(metric_path=["Intelligent Cloud", "Revenue"], period="FY2025")
    source["pages"][0]["tables"][0]["cells"][0]["metric_path"] = []
    source["pages"][0]["tables"][0]["facts"][0]["metric_path"] = []
    recovered, _ = recovery.recover_graph(source, path)
    fact = recovered["pages"][0]["tables"][0]["facts"][0]
    assert fact["metric_path"] == ["Intelligent Cloud", "Revenue"]
    assert fact["normalized_metric_path"] == "intelligent cloud / revenue"


def test_non_fact_numeric_is_never_eligible(tmp_path):
    path = tmp_path / "classification.jsonl.gz"
    _classification(path, "fact-1", "non_fact_numeric")
    recovered, _ = recovery.recover_graph(_source(period="FY2025"), path)
    fact = recovered["pages"][0]["tables"][0]["facts"][0]
    assert fact["fact_eligible"] is False
    assert fact["binding_status"] == "non_fact_numeric"


def test_soft_continuation_is_not_a_period_source(tmp_path):
    path = tmp_path / "classification.jsonl.gz"
    _classification(path, "fact-1")
    source = _source()
    source["pages"][0]["tables"][0]["soft_continuation_group_id"] = "cg-1"
    recovered, summary = recovery.recover_graph(source, path)
    fact = recovered["pages"][0]["tables"][0]["facts"][0]
    assert fact.get("period") is None
    assert summary["soft_continuation_period_inheritance_count"] == 0


def test_recovery_preserves_fact_identity_and_raw_value(tmp_path):
    path = tmp_path / "classification.jsonl.gz"
    _classification(path, "fact-1")
    source = _source(period="FY2025")
    original = source["pages"][0]["tables"][0]["facts"][0].copy()
    recovered, _ = recovery.recover_graph(source, path)
    fact = recovered["pages"][0]["tables"][0]["facts"][0]
    assert fact["fact_id"] == original["fact_id"]
    assert fact["cell_id"] == original["cell_id"]
    assert fact["raw_value"] == original.get("raw_value")
    assert fact["parsed_value"] == original["parsed_value"]
