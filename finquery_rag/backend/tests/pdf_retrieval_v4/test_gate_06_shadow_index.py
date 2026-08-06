"""Contract tests for the V4 Gate 06 shadow-index builder."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluation" / "run_pdf_v4_gate_06.py"
SPEC = importlib.util.spec_from_file_location("run_pdf_v4_gate_06", SCRIPT)
assert SPEC and SPEC.loader
gate06 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate06)


def _unit(unit_type: str, **overrides):
    value = {
        "evidence_unit_id": f"unit-{unit_type}",
        "unit_type": unit_type,
        "document_id": "doc-a",
        "source_pages": [3],
        "source_traceback": {
            "document_id": "doc-a",
            "pdf_page": 3,
            "table_fragment_id": "fragment-a",
            "logical_table_id": "table-a",
            "row_id": "row-a",
            "cell_id": "cell-a",
            "fact_id": "fact-a",
        },
        "statement": "Income statement",
        "title": "Revenue",
        "section_path": ["Results"],
        "metric_path": ["Revenue"],
        "metric_set": ["Revenue"],
        "period_set": ["FY2025"],
        "normalized_period": "FY2025",
        "period": "FY2025",
        "raw_value": "123",
        "raw_text": "Revenue 123",
        "retrieval_text": "Revenue 123",
        "values": ["FY2025: 123"],
        "header_path": ["Years ended", "2025"],
        "currency": "USD",
        "scale": "million",
        "value_kind": "currency",
        "evidence_level": "A",
        "binding_status": "complete",
    }
    value.update(overrides)
    return value


def test_retrieval_view_stable_identity_and_source_group():
    unit = _unit("row")
    assert gate06._view_id(unit) == gate06._view_id(dict(unit))
    assert gate06._source_group_id(unit) == gate06._source_group_id(dict(unit))
    changed = dict(unit, unit_type="cell", evidence_unit_id="unit-cell")
    assert gate06._view_id(changed) != gate06._view_id(unit)


def test_view_text_contracts_do_not_copy_question_or_gold_fields():
    banned = ("case_id", "expected_value", "reference_answer", "gold_source", "review_status", "oracle")
    for unit_type in gate06.UNIT_TYPES:
        unit = _unit(unit_type)
        context = gate06._context(unit)
        text = gate06._view_text(unit, context)
        assert text
        assert not any(token in text.lower() for token in banned)
        if unit_type == "row":
            assert "Metric: Revenue" in text
        if unit_type == "fact":
            assert "Period: FY2025" in text


def test_fact_admission_audit_blocks_below_ninety_percent():
    units = [_unit("fact", evidence_unit_id=f"fact-{i}") for i in range(10)]
    views = {typ: [] for typ in gate06.UNIT_TYPES}
    views["fact"] = [gate06._build_views(units[:8], Path("missing-shadow"))[0]["fact"][0]]
    audit = gate06._fact_admission_audit(units, views, {"fact": {"fact_not_level_a_complete": 2}})
    assert audit["fact_indexed_count"] == 1
    assert audit["fact_total_count"] == 10
    assert audit["admission_blocked"] is True
    assert audit["decision"] == "fact_evidence_admission_blocked"


def test_gate05_stream_header_and_record_count(tmp_path):
    stream = tmp_path / "units.jsonl.gz"
    rows = [{"stream": "header", "record_count": 1}, {"evidence_unit_id": "u", "unit_type": "row"}]
    with stream.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write("\n".join(json.dumps(row) for row in rows).encode("utf-8"))
    header, units = gate06._load_stream(stream)
    assert header["record_count"] == 1
    assert len(units) == 1


def test_traceback_and_runtime_path_contract():
    unit = _unit("section")
    assert gate06._traceback_complete(gate06._traceback(unit))
    assert gate06.DEFAULT_RUNTIME.name == "pdf-retrieval-v4-gate-06"
    assert "artifacts" in gate06.DEFAULT_RUNTIME.parts


def test_soft_continuation_is_metadata_only(tmp_path):
    (tmp_path / "continuation-shadow-predictions.json").write_text(
        json.dumps({"links": [{"continuation_candidate": True, "continuation_group_id": "cg-1", "left_fragment_id": "a", "right_fragment_id": "b"}]}),
        encoding="utf-8",
    )
    groups, fragments = gate06._soft_edges(tmp_path)
    assert groups["cg-1"]["merge_applied"] is False
    assert fragments["a"] == ["cg-1"]


def test_metadata_foreign_keys_are_checked(tmp_path):
    units = [_unit("row"), _unit("cell"), _unit("fact")]
    views_by_type, _ = gate06._build_views(units, tmp_path)
    views = [view for values in views_by_type.values() for view in values]
    manifest = gate06._build_metadata_store(views, {}, tmp_path / "metadata.sqlite")
    assert manifest["foreign_key_failures"] == 0
    assert manifest["foreign_key_audit"] == {
        "row_table_missing": 0,
        "cell_row_missing": 0,
        "fact_cell_missing": 0,
    }
