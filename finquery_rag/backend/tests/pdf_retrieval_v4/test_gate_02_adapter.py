"""Contract tests for the Oracle-blind V4 Gate 02 adapter."""

from __future__ import annotations

from pathlib import Path

from scripts.evaluation.run_pdf_v4_gate_02_predict import (
    _match_native_words,
    _numeric_values,
    _period_from_text,
    parse_table_html,
)


ADAPTER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluation/run_pdf_v4_gate_02_predict.py"


def test_table_html_preserves_rowspan_and_colspan() -> None:
    parsed = parse_table_html(
        """
        <table>
          <tr><th rowspan='2'>Metric</th><th colspan='2'>Years ended</th></tr>
          <tr><th>2025</th><th>2024</th></tr>
          <tr><td>Revenue</td><td>100</td><td>90</td></tr>
        </table>
        """
    )

    assert parsed["row_count"] == 3
    assert parsed["column_count"] == 3
    assert parsed["grid"][1][0]["raw_text"] == "Metric"
    assert parsed["grid"][2][1]["raw_text"] == "100"
    assert parsed["grid"][2][2]["raw_text"] == "90"


def test_period_normalization_is_deterministic() -> None:
    assert _period_from_text("FY2025") == "FY2025"
    assert _period_from_text("Year ended December 31, 2024") == "FY2024"
    assert _period_from_text("not a period") is None


def test_numeric_normalization_does_not_repair_digits() -> None:
    assert _numeric_values("($1,234.50)")[0]["normalized"] == "-1234.50"
    assert _numeric_values("1 234.50% *")[0]["percent"] is True
    assert all(value["normalized"] != "1204" for value in _numeric_values("12O4"))


def test_numeric_continuation_is_rejoined_by_geometry_only() -> None:
    words = [
        {"index": 0, "text": "281,72", "bbox": [427.3, 148.35, 457.74, 159.51]},
        {"index": 1, "text": "4", "bbox": [452.3, 159.85, 457.86, 171.01]},
    ]

    matched = _match_native_words("281,72", words, 0, 1, [400.0, 140.0, 580.0, 180.0], set())
    assert [word["text"] for word in matched] == ["281,72", "4"]


def test_prediction_script_is_oracle_blind_and_non_production() -> None:
    source = ADAPTER_SCRIPT.read_text(encoding="utf-8")
    assert "manual-mapping-review-package" not in source
    assert "labels.golden" not in source
    assert '"runtime_oracle_reads": 0' in source
    assert '"retrieval_runs": 0' in source
    assert '"production_index_writes": 0' in source


def test_adapter_identity_contract_is_source_based() -> None:
    source = ADAPTER_SCRIPT.read_text(encoding="utf-8")
    assert "document_id" in source
    assert "pdf_page" in source
    assert "table_index" in source
    identity_section = source.split("identity_base =", 1)[1].split("cell_id", 1)[0]
    assert "case_id" not in identity_section
