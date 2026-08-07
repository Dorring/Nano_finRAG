"""Contract tests for the Gate 02 R3 full-corpus unified structured adapter.

Covers the R3 identity scheme, integrity checks, full-corpus adapter
builder, probe structural diff reconciliation, full-document context
diff audit, legacy identity continuity, prediction seal safety, Oracle
scoring, D-class presence observation, and finalize acceptance gates.

These tests run without the actual frozen PDFs or MinerU outputs by
using synthetic fixtures.  Protocol/seal safety tests skip when the
generated artifacts do not yet exist on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.adapter_identity import (  # noqa: E402
    build_table_signature,
    cell_id,
    normalize_bbox,
    row_id,
    table_fragment_id,
)
from src.pdf_retrieval_v4.adapter_integrity import (  # noqa: E402
    check_bbox_integrity,
    check_identity_integrity,
    check_page_integrity,
    check_text_integrity,
)
from src.pdf_retrieval_v4.full_corpus_adapter import (  # noqa: E402
    build_page_record,
    build_table_fragment,
    collect_document_metrics,
    collect_structure_metrics,
)
from src.pdf_retrieval_v4.native_alignment import (  # noqa: E402
    column_bands,
    inside,
    union_bbox,
)
from src.pdf_retrieval_v4.table_html_parser import (  # noqa: E402
    compact_text,
    extract_numeric_values,
    norm_text,
    parse_table_html,
    period_from_text,
    period_kind,
    tokenize_text,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


R3_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r3"


def _make_table_html(rows: list[list[str]], header: bool = False) -> str:
    """Build a small HTML table string from a list of cell-text rows."""
    parts = ["<table>"]
    for row in rows:
        parts.append("<tr>")
        for cell in row:
            tag = "th" if header else "td"
            parts.append(f"<{tag}>{cell}</{tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _make_parsed_table(rows: list[list[str]]) -> dict[str, Any]:
    return parse_table_html(_make_table_html(rows))


def _make_mineru_table(rows: list[list[str]], bbox: list[float]) -> dict[str, Any]:
    return {
        "html": _make_table_html(rows),
        "bbox": bbox,
        "parsed": _make_parsed_table(rows),
    }


def _make_cell_record(
    row_index: int,
    column_index: int,
    raw_text: str,
    rowspan: int = 1,
    colspan: int = 1,
    cell_bbox: list[float] | None = None,
    resolved_text: str = "",
    text_source: str = "mineru_table_text",
    native_text: str = "",
    parsed_numeric: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "column_index": column_index,
        "rowspan": rowspan,
        "colspan": colspan,
        "raw_text": raw_text,
        "normalized_text": norm_text(raw_text),
        "cell_bbox": cell_bbox,
        "mineru_text": raw_text,
        "native_text": native_text,
        "resolved_text": resolved_text or raw_text,
        "text_source": text_source,
        "alignment_confidence": 0.0,
        "parsed_numeric": parsed_numeric or [],
        "header_path": [],
        "normalized_period": None,
        "period_kind": None,
        "scale_candidates": [],
    }


# ---------------------------------------------------------------------------
# 1. Identity scheme (adapter_identity)
# ---------------------------------------------------------------------------


class TestAdapterIdentity:
    """R3 identity scheme: source-based, no Question/Gold/Case ID."""

    def test_table_fragment_id_is_stable(self) -> None:
        cells = [_make_cell_record(0, 0, "Revenue", 1, 1)]
        sig = build_table_signature(cells)
        id1 = table_fragment_id("aapl_fy2025", 25, [10.0, 20.0, 300.0, 200.0], sig)
        id2 = table_fragment_id("aapl_fy2025", 25, [10.0, 20.0, 300.0, 200.0], sig)
        assert id1 == id2
        assert id1.startswith("table:")

    def test_table_fragment_id_changes_with_document(self) -> None:
        cells = [_make_cell_record(0, 0, "Revenue")]
        sig = build_table_signature(cells)
        id_a = table_fragment_id("aapl_fy2025", 25, [10, 20, 300, 200], sig)
        id_b = table_fragment_id("jpm_fy2025", 25, [10, 20, 300, 200], sig)
        assert id_a != id_b

    def test_table_fragment_id_changes_with_page(self) -> None:
        cells = [_make_cell_record(0, 0, "Revenue")]
        sig = build_table_signature(cells)
        id1 = table_fragment_id("aapl_fy2025", 25, [10, 20, 300, 200], sig)
        id2 = table_fragment_id("aapl_fy2025", 26, [10, 20, 300, 200], sig)
        assert id1 != id2

    def test_row_id_depends_on_table_fragment_id(self) -> None:
        sig = "Revenue | 100 | 90"
        id1 = row_id("table:abc", 0, sig)
        id2 = row_id("table:xyz", 0, sig)
        assert id1 != id2
        assert id1.startswith("row:")

    def test_cell_id_depends_on_row_id(self) -> None:
        sig = "Revenue|1|1"
        id1 = cell_id("row:abc", 0, sig)
        id2 = cell_id("row:xyz", 0, sig)
        assert id1 != id2
        assert id1.startswith("cell:")

    def test_cell_id_changes_with_column(self) -> None:
        sig = "Revenue|1|1"
        id1 = cell_id("row:abc", 0, sig)
        id2 = cell_id("row:abc", 1, sig)
        assert id1 != id2

    def test_normalize_bbox_rounds_and_handles_invalid(self) -> None:
        assert normalize_bbox([1.234567, 2.789, 3.111, 4.999]) == [
            1.23,
            2.79,
            3.11,
            5.0,
        ]
        assert normalize_bbox(None) == [0.0, 0.0, 0.0, 0.0]
        assert normalize_bbox([1, 2]) == [0.0, 0.0, 0.0, 0.0]

    def test_identity_is_source_based_no_oracle_fields(self) -> None:
        """Identity payload must not include Question/Gold/Case ID."""
        source = (ROOT / "src/pdf_retrieval_v4/adapter_identity.py").read_text(
            encoding="utf-8"
        )
        # The module must not reference Oracle/Question/Case ID fields
        assert "question" not in source.lower()
        assert "gold" not in source.lower()
        assert "case_id" not in source.lower()
        assert "expected_value" not in source.lower()


# ---------------------------------------------------------------------------
# 2. Table HTML parser (table_html_parser)
# ---------------------------------------------------------------------------


class TestTableHtmlParser:
    def test_parse_preserves_rowspan_and_colspan(self) -> None:
        parsed = parse_table_html(
            """
            <table>
              <tr><th rowspan='2'>Metric</th><th colspan='2'>Years</th></tr>
              <tr><th>2025</th><th>2024</th></tr>
              <tr><td>Revenue</td><td>100</td><td>90</td></tr>
            </table>
            """
        )
        assert parsed["row_count"] == 3
        assert parsed["column_count"] == 3
        assert parsed["grid"][1][0]["raw_text"] == "Metric"
        assert parsed["grid"][2][1]["raw_text"] == "100"

    def test_norm_text_strips_tags_and_lowercases(self) -> None:
        assert norm_text("<b>Revenue</b>") == "revenue"
        assert norm_text("  Hello   World  ") == "hello world"

    def test_compact_text_removes_non_alphanumeric(self) -> None:
        assert compact_text("Revenue (Net)") == "revenuenet"
        assert compact_text("$1,234.50") == "123450"

    def test_tokenize_text_extracts_words_and_numbers(self) -> None:
        tokens = tokenize_text("Revenue $1,234 (50.5)%")
        assert "revenue" in tokens
        assert "$1,234" in tokens or "1,234" in tokens
        assert "(50.5)%" in tokens

    def test_period_from_text_extracts_fy(self) -> None:
        assert period_from_text("FY2025") == "FY2025"
        assert period_from_text("Year ended 2024") == "FY2024"
        assert period_from_text("no period") is None

    def test_period_kind_classification(self) -> None:
        assert period_kind("As of December 31") == "instant"
        assert period_kind("Year ended") == "duration"
        assert period_kind("nothing") is None

    def test_extract_numeric_values_does_not_repair_digits(self) -> None:
        values = extract_numeric_values("($1,234.50)")
        assert len(values) == 1
        assert values[0]["normalized"] == "-1234.50"
        # 12O4 (letter O) must not be repaired to 1204; the function may
        # extract "12" and "4" separately but must never produce "1204".
        assert all(v["normalized"] != "1204" for v in extract_numeric_values("12O4"))


# ---------------------------------------------------------------------------
# 3. Native alignment helpers (native_alignment)
# ---------------------------------------------------------------------------


class TestNativeAlignment:
    def test_inside_with_margin(self) -> None:
        word = {"bbox": [10.0, 10.0, 20.0, 20.0]}
        assert inside(word, [5.0, 5.0, 25.0, 25.0]) is True
        assert inside(word, [22.0, 22.0, 30.0, 30.0]) is False
        assert inside(word, [22.0, 22.0, 30.0, 30.0], margin=5.0) is True

    def test_union_bbox_returns_min_max(self) -> None:
        words = [
            {"bbox": [10.0, 20.0, 30.0, 40.0]},
            {"bbox": [15.0, 10.0, 25.0, 50.0]},
        ]
        bbox = union_bbox(words)
        assert bbox == [10.0, 10.0, 30.0, 50.0]

    def test_union_bbox_empty_returns_none(self) -> None:
        assert union_bbox([]) is None

    def test_column_bands_split_evenly(self) -> None:
        bands = column_bands([0.0, 0.0, 100.0, 200.0], 4)
        assert len(bands) == 4
        assert bands[0] == [0.0, 0.0, 25.0, 200.0]
        assert bands[3] == [75.0, 0.0, 100.0, 200.0]

    def test_column_bands_zero_width_returns_empty(self) -> None:
        assert column_bands([0, 0, 100, 200], 0) == []


# ---------------------------------------------------------------------------
# 4. Full-corpus adapter builder (full_corpus_adapter)
# ---------------------------------------------------------------------------


class TestFullCorpusAdapter:
    def test_build_table_fragment_assigns_ids(self) -> None:
        rows = [["Metric", "2025", "2024"], ["Revenue", "100", "90"]]
        table = _make_mineru_table(rows, [10.0, 20.0, 300.0, 200.0])
        fragment = build_table_fragment(table, "aapl_fy2025", 25, 0, words=[])
        assert fragment["table_fragment_id"].startswith("table:")
        assert fragment["row_count"] == 2
        assert fragment["column_count"] == 3
        assert len(fragment["rows"]) == 2
        assert len(fragment["cells"]) == 6
        # Every row must have a row_id and cell_ids
        for row in fragment["rows"]:
            assert row["row_id"].startswith("row:")
            assert len(row["cell_ids"]) == 3
        # Every cell must have cell_id, table_fragment_id, row_id
        for cell in fragment["cells"]:
            assert cell["cell_id"].startswith("cell:")
            assert cell["table_fragment_id"] == fragment["table_fragment_id"]
            assert cell["row_id"].startswith("row:")

    def test_build_table_fragment_parser_backend_label(self) -> None:
        rows = [["A", "B"], ["1", "2"]]
        table = _make_mineru_table(rows, [0.0, 0.0, 100.0, 100.0])
        fragment = build_table_fragment(table, "doc", 1, 0, words=[])
        assert fragment["parser_backend"] == "mineru_hybrid_high"

    def test_build_page_record_basic(self) -> None:
        record = build_page_record("doc", 1, 0, None, [], None)
        assert record["document_id"] == "doc"
        assert record["pdf_page"] == 1
        assert record["page_index"] == 0
        assert record["table_fragment_ids"] == []
        assert record["mineru_page_present"] is True

    def test_collect_structure_metrics_counts(self) -> None:
        rows = [["Metric", "2025"], ["Revenue", "100"]]
        table = _make_mineru_table(rows, [0.0, 0.0, 100.0, 100.0])
        fragment = build_table_fragment(table, "doc", 1, 0, words=[])
        page = build_page_record("doc", 1, 0, None, [fragment], None)
        metrics = collect_structure_metrics([page])
        assert metrics["page_count"] == 1
        assert metrics["table_count"] == 1
        assert metrics["row_count"] == 2
        assert metrics["cell_count"] == 4
        assert metrics["duplicate_table_id_count"] == 0

    def test_collect_document_metrics_groups_by_doc(self) -> None:
        rows = [["A"], ["1"]]
        table = _make_mineru_table(rows, [0.0, 0.0, 50.0, 50.0])
        fragment = build_table_fragment(table, "doc1", 1, 0, words=[])
        page1 = build_page_record("doc1", 1, 0, None, [fragment], None)
        page2 = build_page_record("doc2", 1, 0, None, [], None)
        doc_metrics = collect_document_metrics([page1, page2])
        doc_ids = [m["document_id"] for m in doc_metrics]
        assert doc_ids == ["doc1", "doc2"]


# ---------------------------------------------------------------------------
# 5. Adapter integrity checks (adapter_integrity)
# ---------------------------------------------------------------------------


class TestAdapterIntegrity:
    def test_check_page_integrity_pass(self) -> None:
        pages = [
            {"document_id": "doc", "pdf_page": 1},
            {"document_id": "doc", "pdf_page": 2},
        ]
        result = check_page_integrity(pages, expected_pages=2)
        assert result["passed"] is True
        assert result["missing_page_records"] == 0
        assert result["duplicate_page_records"] == 0

    def test_check_page_integrity_duplicate(self) -> None:
        pages = [
            {"document_id": "doc", "pdf_page": 1},
            {"document_id": "doc", "pdf_page": 1},
        ]
        result = check_page_integrity(pages, expected_pages=2)
        assert result["passed"] is False
        assert result["duplicate_page_records"] == 1

    def test_check_identity_integrity_no_duplicates(self) -> None:
        rows = [["A", "B"], ["1", "2"]]
        table = _make_mineru_table(rows, [0.0, 0.0, 100.0, 100.0])
        fragment = build_table_fragment(table, "doc", 1, 0, words=[])
        page = build_page_record("doc", 1, 0, None, [fragment], None)
        result = check_identity_integrity([page])
        assert result["passed"] is True
        assert result["duplicate_table_id"] == 0
        assert result["duplicate_row_id"] == 0
        assert result["duplicate_cell_id"] == 0
        assert result["row_to_table_missing"] == 0
        assert result["cell_to_row_missing"] == 0

    def test_check_bbox_integrity_valid(self) -> None:
        rows = [["A", "B"], ["1", "2"]]
        table = _make_mineru_table(rows, [10.0, 10.0, 90.0, 90.0])
        fragment = build_table_fragment(table, "doc", 1, 0, words=[])
        page = {
            "document_id": "doc",
            "pdf_page": 1,
            "page_width": 200.0,
            "page_height": 200.0,
            "tables": [fragment],
        }
        result = check_bbox_integrity([page])
        # Rows may have None bbox if no native words matched, which is allowed
        assert result["invalid_table_bbox"] == 0

    def test_check_text_integrity_counts(self) -> None:
        rows = [["A", "B"], ["1", "2"]]
        table = _make_mineru_table(rows, [0.0, 0.0, 100.0, 100.0])
        fragment = build_table_fragment(table, "doc", 1, 0, words=[])
        page = {
            "document_id": "doc",
            "pdf_page": 1,
            "page_width": 200.0,
            "page_height": 200.0,
            "tables": [fragment],
        }
        result = check_text_integrity([page])
        assert "numeric_cell_native_loss" in result
        assert "invalid_numeric_parse" in result
        assert "unresolved_cells" in result


# ---------------------------------------------------------------------------
# 6. Probe structural diff reconciliation (reconcile_probe_structural_diff_r3)
# ---------------------------------------------------------------------------


RECONCILE_SCRIPT = ROOT / "scripts/evaluation/reconcile_probe_structural_diff_r3.py"


class TestProbeStructuralDiffReconciliation:
    def test_script_is_oracle_blind(self) -> None:
        source = RECONCILE_SCRIPT.read_text(encoding="utf-8")
        assert "manual-mapping-review-package" not in source
        assert "labels.golden" not in source

    def test_frozen_document_ids_are_eight(self) -> None:
        source = RECONCILE_SCRIPT.read_text(encoding="utf-8")
        # The frozen set must contain exactly the 8 benchmark document IDs
        for doc_id in [
            "aapl_fy2025",
            "jpm_fy2025",
            "ko_fy2025",
            "msft_fy2025",
            "nvda_fy2025",
            "pfe_fy2024",
            "tsla_fy2025",
            "v_fy2025",
        ]:
            assert doc_id in source

    def test_reconciliation_output_schema_fields(self) -> None:
        """Verify the reconciliation output has the required per-page fields."""
        out_path = R3_OUT / "probe-structural-diff-reconciliation.json"
        if not out_path.is_file():
            pytest.skip("Reconciliation artifact not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["true_missing_page_count"] == 0
        assert data["adapter_blocking"] is False
        for rec in data.get("reconciled_pages", []):
            assert "page_record_present" in rec
            assert "difference_class" in rec
            assert "adapter_blocking" in rec


# ---------------------------------------------------------------------------
# 7. Full-document context diff audit (audit_full_document_context_diff_r3)
# ---------------------------------------------------------------------------


CONTEXT_DIFF_SCRIPT = ROOT / "scripts/evaluation/audit_full_document_context_diff_r3.py"


class TestFullDocumentContextDiffAudit:
    def test_script_is_oracle_blind(self) -> None:
        source = CONTEXT_DIFF_SCRIPT.read_text(encoding="utf-8")
        assert "manual-mapping-review-package" not in source
        assert "labels.golden" not in source

    def test_classification_categories_present(self) -> None:
        source = CONTEXT_DIFF_SCRIPT.read_text(encoding="utf-8")
        for category in [
            "benign_html_normalization",
            "better_table_segmentation",
            "cross_page_context_change",
            "bbox_geometry_shift",
            "row_structure_change",
            "actual_regression",
        ]:
            assert category in source

    def test_html_hash_change_alone_not_blocking(self) -> None:
        """A pure HTML string hash change must not be adapter-blocking."""
        out_path = R3_OUT / "full-document-context-diff-audit.json"
        if not out_path.is_file():
            pytest.skip("Context diff audit not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        # actual_regression must be the only blocking class
        for entry in data.get("html_changed_pages", []):
            if entry["classification"] == "benign_html_normalization":
                assert entry["adapter_blocking"] is False


# ---------------------------------------------------------------------------
# 8. Legacy identity continuity (audit_legacy_identity_continuity_r3)
# ---------------------------------------------------------------------------


LEGACY_SCRIPT = ROOT / "scripts/evaluation/audit_legacy_identity_continuity_r3.py"


class TestLegacyIdentityContinuity:
    def test_script_is_oracle_blind(self) -> None:
        source = LEGACY_SCRIPT.read_text(encoding="utf-8")
        assert "manual-mapping-review-package" not in source
        assert "labels.golden" not in source

    def test_three_layer_comparison_present(self) -> None:
        source = LEGACY_SCRIPT.read_text(encoding="utf-8")
        assert "exact_stable" in source
        assert "structurally_equivalent" in source
        assert "regression" in source.lower()

    def test_corpus_scope_difference_not_regression(self) -> None:
        """Dev-corpus pages must be classified as scope difference, not regression."""
        out_path = R3_OUT / "legacy-probe-identity-continuity.json"
        if not out_path.is_file():
            pytest.skip("Legacy continuity audit not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        # True regression must be 0 for the gate to pass
        assert data["true_regression_count"] == 0


# ---------------------------------------------------------------------------
# 9. Prediction seal safety (seal_pdf_v4_gate_02_r3)
# ---------------------------------------------------------------------------


SEAL_SCRIPT = ROOT / "scripts/evaluation/seal_pdf_v4_gate_02_r3.py"


class TestPredictionSealSafety:
    def test_seal_script_is_oracle_blind(self) -> None:
        source = SEAL_SCRIPT.read_text(encoding="utf-8")
        assert "manual-mapping-review-package" not in source
        assert "labels.golden" not in source

    def test_seal_has_zero_safety_reads(self) -> None:
        seal_path = R3_OUT / "adapter-prediction-seal.json"
        if not seal_path.is_file():
            pytest.skip("Seal not yet generated")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        assert seal["question_reads_before_seal"] == 0
        assert seal["gold_reads_before_seal"] == 0
        assert seal["governance_reads_before_seal"] == 0
        assert seal["expected_value_reads_before_seal"] == 0
        assert seal["reference_answer_reads_before_seal"] == 0
        assert seal["header_graph_runs"] == 0
        assert seal["evidence_unit_builds"] == 0
        assert seal["index_builds"] == 0
        assert seal["retrieval_runs"] == 0
        assert seal["reranker_calls"] == 0
        assert seal["answer_generation_calls"] == 0
        assert seal["production_index_writes"] == 0
        assert seal["production_config_modified"] is False
        assert seal["production_switch_allowed"] is False
        assert seal["sealed"] is True

    def test_seal_records_prediction_hash(self) -> None:
        seal_path = R3_OUT / "adapter-prediction-seal.json"
        if not seal_path.is_file():
            pytest.skip("Seal not yet generated")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        assert seal.get("prediction_hash")
        assert seal.get("input_manifest_hash")
        assert seal.get("protocol_hash")


# ---------------------------------------------------------------------------
# 10. Oracle regression scoring (score_pdf_v4_gate_02_r3_oracle)
# ---------------------------------------------------------------------------


ORACLE_SCRIPT = ROOT / "scripts/evaluation/score_pdf_v4_gate_02_r3_oracle.py"


class TestOracleRegressionScoring:
    def test_scoring_script_runs_after_seal(self) -> None:
        source = ORACLE_SCRIPT.read_text(encoding="utf-8")
        assert "adapter-prediction-seal.json" in source
        assert "sealed" in source

    def test_scoring_targets_22_22(self) -> None:
        out_path = R3_OUT / "post-seal-oracle-regression.json"
        if not out_path.is_file():
            pytest.skip("Oracle regression not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["oracle_record_count"] == 22
        for key in [
            "table_recovery",
            "row_recovery",
            "numeric_exact",
            "scale_recoverability",
            "source_traceback",
        ]:
            assert key in data

    def test_scoring_script_does_not_read_gold_before_seal(self) -> None:
        source = ORACLE_SCRIPT.read_text(encoding="utf-8")
        # Must verify the seal before loading Oracle records
        assert "sealed" in source
        assert "adapter-prediction-seal.json" in source


# ---------------------------------------------------------------------------
# 11. D-class / B-class presence observation (observe_d_class_presence_r3)
# ---------------------------------------------------------------------------


D_CLASS_SCRIPT = ROOT / "scripts/evaluation/observe_d_class_presence_r3.py"


class TestDClassPresenceObservation:
    def test_script_is_observation_only(self) -> None:
        source = D_CLASS_SCRIPT.read_text(encoding="utf-8")
        assert "manual-mapping-review-package" not in source
        assert "labels.golden" not in source

    def test_script_runs_after_seal(self) -> None:
        source = D_CLASS_SCRIPT.read_text(encoding="utf-8")
        assert "adapter-prediction-seal.json" in source
        assert "sealed" in source

    def test_d_class_presence_report_fields(self) -> None:
        out_path = R3_OUT / "d-class-structural-presence.json"
        if not out_path.is_file():
            pytest.skip("D-class presence not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert "d_class_total" in data
        assert "d_class_page_present" in data
        assert "d_class_table_present" in data
        assert "d_class_row_present" in data
        assert "b_class_total" in data
        assert "b_class_row_cell_exists" in data


# ---------------------------------------------------------------------------
# 11.1 Benchmark structural presence closure (audit_benchmark_structural_presence_r31)
# ---------------------------------------------------------------------------


R31_SCRIPT = ROOT / "scripts/evaluation/audit_benchmark_structural_presence_r31.py"


class TestBenchmarkStructuralPresenceClosure:
    """Gate 02 R3.1: 33-record 5-layer structural presence audit."""

    def test_script_is_observation_only(self) -> None:
        source = R31_SCRIPT.read_text(encoding="utf-8")
        assert "manual-mapping-review-package" not in source
        assert "labels.golden" not in source

    def test_script_runs_after_seal(self) -> None:
        source = R31_SCRIPT.read_text(encoding="utf-8")
        assert "adapter-prediction-seal.json" in source
        assert "sealed" in source

    def test_script_does_not_modify_adapter(self) -> None:
        source = R31_SCRIPT.read_text(encoding="utf-8")
        assert "re-run" not in source.lower() or "does NOT" in source

    def test_script_loads_33_records(self) -> None:
        source = R31_SCRIPT.read_text(encoding="utf-8")
        assert "structurally_absent" in source
        assert "strict_mapped_not_retrieved" in source
        assert "b-class-detail.json" in source
        assert "gold-coverage-classification.json" in source

    def test_five_layer_coverage_check_present(self) -> None:
        source = R31_SCRIPT.read_text(encoding="utf-8")
        for layer in [
            "page_present",
            "table_present",
            "row_present",
            "cell_present",
            "candidate_compatible_structure",
        ]:
            assert layer in source

    def test_decision_thresholds_present(self) -> None:
        source = R31_SCRIPT.read_text(encoding="utf-8")
        assert "full_corpus_benchmark_structural_presence_closed" in source
        assert "full_corpus_structural_presence_insufficient" in source
        assert "full_corpus_financial_semantic_graph" in source
        assert "stop_and_classify_missing_evidence_shapes" in source

    def test_closure_artifact_fields(self) -> None:
        out_path = R3_OUT / "benchmark-structural-presence-closure.json"
        if not out_path.is_file():
            pytest.skip("R3.1 closure artifact not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["seal_verified"] is True
        assert data["d_class_metrics"]["total"] == 16
        assert data["b_class_metrics"]["total"] == 17
        for field in [
            "page_present",
            "table_present",
            "row_present",
            "cell_present",
            "candidate_compatible_structure",
        ]:
            assert field in data["d_class_metrics"]
            assert field in data["b_class_metrics"]
        assert "newly_structurally_recoverable" in data
        assert data["production_switch_allowed"] is False
        assert data["decision"] == "full_corpus_benchmark_structural_presence_closed"
        assert data["next_gate"] == "full_corpus_financial_semantic_graph"

    def test_per_record_has_five_layer_fields(self) -> None:
        out_path = R3_OUT / "benchmark-structural-presence-closure.json"
        if not out_path.is_file():
            pytest.skip("R3.1 closure artifact not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        for record in data["d_class_records"] + data["b_class_records"]:
            assert "case_id" in record
            assert "gold_source_identity" in record
            assert "benchmark_class" in record
            assert "document_id" in record
            assert "pdf_page" in record
            assert "page_present" in record
            assert "table_present" in record
            assert "row_present" in record
            assert "cell_present" in record
            assert "candidate_compatible_structure" in record
            assert "coverage_level" in record

    def test_strength_is_strong_or_acceptable(self) -> None:
        out_path = R3_OUT / "benchmark-structural-presence-closure.json"
        if not out_path.is_file():
            pytest.skip("R3.1 closure artifact not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["strength"] in ("strong", "acceptable")


# ---------------------------------------------------------------------------
# 11.2 Target-specific Structural Alignment (audit_target_structural_alignment_r32)
# ---------------------------------------------------------------------------


R32_SCRIPT = ROOT / "scripts/evaluation/audit_target_structural_alignment_r32.py"


class TestTargetStructuralAlignment:
    """Gate 02 R3.2: Target-specific structural alignment audit."""

    def test_script_is_observation_only(self) -> None:
        source = R32_SCRIPT.read_text(encoding="utf-8")
        assert "manual-mapping-review-package" not in source
        assert "labels.golden" not in source

    def test_script_runs_after_seal(self) -> None:
        source = R32_SCRIPT.read_text(encoding="utf-8")
        assert "adapter-prediction-seal.json" in source
        assert "sealed" in source

    def test_script_does_not_modify_adapter(self) -> None:
        source = R32_SCRIPT.read_text(encoding="utf-8")
        assert "re-run" not in source.lower() or "does NOT" in source

    def test_script_requires_r31_closure(self) -> None:
        source = R32_SCRIPT.read_text(encoding="utf-8")
        assert "benchmark-structural-presence-closure.json" in source

    def test_script_loads_33_records(self) -> None:
        source = R32_SCRIPT.read_text(encoding="utf-8")
        assert "structurally_absent" in source
        assert "gold-coverage-classification.json" in source
        assert "b-class-detail.json" in source

    def test_five_layer_t0_t4_present(self) -> None:
        source = R32_SCRIPT.read_text(encoding="utf-8")
        for layer in [
            "target_page_present",
            "target_block_present",
            "target_row_present",
            "target_cells_present",
            "target_candidate_alignable",
        ]:
            assert layer in source

    def test_numeric_and_text_matching_present(self) -> None:
        source = R32_SCRIPT.read_text(encoding="utf-8")
        assert "_extract_numeric_tokens" in source
        assert "_extract_text_tokens" in source
        assert "numeric_recall" in source
        assert "metric_token_recall" in source
        assert "_counter_recall" in source

    def test_match_strategies_present(self) -> None:
        source = R32_SCRIPT.read_text(encoding="utf-8")
        assert "single_row" in source
        assert "table_block" in source
        assert "multi_row_block" in source
        assert "_score_table_match" in source
        assert "_score_multi_row_match" in source

    def test_decision_thresholds_present(self) -> None:
        source = R32_SCRIPT.read_text(encoding="utf-8")
        assert "target_structural_alignment_closed" in source
        assert "target_structural_alignment_insufficient" in source
        assert "full_corpus_financial_semantic_graph" in source
        assert "stop_and_classify_missing_evidence_shapes" in source

    def test_failure_reasons_classified(self) -> None:
        source = R32_SCRIPT.read_text(encoding="utf-8")
        for reason in [
            "target_table_missing",
            "target_row_missing",
            "target_numeric_cells_missing",
            "multiple_structural_matches",
            "candidate_is_multi_row_block",
            "candidate_is_narrative",
        ]:
            assert reason in source

    def test_alignment_artifact_fields(self) -> None:
        out_path = R3_OUT / "target-structural-alignment.json"
        if not out_path.is_file():
            pytest.skip("R3.2 alignment artifact not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["seal_verified"] is True
        assert data["r31_closure_verified"] is True
        assert data["d_class_metrics"]["total"] == 16
        assert data["b_class_metrics"]["total"] == 17
        assert "false_structural_alignment" in data
        assert data["production_switch_allowed"] is False
        assert data["decision"] == "target_structural_alignment_closed"
        assert data["next_gate"] == "full_corpus_financial_semantic_graph"

    def test_per_record_has_match_evidence(self) -> None:
        out_path = R3_OUT / "target-structural-alignment.json"
        if not out_path.is_file():
            pytest.skip("R3.2 alignment artifact not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        for record in data["d_class_records"] + data["b_class_records"]:
            assert "alignment_grade" in record
            assert "match_evidence" in record
            ev = record["match_evidence"]
            assert "numeric_recall" in ev
            assert "metric_token_recall" in ev
            assert "match_strategy" in ev
            assert "ambiguous" in ev
            assert "tiebreak_used" in ev

    def test_no_false_structural_alignment(self) -> None:
        out_path = R3_OUT / "target-structural-alignment.json"
        if not out_path.is_file():
            pytest.skip("R3.2 alignment artifact not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["false_structural_alignment"] == 0

    def test_strength_is_strong_or_acceptable(self) -> None:
        out_path = R3_OUT / "target-structural-alignment.json"
        if not out_path.is_file():
            pytest.skip("R3.2 alignment artifact not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["strength"] in ("strong", "acceptable")


# ---------------------------------------------------------------------------
# 11.3 Structural Ambiguity Closure R1 (audit_structural_ambiguity_r32_r1)
# ---------------------------------------------------------------------------


R32_R1_SCRIPT = ROOT / "scripts/evaluation/audit_structural_ambiguity_r32_r1.py"


class TestStructuralAmbiguityClosureR1:
    """Gate 02 R3.2 R1: Structural Ambiguity Closure."""

    def test_script_is_observation_only(self) -> None:
        source = R32_R1_SCRIPT.read_text(encoding="utf-8")
        assert "manual-mapping-review-package" not in source
        assert "labels.golden" not in source
        assert "questions.golden" not in source

    def test_script_does_not_rerun_adapter(self) -> None:
        source = R32_R1_SCRIPT.read_text(encoding="utf-8")
        assert "adapter-prediction-seal.json" in source
        assert "does NOT modify the adapter" in source

    def test_script_requires_r32_closure(self) -> None:
        source = R32_R1_SCRIPT.read_text(encoding="utf-8")
        assert "target-structural-alignment.json" in source
        assert "target_structural_alignment_closed" in source

    def test_tiebreak_disambiguation_present(self) -> None:
        source = R32_R1_SCRIPT.read_text(encoding="utf-8")
        assert "_disambiguate_tiebreak" in source
        for status in ("equivalent_set", "unique", "ambiguous"):
            assert status in source

    def test_low_tr_enhancement_present(self) -> None:
        source = R32_R1_SCRIPT.read_text(encoding="utf-8")
        assert "_audit_low_tr" in source
        for signal in (
            "row_label_compatible",
            "header_path_compatible",
            "table_title_compatible",
            "period_axis_compatible",
            "bbox_strong",
            "block_signature_unique",
        ):
            assert signal in source

    def test_decision_thresholds_present(self) -> None:
        source = R32_R1_SCRIPT.read_text(encoding="utf-8")
        assert "target_structural_alignment_ambiguity_closed" in source
        assert "target_structural_alignment_ambiguity_insufficient" in source
        assert "full_corpus_financial_semantic_graph" in source
        assert "stop_and_classify_missing_evidence_shapes" in source

    def test_artifact_fields(self) -> None:
        out_path = R3_OUT / "target-structural-alignment-r1.json"
        if not out_path.is_file():
            pytest.skip("R3.2 R1 alignment artifact not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["seal_verified"] is True
        assert data["r32_closure_verified"] is True
        assert data["d_class_metrics"]["total"] == 16
        assert data["b_class_metrics"]["total"] == 17
        assert "false_structural_alignment" in data
        assert "arbitrary_row_index_resolutions" in data
        assert data["production_switch_allowed"] is False

    def test_acceptance_gate_passed(self) -> None:
        out_path = R3_OUT / "r3-2-r1-acceptance.json"
        if not out_path.is_file():
            pytest.skip("R3.2 R1 acceptance not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["all_passed"] is True
        assert data["decision"] == "target_structural_alignment_ambiguity_closed"
        assert data["strength"] in ("strong", "acceptable")
        assert data["next_gate"] == "full_corpus_financial_semantic_graph"
        assert data["production_switch_allowed"] is False

    def test_no_arbitrary_row_index_resolution(self) -> None:
        out_path = R3_OUT / "target-structural-alignment-r1.json"
        if not out_path.is_file():
            pytest.skip("R3.2 R1 alignment artifact not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["arbitrary_row_index_resolutions"] == 0
        assert data["false_structural_alignment"] == 0

    def test_ambiguity_closure_no_ambiguous(self) -> None:
        out_path = R3_OUT / "ambiguity-closure.json"
        if not out_path.is_file():
            pytest.skip("R3.2 R1 ambiguity-closure not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["ambiguous_count"] == 0
        for record in data["records"]:
            assert record["alignment_status"] in ("equivalent_set", "unique")
            assert record["structure_recoverable"] is True

    def test_relaxed_text_match_no_downgrade(self) -> None:
        out_path = R3_OUT / "relaxed-text-match-audit.json"
        if not out_path.is_file():
            pytest.skip("R3.2 R1 relaxed-text-match not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["enhanced_grade_b"] == 0
        for record in data["records"]:
            assert record["enhanced_grade"] == "A"
            assert record["structure_recoverable"] is True


# ---------------------------------------------------------------------------
# 12. Finalize acceptance gates (finalize_pdf_v4_gate_02_r3)
# ---------------------------------------------------------------------------


FINALIZE_SCRIPT = ROOT / "scripts/evaluation/finalize_pdf_v4_gate_02_r3.py"


class TestFinalizeAcceptance:
    def test_finalize_script_is_oracle_blind(self) -> None:
        source = FINALIZE_SCRIPT.read_text(encoding="utf-8")
        assert "manual-mapping-review-package" not in source
        assert "labels.golden" not in source

    def test_finalize_has_five_acceptance_gates(self) -> None:
        source = FINALIZE_SCRIPT.read_text(encoding="utf-8")
        assert "coverage_gate" in source
        assert "identity_gate" in source
        assert "oracle_gate" in source
        assert "regression_gate" in source
        assert "safety_gate" in source

    def test_finalize_production_switch_not_allowed(self) -> None:
        source = FINALIZE_SCRIPT.read_text(encoding="utf-8")
        assert "production_switch_allowed" in source

    def test_acceptance_decision_passed(self) -> None:
        out_path = R3_OUT / "acceptance.json"
        if not out_path.is_file():
            pytest.skip("Acceptance not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["decision"] == "full_corpus_unified_structured_adapter_passed"
        assert data["all_passed"] is True

    def test_next_gate_is_financial_semantic_graph(self) -> None:
        out_path = R3_OUT / "next-gate.json"
        if not out_path.is_file():
            pytest.skip("Next-gate not yet generated")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["next_gate"] == "full_corpus_financial_semantic_graph"
        assert data["production_switch_allowed"] is False


# ---------------------------------------------------------------------------
# 13. Protocol safety (gate-02-r3-protocol.json)
# ---------------------------------------------------------------------------


class TestProtocolSafety:
    def test_protocol_has_zero_forbidden_runs(self) -> None:
        protocol_path = R3_OUT / "gate-02-r3-protocol.json"
        if not protocol_path.is_file():
            pytest.skip("Protocol not yet generated")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        assert protocol.get("runtime_oracle_reads") == 0
        assert protocol.get("runtime_question_reads") == 0
        assert protocol.get("runtime_governance_reads") == 0
        assert protocol.get("expected_value_reads") == 0
        assert protocol.get("header_graph_runs") == 0
        assert protocol.get("evidence_unit_builds") == 0
        assert protocol.get("index_builds") == 0
        assert protocol.get("retrieval_runs") == 0
        assert protocol.get("reranker_calls") == 0
        assert protocol.get("answer_generation_calls") == 0
        assert protocol.get("production_index_writes") == 0
        assert protocol.get("production_config_modified") is False
        assert protocol.get("production_switch_allowed") is False

    def test_protocol_forbidden_list_complete(self) -> None:
        protocol_path = R3_OUT / "gate-02-r3-protocol.json"
        if not protocol_path.is_file():
            pytest.skip("Protocol not yet generated")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        forbidden = protocol.get("forbidden", [])
        for item in [
            "oracle",
            "gold",
            "expected_value",
            "question",
            "case_id",
            "retrieval",
            "index",
            "reranker",
            "answer_generation",
            "header_graph",
            "evidence_unit",
            "candidate_view",
            "temporal_binding",
            "metric_hierarchy",
        ]:
            assert item in forbidden


# ---------------------------------------------------------------------------
# 14. Input integrity (input-integrity.json)
# ---------------------------------------------------------------------------


class TestInputIntegrity:
    def test_input_integrity_has_eight_documents(self) -> None:
        path = R3_OUT / "input-integrity.json"
        if not path.is_file():
            pytest.skip("Input integrity not yet generated")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["document_count"] == 8
        assert data["all_documents_present"] is True
        assert data["all_middle_json_present"] is True

    def test_input_integrity_records_r2_seal_hash(self) -> None:
        path = R3_OUT / "input-integrity.json"
        if not path.is_file():
            pytest.skip("Input integrity not yet generated")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("r2_seal_sha256")


# ---------------------------------------------------------------------------
# 15. Adapter prediction manifest (adapter-prediction-manifest.json)
# ---------------------------------------------------------------------------


class TestAdapterPredictionManifest:
    def test_manifest_records_counts_and_hashes(self) -> None:
        path = R3_OUT / "adapter-prediction-manifest.json"
        if not path.is_file():
            pytest.skip("Prediction manifest not yet generated")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["prediction_page_count"] == 1348
        assert data["duplicate_table_id_count"] == 0
        assert data["duplicate_row_id_count"] == 0
        assert data["duplicate_cell_id_count"] == 0
        assert data.get("predictions_hash")
        assert data.get("table_identity_hash")
        assert data.get("row_identity_hash")
        assert data.get("cell_identity_hash")


# ---------------------------------------------------------------------------
# 16. Structure metrics (full-corpus-structure-metrics.json)
# ---------------------------------------------------------------------------


class TestStructureMetrics:
    def test_structure_metrics_has_required_fields(self) -> None:
        path = R3_OUT / "full-corpus-structure-metrics.json"
        if not path.is_file():
            pytest.skip("Structure metrics not yet generated")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["page_count"] == 1348
        for field in [
            "pages_with_native_text",
            "pages_with_tables",
            "table_count",
            "row_count",
            "cell_count",
            "native_aligned_cell_count",
            "mineru_text_fallback_count",
            "ocr_fallback_count",
            "unresolved_cell_count",
            "numeric_cell_count",
            "numeric_parse_success_count",
            "scale_candidate_count",
        ]:
            assert field in data

    def test_document_metrics_has_eight_entries(self) -> None:
        path = R3_OUT / "document-structure-metrics.json"
        if not path.is_file():
            pytest.skip("Document metrics not yet generated")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data) == 8
        doc_ids = [d["document_id"] for d in data]
        assert doc_ids == sorted(doc_ids)
