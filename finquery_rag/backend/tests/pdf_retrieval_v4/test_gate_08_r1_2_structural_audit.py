"""Tests for Gate 08 R1.2 structural presence auditor."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.pdf_retrieval_v4.structural_presence_auditor import (
    StructuralPresenceAuditor,
)


# ------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------


@pytest.fixture
def temp_metadata_db(tmp_path: Path) -> Path:
    """Create a temporary metadata DB with test views."""
    db_path = tmp_path / "metadata.sqlite"
    conn = sqlite3.connect(str(db_path))

    conn.execute(
        "CREATE TABLE retrieval_views ("
        "retrieval_view_id TEXT, "
        "evidence_unit_id TEXT, "
        "unit_type TEXT, "
        "retrieval_text TEXT, "
        "metadata_json TEXT)"
    )
    conn.execute(
        "CREATE TABLE table_rows ("
        "row_id TEXT, "
        "logical_table_id TEXT, "
        "member_view_ids_json TEXT)"
    )
    conn.execute(
        "CREATE TABLE row_cells ("
        "cell_id TEXT, "
        "row_id TEXT, "
        "member_view_ids_json TEXT)"
    )
    conn.execute(
        "CREATE TABLE facts ("
        "fact_id TEXT, "
        "cell_id TEXT, "
        "member_view_ids_json TEXT)"
    )

    # Insert test views
    views = [
        # Table view on page 26
        {
            "view_id": "view:table1",
            "unit_type": "table",
            "text": "Consolidated Statements",
            "metadata": {
                "document_id": "aapl_fy2025",
                "pdf_pages": [26],
                "logical_table_id": "tbl:1",
            },
        },
        # Row view on page 26 matching "total net sales"
        {
            "view_id": "view:row1",
            "unit_type": "row",
            "text": "Total net sales",
            "metadata": {
                "document_id": "aapl_fy2025",
                "pdf_pages": [26],
                "logical_table_id": "tbl:1",
                "row_id": "row:1",
                "metric_path": "Total net sales",
            },
        },
        # Fact view on page 26
        {
            "view_id": "view:fact1",
            "unit_type": "atomic_fact",
            "text": "Total net sales FY2025 416191",
            "metadata": {
                "document_id": "aapl_fy2025",
                "pdf_pages": [26],
                "logical_table_id": "tbl:1",
                "row_id": "row:1",
                "fact_id": "fact:1",
                "periods": ["FY2025"],
            },
        },
        # Narrative view on page 30
        {
            "view_id": "view:narr1",
            "unit_type": "section",
            "text": "Risk Factors",
            "metadata": {
                "document_id": "aapl_fy2025",
                "pdf_pages": [30],
            },
        },
    ]

    for v in views:
        conn.execute(
            "INSERT INTO retrieval_views VALUES (?, ?, ?, ?, ?)",
            (
                v["view_id"],
                f"eu:{v['view_id']}",
                v["unit_type"],
                v["text"],
                json.dumps(v["metadata"]),
            ),
        )

    # Insert table_rows
    conn.execute(
        "INSERT INTO table_rows VALUES (?, ?, ?)",
        ("row:1", "tbl:1", json.dumps(["view:row1"])),
    )

    # Insert row_cells
    conn.execute(
        "INSERT INTO row_cells VALUES (?, ?, ?)",
        ("cell:1", "row:1", json.dumps(["view:fact1"])),
    )

    # Insert facts
    conn.execute(
        "INSERT INTO facts VALUES (?, ?, ?)",
        ("fact:1", "cell:1", json.dumps(["view:fact1"])),
    )

    conn.commit()
    conn.close()
    return db_path


# ------------------------------------------------------------------
# StructuralPresenceAuditor tests
# ------------------------------------------------------------------


class TestStructuralPresenceAuditorInit:
    def test_loads_metadata(self, temp_metadata_db: Path) -> None:
        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            assert auditor.total_views == 4

    def test_context_manager(self, temp_metadata_db: Path) -> None:
        auditor = StructuralPresenceAuditor(temp_metadata_db)
        auditor.close()
        # Should not raise


class TestLayerChecks:
    def test_layer1_page_present(self, temp_metadata_db: Path) -> None:
        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            audit = auditor.audit_gold_source(
                case_id="test_case",
                source_index=0,
                gold_candidate_key="candidate:v1:test1",
                gold_document_id="aapl_fy2025",
                gold_page=26,
                gold_row_label="Total net sales",
                gold_period="FY2025",
                r1_strict_mapped=False,
            )
            assert audit.layer1_page_present is True

    def test_layer1_page_absent(self, temp_metadata_db: Path) -> None:
        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            audit = auditor.audit_gold_source(
                case_id="test_case",
                source_index=0,
                gold_candidate_key="candidate:v1:test1",
                gold_document_id="aapl_fy2025",
                gold_page=999,
                r1_strict_mapped=False,
            )
            assert audit.layer1_page_present is False
            assert audit.failure_class == "S4_mineru_structure_missing"
            assert audit.pdf_reprocessing_required is True

    def test_layer2_table_present(self, temp_metadata_db: Path) -> None:
        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            audit = auditor.audit_gold_source(
                case_id="test_case",
                source_index=0,
                gold_candidate_key="candidate:v1:test1",
                gold_document_id="aapl_fy2025",
                gold_page=26,
                gold_row_label="Total net sales",
                gold_period="FY2025",
                r1_strict_mapped=False,
            )
            assert audit.layer2_table_present is True

    def test_layer3_row_present(self, temp_metadata_db: Path) -> None:
        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            audit = auditor.audit_gold_source(
                case_id="test_case",
                source_index=0,
                gold_candidate_key="candidate:v1:test1",
                gold_document_id="aapl_fy2025",
                gold_page=26,
                gold_row_label="Total net sales",
                gold_period="FY2025",
                r1_strict_mapped=False,
            )
            assert audit.layer3_row_present is True

    def test_layer3_row_absent_mismatch(self, temp_metadata_db: Path) -> None:
        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            audit = auditor.audit_gold_source(
                case_id="test_case",
                source_index=0,
                gold_candidate_key="candidate:v1:test1",
                gold_document_id="aapl_fy2025",
                gold_page=26,
                gold_row_label="Nonexistent Metric",
                gold_period="FY2025",
                r1_strict_mapped=False,
            )
            assert audit.layer3_row_present is False


class TestFailureClassification:
    def test_s4_page_missing(self, temp_metadata_db: Path) -> None:
        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            audit = auditor.audit_gold_source(
                case_id="test_case",
                source_index=0,
                gold_candidate_key="candidate:v1:test1",
                gold_document_id="aapl_fy2025",
                gold_page=999,
                r1_strict_mapped=False,
            )
            assert audit.failure_class == "S4_mineru_structure_missing"
            assert audit.pdf_reprocessing_required is True
            assert audit.recommended_action == "targeted_pdf_reprocessing"

    def test_s5_narrative_evidence(
        self, temp_metadata_db: Path
    ) -> None:
        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            audit = auditor.audit_gold_source(
                case_id="test_case",
                source_index=0,
                gold_candidate_key="candidate:v1:test1",
                gold_document_id="aapl_fy2025",
                gold_page=30,
                gold_evidence_type="narrative",
                r1_strict_mapped=False,
            )
            assert audit.failure_class == "S5_narrative_evidence"
            assert audit.pdf_reprocessing_required is False

    def test_s1_bridge_missing(
        self, temp_metadata_db: Path
    ) -> None:
        """Structure exists (table+row+fact) but no candidate bridge."""
        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            audit = auditor.audit_gold_source(
                case_id="test_case",
                source_index=0,
                gold_candidate_key="candidate:v1:test1",
                gold_document_id="aapl_fy2025",
                gold_page=26,
                gold_row_label="Total net sales",
                gold_period="FY2025",
                r1_strict_mapped=False,
                r1_matched_view_id="view:fact1",
            )
            assert audit.layer5_candidate_bridge is False
            # Structure exists but no bridge
            assert audit.failure_class == "S1_bridge_missing"
            assert audit.recommended_action == "candidate_bridge_expansion"

    def test_s2_granularity_mismatch(
        self, temp_metadata_db: Path
    ) -> None:
        """Table exists but row not found."""
        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            audit = auditor.audit_gold_source(
                case_id="test_case",
                source_index=0,
                gold_candidate_key="candidate:v1:test1",
                gold_document_id="aapl_fy2025",
                gold_page=26,
                gold_row_label="Nonexistent Row",
                gold_period="FY2025",
                r1_strict_mapped=False,
            )
            assert audit.layer2_table_present is True
            assert audit.layer3_row_present is False
            assert audit.failure_class == "S2_candidate_granularity_mismatch"

    def test_s6_candidate_mapping_error(
        self, temp_metadata_db: Path
    ) -> None:
        """Strict mapping exists but not retrieved."""
        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            audit = auditor.audit_gold_source(
                case_id="test_case",
                source_index=0,
                gold_candidate_key="candidate:v1:test1",
                gold_document_id="aapl_fy2025",
                gold_page=26,
                gold_row_label="Total net sales",
                gold_period="FY2025",
                r1_strict_mapped=True,
                r1_matched_view_id="view:fact1",
            )
            assert audit.layer5_candidate_bridge is True
            assert audit.failure_class == "S6_candidate_mapping_error"


class TestAuditSerialization:
    def test_to_dict(self, temp_metadata_db: Path) -> None:
        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            audit = auditor.audit_gold_source(
                case_id="test_case",
                source_index=0,
                gold_candidate_key="candidate:v1:test1",
                gold_document_id="aapl_fy2025",
                gold_page=26,
                gold_row_label="Total net sales",
                gold_period="FY2025",
                r1_strict_mapped=False,
            )
            d = audit.to_dict()
            assert d["gold_source_identity"] == "test_case#0"
            assert d["document_id"] == "aapl_fy2025"
            assert d["pdf_page"] == 26
            assert "layer1_page_present" in d
            assert "failure_class" in d
            assert "recommended_action" in d
            assert "pdf_reprocessing_required" in d

    def test_audit_is_frozen(self, temp_metadata_db: Path) -> None:
        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            audit = auditor.audit_gold_source(
                case_id="test_case",
                source_index=0,
                gold_candidate_key="candidate:v1:test1",
                gold_document_id="aapl_fy2025",
                gold_page=26,
                r1_strict_mapped=False,
            )
            with pytest.raises(AttributeError):
                audit.failure_class = "modified"  # type: ignore[misc]


class TestMutualExclusivity:
    def test_each_audit_has_exactly_one_class(
        self, temp_metadata_db: Path
    ) -> None:
        """Each audit result must have exactly one failure_class."""
        test_cases = [
            {"page": 999, "expected_class": "S4_mineru_structure_missing"},
            {
                "page": 30,
                "evidence_type": "narrative",
                "expected_class": "S5_narrative_evidence",
            },
            {
                "page": 26,
                "row_label": "Total net sales",
                "period": "FY2025",
                "strict_mapped": False,
                "expected_class": "S1_bridge_missing",
            },
            {
                "page": 26,
                "row_label": "Nonexistent",
                "period": "FY2025",
                "strict_mapped": False,
                "expected_class": "S2_candidate_granularity_mismatch",
            },
            {
                "page": 26,
                "row_label": "Total net sales",
                "period": "FY2025",
                "strict_mapped": True,
                "expected_class": "S6_candidate_mapping_error",
            },
        ]

        with StructuralPresenceAuditor(temp_metadata_db) as auditor:
            for i, tc in enumerate(test_cases):
                audit = auditor.audit_gold_source(
                    case_id=f"case_{i}",
                    source_index=0,
                    gold_candidate_key=f"candidate:v1:test{i}",
                    gold_document_id="aapl_fy2025",
                    gold_page=tc.get("page"),
                    gold_row_label=tc.get("row_label"),
                    gold_period=tc.get("period"),
                    gold_evidence_type=tc.get("evidence_type"),
                    r1_strict_mapped=tc.get("strict_mapped", False),
                    r1_matched_view_id=(
                        "view:fact1" if tc.get("strict_mapped") else None
                    ),
                )
                assert audit.failure_class == tc["expected_class"], (
                    f"Case {i}: expected {tc['expected_class']}, "
                    f"got {audit.failure_class}"
                )
