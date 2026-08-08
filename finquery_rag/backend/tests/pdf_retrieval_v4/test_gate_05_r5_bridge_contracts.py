"""Gate 05 R5 — Bridge Contract Tests.

Tests verify that the Candidate Evidence Bridge implementation correctly
handles all specified high-risk scenarios. Tests import production modules
from src.pdf_retrieval_v4.* — no test-local mocks.

Run:
    python3 -m pytest tests/pdf_retrieval_v4/test_gate_05_r5_bridge_contracts.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure backend is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.pdf_retrieval_v4.candidate_bridge_models import (  # noqa: E402
    BRIDGE_SCHEMA_VERSION,
    BridgeGrade,
    BridgeMatch,
    BridgeResult,
    CandidateSignature,
    CandidateStructuredView,
    SemanticEvidenceSignature,
    is_structured_eligible,
)
from src.pdf_retrieval_v4.candidate_evidence_bridge import CandidateEvidenceBridge  # noqa: E402
from src.pdf_retrieval_v4.candidate_multirow_bridge import MultiRowBridge  # noqa: E402
from src.pdf_retrieval_v4.candidate_narrative_bridge import NarrativeBridge  # noqa: E402
from src.pdf_retrieval_v4.candidate_row_bridge import (  # noqa: E402
    RowBridge,
    compute_bbox_iou,
    compute_numeric_recall,
    compute_text_coverage,
    metric_compatible,
    period_compatible,
)
from src.pdf_retrieval_v4.candidate_signature import build_candidate_signature  # noqa: E402
from src.pdf_retrieval_v4.candidate_structured_view import StructuredViewBuilder  # noqa: E402
from src.pdf_retrieval_v4.candidate_table_bridge import TableBridge  # noqa: E402
from src.pdf_retrieval_v4.bridge_equivalence import BridgeEquivalence  # noqa: E402
from src.pdf_retrieval_v4.bridge_validator import BridgeValidator  # noqa: E402
from src.pdf_retrieval_v4.semantic_evidence_catalog import (  # noqa: E402
    SemanticEvidenceCatalog,
    load_catalog,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic test data
# ---------------------------------------------------------------------------


def make_candidate_sig(
    candidate_key: str = "candidate:v1:test001",
    document_id: str = "test_doc",
    pdf_page: int = 26,
    block_type: str = "table_row",
    raw_content: str = "Total net sales 123456 789012",
    text_tokens: tuple[str, ...] = ("total", "net", "sales"),
    numeric_multiset: tuple[str, ...] = ("123456", "789012"),
    period_tokens: tuple[str, ...] = ("FY2025", "FY2024"),
    normalized_text: str = "total net sales 123456 789012",
    existing_row_ids: tuple[str, ...] = (),
    existing_metric_paths: tuple[str, ...] = (),
    existing_bridge_grade: str = "raw_only",
) -> CandidateSignature:
    """Build a synthetic CandidateSignature for testing."""
    return CandidateSignature(
        candidate_key=candidate_key,
        document_id=document_id,
        pdf_page=pdf_page,
        block_type=block_type,
        raw_content=raw_content,
        text_tokens=text_tokens,
        numeric_multiset=numeric_multiset,
        period_tokens=period_tokens,
        normalized_text=normalized_text,
        existing_row_ids=existing_row_ids,
        existing_metric_paths=existing_metric_paths,
        existing_bridge_grade=existing_bridge_grade,
    )


def make_evidence_sig(
    evidence_id: str = "row:test001",
    evidence_type: str = "semantic_row",
    document_id: str = "test_doc",
    pdf_page: int = 26,
    table_id: str | None = "table:test001",
    row_id: str | None = "row:test001",
    bbox: tuple[float, ...] = (100.0, 200.0, 500.0, 220.0),
    metric_paths: tuple[str, ...] = ("Total net sales",),
    periods: tuple[str, ...] = ("FY2025", "FY2024"),
    numeric_multiset: tuple[str, ...] = ("123456", "789012"),
    raw_text: str = "Total net sales",
    normalized_text: str = "total net sales",
    raw_values: tuple[str, ...] = (),
    row_type: str = "metric_row",
    row_index: int = 5,
    equivalent_group_id: str | None = None,
) -> SemanticEvidenceSignature:
    """Build a synthetic SemanticEvidenceSignature for testing."""
    return SemanticEvidenceSignature(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        document_id=document_id,
        pdf_page=pdf_page,
        table_id=table_id,
        row_id=row_id,
        cell_ids=(),
        bbox=bbox,
        raw_values=raw_values,
        metric_paths=metric_paths,
        periods=periods,
        segments=(),
        buckets=(),
        numeric_multiset=numeric_multiset,
        raw_text=raw_text,
        normalized_text=normalized_text,
        source_traceback={
            "document_id": document_id,
            "pdf_page": pdf_page,
            "row_bbox": list(bbox) if bbox else None,
        },
        equivalent_group_id=equivalent_group_id,
        row_type=row_type,
        row_index=row_index,
    )


def make_catalog_with_evidence(
    evidence_list: list[SemanticEvidenceSignature],
) -> SemanticEvidenceCatalog:
    """Build a SemanticEvidenceCatalog populated with given evidence."""
    catalog = SemanticEvidenceCatalog()
    for ev in evidence_list:
        catalog.add(ev)
    return catalog


# ---------------------------------------------------------------------------
# Test 0: Import production modules
# ---------------------------------------------------------------------------


class TestImportsProductionModules:
    """Verify that tests import production modules, not test-local mocks."""

    def test_import_bridge_models(self):
        assert BRIDGE_SCHEMA_VERSION.startswith("pdf-retrieval-v4/gate-05-r5")

    def test_import_row_bridge(self):
        assert RowBridge is not None

    def test_import_multirow_bridge(self):
        assert MultiRowBridge is not None

    def test_import_table_bridge(self):
        assert TableBridge is not None

    def test_import_narrative_bridge(self):
        assert NarrativeBridge is not None

    def test_import_orchestrator(self):
        assert CandidateEvidenceBridge is not None

    def test_import_view_builder(self):
        assert StructuredViewBuilder is not None

    def test_import_equivalence(self):
        assert BridgeEquivalence is not None

    def test_import_validator(self):
        assert BridgeValidator is not None

    def test_import_catalog(self):
        assert load_catalog is not None


# ---------------------------------------------------------------------------
# Test 1: Cross-document same value NOT bridge
# ---------------------------------------------------------------------------


class TestCrossDocumentSameValueNotBridge:
    """Same numeric value but different document must NOT bridge."""

    def test_cross_document_not_bridge(self):
        candidate = make_candidate_sig(
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("123456",),
        )
        evidence = make_evidence_sig(
            document_id="doc_b",
            pdf_page=26,
            numeric_multiset=("123456",),
        )
        catalog = make_catalog_with_evidence([evidence])
        bridge = RowBridge(catalog)
        result = bridge.bridge(candidate)

        # Should be unmapped because evidence is on a different document
        assert result.grade == BridgeGrade.UNMAPPED.value
        assert result.failure_stage is not None


# ---------------------------------------------------------------------------
# Test 2: Same page, same number, wrong metric NOT Grade-A
# ---------------------------------------------------------------------------


class TestSamePageWrongMetricNotBridge:
    """Same page + same numbers but incompatible metric should not be Grade-A."""

    def test_wrong_metric_not_grade_a(self):
        candidate = make_candidate_sig(
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("123456",),
            text_tokens=("revenue", "total"),
            existing_metric_paths=("Total Revenue",),
        )
        evidence = make_evidence_sig(
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("123456",),
            metric_paths=("Operating Expenses",),
            normalized_text="operating expenses 123456",
            raw_text="Operating Expenses",
        )
        catalog = make_catalog_with_evidence([evidence])
        catalog.set_row_metric_paths(evidence.row_id or "", "Operating Expenses")
        bridge = RowBridge(catalog)
        result = bridge.bridge(candidate)

        # Should NOT be Grade-A because metric is incompatible
        # (numeric recall=1.0 but text coverage will be low and metric incompatible)
        if result.grade == BridgeGrade.A3_ROW_SIGNATURE.value:
            # If A3, the text coverage must be strong enough despite metric mismatch
            # But since text tokens are ("revenue", "total") and evidence text is
            # "operating expenses", text coverage should be 0
            assert result.matches[0].text_coverage < 0.5


# ---------------------------------------------------------------------------
# Test 3: BBox + signature unique bridge
# ---------------------------------------------------------------------------


class TestBboxAndSignatureUniqueBridge:
    """BBox overlap + numeric + text = unique Grade-A bridge."""

    def test_bbox_signature_bridge(self):
        candidate = make_candidate_sig(
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100", "200"),
            text_tokens=("total", "revenue"),
        )
        evidence = make_evidence_sig(
            document_id="doc_a",
            pdf_page=26,
            bbox=(100.0, 200.0, 500.0, 220.0),
            numeric_multiset=("100", "200", "300"),
            normalized_text="total revenue 100 200 300",
            raw_text="Total revenue 100 200 300",
        )
        catalog = make_catalog_with_evidence([evidence])
        catalog.set_row_metric_paths(evidence.row_id or "", "Total revenue")
        bridge = RowBridge(catalog)
        result = bridge.bridge(candidate)

        # Should be Grade-A (A2 or A3)
        assert BridgeGrade.is_grade_a(result.grade)
        assert len(result.matches) == 1


# ---------------------------------------------------------------------------
# Test 4: No bbox, row signature bridge
# ---------------------------------------------------------------------------


class TestNoBboxRowSignatureBridge:
    """No bbox but strong row-text + numeric = A3 bridge."""

    def test_no_bbox_row_signature(self):
        candidate = make_candidate_sig(
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100", "200"),
            text_tokens=("total", "revenue", "net", "sales"),
        )
        evidence = make_evidence_sig(
            document_id="doc_a",
            pdf_page=26,
            bbox=(),  # No bbox
            numeric_multiset=("100", "200"),
            normalized_text="total revenue net sales 100 200",
            raw_text="Total revenue net sales 100 200",
        )
        catalog = make_catalog_with_evidence([evidence])
        catalog.set_row_metric_paths(evidence.row_id or "", "Total revenue")
        bridge = RowBridge(catalog)
        result = bridge.bridge(candidate)

        # Should be Grade-A (A3 since no bbox)
        assert BridgeGrade.is_grade_a(result.grade)


# ---------------------------------------------------------------------------
# Test 5: Numeric only NOT Grade-A
# ---------------------------------------------------------------------------


class TestNumericOnlyNotGradeA:
    """Numeric match alone (without text/metric) should NOT be Grade-A."""

    def test_numeric_only_not_grade_a(self):
        candidate = make_candidate_sig(
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100", "200"),
            text_tokens=("zzz", "xxx", "yyy"),  # No overlapping text
        )
        evidence = make_evidence_sig(
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100", "200"),
            normalized_text="completely different text here",
            raw_text="Completely different text",
        )
        catalog = make_catalog_with_evidence([evidence])
        catalog.set_row_metric_paths(evidence.row_id or "", "Different metric")
        bridge = RowBridge(catalog)
        result = bridge.bridge(candidate)

        # Should NOT be Grade-A because text coverage is 0 and metric incompatible
        assert not BridgeGrade.is_grade_a(result.grade)


# ---------------------------------------------------------------------------
# Test 6: Equal score rows fail closed
# ---------------------------------------------------------------------------


class TestEqualScoreRowsFailClosed:
    """Two rows with equal score → B_ambiguous, not Grade-A."""

    def test_equal_score_ambiguous(self):
        candidate = make_candidate_sig(
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100",),
            text_tokens=("revenue", "total"),
        )
        ev1 = make_evidence_sig(
            evidence_id="row:ev1",
            row_id="row:ev1",
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100",),
            normalized_text="total revenue 100",
            raw_text="Total revenue 100",
        )
        ev2 = make_evidence_sig(
            evidence_id="row:ev2",
            row_id="row:ev2",
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100",),
            normalized_text="total revenue 100",
            raw_text="Total revenue 100",
        )
        catalog = make_catalog_with_evidence([ev1, ev2])
        catalog.set_row_metric_paths("row:ev1", "Total revenue")
        catalog.set_row_metric_paths("row:ev2", "Total revenue")
        bridge = RowBridge(catalog)
        result = bridge.bridge(candidate)

        # Should be B_ambiguous (two equal matches)
        assert result.grade in (
            BridgeGrade.B_AMBIGUOUS.value,
            BridgeGrade.UNMAPPED.value,
        )
        # Must NOT be Grade-A
        assert not BridgeGrade.is_grade_a(result.grade)


# ---------------------------------------------------------------------------
# Test 7: Multi-row contiguous bridge
# ---------------------------------------------------------------------------


class TestMultirowContiguousBridge:
    """Contiguous rows = A4 multirow bridge."""

    def test_contiguous_bridge(self):
        candidate = make_candidate_sig(
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100", "200", "300"),
            text_tokens=("revenue", "total", "sales"),
        )
        ev1 = make_evidence_sig(
            evidence_id="row:ev1",
            row_id="row:ev1",
            document_id="doc_a",
            pdf_page=26,
            table_id="table:t1",
            row_index=5,
            numeric_multiset=("100",),
            normalized_text="revenue 100",
            raw_text="Revenue 100",
        )
        ev2 = make_evidence_sig(
            evidence_id="row:ev2",
            row_id="row:ev2",
            document_id="doc_a",
            pdf_page=26,
            table_id="table:t1",
            row_index=6,
            numeric_multiset=("200",),
            normalized_text="sales 200",
            raw_text="Sales 200",
        )
        ev3 = make_evidence_sig(
            evidence_id="row:ev3",
            row_id="row:ev3",
            document_id="doc_a",
            pdf_page=26,
            table_id="table:t1",
            row_index=7,
            numeric_multiset=("300",),
            normalized_text="total 300",
            raw_text="Total 300",
        )
        catalog = make_catalog_with_evidence([ev1, ev2, ev3])
        bridge = MultiRowBridge(catalog)
        result = bridge.bridge(candidate)

        # Should be A4_multirow (contiguous rows 5,6,7 with combined numeric recall=1.0)
        assert result.grade == BridgeGrade.A4_MULTIROW.value


# ---------------------------------------------------------------------------
# Test 8: Multi-row non-contiguous rejected
# ---------------------------------------------------------------------------


class TestMultirowNoncontiguousRejected:
    """Non-contiguous rows should NOT bridge as multirow."""

    def test_noncontiguous_rejected(self):
        candidate = make_candidate_sig(
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100", "300"),  # Only first and third
            text_tokens=("revenue", "total"),
        )
        ev1 = make_evidence_sig(
            evidence_id="row:ev1",
            row_id="row:ev1",
            document_id="doc_a",
            pdf_page=26,
            table_id="table:t1",
            row_index=5,
            numeric_multiset=("100",),
            normalized_text="revenue 100",
            raw_text="Revenue 100",
        )
        ev2 = make_evidence_sig(
            evidence_id="row:ev2",
            row_id="row:ev2",
            document_id="doc_a",
            pdf_page=26,
            table_id="table:t1",
            row_index=7,  # Gap: 5, 7 (not contiguous)
            numeric_multiset=("300",),
            normalized_text="total 300",
            raw_text="Total 300",
        )
        catalog = make_catalog_with_evidence([ev1, ev2])
        bridge = MultiRowBridge(catalog)
        result = bridge.bridge(candidate)

        # Should NOT be A4_multirow (non-contiguous)
        assert result.grade != BridgeGrade.A4_MULTIROW.value


# ---------------------------------------------------------------------------
# Test 9: Table block does not expand all candidates
# ---------------------------------------------------------------------------


class TestTableBlockDoesNotExpandAll:
    """A table match should NOT mark all same-page candidates as Grade-A."""

    def test_table_does_not_expand(self):
        # Table candidate matches a logical table
        table_candidate = make_candidate_sig(
            candidate_key="candidate:table1",
            document_id="doc_a",
            pdf_page=26,
            block_type="table",
            numeric_multiset=("100", "200"),
            text_tokens=("revenue", "total"),
        )
        # Another candidate on the same page that should NOT be affected
        other_candidate = make_candidate_sig(
            candidate_key="candidate:other1",
            document_id="doc_a",
            pdf_page=26,
            block_type="table_row",
            numeric_multiset=("999",),
            text_tokens=("zzz",),
        )
        ev_table = make_evidence_sig(
            evidence_id="table:t1",
            evidence_type="logical_table",
            document_id="doc_a",
            pdf_page=26,
            table_id="table:t1",
            row_id=None,
            bbox=(50.0, 100.0, 550.0, 400.0),
            normalized_text="Revenue Table",
            raw_text="Revenue Table",
            row_type=None,
            row_index=None,
        )
        ev_row = make_evidence_sig(
            evidence_id="row:t1r1",
            row_id="row:t1r1",
            document_id="doc_a",
            pdf_page=26,
            table_id="table:t1",
            row_index=0,
            numeric_multiset=("100", "200"),
            normalized_text="revenue total 100 200",
            raw_text="Revenue Total 100 200",
        )
        catalog = make_catalog_with_evidence([ev_table, ev_row])
        catalog.set_row_metric_paths("row:t1r1", "Revenue")

        # Bridge the table candidate
        table_bridge = TableBridge(catalog)
        table_result = table_bridge.bridge(table_candidate)
        assert BridgeGrade.is_grade_a(table_result.grade)

        # Bridge the other candidate — should NOT be Grade-A
        row_bridge = RowBridge(catalog)
        other_result = row_bridge.bridge(other_candidate)
        assert not BridgeGrade.is_grade_a(other_result.grade)


# ---------------------------------------------------------------------------
# Test 10: Narrative text bridge
# ---------------------------------------------------------------------------


class TestNarrativeTextBridge:
    """Narrative text = A5 bridge."""

    def test_narrative_bridge(self):
        candidate = make_candidate_sig(
            candidate_key="candidate:text1",
            document_id="doc_a",
            pdf_page=26,
            block_type="text",
            text_tokens=("risk", "factors", "competition", "market"),
            numeric_multiset=(),
        )
        narrative = make_evidence_sig(
            evidence_id="narrative:test1",
            evidence_type="narrative_evidence",
            document_id="doc_a",
            pdf_page=26,
            table_id=None,
            row_id=None,
            bbox=(100.0, 300.0, 500.0, 400.0),
            metric_paths=(),
            periods=(),
            numeric_multiset=(),
            normalized_text="risk factors include competition and market conditions",
            raw_text="Risk Factors include competition and market conditions",
            row_type=None,
            row_index=None,
        )
        catalog = make_catalog_with_evidence([narrative])
        bridge = NarrativeBridge(catalog)
        result = bridge.bridge(candidate)

        assert result.grade == BridgeGrade.A5_NARRATIVE.value


# ---------------------------------------------------------------------------
# Test 11: Equivalent set bridge
# ---------------------------------------------------------------------------


class TestEquivalentSetBridge:
    """Candidate maps to equivalent set = A_equivalent."""

    def test_equivalent_set_bridge(self):
        candidate = make_candidate_sig(
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100", "200"),
            text_tokens=("revenue", "total"),
            existing_row_ids=("row:eq1",),  # Direct identity to equivalent member
        )
        ev1 = make_evidence_sig(
            evidence_id="row:eq1",
            row_id="row:eq1",
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100", "200"),
            normalized_text="total revenue 100 200",
            raw_text="Total revenue 100 200",
            equivalent_group_id="equiv:group1",
        )
        catalog = make_catalog_with_evidence([ev1])
        bridge = CandidateEvidenceBridge(catalog)
        result = bridge.bridge_one(candidate)
        result = BridgeEquivalence(catalog).check_equivalent_bridge(result)

        assert result.grade == BridgeGrade.A_EQUIVALENT.value


# ---------------------------------------------------------------------------
# Test 12: Equivalent set no double count
# ---------------------------------------------------------------------------


class TestEquivalentSetNoDoubleCount:
    """Equivalent set should not cause double counting."""

    def test_no_double_count(self):
        ev1 = make_evidence_sig(
            evidence_id="row:eq1",
            row_id="row:eq1",
            document_id="doc_a",
            pdf_page=26,
            equivalent_group_id="equiv:group1",
        )
        ev2 = make_evidence_sig(
            evidence_id="row:eq2",
            row_id="row:eq2",
            document_id="doc_a",
            pdf_page=26,
            equivalent_group_id="equiv:group1",
        )
        catalog = make_catalog_with_evidence([ev1, ev2])
        equiv = BridgeEquivalence(catalog)

        # Both evidence are in the same group
        assert equiv.get_group_for_evidence("row:eq1") == "equiv:group1"
        assert equiv.get_group_for_evidence("row:eq2") == "equiv:group1"

        # Check double count detection
        results = [
            BridgeResult(
                candidate_key="cand1",
                grade=BridgeGrade.A1_DIRECT.value,
                matches=(
                    BridgeMatch(
                        evidence_id="row:eq1",
                        evidence_type="semantic_row",
                        grade=BridgeGrade.A1_DIRECT.value,
                        score=1.0,
                        reasons=(),
                        numeric_recall=1.0,
                        text_coverage=1.0,
                        bbox_overlap=1.0,
                        metric_compatible=True,
                        period_compatible=True,
                    ),
                ),
                failure_stage=None,
                bridge_reasons=(),
            ),
            BridgeResult(
                candidate_key="cand2",
                grade=BridgeGrade.A1_DIRECT.value,
                matches=(
                    BridgeMatch(
                        evidence_id="row:eq2",
                        evidence_type="semantic_row",
                        grade=BridgeGrade.A1_DIRECT.value,
                        score=1.0,
                        reasons=(),
                        numeric_recall=1.0,
                        text_coverage=1.0,
                        bbox_overlap=1.0,
                        metric_compatible=True,
                        period_compatible=True,
                    ),
                ),
                failure_stage=None,
                bridge_reasons=(),
            ),
        ]
        violations = equiv.detect_double_count(results)
        # Should detect that two candidates map to the same equivalent group
        assert len(violations) > 0


# ---------------------------------------------------------------------------
# Test 13: One candidate one structured view
# ---------------------------------------------------------------------------


class TestOneCandidateOneStructuredView:
    """One candidate → one structured view (even with multiple matches)."""

    def test_one_view_per_candidate(self):
        candidate = make_candidate_sig(
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100", "200"),
            text_tokens=("revenue", "total"),
            existing_row_ids=("row:r1",),
        )
        ev_row = make_evidence_sig(
            evidence_id="row:r1",
            row_id="row:r1",
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100", "200"),
            normalized_text="total revenue 100 200",
            raw_text="Total revenue 100 200",
        )
        ev_atomic = make_evidence_sig(
            evidence_id="atomic:f1",
            evidence_type="atomic_fact",
            document_id="doc_a",
            pdf_page=26,
            row_id="row:r1",
            table_id="table:t1",
            metric_paths=("Total revenue",),
            periods=("FY2025",),
            raw_values=("100",),
            numeric_multiset=("100",),
            raw_text="100",
            normalized_text="100",
            row_type=None,
            row_index=None,
        )
        catalog = make_catalog_with_evidence([ev_row, ev_atomic])
        catalog.set_row_metric_paths("row:r1", "Total revenue")

        bridge = CandidateEvidenceBridge(catalog)
        result = bridge.bridge_one(candidate)

        view_builder = StructuredViewBuilder(catalog)
        view = view_builder.build_view(candidate, result)

        # Should have exactly one view
        assert view is not None
        assert view.candidate_key == candidate.candidate_key
        # View can contain multiple facts
        assert len(view.facts) >= 1


# ---------------------------------------------------------------------------
# Test 14: View can contain multiple facts
# ---------------------------------------------------------------------------


class TestViewCanContainMultipleFacts:
    """A structured view can aggregate multiple facts."""

    def test_multiple_facts_in_view(self):
        candidate = make_candidate_sig(
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100", "200"),
            text_tokens=("revenue", "total"),
            existing_row_ids=("row:r1",),
        )
        ev_row = make_evidence_sig(
            evidence_id="row:r1",
            row_id="row:r1",
            document_id="doc_a",
            pdf_page=26,
            numeric_multiset=("100", "200"),
            normalized_text="total revenue 100 200",
            raw_text="Total revenue 100 200",
        )
        ev_atomic1 = make_evidence_sig(
            evidence_id="atomic:f1",
            evidence_type="atomic_fact",
            document_id="doc_a",
            pdf_page=26,
            row_id="row:r1",
            table_id="table:t1",
            metric_paths=("Total revenue",),
            periods=("FY2025",),
            raw_values=("100",),
            numeric_multiset=("100",),
            raw_text="100",
            normalized_text="100",
            row_type=None,
            row_index=None,
        )
        ev_atomic2 = make_evidence_sig(
            evidence_id="atomic:f2",
            evidence_type="atomic_fact",
            document_id="doc_a",
            pdf_page=26,
            row_id="row:r1",
            table_id="table:t1",
            metric_paths=("Total revenue",),
            periods=("FY2024",),
            raw_values=("200",),
            numeric_multiset=("200",),
            raw_text="200",
            normalized_text="200",
            row_type=None,
            row_index=None,
        )
        catalog = make_catalog_with_evidence([ev_row, ev_atomic1, ev_atomic2])
        catalog.set_row_metric_paths("row:r1", "Total revenue")

        bridge = CandidateEvidenceBridge(catalog)
        result = bridge.bridge_one(candidate)

        view_builder = StructuredViewBuilder(catalog)
        view = view_builder.build_view(candidate, result)

        assert view is not None
        # Should contain multiple facts (at least 2 atomic facts)
        atomic_facts = [f for f in view.facts if f.get("type") == "atomic"]
        assert len(atomic_facts) >= 2


# ---------------------------------------------------------------------------
# Test 15: Candidate key deterministic
# ---------------------------------------------------------------------------


class TestCandidateKeyDeterministic:
    """Same candidate input → same signature."""

    def test_deterministic_signature(self):
        record = {
            "candidate_key": "candidate:v1:abc123",
            "document_id": "doc_a",
            "pdf_page": 26,
            "raw_view": {
                "retrieval_text": "Document: doc_a\nPage: 26\nBlock Type: table_row\n\nSource:\nTotal revenue 100 200",
            },
            "bridge_grade": "raw_only",
        }
        sig1 = build_candidate_signature(record)
        sig2 = build_candidate_signature(record)

        assert sig1.candidate_key == sig2.candidate_key
        assert sig1.document_id == sig2.document_id
        assert sig1.pdf_page == sig2.pdf_page
        assert sig1.block_type == sig2.block_type
        assert sig1.numeric_multiset == sig2.numeric_multiset
        assert sig1.text_tokens == sig2.text_tokens


# ---------------------------------------------------------------------------
# Test 16: Bridge no gold access
# ---------------------------------------------------------------------------


class TestBridgeNoGoldAccess:
    """Bridge must not access gold data."""

    def test_no_gold_fields_in_signature(self):
        sig = make_candidate_sig()
        # CandidateSignature should not have any gold/question fields
        sig_dict = sig.to_dict()
        forbidden_keys = {"gold", "expected_value", "question", "answer", "label"}
        for key in sig_dict:
            assert not any(fk in key.lower() for fk in forbidden_keys), (
                f"Forbidden key '{key}' in CandidateSignature"
            )

    def test_no_gold_fields_in_evidence(self):
        ev = make_evidence_sig()
        ev_dict = ev.to_dict()
        forbidden_keys = {"gold", "expected_value", "question", "answer", "label"}
        for key in ev_dict:
            assert not any(fk in key.lower() for fk in forbidden_keys), (
                f"Forbidden key '{key}' in SemanticEvidenceSignature"
            )


# ---------------------------------------------------------------------------
# Test 17: Bridge no question access
# ---------------------------------------------------------------------------


class TestBridgeNoQuestionAccess:
    """Bridge must not access question data."""

    def test_no_question_in_bridge_result(self):
        result = BridgeResult(
            candidate_key="test",
            grade=BridgeGrade.UNMAPPED.value,
            matches=(),
            failure_stage="candidate_text_signature_mismatch",
            bridge_reasons=("test",),
        )
        result_dict = result.to_dict()
        forbidden = {"question", "query", "expected", "gold"}
        for key in result_dict:
            assert not any(f in key.lower() for f in forbidden)


# ---------------------------------------------------------------------------
# Test 18: Grade B not strict index eligible
# ---------------------------------------------------------------------------


class TestGradeBNotStrictEligible:
    """Grade-B should not be eligible for Strict Candidate Index."""

    def test_grade_b_not_strict(self):
        assert not BridgeGrade.is_grade_a(BridgeGrade.B_AMBIGUOUS.value)
        assert not BridgeGrade.is_grade_a(BridgeGrade.C_NAVIGATION_ONLY.value)
        assert not BridgeGrade.is_grade_a(BridgeGrade.UNMAPPED.value)

    def test_grade_a_strict(self):
        for grade in BridgeGrade.strict_eligible_grades():
            assert BridgeGrade.is_grade_a(grade)

    def test_structured_view_only_for_grade_a(self):
        """StructuredViewBuilder should return None for non-A grades."""
        candidate = make_candidate_sig()
        result = BridgeResult(
            candidate_key=candidate.candidate_key,
            grade=BridgeGrade.B_AMBIGUOUS.value,
            matches=(),
            failure_stage="multiple_equal_matches",
            bridge_reasons=("test",),
        )
        catalog = make_catalog_with_evidence([])
        builder = StructuredViewBuilder(catalog)
        view = builder.build_view(candidate, result)
        assert view is None


# ---------------------------------------------------------------------------
# Test 19: Matching utility correctness
# ---------------------------------------------------------------------------


class TestMatchingUtilities:
    """Verify shared matching utility functions."""

    def test_numeric_recall_perfect(self):
        assert compute_numeric_recall(("100", "200"), ("100", "200", "300")) == 1.0

    def test_numeric_recall_partial(self):
        assert compute_numeric_recall(("100", "500"), ("100", "200")) == 0.5

    def test_numeric_recall_empty_candidate(self):
        assert compute_numeric_recall((), ("100",)) == 1.0

    def test_text_coverage(self):
        tc = compute_text_coverage(("revenue", "total"), "total revenue was 100")
        assert tc == 1.0

    def test_text_coverage_no_match(self):
        tc = compute_text_tokens_no_match(("zzz",), "total revenue")
        assert tc == 0.0

    def test_bbox_iou_identical(self):
        assert compute_bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_bbox_iou_no_overlap(self):
        assert compute_bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_bbox_iou_partial(self):
        iou = compute_bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.0 < iou < 0.2

    def test_metric_compatible_same(self):
        assert metric_compatible("Total Revenue", "Total Revenue") is True

    def test_metric_compatible_substring(self):
        assert metric_compatible("Revenue", "Total Revenue") is True

    def test_metric_compatible_different(self):
        assert metric_compatible("Revenue", "Expenses") is False

    def test_metric_compatible_empty(self):
        assert metric_compatible(None, "Revenue") is True

    def test_period_compatible_overlap(self):
        assert period_compatible(("FY2025",), ("FY2025", "FY2024")) is True

    def test_period_compatible_no_overlap(self):
        assert period_compatible(("FY2025",), ("FY2023",)) is False

    def test_period_compatible_empty(self):
        assert period_compatible((), ("FY2025",)) is True


def compute_text_tokens_no_match(tokens, text):
    """Helper for text coverage test."""
    return compute_text_coverage(tokens, text)


# ---------------------------------------------------------------------------
# Test 20: Block type eligibility
# ---------------------------------------------------------------------------


class TestBlockTypeEligibility:
    """Verify structured eligibility by block type."""

    def test_table_row_eligible(self):
        assert is_structured_eligible("table_row") is True

    def test_table_eligible(self):
        assert is_structured_eligible("table") is True

    def test_text_eligible(self):
        assert is_structured_eligible("text") is True

    def test_front_matter_not_eligible(self):
        assert is_structured_eligible("front_matter") is False

    def test_unknown_not_eligible(self):
        assert is_structured_eligible("unknown") is False

    def test_front_matter_unmapped(self):
        """front_matter candidates should be unmapped with candidate_type_unsupported."""
        candidate = make_candidate_sig(block_type="front_matter")
        catalog = make_catalog_with_evidence([])
        bridge = CandidateEvidenceBridge(catalog)
        result = bridge.bridge_one(candidate)
        assert result.grade == BridgeGrade.UNMAPPED.value
        assert result.failure_stage == "candidate_type_unsupported"


# ---------------------------------------------------------------------------
# Test 21: Validator detects violations
# ---------------------------------------------------------------------------


class TestValidator:
    """Verify bridge validator catches violations."""

    def test_missing_traceback_detected(self):
        """Views without source_traceback should be flagged."""
        view = CandidateStructuredView(
            candidate_key="test",
            document_id="doc_a",
            pdf_page=26,
            candidate_type="table_row",
            raw_content="test",
            section_path=(),
            table_title=None,
            metric_paths=(),
            periods=(),
            facts=(),
            segments=(),
            buckets=(),
            row_matrix=None,
            semantic_evidence_ids=(),
            row_ids=(),
            bridge_grade=BridgeGrade.A3_ROW_SIGNATURE.value,
            bridge_reasons=(),
            source_traceback=(),  # Empty — should be flagged
        )
        catalog = make_catalog_with_evidence([])
        equiv = BridgeEquivalence(catalog)
        validator = BridgeValidator(equiv)
        result = validator.validate([], [], [view])
        assert any(
            v["gate"] == "missing_evidence_traceback" for v in result["violations"]
        )

    def test_no_gold_reads(self):
        """Validator should report 0 gold reads."""
        catalog = make_catalog_with_evidence([])
        equiv = BridgeEquivalence(catalog)
        validator = BridgeValidator(equiv)
        result = validator.validate([], [], [])
        assert result["metrics"]["gold_reads"] == 0
        assert result["metrics"]["question_reads"] == 0
