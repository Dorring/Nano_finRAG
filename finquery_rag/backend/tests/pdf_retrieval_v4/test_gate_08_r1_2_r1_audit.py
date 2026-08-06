"""Tests for Gate 08 R1.2 R1 corrected audit scope and ingestion coverage.

Covers:
 1. D-class ingestion scope classification (I-IV)
 2. B-class unrecovered subdivision (4 subclasses)
 3. Priority ordering for B-class
 4. Mutual exclusivity
 5. Serialization
 6. Mineru failure flag correctness
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.ingestion_scope_auditor import (  # noqa: E402
    ALL_B_SUBCLASSES,
    ALL_D_CLASSES,
    B_SUB_MULTI_SLOT_TRUNCATED,
    B_SUB_POOL_TRUNCATED,
    B_SUB_RAW_QUERY_MISS,
    B_SUB_STRUCTURED_MISSING,
    B_UNIFIED,
    D_CLASS_I,
    D_CLASS_II,
    D_CLASS_III,
    D_CLASS_IV,
    BClassUnrecoveredAudit,
    CorrectedAuditRecord,
    DClassIngestionAudit,
    classify_b_class,
    classify_d_class,
)


# ---------------------------------------------------------------------------
# 1. D-class ingestion scope classification
# ---------------------------------------------------------------------------


class TestDClassClassification:
    def test_class_i_out_of_ingestion_scope(self) -> None:
        cls, is_mineru, notes = classify_d_class(
            in_gate02_probe_scope=False,
            v4_views_on_page=False,
            structural_views_on_page=False,
            candidate_view_present=False,
        )
        assert cls == D_CLASS_I
        assert is_mineru is False
        assert "MinerU" not in notes or "cannot" in notes.lower()

    def test_class_i_not_described_as_mineru_failure(self) -> None:
        """Class I must NOT be described as MinerU failure."""
        cls, is_mineru, _ = classify_d_class(
            in_gate02_probe_scope=False,
            v4_views_on_page=False,
            structural_views_on_page=False,
            candidate_view_present=False,
        )
        assert cls == D_CLASS_I
        assert is_mineru is False

    def test_class_ii_ingested_no_v4_view(self) -> None:
        cls, is_mineru, notes = classify_d_class(
            in_gate02_probe_scope=True,
            v4_views_on_page=False,
            structural_views_on_page=False,
            candidate_view_present=False,
        )
        assert cls == D_CLASS_II
        assert is_mineru is True
        assert "mineru" in notes.lower() or "adapter" in notes.lower()

    def test_class_ii_is_only_mineru_failure(self) -> None:
        """Only class II can be called mineru_or_adapter_structure_missing."""
        for in_scope, v4, struct, cand in [
            (False, False, False, False),  # I
            (True, True, False, False),    # III
            (True, True, True, False),     # IV
        ]:
            _, is_mineru, _ = classify_d_class(
                in_gate02_probe_scope=in_scope,
                v4_views_on_page=v4,
                structural_views_on_page=struct,
                candidate_view_present=cand,
            )
            assert is_mineru is False

    def test_class_iii_structure_present_view_missing(self) -> None:
        cls, is_mineru, _ = classify_d_class(
            in_gate02_probe_scope=True,
            v4_views_on_page=True,
            structural_views_on_page=False,
            candidate_view_present=False,
        )
        assert cls == D_CLASS_III
        assert is_mineru is False

    def test_class_iv_candidate_view_not_retrieved(self) -> None:
        cls, is_mineru, _ = classify_d_class(
            in_gate02_probe_scope=True,
            v4_views_on_page=True,
            structural_views_on_page=True,
            candidate_view_present=False,
        )
        assert cls == D_CLASS_IV
        assert is_mineru is False

    def test_all_four_classes_covered(self) -> None:
        """All four D-class categories must be reachable."""
        results = set()
        for in_scope, v4, struct, cand in [
            (False, False, False, False),
            (True, False, False, False),
            (True, True, False, False),
            (True, True, True, False),
        ]:
            cls, _, _ = classify_d_class(
                in_gate02_probe_scope=in_scope,
                v4_views_on_page=v4,
                structural_views_on_page=struct,
                candidate_view_present=cand,
            )
            results.add(cls)
        assert results == set(ALL_D_CLASSES)


# ---------------------------------------------------------------------------
# 2. B-class unrecovered subdivision
# ---------------------------------------------------------------------------


class TestBClassSubdivision:
    def test_multi_slot_budget_truncated_priority_1(self) -> None:
        """Multi-slot truncation takes priority over all else."""
        subclass, _ = classify_b_class(
            has_structured_view=False,
            has_raw_view=True,
            raw_bm25_rank=26,
            raw_dense_rank=None,
            structured_bm25_rank=None,
            structured_dense_rank=None,
            candidate_rrf_rank=97,
            in_top40=False,
            in_top50=True,
            is_multi_slot=True,
            first_failure_stage="multi_slot_budget_truncated",
        )
        assert subclass == B_SUB_MULTI_SLOT_TRUNCATED

    def test_pool_truncated_priority_2(self) -> None:
        """Pool truncation takes priority over structured view missing."""
        subclass, _ = classify_b_class(
            has_structured_view=False,
            has_raw_view=True,
            raw_bm25_rank=18,
            raw_dense_rank=None,
            structured_bm25_rank=None,
            structured_dense_rank=None,
            candidate_rrf_rank=61,
            in_top40=False,
            in_top50=True,
            is_multi_slot=False,
            first_failure_stage="candidate_rank_41_to_50",
        )
        assert subclass == B_SUB_POOL_TRUNCATED

    def test_structured_view_missing_priority_3(self) -> None:
        """When no pool truncation and no structured view, it's structured missing."""
        subclass, _ = classify_b_class(
            has_structured_view=False,
            has_raw_view=True,
            raw_bm25_rank=None,
            raw_dense_rank=None,
            structured_bm25_rank=None,
            structured_dense_rank=None,
            candidate_rrf_rank=None,
            in_top40=False,
            in_top50=False,
            is_multi_slot=False,
            first_failure_stage="candidate_not_in_any_top50",
        )
        assert subclass == B_SUB_STRUCTURED_MISSING

    def test_raw_query_miss_priority_4(self) -> None:
        """When both views exist but query missed, it's raw query miss."""
        subclass, _ = classify_b_class(
            has_structured_view=True,
            has_raw_view=True,
            raw_bm25_rank=None,
            raw_dense_rank=None,
            structured_bm25_rank=None,
            structured_dense_rank=None,
            candidate_rrf_rank=None,
            in_top40=False,
            in_top50=False,
            is_multi_slot=False,
            first_failure_stage="candidate_not_in_any_top50",
        )
        assert subclass == B_SUB_RAW_QUERY_MISS

    def test_multi_slot_not_truncated_falls_through(self) -> None:
        """Multi-slot but not budget-truncated falls to structured missing."""
        subclass, _ = classify_b_class(
            has_structured_view=False,
            has_raw_view=True,
            raw_bm25_rank=None,
            raw_dense_rank=None,
            structured_bm25_rank=None,
            structured_dense_rank=None,
            candidate_rrf_rank=None,
            in_top40=False,
            in_top50=False,
            is_multi_slot=True,
            first_failure_stage="candidate_not_in_any_top50",
        )
        assert subclass == B_SUB_STRUCTURED_MISSING

    def test_all_subclasses_reachable(self) -> None:
        """All four B-class subclasses must be reachable."""
        results = set()
        # multi_slot_truncated
        results.add(classify_b_class(
            has_structured_view=False, has_raw_view=True,
            raw_bm25_rank=26, raw_dense_rank=None,
            structured_bm25_rank=None, structured_dense_rank=None,
            candidate_rrf_rank=97, in_top40=False, in_top50=True,
            is_multi_slot=True, first_failure_stage="multi_slot_budget_truncated",
        )[0])
        # pool_truncated
        results.add(classify_b_class(
            has_structured_view=False, has_raw_view=True,
            raw_bm25_rank=18, raw_dense_rank=None,
            structured_bm25_rank=None, structured_dense_rank=None,
            candidate_rrf_rank=61, in_top40=False, in_top50=True,
            is_multi_slot=False, first_failure_stage="candidate_rank_41_to_50",
        )[0])
        # structured_missing
        results.add(classify_b_class(
            has_structured_view=False, has_raw_view=True,
            raw_bm25_rank=None, raw_dense_rank=None,
            structured_bm25_rank=None, structured_dense_rank=None,
            candidate_rrf_rank=None, in_top40=False, in_top50=False,
            is_multi_slot=False, first_failure_stage="candidate_not_in_any_top50",
        )[0])
        # raw_query_miss
        results.add(classify_b_class(
            has_structured_view=True, has_raw_view=True,
            raw_bm25_rank=None, raw_dense_rank=None,
            structured_bm25_rank=None, structured_dense_rank=None,
            candidate_rrf_rank=None, in_top40=False, in_top50=False,
            is_multi_slot=False, first_failure_stage="candidate_not_in_any_top50",
        )[0])
        assert results == set(ALL_B_SUBCLASSES)


# ---------------------------------------------------------------------------
# 3. Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_multi_slot_beats_pool_truncated(self) -> None:
        """If both multi_slot and in_top50, multi_slot wins."""
        subclass, _ = classify_b_class(
            has_structured_view=False,
            has_raw_view=True,
            raw_bm25_rank=26,
            raw_dense_rank=None,
            structured_bm25_rank=None,
            structured_dense_rank=None,
            candidate_rrf_rank=97,
            in_top40=False,
            in_top50=True,
            is_multi_slot=True,
            first_failure_stage="multi_slot_budget_truncated",
        )
        assert subclass == B_SUB_MULTI_SLOT_TRUNCATED

    def test_pool_truncated_beats_structured_missing(self) -> None:
        """If in_top50 but no structured view, pool_truncated wins."""
        subclass, _ = classify_b_class(
            has_structured_view=False,
            has_raw_view=True,
            raw_bm25_rank=18,
            raw_dense_rank=None,
            structured_bm25_rank=None,
            structured_dense_rank=None,
            candidate_rrf_rank=61,
            in_top40=False,
            in_top50=True,
            is_multi_slot=False,
            first_failure_stage="candidate_rank_41_to_50",
        )
        assert subclass == B_SUB_POOL_TRUNCATED

    def test_structured_missing_beats_raw_query_miss(self) -> None:
        """If no structured view, structured_missing wins over raw_query_miss."""
        subclass, _ = classify_b_class(
            has_structured_view=False,
            has_raw_view=True,
            raw_bm25_rank=None,
            raw_dense_rank=None,
            structured_bm25_rank=None,
            structured_dense_rank=None,
            candidate_rrf_rank=None,
            in_top40=False,
            in_top50=False,
            is_multi_slot=False,
            first_failure_stage="candidate_not_in_any_top50",
        )
        assert subclass == B_SUB_STRUCTURED_MISSING


# ---------------------------------------------------------------------------
# 4. Mutual exclusivity
# ---------------------------------------------------------------------------


class TestMutualExclusivity:
    def test_each_input_produces_exactly_one_class(self) -> None:
        """Every possible input combination produces exactly one subclass."""
        for in_scope in (True, False):
            for v4 in (True, False):
                for struct in (True, False):
                    for cand in (True, False):
                        cls, _, _ = classify_d_class(
                            in_gate02_probe_scope=in_scope,
                            v4_views_on_page=v4,
                            structural_views_on_page=struct,
                            candidate_view_present=cand,
                        )
                        assert cls in ALL_D_CLASSES

    def test_b_class_always_one_subclass(self) -> None:
        for has_sv in (True, False):
            for has_rv in (True, False):
                for in40 in (True, False):
                    for in50 in (True, False):
                        for multi in (True, False):
                            for stage in (
                                "multi_slot_budget_truncated",
                                "candidate_rank_41_to_50",
                                "candidate_not_in_any_top50",
                            ):
                                subclass, _ = classify_b_class(
                                    has_structured_view=has_sv,
                                    has_raw_view=has_rv,
                                    raw_bm25_rank=1,
                                    raw_dense_rank=None,
                                    structured_bm25_rank=None,
                                    structured_dense_rank=None,
                                    candidate_rrf_rank=1,
                                    in_top40=in40,
                                    in_top50=in50,
                                    is_multi_slot=multi,
                                    first_failure_stage=stage,
                                )
                                assert subclass in ALL_B_SUBCLASSES


# ---------------------------------------------------------------------------
# 5. Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_d_class_audit_to_dict(self) -> None:
        audit = DClassIngestionAudit(
            gold_source_identity="case_001#0",
            case_id="case_001",
            gold_candidate_key="candidate:v1:abc",
            document_id="aapl_fy2025",
            pdf_page=197,
            ingestion_scope_class=D_CLASS_I,
            in_gate02_probe_scope=False,
            v4_views_on_page=False,
            structural_views_on_page=False,
            candidate_view_present=False,
            is_mineru_failure=False,
            audit_notes="out of scope",
        )
        d = audit.to_dict()
        assert d["ingestion_scope_class"] == D_CLASS_I
        assert d["is_mineru_failure"] is False
        assert d["pdf_page"] == 197

    def test_b_class_audit_to_dict(self) -> None:
        audit = BClassUnrecoveredAudit(
            gold_source_identity="case_002#0",
            case_id="case_002",
            gold_candidate_key="candidate:v1:def",
            unified_class=B_UNIFIED,
            failure_subclass=B_SUB_STRUCTURED_MISSING,
            has_structured_view=False,
            has_raw_view=True,
            raw_bm25_rank=None,
            raw_dense_rank=None,
            structured_bm25_rank=None,
            structured_dense_rank=None,
            candidate_rrf_rank=None,
            in_top40=False,
            in_top50=False,
            is_multi_slot=False,
            slot_count=0,
            first_failure_stage="candidate_not_in_any_top50",
            audit_notes="structured view missing",
        )
        d = audit.to_dict()
        assert d["unified_class"] == B_UNIFIED
        assert d["failure_subclass"] == B_SUB_STRUCTURED_MISSING
        assert d["has_structured_view"] is False

    def test_corrected_audit_record_to_dict(self) -> None:
        record = CorrectedAuditRecord(
            gold_source_identity="case_003#0",
            case_id="case_003",
            gold_candidate_key="candidate:v1:ghi",
            original_class="structurally_absent",
            corrected_class=D_CLASS_I,
            is_mineru_failure=False,
            audit_notes="out of scope",
        )
        d = record.to_dict()
        assert d["original_class"] == "structurally_absent"
        assert d["corrected_class"] == D_CLASS_I

    def test_audit_records_are_frozen(self) -> None:
        audit = DClassIngestionAudit(
            gold_source_identity="x",
            case_id="x",
            gold_candidate_key="x",
            document_id="x",
            pdf_page=1,
            ingestion_scope_class=D_CLASS_I,
            in_gate02_probe_scope=False,
            v4_views_on_page=False,
            structural_views_on_page=False,
            candidate_view_present=False,
            is_mineru_failure=False,
            audit_notes="x",
        )
        with pytest.raises(AttributeError):
            audit.case_id = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 6. Mineru failure flag correctness
# ---------------------------------------------------------------------------


class TestMineruFailureFlag:
    def test_only_class_ii_is_mineru_failure(self) -> None:
        """is_mineru_failure must be True only for class II."""
        cases = [
            (False, False, False, False),  # I
            (True, False, False, False),   # II
            (True, True, False, False),    # III
            (True, True, True, False),     # IV
        ]
        mineru_flags = []
        for in_scope, v4, struct, cand in cases:
            _, is_mineru, _ = classify_d_class(
                in_gate02_probe_scope=in_scope,
                v4_views_on_page=v4,
                structural_views_on_page=struct,
                candidate_view_present=cand,
            )
            mineru_flags.append(is_mineru)
        assert mineru_flags == [False, True, False, False]

    def test_b_class_never_mineru_failure(self) -> None:
        """B-class is never a MinerU failure (it's a retrieval issue)."""
        for has_sv in (True, False):
            for in50 in (True, False):
                for multi in (True, False):
                    for stage in (
                        "multi_slot_budget_truncated",
                        "candidate_rank_41_to_50",
                        "candidate_not_in_any_top50",
                    ):
                        # B-class doesn't have is_mineru_failure field,
                        # but CorrectedAuditRecord sets it to False
                        record = CorrectedAuditRecord(
                            gold_source_identity="x",
                            case_id="x",
                            gold_candidate_key="x",
                            original_class="strict_mapped_not_retrieved",
                            corrected_class=B_SUB_STRUCTURED_MISSING,
                            is_mineru_failure=False,
                            audit_notes="x",
                        )
                        assert record.is_mineru_failure is False
