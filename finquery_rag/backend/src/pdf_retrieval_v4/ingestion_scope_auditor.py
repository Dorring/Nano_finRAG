"""Corrected audit module for Gate 08 R1.2 R1.

Replaces the S1-S6 classification from the initial R1.2 audit with
two corrected classification schemes:

D-class (structurally absent, 16 Gold):
  I.   out_of_ingestion_scope
       Page not in Gate 02's 87-page probe input scope.
  II.  ingested_page_no_v4_view
       Page was input to MinerU and Adapter but no V4 structured view.
       Only this class may be called mineru_or_adapter_structure_missing.
  III. v4_structure_present_candidate_view_missing
       Table/Row/Fact exists but no Candidate Structured View.
  IV.  candidate_view_present_not_retrieved
       Structured View exists but not in retrieval results.

B-class (strict-mapped-not-retrieved, 17 unrecovered Gold):
  All uniformly marked strict_mapped_candidate_not_retrieved, then
  subdivided into:
    candidate_structured_view_missing
    candidate_raw_view_query_miss
    candidate_pool_truncated
    multi_slot_budget_truncated

Audit scope is strictly 33 Gold = 17 unrecovered B-class + 16 D-class.
The 5 B-class recovered by R2 are NOT audited.

No MinerU runs, no retriever runs, no structure modifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# D-class ingestion scope classification (I-IV)
# ---------------------------------------------------------------------------

D_CLASS_I = "out_of_ingestion_scope"
D_CLASS_II = "ingested_page_no_v4_view"
D_CLASS_III = "v4_structure_present_candidate_view_missing"
D_CLASS_IV = "candidate_view_present_not_retrieved"

ALL_D_CLASSES = (D_CLASS_I, D_CLASS_II, D_CLASS_III, D_CLASS_IV)


@dataclass(frozen=True)
class DClassIngestionAudit:
    """Ingestion scope audit result for one D-class Gold source."""

    gold_source_identity: str
    case_id: str
    gold_candidate_key: str
    document_id: str
    pdf_page: int | None

    ingestion_scope_class: str  # D_CLASS_I .. D_CLASS_IV
    in_gate02_probe_scope: bool
    v4_views_on_page: bool
    structural_views_on_page: bool
    candidate_view_present: bool

    is_mineru_failure: bool  # True ONLY for class II
    audit_notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gold_source_identity": self.gold_source_identity,
            "case_id": self.case_id,
            "gold_candidate_key": self.gold_candidate_key,
            "document_id": self.document_id,
            "pdf_page": self.pdf_page,
            "ingestion_scope_class": self.ingestion_scope_class,
            "in_gate02_probe_scope": self.in_gate02_probe_scope,
            "v4_views_on_page": self.v4_views_on_page,
            "structural_views_on_page": self.structural_views_on_page,
            "candidate_view_present": self.candidate_view_present,
            "is_mineru_failure": self.is_mineru_failure,
            "audit_notes": self.audit_notes,
        }


def classify_d_class(
    *,
    in_gate02_probe_scope: bool,
    v4_views_on_page: bool,
    structural_views_on_page: bool,
    candidate_view_present: bool,
) -> tuple[str, bool, str]:
    """Classify a D-class Gold source into I-IV.

    Returns (ingestion_scope_class, is_mineru_failure, audit_notes).
    """
    if not in_gate02_probe_scope:
        return (
            D_CLASS_I,
            False,
            "Page not in Gate 02's 87-page probe input scope; "
            "cannot be described as a MinerU failure.",
        )

    if not v4_views_on_page:
        return (
            D_CLASS_II,
            True,
            "Page was ingested by MinerU and Adapter but no V4 "
            "structured view exists; mineru_or_adapter_structure_missing.",
        )

    if not structural_views_on_page:
        return (
            D_CLASS_III,
            False,
            "V4 structure exists on page but no Candidate Structured "
            "View was generated for this Gold source.",
        )

    if not candidate_view_present:
        return (
            D_CLASS_IV,
            False,
            "Candidate Structured View exists but was not retrieved "
            "in the combined pool.",
        )

    # Should not reach here for D-class (structurally_absent)
    return (
        D_CLASS_IV,
        False,
        "Candidate view present and retrieved; classification may need "
        "re-evaluation.",
    )


# ---------------------------------------------------------------------------
# B-class unrecovered subdivision
# ---------------------------------------------------------------------------

B_UNIFIED = "strict_mapped_candidate_not_retrieved"

B_SUB_STRUCTURED_MISSING = "candidate_structured_view_missing"
B_SUB_RAW_QUERY_MISS = "candidate_raw_view_query_miss"
B_SUB_POOL_TRUNCATED = "candidate_pool_truncated"
B_SUB_MULTI_SLOT_TRUNCATED = "multi_slot_budget_truncated"

ALL_B_SUBCLASSES = (
    B_SUB_STRUCTURED_MISSING,
    B_SUB_RAW_QUERY_MISS,
    B_SUB_POOL_TRUNCATED,
    B_SUB_MULTI_SLOT_TRUNCATED,
)


@dataclass(frozen=True)
class BClassUnrecoveredAudit:
    """Subdivision audit for one unrecovered B-class Gold source."""

    gold_source_identity: str
    case_id: str
    gold_candidate_key: str

    unified_class: str  # always strict_mapped_candidate_not_retrieved
    failure_subclass: str

    has_structured_view: bool
    has_raw_view: bool
    raw_bm25_rank: int | None
    raw_dense_rank: int | None
    structured_bm25_rank: int | None
    structured_dense_rank: int | None
    candidate_rrf_rank: int | None
    in_top40: bool
    in_top50: bool
    is_multi_slot: bool
    slot_count: int
    first_failure_stage: str

    audit_notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gold_source_identity": self.gold_source_identity,
            "case_id": self.case_id,
            "gold_candidate_key": self.gold_candidate_key,
            "unified_class": self.unified_class,
            "failure_subclass": self.failure_subclass,
            "has_structured_view": self.has_structured_view,
            "has_raw_view": self.has_raw_view,
            "raw_bm25_rank": self.raw_bm25_rank,
            "raw_dense_rank": self.raw_dense_rank,
            "structured_bm25_rank": self.structured_bm25_rank,
            "structured_dense_rank": self.structured_dense_rank,
            "candidate_rrf_rank": self.candidate_rrf_rank,
            "in_top40": self.in_top40,
            "in_top50": self.in_top50,
            "is_multi_slot": self.is_multi_slot,
            "slot_count": self.slot_count,
            "first_failure_stage": self.first_failure_stage,
            "audit_notes": self.audit_notes,
        }


def classify_b_class(
    *,
    has_structured_view: bool,
    has_raw_view: bool,
    raw_bm25_rank: int | None,
    raw_dense_rank: int | None,
    structured_bm25_rank: int | None,
    structured_dense_rank: int | None,
    candidate_rrf_rank: int | None,
    in_top40: bool,
    in_top50: bool,
    is_multi_slot: bool,
    first_failure_stage: str,
) -> tuple[str, str]:
    """Subdivide an unrecovered B-class Gold source.

    Returns (failure_subclass, audit_notes).

    Priority:
      1. multi_slot_budget_truncated
      2. candidate_pool_truncated
      3. candidate_structured_view_missing
      4. candidate_raw_view_query_miss
    """
    # Priority 1: multi-slot budget truncated
    if is_multi_slot and first_failure_stage == "multi_slot_budget_truncated":
        return (
            B_SUB_MULTI_SLOT_TRUNCATED,
            "Multi-slot case where slot budget was truncated; candidate "
            "appeared in raw lane but RRF rank exceeded slot budget.",
        )

    # Priority 2: pool truncated (in top50 but not top40)
    if in_top50 and not in_top40:
        return (
            B_SUB_POOL_TRUNCATED,
            "Candidate appeared in lane top-50 but was truncated at "
            "the final pool top-40 cutoff.",
        )

    # Priority 3: structured view missing
    if not has_structured_view:
        return (
            B_SUB_STRUCTURED_MISSING,
            "No Candidate Structured View exists for this candidate; "
            "structured lanes cannot contribute. Raw view exists but "
            "raw query also missed.",
        )

    # Priority 4: raw view query miss (both views exist but query missed)
    return (
        B_SUB_RAW_QUERY_MISS,
        "Both raw and structured views exist but query did not retrieve "
        "the candidate in any lane.",
    )


# ---------------------------------------------------------------------------
# Combined audit record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectedAuditRecord:
    """Combined corrected audit record for one Gold source."""

    gold_source_identity: str
    case_id: str
    gold_candidate_key: str
    original_class: str  # B or D from R1.1
    corrected_class: str  # D_CLASS_I..IV or B_SUB_*
    is_mineru_failure: bool
    audit_notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gold_source_identity": self.gold_source_identity,
            "case_id": self.case_id,
            "gold_candidate_key": self.gold_candidate_key,
            "original_class": self.original_class,
            "corrected_class": self.corrected_class,
            "is_mineru_failure": self.is_mineru_failure,
            "audit_notes": self.audit_notes,
        }
