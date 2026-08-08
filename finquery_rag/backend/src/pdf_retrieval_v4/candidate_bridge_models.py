"""Gate 05 R5 — Candidate Evidence Bridge data models.

Defines the stable, deterministic data structures for mapping Production
Candidates to Full-corpus Semantic Graph evidence WITHOUT reading
Question / Gold / Governance data.

Core types
----------
- ``BridgeGrade``           — enumeration of bridge quality grades
- ``CandidateSignature``    — normalized features extracted from a Production Candidate
- ``SemanticEvidenceSignature`` — normalized features extracted from a Semantic Evidence unit
- ``BridgeResult``          — outcome of matching one candidate to evidence
- ``CandidateStructuredView`` — aggregated structured view for a Grade-A candidate
- ``FailureStage``          — first-failure classification for unmapped candidates

Conventions
-----------
- All dataclasses are frozen (hashable, deterministic).
- No Question / Gold / Expected-Value fields participate in any identity.
- ``bridge_grade`` only allows ``A*`` for Strict Candidate Index eligibility.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

BRIDGE_SCHEMA_VERSION = "pdf-retrieval-v4/gate-05-r5/candidate-bridge/v1"


# ---------------------------------------------------------------------------
# Bridge Grade
# ---------------------------------------------------------------------------


class BridgeGrade(str, Enum):
    """Quality grade for a candidate↔evidence bridge.

    Only ``A*`` grades are eligible for Strict Candidate Structured View.
    """

    A1_DIRECT = "A1_direct"
    A2_BBOX_SIGNATURE = "A2_bbox_signature"
    A3_ROW_SIGNATURE = "A3_row_signature"
    A4_MULTIROW = "A4_multirow"
    A5_NARRATIVE = "A5_narrative"
    A_EQUIVALENT = "A_equivalent"
    B_AMBIGUOUS = "B_ambiguous"
    C_NAVIGATION_ONLY = "C_navigation_only"
    UNMAPPED = "unmapped"

    @classmethod
    def is_grade_a(cls, grade: str) -> bool:
        """Return True if the grade is any A* variant."""
        return grade.startswith("A")

    @classmethod
    def strict_eligible_grades(cls) -> tuple[str, ...]:
        """Grades that qualify for Strict Candidate Index."""
        return (
            cls.A1_DIRECT.value,
            cls.A2_BBOX_SIGNATURE.value,
            cls.A3_ROW_SIGNATURE.value,
            cls.A4_MULTIROW.value,
            cls.A5_NARRATIVE.value,
            cls.A_EQUIVALENT.value,
        )


# ---------------------------------------------------------------------------
# Failure Stage classification
# ---------------------------------------------------------------------------

FAILURE_STAGES = (
    "candidate_type_unsupported",
    "candidate_bbox_missing",
    "candidate_text_signature_mismatch",
    "numeric_signature_mismatch",
    "metric_signature_mismatch",
    "period_signature_mismatch",
    "multirow_required",
    "narrative_bridge_missing",
    "multiple_equal_matches",
    "semantic_evidence_fanout_ambiguous",
    "legacy_candidate_granularity_mismatch",
)


# ---------------------------------------------------------------------------
# Candidate Signature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSignature:
    """Normalized features extracted from a Production Candidate.

    Used as the query-side signature for bridge matching.
    No Question / Gold fields are present.
    """

    candidate_key: str
    document_id: str
    pdf_page: int
    block_type: str  # table_row, table, text, front_matter
    raw_content: str
    text_tokens: tuple[str, ...]
    numeric_multiset: tuple[str, ...]
    period_tokens: tuple[str, ...]
    normalized_text: str
    # Optional: existing structural mapping from prior gates
    existing_row_ids: tuple[str, ...] = ()
    existing_logical_table_ids: tuple[str, ...] = ()
    existing_metric_paths: tuple[str, ...] = ()
    existing_bridge_grade: str = "raw_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "document_id": self.document_id,
            "pdf_page": self.pdf_page,
            "block_type": self.block_type,
            "raw_content": self.raw_content,
            "text_tokens": list(self.text_tokens),
            "numeric_multiset": list(self.numeric_multiset),
            "period_tokens": list(self.period_tokens),
            "normalized_text": self.normalized_text,
            "existing_row_ids": list(self.existing_row_ids),
            "existing_logical_table_ids": list(self.existing_logical_table_ids),
            "existing_metric_paths": list(self.existing_metric_paths),
            "existing_bridge_grade": self.existing_bridge_grade,
        }


# ---------------------------------------------------------------------------
# Semantic Evidence Signature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticEvidenceSignature:
    """Normalized features extracted from a Semantic Evidence unit.

    One signature per evidence row/fact/narrative block from Gate 03 R2.
    Used as the target-side signature for bridge matching.
    """

    evidence_id: str
    evidence_type: str  # semantic_row, atomic_fact, comparison_fact, bucket_fact, row_matrix, narrative_evidence, logical_table
    document_id: str
    pdf_page: int
    table_id: str | None
    row_id: str | None
    cell_ids: tuple[str, ...]
    bbox: tuple[float, ...]
    metric_paths: tuple[str, ...]
    periods: tuple[str, ...]
    segments: tuple[str, ...]
    buckets: tuple[str, ...]
    raw_values: tuple[str, ...]
    numeric_multiset: tuple[str, ...]
    raw_text: str
    normalized_text: str
    source_traceback: dict[str, Any]
    equivalent_group_id: str | None = None
    # For semantic rows: row_type and row_index
    row_type: str | None = None
    row_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "document_id": self.document_id,
            "pdf_page": self.pdf_page,
            "table_id": self.table_id,
            "row_id": self.row_id,
            "cell_ids": list(self.cell_ids),
            "bbox": list(self.bbox),
            "metric_paths": list(self.metric_paths),
            "periods": list(self.periods),
            "segments": list(self.segments),
            "buckets": list(self.buckets),
            "raw_values": list(self.raw_values),
            "numeric_multiset": list(self.numeric_multiset),
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "source_traceback": self.source_traceback,
            "equivalent_group_id": self.equivalent_group_id,
            "row_type": self.row_type,
            "row_index": self.row_index,
        }


# ---------------------------------------------------------------------------
# Bridge Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BridgeMatch:
    """A single candidate↔evidence match with score breakdown."""

    evidence_id: str
    evidence_type: str
    grade: str
    score: float
    reasons: tuple[str, ...]
    # Detailed score components
    numeric_recall: float
    text_coverage: float
    bbox_overlap: float
    metric_compatible: bool
    period_compatible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "grade": self.grade,
            "score": self.score,
            "reasons": list(self.reasons),
            "numeric_recall": self.numeric_recall,
            "text_coverage": self.text_coverage,
            "bbox_overlap": self.bbox_overlap,
            "metric_compatible": self.metric_compatible,
            "period_compatible": self.period_compatible,
        }


@dataclass(frozen=True)
class BridgeResult:
    """Outcome of bridging one candidate to evidence.

    If ``grade`` is A*, ``matches`` contains exactly the matched evidence.
    If ``grade`` is B_ambiguous, ``matches`` contains the tied candidates.
    If ``grade`` is unmapped, ``matches`` is empty and ``failure_stage`` is set.
    """

    candidate_key: str
    grade: str
    matches: tuple[BridgeMatch, ...]
    failure_stage: str | None
    bridge_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "grade": self.grade,
            "matches": [m.to_dict() for m in self.matches],
            "failure_stage": self.failure_stage,
            "bridge_reasons": list(self.bridge_reasons),
        }


# ---------------------------------------------------------------------------
# Candidate Structured View
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateStructuredView:
    """Aggregated structured view for a Grade-A candidate.

    One candidate → one structured view.
    Internal facts can be multiple (aggregated from Atomic/Comparison/Bucket/RowMatrix).
    """

    candidate_key: str
    document_id: str
    pdf_page: int
    candidate_type: str  # table_row, table, text
    raw_content: str
    section_path: tuple[str, ...]
    table_title: str | None
    metric_paths: tuple[str, ...]
    periods: tuple[str, ...]
    facts: tuple[dict[str, Any], ...]
    segments: tuple[str, ...]
    buckets: tuple[str, ...]
    row_matrix: dict[str, Any] | None
    semantic_evidence_ids: tuple[str, ...]
    row_ids: tuple[str, ...]
    bridge_grade: str
    bridge_reasons: tuple[str, ...]
    source_traceback: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "document_id": self.document_id,
            "pdf_page": self.pdf_page,
            "candidate_type": self.candidate_type,
            "raw_content": self.raw_content,
            "section_path": list(self.section_path),
            "table_title": self.table_title,
            "metric_paths": list(self.metric_paths),
            "periods": list(self.periods),
            "facts": [dict(f) for f in self.facts],
            "segments": list(self.segments),
            "buckets": list(self.buckets),
            "row_matrix": self.row_matrix,
            "semantic_evidence_ids": list(self.semantic_evidence_ids),
            "row_ids": list(self.row_ids),
            "bridge_grade": self.bridge_grade,
            "bridge_reasons": list(self.bridge_reasons),
            "source_traceback": [dict(t) for t in self.source_traceback],
        }


# ---------------------------------------------------------------------------
# Bridge Eligibility
# ---------------------------------------------------------------------------

# Candidate types eligible for structured bridging
STRUCTURED_ELIGIBLE_BLOCK_TYPES = ("table_row", "table", "text")

# Block types that are NOT eligible (raw_only)
RAW_ONLY_BLOCK_TYPES = ("front_matter",)

# Semantic row types eligible for bridging (financial data rows)
BRIDGE_ELIGIBLE_ROW_TYPES = ("metric_row", "subtotal", "total")


def is_structured_eligible(block_type: str) -> bool:
    """Check if a candidate block type is eligible for structured bridging."""
    return block_type in STRUCTURED_ELIGIBLE_BLOCK_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_bridge_manifest_hash(
    candidate_count: int,
    grade_a_count: int,
    grade_b_count: int,
    unmapped_count: int,
    view_hash: str,
) -> str:
    """Build deterministic bridge manifest hash."""
    payload = _stable_json(
        {
            "schema": BRIDGE_SCHEMA_VERSION,
            "candidate_count": candidate_count,
            "grade_a_count": grade_a_count,
            "grade_b_count": grade_b_count,
            "unmapped_count": unmapped_count,
            "view_hash": view_hash,
        }
    )
    return _sha256(payload)


def build_candidate_view_hash(views: list[dict[str, Any]]) -> str:
    """Build deterministic hash over all structured views."""
    # Sort by candidate_key for determinism
    sorted_views = sorted(views, key=lambda v: v.get("candidate_key", ""))
    payload = _stable_json(
        {
            "schema": BRIDGE_SCHEMA_VERSION,
            "view_count": len(sorted_views),
            "views": sorted_views,
        }
    )
    return _sha256(payload)
