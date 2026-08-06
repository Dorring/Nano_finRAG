"""Candidate-aligned Dual-view data models for Gate 08 R2.

Each Production Candidate produces two retrieval views sharing the same
candidate_key:

  - Raw View:       preserves original candidate text for lexical matching
  - Structured View: aggregates V4 structural metadata (metric, period, facts)

View IDs are deterministic hashes of (candidate_key, view_schema_version).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

VIEW_SCHEMA_VERSION = "candidate-aligned-v1"


def _view_id(candidate_key: str, view_type: str) -> str:
    """Deterministic view ID from candidate_key and view type."""
    raw = f"{candidate_key}|{VIEW_SCHEMA_VERSION}|{view_type}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"cav:{digest}"


@dataclass(frozen=True)
class CandidateAlignedView:
    """A single candidate-aligned retrieval view (raw or structured)."""

    candidate_key: str
    view_type: str  # "raw" or "structured"
    view_id: str
    retrieval_text: str
    document_id: str
    pdf_page: int | None = None
    logical_table_ids: tuple[str, ...] = ()
    row_ids: tuple[str, ...] = ()
    fact_ids: tuple[str, ...] = ()
    metric_paths: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    temporal_types: tuple[str, ...] = ()
    bridge_grade: str = "raw_only"  # A1|A2|A3|raw_only

    @property
    def has_structured_mapping(self) -> bool:
        return self.bridge_grade != "raw_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "view_type": self.view_type,
            "view_id": self.view_id,
            "retrieval_text": self.retrieval_text,
            "document_id": self.document_id,
            "pdf_page": self.pdf_page,
            "logical_table_ids": list(self.logical_table_ids),
            "row_ids": list(self.row_ids),
            "fact_ids": list(self.fact_ids),
            "metric_paths": list(self.metric_paths),
            "periods": list(self.periods),
            "temporal_types": list(self.temporal_types),
            "bridge_grade": self.bridge_grade,
            "has_structured_mapping": self.has_structured_mapping,
        }


@dataclass(frozen=True)
class CandidateViewPair:
    """Raw + Structured views for a single candidate."""

    candidate_key: str
    raw_view: CandidateAlignedView
    structured_view: CandidateAlignedView | None  # None when no structural mapping

    @property
    def document_id(self) -> str:
        return self.raw_view.document_id

    @property
    def pdf_page(self) -> int | None:
        return self.raw_view.pdf_page

    @property
    def bridge_grade(self) -> str:
        return self.raw_view.bridge_grade

    def all_views(self) -> list[CandidateAlignedView]:
        result = [self.raw_view]
        if self.structured_view is not None:
            result.append(self.structured_view)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "raw_view_id": self.raw_view.view_id,
            "structured_view_id": (
                self.structured_view.view_id if self.structured_view else None
            ),
            "document_id": self.document_id,
            "pdf_page": self.pdf_page,
            "logical_table_ids": list(self.raw_view.logical_table_ids),
            "row_ids": list(self.raw_view.row_ids),
            "fact_ids": list(self.raw_view.fact_ids),
            "metric_paths": list(self.raw_view.metric_paths),
            "periods": list(self.raw_view.periods),
            "temporal_types": list(self.raw_view.temporal_types),
            "bridge_grade": self.bridge_grade,
        }


def make_raw_view_id(candidate_key: str) -> str:
    return _view_id(candidate_key, "raw")


def make_structured_view_id(candidate_key: str) -> str:
    return _view_id(candidate_key, "structured")
