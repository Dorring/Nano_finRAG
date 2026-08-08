"""Gate 03 R2 — Semantic Graph data models.

Defines the stable, deterministic data structures produced by the
five-pass Full-corpus Financial Semantic Graph pipeline:

  Pass A — Logical Table + Row Classification
  Pass B — Header / Metric Graph
  Pass C — Temporal / Dimension Graph
  Pass D — Scale / Currency Resolution
  Pass E — Typed Evidence Emission

All identity fields are derived ONLY from source structure
(document_id, table_id, row_id, cell_id, axis identity, schema version).
Question / gold / expected_value MUST NOT participate in any identity.

Conventions
-----------
- Every dataclass is frozen (hashable, deterministic).
- ``to_dict`` serializes for JSON artifacts (sorted keys).
- ``semantic_fact_id`` / ``semantic_evidence_id`` use sha256 over
  structural fields only.
- ``source_traceback`` always records the physical provenance needed to
  round-trip back to the sealed adapter predictions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Schema version — bump when identity-relevant fields change
# ---------------------------------------------------------------------------

SEMANTIC_SCHEMA_VERSION = "pdf-retrieval-v4/gate-03-r2/semantic-graph/v1"

# ---------------------------------------------------------------------------
# Enumerations (as string constants for JSON-friendliness)
# ---------------------------------------------------------------------------

STATEMENT_TYPES = (
    "income_statement",
    "balance_sheet",
    "cash_flow",
    "segment_table",
    "maturity_table",
    "aging_table",
    "share_table",
    "operating_metric_table",
    "other_financial_table",
    "unknown",
)

ROW_TYPES = (
    "metric_row",
    "group_header",
    "section_header",
    "column_header",
    "subtotal",
    "total",
    "note",
    "spacer",
    "unknown",
)

# Rows that are eligible to enter the Metric Coverage denominator
FINANCIAL_DATA_ROW_TYPES = ("metric_row", "subtotal", "total")

TEMPORAL_KINDS = (
    "point",
    "duration",
    "comparison",
    "bucket",
    "segment",
    "category",
    "non_temporal",
    "unknown",
)

SCALE_STATUS = ("resolved", "candidate", "conflict", "missing")
SCALE_LEVELS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
AUTO_RESOLVE_LEVELS = ("S0", "S1", "S2", "S3")

CURRENCY_STATUS = ("resolved", "unresolved", "conflict")

METRIC_STATUS = ("resolved", "ambiguous", "missing")

EVIDENCE_TYPES = (
    "atomic_fact",
    "comparison_fact",
    "bucket_fact",
    "row_matrix",
    "narrative_evidence",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Pass A — Logical Table + Semantic Row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicalTable:
    """A physical table fragment annotated with financial semantics."""

    table_fragment_id: str
    document_id: str
    pdf_page: int
    table_index: int
    statement_type: str
    table_title: str
    row_count: int
    column_count: int
    scale_candidates: tuple[str, ...]
    source_traceback: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_fragment_id": self.table_fragment_id,
            "document_id": self.document_id,
            "pdf_page": self.pdf_page,
            "table_index": self.table_index,
            "statement_type": self.statement_type,
            "table_title": self.table_title,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "scale_candidates": list(self.scale_candidates),
            "source_traceback": self.source_traceback,
        }


@dataclass(frozen=True)
class SemanticRow:
    """A physical row annotated with row-type and metric-graph eligibility."""

    row_id: str
    table_fragment_id: str
    document_id: str
    pdf_page: int
    row_index: int
    row_type: str
    raw_label: str
    parent_row_id: str | None
    semantic_eligible: bool
    source_traceback: dict[str, Any]

    @property
    def is_financial_data_row(self) -> bool:
        return self.row_type in FINANCIAL_DATA_ROW_TYPES

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "table_fragment_id": self.table_fragment_id,
            "document_id": self.document_id,
            "pdf_page": self.pdf_page,
            "row_index": self.row_index,
            "row_type": self.row_type,
            "raw_label": self.raw_label,
            "parent_row_id": self.parent_row_id,
            "semantic_eligible": self.semantic_eligible,
            "source_traceback": self.source_traceback,
        }


# ---------------------------------------------------------------------------
# Pass B — Metric Graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricPath:
    """Resolved multi-level metric path for a financial data row."""

    row_id: str
    table_fragment_id: str
    raw_row_label: str
    leaf_metric: str
    metric_path: str
    metric_path_segments: tuple[str, ...]
    metric_depth: int
    parent_metric_row_id: str | None
    metric_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "table_fragment_id": self.table_fragment_id,
            "raw_row_label": self.raw_row_label,
            "leaf_metric": self.leaf_metric,
            "metric_path": self.metric_path,
            "metric_path_segments": list(self.metric_path_segments),
            "metric_depth": self.metric_depth,
            "parent_metric_row_id": self.parent_metric_row_id,
            "metric_status": self.metric_status,
        }


# ---------------------------------------------------------------------------
# Pass C — Temporal / Dimension Axis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticAxisBinding:
    """Temporal/dimension binding for a single cell's value column."""

    cell_id: str
    row_id: str
    table_fragment_id: str
    column_index: int
    temporal_kind: str
    period_start: str | None
    period_end: str | None
    normalized_period: str | None
    comparison_role: str | None
    bucket_label: str | None
    segment_label: str | None
    category_label: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "row_id": self.row_id,
            "table_fragment_id": self.table_fragment_id,
            "column_index": self.column_index,
            "temporal_kind": self.temporal_kind,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "normalized_period": self.normalized_period,
            "comparison_role": self.comparison_role,
            "bucket_label": self.bucket_label,
            "segment_label": self.segment_label,
            "category_label": self.category_label,
        }


# ---------------------------------------------------------------------------
# Pass D — Scale + Currency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleResolution:
    """Resolved numeric scale for a table or cell."""

    table_fragment_id: str
    scale: float | None
    scale_unit: str | None
    scale_level: str
    scale_status: str
    raw_candidates: tuple[str, ...]
    source: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_fragment_id": self.table_fragment_id,
            "scale": self.scale,
            "scale_unit": self.scale_unit,
            "scale_level": self.scale_level,
            "scale_status": self.scale_status,
            "raw_candidates": list(self.raw_candidates),
            "source": self.source,
        }


@dataclass(frozen=True)
class CurrencyResolution:
    """Resolved currency for a table."""

    table_fragment_id: str
    currency_symbol: str | None
    currency_code: str | None
    currency_source: str | None
    currency_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_fragment_id": self.table_fragment_id,
            "currency_symbol": self.currency_symbol,
            "currency_code": self.currency_code,
            "currency_source": self.currency_source,
            "currency_status": self.currency_status,
        }


# ---------------------------------------------------------------------------
# Pass E — Typed Evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceTraceback:
    """Physical provenance for a semantic evidence unit."""

    document_id: str
    pdf_page: int
    table_fragment_id: str | None
    row_id: str | None
    cell_id: str | None
    bbox: list[float] | None
    raw_text: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "pdf_page": self.pdf_page,
            "table_fragment_id": self.table_fragment_id,
            "row_id": self.row_id,
            "cell_id": self.cell_id,
            "bbox": self.bbox,
            "raw_text": self.raw_text,
        }


@dataclass(frozen=True)
class AtomicFact:
    """Metric × one明确 Axis × Value."""

    semantic_fact_id: str
    document_id: str
    table_fragment_id: str
    row_id: str
    cell_id: str
    metric_path: str
    leaf_metric: str
    temporal_kind: str
    normalized_period: str | None
    period_start: str | None
    period_end: str | None
    value_raw: str
    value_normalized: str | None
    scale: float | None
    scale_unit: str | None
    currency_code: str | None
    equivalent_group_id: str | None
    source_traceback: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonFact:
    """A comparison relationship (e.g. % change) between atomic values."""

    semantic_fact_id: str
    document_id: str
    table_fragment_id: str
    row_id: str
    metric_path: str
    leaf_metric: str
    comparison_role: str
    base_period: str | None
    compared_period: str | None
    base_value: str | None
    compared_value: str | None
    reported_change: str | None
    value_normalized: str | None
    source_traceback: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BucketFact:
    """A value bound to a bucket dimension (maturity / aging / rating / range)."""

    semantic_fact_id: str
    document_id: str
    table_fragment_id: str
    row_id: str
    cell_id: str
    metric_path: str
    leaf_metric: str
    bucket_label: str
    bucket_kind: str
    value_raw: str
    value_normalized: str | None
    scale: float | None
    scale_unit: str | None
    currency_code: str | None
    source_traceback: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RowMatrix:
    """A metric row's multi-column value matrix for multi-period operands."""

    semantic_fact_id: str
    document_id: str
    table_fragment_id: str
    row_id: str
    metric_path: str
    leaf_metric: str
    dimensions: tuple[dict[str, Any], ...]
    scale: float | None
    scale_unit: str | None
    currency_code: str | None
    equivalent_group_id: str | None
    source_traceback: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_fact_id": self.semantic_fact_id,
            "document_id": self.document_id,
            "table_fragment_id": self.table_fragment_id,
            "row_id": self.row_id,
            "metric_path": self.metric_path,
            "leaf_metric": self.leaf_metric,
            "dimensions": [dict(d) for d in self.dimensions],
            "scale": self.scale,
            "scale_unit": self.scale_unit,
            "currency_code": self.currency_code,
            "equivalent_group_id": self.equivalent_group_id,
            "source_traceback": self.source_traceback,
        }


@dataclass(frozen=True)
class NarrativeEvidence:
    """Section / heading / paragraph evidence (no LLM summary)."""

    semantic_evidence_id: str
    document_id: str
    pdf_page: int
    section_path: str
    heading: str
    raw_text: str
    bbox: list[float] | None
    evidence_subtype: str
    source_traceback: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Identity builders (structural fields only — no question/gold/expected)
# ---------------------------------------------------------------------------


def build_atomic_fact_id(
    document_id: str,
    table_fragment_id: str,
    row_id: str,
    cell_id: str,
) -> str:
    payload = _stable_json(
        {
            "schema": SEMANTIC_SCHEMA_VERSION,
            "type": "atomic_fact",
            "document_id": document_id,
            "table_fragment_id": table_fragment_id,
            "row_id": row_id,
            "cell_id": cell_id,
        }
    )
    return "atomic:" + _sha256(payload)


def build_comparison_fact_id(
    document_id: str,
    table_fragment_id: str,
    row_id: str,
    comparison_role: str,
    cell_id: str | None,
) -> str:
    payload = _stable_json(
        {
            "schema": SEMANTIC_SCHEMA_VERSION,
            "type": "comparison_fact",
            "document_id": document_id,
            "table_fragment_id": table_fragment_id,
            "row_id": row_id,
            "comparison_role": comparison_role,
            "cell_id": cell_id or "",
        }
    )
    return "comparison:" + _sha256(payload)


def build_bucket_fact_id(
    document_id: str,
    table_fragment_id: str,
    row_id: str,
    cell_id: str,
) -> str:
    payload = _stable_json(
        {
            "schema": SEMANTIC_SCHEMA_VERSION,
            "type": "bucket_fact",
            "document_id": document_id,
            "table_fragment_id": table_fragment_id,
            "row_id": row_id,
            "cell_id": cell_id,
        }
    )
    return "bucket:" + _sha256(payload)


def build_row_matrix_id(
    document_id: str,
    table_fragment_id: str,
    row_id: str,
) -> str:
    payload = _stable_json(
        {
            "schema": SEMANTIC_SCHEMA_VERSION,
            "type": "row_matrix",
            "document_id": document_id,
            "table_fragment_id": table_fragment_id,
            "row_id": row_id,
        }
    )
    return "matrix:" + _sha256(payload)


def build_narrative_evidence_id(
    document_id: str,
    pdf_page: int,
    section_path: str,
    heading: str,
    raw_text: str,
) -> str:
    payload = _stable_json(
        {
            "schema": SEMANTIC_SCHEMA_VERSION,
            "type": "narrative_evidence",
            "document_id": document_id,
            "pdf_page": pdf_page,
            "section_path": section_path,
            "heading": heading,
            "raw_text": raw_text[:500],
        }
    )
    return "narrative:" + _sha256(payload)


def build_equivalent_group_id(row_ids: list[str]) -> str:
    """Build a deterministic equivalent-set group id from physical row ids."""
    payload = _stable_json(
        {
            "schema": SEMANTIC_SCHEMA_VERSION,
            "type": "equivalent_set",
            "row_ids": sorted(row_ids),
        }
    )
    return "equiv:" + _sha256(payload)


def canonical_semantic_fact_id(
    physical_fact_ids: list[str],
) -> str:
    """Collapse multiple physical facts from an equivalent set into one canonical id."""
    payload = _stable_json(
        {
            "schema": SEMANTIC_SCHEMA_VERSION,
            "type": "canonical_equivalent",
            "facts": sorted(physical_fact_ids),
        }
    )
    return "canonical:" + _sha256(payload)
