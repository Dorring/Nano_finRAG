"""Pure contracts and fail-closed gates for NF-OPT-08 shadow reingestion."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import hashlib
import json
from typing import Iterable


ENABLE_TABLE_FACT_EXTRACTION = False


class ParserCapabilityStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class StructuredTableCell:
    row_index: int
    column_index: int
    row_span: int
    column_span: int
    raw_text: str
    normalized_text: str
    numeric_value: Decimal | None
    value_type: str
    bbox: tuple[float, float, float, float] | None
    confidence: float | None


@dataclass(frozen=True)
class StructuredTableRecord:
    shadow_table_id: str
    document_id: str
    pdf_sha256: str
    pdf_page_start: int
    pdf_page_end: int
    table_title: str | None
    table_caption: str | None
    cells: tuple[StructuredTableCell, ...]
    currency: str | None
    scale: str | None
    parser_name: str
    parser_version: str
    parser_artifact_hash: str
    source_page_image_hash: str | None


@dataclass(frozen=True)
class ShadowEvidenceMapping:
    legacy_candidate_key: str
    shadow_table_id: str
    shadow_row_id: str
    shadow_cell_ids: tuple[str, ...]
    relation: str
    document_match: bool
    page_match: bool
    metric_match: bool
    period_match: bool
    value_match: bool
    scale_match: bool
    reviewer: str | None
    reviewed_at: str | None
    verified: bool


def stable_shadow_id(kind: str, *parts: object) -> str:
    payload = json.dumps([kind, *parts], sort_keys=True, separators=(",", ":"))
    return "shadow:" + hashlib.sha256(payload.encode()).hexdigest()


def mapping_is_manually_verified(mapping: ShadowEvidenceMapping) -> bool:
    """Require every semantic check and a named, dated human review."""
    return bool(
        mapping.verified
        and mapping.reviewer
        and mapping.reviewed_at
        and mapping.shadow_table_id
        and mapping.shadow_row_id
        and mapping.shadow_cell_ids
        and mapping.relation
        and mapping.document_match
        and mapping.page_match
        and mapping.metric_match
        and mapping.period_match
        and mapping.value_match
        and mapping.scale_match
    )


def parser_capability_gate(records: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(records)
    metrics = {
        "source_count": len(rows),
        "table_detected_count": sum(bool(row.get("table_detected")) for row in rows),
        "correct_table_boundary_count": sum(
            bool(row.get("correct_table_boundary")) for row in rows
        ),
        "required_row_recovery_count": sum(
            bool(row.get("required_row_recovered")) for row in rows
        ),
        "required_cell_recovery_count": sum(
            bool(row.get("required_cells_recovered")) for row in rows
        ),
        "period_recovery_count": sum(
            bool(row.get("period_recovered")) for row in rows
        ),
        "scale_recovery_count": sum(bool(row.get("scale_recovered")) for row in rows),
        "currency_recovery_count": sum(
            bool(row.get("currency_recovered")) for row in rows
        ),
        "evidence_page_accuracy_count": sum(
            bool(row.get("evidence_page_correct")) for row in rows
        ),
        "wrong_table_selection_count": sum(
            bool(row.get("wrong_table_selected")) for row in rows
        ),
        "wrong_row_mapping_count": sum(bool(row.get("wrong_row_mapped")) for row in rows),
        "wrong_column_mapping_count": sum(
            bool(row.get("wrong_column_mapped")) for row in rows
        ),
        "cross_table_join_count": sum(bool(row.get("cross_table_join")) for row in rows),
        "page_mismatch_count": sum(bool(row.get("page_mismatch")) for row in rows),
    }
    passed = (
        metrics["source_count"] == 22
        and metrics["table_detected_count"] >= 21
        and metrics["required_row_recovery_count"] >= 20
        and metrics["required_cell_recovery_count"] >= 20
        and metrics["period_recovery_count"] >= 20
        and metrics["evidence_page_accuracy_count"] == 22
        and metrics["wrong_table_selection_count"] == 0
        and metrics["wrong_row_mapping_count"] == 0
        and metrics["wrong_column_mapping_count"] == 0
        and metrics["cross_table_join_count"] == 0
        and metrics["page_mismatch_count"] == 0
    )
    return {**metrics, "gate_passed": passed}


def combined_table_count(pymupdf_count: int, camelot_count: int) -> int:
    """Native parser capability is the union of its two fixed detectors."""
    return max(0, pymupdf_count) + max(0, camelot_count)


def require_safe_parser_inputs(inputs: dict[str, object]) -> None:
    """Parser inputs may contain a PDF locator and page, never answer metadata."""
    forbidden = {
        "case_id",
        "expected_answer",
        "expected_operand_value",
        "expected_scale",
        "expected_operation_result",
        "gold",
        "label",
    }
    overlap = forbidden & set(inputs)
    if overlap:
        raise ValueError("parser input contains prohibited benchmark metadata")
