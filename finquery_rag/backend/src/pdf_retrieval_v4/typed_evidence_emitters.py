"""Gate 03 R2 Pass E — Typed Evidence Emission.

Emits five types of semantic evidence from the semantic graph:

  1. Atomic Fact     — Metric × one Axis × Value
  2. Comparison Fact — % change / yoy between periods
  3. Bucket Fact     — value bound to a bucket dimension
  4. Row Matrix      — multi-column value matrix for multi-period operands
  5. Narrative Evidence — section/heading/paragraph (no LLM summary)

Admission gates:
  Atomic Fact Admission     >= 85%
  Typed Evidence Admission  >= 97%  (Atomic + Comparison + Bucket + RowMatrix)
"""

from __future__ import annotations

from typing import Any

from src.pdf_retrieval_v4.semantic_graph_models import (
    AtomicFact,
    BucketFact,
    ComparisonFact,
    NarrativeEvidence,
    RowMatrix,
    SemanticAxisBinding,
    SemanticRow,
    MetricPath,
    ScaleResolution,
    CurrencyResolution,
    build_atomic_fact_id,
    build_bucket_fact_id,
    build_comparison_fact_id,
    build_narrative_evidence_id,
    build_row_matrix_id,
)
from src.pdf_retrieval_v4.table_html_parser import norm_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_numeric_value(cell: dict[str, Any]) -> tuple[str, str | None]:
    """Extract (raw_value, normalized_value) from a cell."""
    parsed = cell.get("parsed_numeric") or []
    raw_text = str(cell.get("resolved_text") or "").strip()
    if parsed:
        first = parsed[0]
        return raw_text, str(first.get("normalized") or "")
    return raw_text, None


# ---------------------------------------------------------------------------
# 1. Atomic Fact Emitter
# ---------------------------------------------------------------------------


def emit_atomic_facts(
    semantic_rows: list[SemanticRow],
    metric_paths: list[MetricPath],
    axis_bindings: list[SemanticAxisBinding],
    cells: list[dict[str, Any]],
    scale: ScaleResolution,
    currency: CurrencyResolution,
    equivalence_map: dict[str, str],
) -> list[AtomicFact]:
    """Emit AtomicFact for each (financial-data row, numeric cell with temporal axis).

    An atomic fact is generated when:
    - The row is a financial-data row (metric_row / subtotal / total)
    - The cell has a parsed numeric value
    - The cell's temporal kind is point, duration, or comparison
      (not bucket/segment/category/non_temporal/unknown)
    """
    mp_by_row: dict[str, MetricPath] = {mp.row_id: mp for mp in metric_paths}
    axis_by_cell: dict[str, SemanticAxisBinding] = {
        ab.cell_id: ab for ab in axis_bindings
    }

    sr_by_id: dict[str, SemanticRow] = {sr.row_id: sr for sr in semantic_rows}
    facts: list[AtomicFact] = []

    for cell in cells:
        col = int(cell.get("column_index") or 0)
        if col == 0:
            continue  # Skip metric label column

        row_id = str(cell.get("row_id") or "")
        cell_id = str(cell.get("cell_id") or "")

        # Check if row is financial-data
        sr = sr_by_id.get(row_id)
        if not sr or not sr.is_financial_data_row:
            continue

        # Use SemanticRow as authoritative source for document_id / table_fragment_id
        document_id = sr.document_id
        table_fragment_id = sr.table_fragment_id

        # Check for numeric value - require a normalizable value for admission
        raw_val, norm_val = _get_numeric_value(cell)
        if not raw_val or norm_val is None:
            continue

        # Check temporal axis
        axis = axis_by_cell.get(cell_id)
        if not axis:
            continue

        if axis.temporal_kind not in ("point", "duration", "comparison"):
            continue

        # Get metric path
        mp = mp_by_row.get(row_id)
        if not mp:
            continue

        # Skip if metric_status is missing
        if mp.metric_status == "missing":
            continue

        equiv_group = equivalence_map.get(row_id)

        fact = AtomicFact(
            semantic_fact_id=build_atomic_fact_id(
                document_id, table_fragment_id, row_id, cell_id
            ),
            document_id=document_id,
            table_fragment_id=table_fragment_id,
            row_id=row_id,
            cell_id=cell_id,
            metric_path=mp.metric_path,
            leaf_metric=mp.leaf_metric,
            temporal_kind=axis.temporal_kind,
            normalized_period=axis.normalized_period,
            period_start=axis.period_start,
            period_end=axis.period_end,
            value_raw=raw_val,
            value_normalized=norm_val,
            scale=scale.scale if scale.scale_status == "resolved" else None,
            scale_unit=scale.scale_unit if scale.scale_status == "resolved" else None,
            currency_code=currency.currency_code
            if currency.currency_status == "resolved"
            else None,
            equivalent_group_id=equiv_group,
            source_traceback={
                "document_id": document_id,
                "pdf_page": sr.pdf_page,
                "table_fragment_id": table_fragment_id,
                "row_id": row_id,
                "cell_id": cell_id,
                "bbox": cell.get("cell_bbox"),
                "raw_text": raw_val,
            },
        )
        facts.append(fact)

    return facts


# ---------------------------------------------------------------------------
# 2. Comparison Fact Emitter
# ---------------------------------------------------------------------------


def emit_comparison_facts(
    semantic_rows: list[SemanticRow],
    metric_paths: list[MetricPath],
    axis_bindings: list[SemanticAxisBinding],
    cells: list[dict[str, Any]],
    scale: ScaleResolution,
    equivalence_map: dict[str, str],
) -> list[ComparisonFact]:
    """Emit ComparisonFact for comparison columns (% change).

    A comparison fact links a base period value to a compared period value
    with a reported change (absolute or percentage).
    """
    mp_by_row: dict[str, MetricPath] = {mp.row_id: mp for mp in metric_paths}
    axis_by_cell: dict[str, SemanticAxisBinding] = {
        ab.cell_id: ab for ab in axis_bindings
    }

    sr_by_id: dict[str, SemanticRow] = {sr.row_id: sr for sr in semantic_rows}
    facts: list[ComparisonFact] = []

    # Group cells by row, then find comparison columns
    cells_by_row: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        rid = str(cell.get("row_id") or "")
        cells_by_row.setdefault(rid, []).append(cell)

    for row_id, row_cells in cells_by_row.items():
        sr = sr_by_id.get(row_id)
        if not sr or not sr.is_financial_data_row:
            continue

        mp = mp_by_row.get(row_id)
        if not mp or mp.metric_status == "missing":
            continue

        document_id = sr.document_id
        table_fragment_id = sr.table_fragment_id

        # Find comparison cells in this row
        comparison_cells: list[tuple[dict[str, Any], SemanticAxisBinding]] = []
        for cell in row_cells:
            cell_id = str(cell.get("cell_id") or "")
            axis = axis_by_cell.get(cell_id)
            if axis and axis.temporal_kind == "comparison":
                comparison_cells.append((cell, axis))

        if not comparison_cells:
            continue

        # Find base and compared values from non-comparison numeric cells
        base_value: str | None = None
        base_period: str | None = None
        compared_value: str | None = None
        compared_period: str | None = None

        for cell in row_cells:
            cell_id = str(cell.get("cell_id") or "")
            axis = axis_by_cell.get(cell_id)
            if not axis or axis.temporal_kind not in ("point", "duration"):
                continue
            raw_val, _ = _get_numeric_value(cell)
            if not raw_val:
                continue
            # First temporal numeric cell is base, second is compared
            if base_value is None:
                base_value = raw_val
                base_period = axis.normalized_period
            elif compared_value is None:
                compared_value = raw_val
                compared_period = axis.normalized_period

        for comp_cell, comp_axis in comparison_cells:
            raw_val, norm_val = _get_numeric_value(comp_cell)
            comp_cell_id = str(comp_cell.get("cell_id") or "")

            fact = ComparisonFact(
                semantic_fact_id=build_comparison_fact_id(
                    document_id,
                    table_fragment_id,
                    row_id,
                    comp_axis.comparison_role or "percent_change",
                    comp_cell_id,
                ),
                document_id=document_id,
                table_fragment_id=table_fragment_id,
                row_id=row_id,
                metric_path=mp.metric_path,
                leaf_metric=mp.leaf_metric,
                comparison_role=comp_axis.comparison_role or "percent_change",
                base_period=base_period,
                compared_period=compared_period,
                base_value=base_value,
                compared_value=compared_value,
                reported_change=raw_val,
                value_normalized=norm_val,
                source_traceback={
                    "document_id": document_id,
                    "pdf_page": sr.pdf_page,
                    "table_fragment_id": table_fragment_id,
                    "row_id": row_id,
                    "cell_id": comp_cell_id,
                    "bbox": comp_cell.get("cell_bbox"),
                    "raw_text": raw_val,
                },
            )
            facts.append(fact)

    return facts


# ---------------------------------------------------------------------------
# 3. Bucket Fact Emitter
# ---------------------------------------------------------------------------


def emit_bucket_facts(
    semantic_rows: list[SemanticRow],
    metric_paths: list[MetricPath],
    axis_bindings: list[SemanticAxisBinding],
    cells: list[dict[str, Any]],
    scale: ScaleResolution,
    currency: CurrencyResolution,
) -> list[BucketFact]:
    """Emit BucketFact for cells with temporal_kind == 'bucket'."""
    mp_by_row: dict[str, MetricPath] = {mp.row_id: mp for mp in metric_paths}
    axis_by_cell: dict[str, SemanticAxisBinding] = {
        ab.cell_id: ab for ab in axis_bindings
    }

    sr_by_id: dict[str, SemanticRow] = {sr.row_id: sr for sr in semantic_rows}
    facts: list[BucketFact] = []

    for cell in cells:
        col = int(cell.get("column_index") or 0)
        if col == 0:
            continue

        cell_id = str(cell.get("cell_id") or "")
        row_id = str(cell.get("row_id") or "")

        axis = axis_by_cell.get(cell_id)
        if not axis or axis.temporal_kind != "bucket":
            continue

        sr = sr_by_id.get(row_id)
        if not sr or not sr.is_financial_data_row:
            continue

        document_id = sr.document_id
        table_fragment_id = sr.table_fragment_id

        mp = mp_by_row.get(row_id)
        if not mp or mp.metric_status == "missing":
            continue

        raw_val, norm_val = _get_numeric_value(cell)
        if not raw_val:
            continue

        bucket_label = axis.bucket_label or str(cell.get("resolved_text") or "")[:200]
        # Determine bucket kind from context
        bucket_kind = "maturity"
        label_lower = norm_text(bucket_label)
        if any(
            w in label_lower
            for w in ("aging", "past due", "impaired", "30 day", "60 day", "90 day")
        ):
            bucket_kind = "aging"
        elif any(w in label_lower for w in ("rating", "grade", "tier")):
            bucket_kind = "rating"
        elif any(w in label_lower for w in ("range", "band")):
            bucket_kind = "range"

        fact = BucketFact(
            semantic_fact_id=build_bucket_fact_id(
                document_id, table_fragment_id, row_id, cell_id
            ),
            document_id=document_id,
            table_fragment_id=table_fragment_id,
            row_id=row_id,
            cell_id=cell_id,
            metric_path=mp.metric_path,
            leaf_metric=mp.leaf_metric,
            bucket_label=bucket_label,
            bucket_kind=bucket_kind,
            value_raw=raw_val,
            value_normalized=norm_val,
            scale=scale.scale if scale.scale_status == "resolved" else None,
            scale_unit=scale.scale_unit if scale.scale_status == "resolved" else None,
            currency_code=currency.currency_code
            if currency.currency_status == "resolved"
            else None,
            source_traceback={
                "document_id": document_id,
                "pdf_page": sr.pdf_page,
                "table_fragment_id": table_fragment_id,
                "row_id": row_id,
                "cell_id": cell_id,
                "bbox": cell.get("cell_bbox"),
                "raw_text": raw_val,
            },
        )
        facts.append(fact)

    return facts


# ---------------------------------------------------------------------------
# 4. Row Matrix Emitter
# ---------------------------------------------------------------------------


def emit_row_matrices(
    semantic_rows: list[SemanticRow],
    metric_paths: list[MetricPath],
    axis_bindings: list[SemanticAxisBinding],
    cells: list[dict[str, Any]],
    scale: ScaleResolution,
    currency: CurrencyResolution,
    equivalence_map: dict[str, str],
) -> list[RowMatrix]:
    """Emit RowMatrix for each financial-data row with multiple temporal columns.

    A RowMatrix captures all (axis, value) pairs for a single metric row,
    enabling multi-period operand computation without depending on multiple
    Atomic Facts being recalled simultaneously.
    """
    mp_by_row: dict[str, MetricPath] = {mp.row_id: mp for mp in metric_paths}
    axis_by_cell: dict[str, SemanticAxisBinding] = {
        ab.cell_id: ab for ab in axis_bindings
    }

    cells_by_row: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        rid = str(cell.get("row_id") or "")
        cells_by_row.setdefault(rid, []).append(cell)

    facts: list[RowMatrix] = []

    for sr in semantic_rows:
        if not sr.is_financial_data_row:
            continue

        mp = mp_by_row.get(sr.row_id)
        if not mp or mp.metric_status == "missing":
            continue

        row_cells = cells_by_row.get(sr.row_id, [])
        dimensions: list[dict[str, Any]] = []

        for cell in row_cells:
            col = int(cell.get("column_index") or 0)
            if col == 0:
                continue

            cell_id = str(cell.get("cell_id") or "")
            axis = axis_by_cell.get(cell_id)
            if not axis:
                continue

            raw_val, norm_val = _get_numeric_value(cell)
            if not raw_val:
                continue

            dimensions.append(
                {
                    "cell_id": cell_id,
                    "column_index": col,
                    "temporal_kind": axis.temporal_kind,
                    "normalized_period": axis.normalized_period,
                    "period_start": axis.period_start,
                    "period_end": axis.period_end,
                    "comparison_role": axis.comparison_role,
                    "bucket_label": axis.bucket_label,
                    "segment_label": axis.segment_label,
                    "value_raw": raw_val,
                    "value_normalized": norm_val,
                }
            )

        if len(dimensions) < 2:
            continue  # RowMatrix only for multi-dimension rows

        equiv_group = equivalence_map.get(sr.row_id)

        fact = RowMatrix(
            semantic_fact_id=build_row_matrix_id(
                sr.document_id, sr.table_fragment_id, sr.row_id
            ),
            document_id=sr.document_id,
            table_fragment_id=sr.table_fragment_id,
            row_id=sr.row_id,
            metric_path=mp.metric_path,
            leaf_metric=mp.leaf_metric,
            dimensions=tuple(dimensions),
            scale=scale.scale if scale.scale_status == "resolved" else None,
            scale_unit=scale.scale_unit if scale.scale_status == "resolved" else None,
            currency_code=currency.currency_code
            if currency.currency_status == "resolved"
            else None,
            equivalent_group_id=equiv_group,
            source_traceback={
                "document_id": sr.document_id,
                "pdf_page": sr.pdf_page,
                "table_fragment_id": sr.table_fragment_id,
                "row_id": sr.row_id,
                "cell_id": None,
                "bbox": None,
                "raw_text": None,
            },
        )
        facts.append(fact)

    return facts


# ---------------------------------------------------------------------------
# 5. Narrative Evidence Emitter
# ---------------------------------------------------------------------------


def emit_narrative_evidence(
    narrative_blocks: list[dict[str, Any]],
) -> list[NarrativeEvidence]:
    """Emit NarrativeEvidence from pre-extracted narrative blocks.

    Parameters
    ----------
    narrative_blocks
        List of dicts, each with keys:
        ``document_id``, ``pdf_page``, ``section_path``, ``heading``,
        ``raw_text``, ``bbox``, ``evidence_subtype``.

    No LLM summarization is performed — raw_text is stored as-is.
    """
    facts: list[NarrativeEvidence] = []

    for block in narrative_blocks:
        document_id = str(block.get("document_id") or "")
        pdf_page = int(block.get("pdf_page") or 0)
        section_path = str(block.get("section_path") or "")
        heading = str(block.get("heading") or "")
        raw_text = str(block.get("raw_text") or "")
        bbox = block.get("bbox")
        evidence_subtype = str(block.get("evidence_subtype") or "paragraph")

        if not raw_text.strip():
            continue

        fact = NarrativeEvidence(
            semantic_evidence_id=build_narrative_evidence_id(
                document_id, pdf_page, section_path, heading, raw_text
            ),
            document_id=document_id,
            pdf_page=pdf_page,
            section_path=section_path,
            heading=heading,
            raw_text=raw_text[:2000],  # Cap to avoid huge artifacts
            bbox=bbox,
            evidence_subtype=evidence_subtype,
            source_traceback={
                "document_id": document_id,
                "pdf_page": pdf_page,
                "table_fragment_id": None,
                "row_id": None,
                "cell_id": None,
                "bbox": bbox,
                "raw_text": raw_text[:2000],
            },
        )
        facts.append(fact)

    return facts
