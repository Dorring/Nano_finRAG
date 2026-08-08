"""Gate 03 R2 Pass A — Semantic Row Classification.

Classifies each physical row into a ``row_type`` using ONLY
document-internal signals: row label text, numeric content, indentation,
and header-position.  No question / gold / company-specific rules.

Row types (see ``ROW_TYPES`` in ``semantic_graph_models``):
  metric_row, group_header, section_header, column_header,
  subtotal, total, note, spacer, unknown

Only ``metric_row``, ``subtotal``, and ``total`` are
``semantic_eligible`` and enter the Metric Coverage denominator.
"""

from __future__ import annotations

import re
from typing import Any

from src.pdf_retrieval_v4.semantic_graph_models import (
    FINANCIAL_DATA_ROW_TYPES,
    SemanticRow,
)
from src.pdf_retrieval_v4.table_html_parser import norm_text

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_TOTAL_RE = re.compile(r"^(total|total\s+|net\s+total|grand\s+total)", re.IGNORECASE)
_SUBTOTAL_RE = re.compile(r"^(sub[-\s]?total|subtotal)", re.IGNORECASE)
_SECTION_HEADER_RE = re.compile(
    r"^[A-Z][A-Z\s,&\-]{3,}$"  # ALL CAPS short label
)
_NOTE_RE = re.compile(
    r"^(\([\d]+\)|note\s+\d|see note|see\s+accompanying)", re.IGNORECASE
)
_SPACER_RE = re.compile(r"^[\s—\-\.]*$")

# Labels that are almost never metric rows
_NON_METRIC_LABELS = (
    "title of each class",
    "trading symbol",
    "name of each exchange",
    "years ended",
    "as of",
    "in millions",
    "in thousands",
    "in billions",
    "dollars in millions",
    "(in millions)",
    "(in thousands)",
    "(in billions)",
    "amounts in",
    "fiscal year",
    "three months ended",
    "six months ended",
    "nine months ended",
    "twelve months ended",
    "year ended",
    "quarter ended",
    "periods",
    "description",
    "category",
    "type",
)

# Labels that indicate a group/header row (parent metric)
_GROUP_HEADER_HINTS = (
    "operating activities",
    "investing activities",
    "financing activities",
    "current assets",
    "current liabilities",
    "non-current assets",
    "non-current liabilities",
    "assets",
    "liabilities",
    "stockholders' equity",
    "shareholders' equity",
    "operating expenses",
    "operating income",
    "other income",
    "other expense",
)


def _has_numeric(cells: list[dict[str, Any]]) -> bool:
    """Return True if any non-first-column cell has a parsed numeric value."""
    for cell in cells:
        col = int(cell.get("column_index") or 0)
        if col == 0:
            continue
        parsed = cell.get("parsed_numeric") or []
        if parsed:
            return True
    return False


def _is_column_header_row(
    row: dict[str, Any],
    cells: list[dict[str, Any]],
) -> bool:
    """Detect a column-header row: first cell is a period/label, other cells
    are period labels or empty, and no numeric data values."""
    metric_text = norm_text(row.get("metric_text") or "")
    if not metric_text:
        return False
    # Column header rows often have period-like text
    period_hints = (
        "year ended",
        "as of",
        "years ended",
        "in millions",
        "in thousands",
        "fiscal year",
        "three months",
        "six months",
        "nine months",
        "twelve months",
        "quarter ended",
        "periods",
        "months ended",
    )
    if any(hint in metric_text for hint in period_hints):
        return True
    # If the row has no numeric values AND first cell is a known label
    if not _has_numeric(cells):
        if metric_text in _NON_METRIC_LABELS:
            return True
    return False


def _detect_parent_row(
    row: dict[str, Any],
    cells: list[dict[str, Any]],
    prev_data_row_id: str | None,
) -> str | None:
    """Detect if this row is a group_header that parents subsequent metric rows.

    A group_header has:
    - A non-empty label
    - No numeric values in value columns (or all empty)
    - A label that matches known group header hints
    """
    metric_text = norm_text(row.get("metric_text") or "")
    if not metric_text:
        return None
    if _has_numeric(cells):
        return None
    for hint in _GROUP_HEADER_HINTS:
        if hint in metric_text:
            return prev_data_row_id  # parent is the last data row before this
    return None


def classify_row(
    row: dict[str, Any],
    cells: list[dict[str, Any]],
    table_fragment_id: str,
    document_id: str,
    pdf_page: int,
    prev_data_row_id: str | None,
) -> SemanticRow:
    """Classify a single physical row into a SemanticRow.

    Parameters
    ----------
    row
        Row dict from adapter-predictions.
    cells
        Cell dicts belonging to this row (filtered by row_index).
    table_fragment_id, document_id, pdf_page
        Parent table/page context.
    prev_data_row_id
        The row_id of the previous financial-data row in the same table
        (used for parent metric linking).  None if this is the first row.
    """
    row_id_val = str(row.get("row_id") or "")
    row_index = int(row.get("row_index") or 0)
    metric_text = str(row.get("metric_text") or "")
    raw_label = metric_text[:300]
    normed_label = norm_text(raw_label)

    has_num = _has_numeric(cells)
    parent_row_id: str | None = None

    # 1. Spacer: empty or only dashes/dots
    if not normed_label or _SPACER_RE.match(raw_label.strip()):
        row_type = "spacer"

    # 2. Note: parenthesized number or "see note"
    elif _NOTE_RE.match(raw_label.strip()):
        row_type = "note"

    # 3. Column header: period labels, scale labels
    elif _is_column_header_row(row, cells):
        row_type = "column_header"

    # 4. Total
    elif _TOTAL_RE.match(raw_label.strip()) or _TOTAL_RE.match(normed_label):
        row_type = "total"
        parent_row_id = prev_data_row_id

    # 5. Subtotal
    elif _SUBTOTAL_RE.match(raw_label.strip()) or _SUBTOTAL_RE.match(normed_label):
        row_type = "subtotal"
        parent_row_id = prev_data_row_id

    # 6. Group header: has label, no numerics, matches group hints
    elif not has_num and any(hint in normed_label for hint in _GROUP_HEADER_HINTS):
        row_type = "group_header"

    # 7. Section header: ALL CAPS short label, no numerics
    elif not has_num and _SECTION_HEADER_RE.match(raw_label.strip()):
        row_type = "section_header"

    # 8. Metric row: has a label AND has numeric values in value columns
    elif has_num and normed_label and normed_label not in _NON_METRIC_LABELS:
        row_type = "metric_row"
        parent_row_id = prev_data_row_id

    # 9. Non-metric with numerics but label is a known non-metric
    elif normed_label in _NON_METRIC_LABELS:
        row_type = "column_header"

    # 10. Fallback
    else:
        row_type = "unknown"

    semantic_eligible = row_type in FINANCIAL_DATA_ROW_TYPES

    source_traceback = {
        "row_id": row_id_val,
        "table_fragment_id": table_fragment_id,
        "document_id": document_id,
        "pdf_page": pdf_page,
        "row_index": row_index,
        "row_bbox": row.get("row_bbox") or [],
    }

    return SemanticRow(
        row_id=row_id_val,
        table_fragment_id=table_fragment_id,
        document_id=document_id,
        pdf_page=pdf_page,
        row_index=row_index,
        row_type=row_type,
        raw_label=raw_label,
        parent_row_id=parent_row_id,
        semantic_eligible=semantic_eligible,
        source_traceback=source_traceback,
    )


def classify_table_rows(
    table: dict[str, Any],
    logical_table_id: str,
    document_id: str,
    pdf_page: int,
) -> list[SemanticRow]:
    """Classify all rows in a table fragment.

    Returns the list of SemanticRow in row_index order, with parent_row_id
    linking each financial-data row to the preceding group/header row.
    """
    rows = table.get("rows") or []
    cells = table.get("cells") or []

    # Group cells by row_index
    cells_by_row: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        ri = int(cell.get("row_index") or 0)
        cells_by_row.setdefault(ri, []).append(cell)

    result: list[SemanticRow] = []
    prev_group_row_id: str | None = None

    for row in rows:
        ri = int(row.get("row_index") or 0)
        row_cells = cells_by_row.get(ri, [])
        sr = classify_row(
            row,
            row_cells,
            logical_table_id,
            document_id,
            pdf_page,
            prev_group_row_id,
        )
        result.append(sr)
        # Track group headers as potential parents for subsequent metric rows
        if sr.row_type == "group_header":
            prev_group_row_id = sr.row_id
        elif sr.is_financial_data_row:
            # Keep the group header as parent until a new one appears
            pass

    return result
