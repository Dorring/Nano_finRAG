"""Gate 03 R2 Pass C — Temporal / Dimension Graph.

Classifies each value-column's temporal/dimension semantics into one of::

    point, duration, comparison, bucket, segment, category,
    non_temporal, unknown

and produces a ``SemanticAxisBinding`` per cell.

Key principle: **fail-closed**.  When the temporal axis cannot be
determined, ``temporal_kind = "unknown"`` — never force-bind to increase
admission.
"""

from __future__ import annotations

import re
from typing import Any

from src.pdf_retrieval_v4.semantic_graph_models import SemanticAxisBinding

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# "As of Dec 31, 2025" → point
_POINT_RE = re.compile(r"\bas\s+of\b|\bat\s+\w+\s+\d", re.IGNORECASE)
# "Year ended Dec 31, 2025" → duration
_DURATION_RE = re.compile(
    r"\b(year|years|months|quarter|six months|nine months|three months|"
    r"twelve months)\s+(ended|ending)\b",
    re.IGNORECASE,
)
# "FY2025", "Fiscal 2025" → duration (annual)
_FISCAL_RE = re.compile(r"\bfy\s*\d{4}\b|\bfiscal\s+(year\s+)?\d{4}\b", re.IGNORECASE)
# "% change", "% Change", "Change" → comparison
_COMPARISON_RE = re.compile(
    r"%\s*change|change\s*%|percent\s*change|yoy|year.over.year|\bincrease\b|\bdecrease\b",
    re.IGNORECASE,
)
# Bucket: "Less than 1 year", "1-3 years", "More than 5 years"
_BUCKET_RE = re.compile(
    r"less\s+than|more\s+than|over\s+\d|\d\s*[-–to]+\s*\d\s*year"
    r"|range|rating|grade|tier",
    re.IGNORECASE,
)
# Segment: "Americas", "EMEA", "APAC", "U.S.", "International"
_SEGMENT_RE = re.compile(
    r"\b(americas|emea|apac|europe|asia|japan|china|canada|"
    r"international|u\.s\.|united states|domestic|foreign|"
    r"greater china|rest of)\b",
    re.IGNORECASE,
)
# Category: "Product", "Service", "Hardware", "Software"
_CATEGORY_RE = re.compile(
    r"\b(product|products|service|services|hardware|software|"
    r"subscription|license|advertising|other)\b",
    re.IGNORECASE,
)
# Date extraction for period_start/period_end
_DATE_RE = re.compile(r"(\w+\s+\d{1,2},?\s+\d{4}|\d{4})")

# Month names for date normalization
_MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "sept": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def _extract_dates(text: str) -> list[str]:
    """Extract date strings from text, returning ISO-like YYYY-MM-DD or YYYY."""
    dates: list[str] = []
    for match in _DATE_RE.finditer(text):
        raw = match.group(1).strip()
        # Try "Month DD, YYYY"
        m = re.match(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", raw)
        if m:
            month_name = m.group(1).lower()
            day = m.group(2).zfill(2)
            year = m.group(3)
            month = _MONTHS.get(month_name)
            if month:
                dates.append(f"{year}-{month}-{day}")
                continue
        # Try bare year
        if re.fullmatch(r"\d{4}", raw):
            dates.append(raw)
    return dates


def _classify_column_temporal(
    header_path: list[str],
    normalized_period: str | None,
    period_kind: str | None,
    cell_text: str,
) -> tuple[str, str | None, str | None, str | None]:
    """Classify a single column's temporal kind.

    Returns (temporal_kind, period_start, period_end, comparison_role).
    """
    header_text = " ".join(str(h) for h in header_path)
    combined = header_text + " " + cell_text

    # 1. Comparison: % change columns
    if _COMPARISON_RE.search(combined):
        return ("comparison", None, None, "percent_change")

    # 2. Bucket: maturity/aging ranges
    if _BUCKET_RE.search(combined):
        return ("bucket", None, None, None)

    # 3. Segment: geographic/business segment
    if _SEGMENT_RE.search(combined):
        return ("segment", None, None, None)

    # 4. Category: product/service
    if _CATEGORY_RE.search(combined) and not _POINT_RE.search(combined):
        # Only category if no temporal signal
        if not (_DURATION_RE.search(combined) or _POINT_RE.search(combined)):
            return ("category", None, None, None)

    # 5. Point: "As of <date>"
    if _POINT_RE.search(combined):
        dates = _extract_dates(combined)
        period_start = dates[0] if dates else None
        return ("point", period_start, period_start, None)

    # 6. Duration: "Year ended <date>"
    if _DURATION_RE.search(combined):
        dates = _extract_dates(combined)
        period_end = dates[-1] if dates else None
        period_start = dates[0] if len(dates) > 1 else None
        return ("duration", period_start, period_end, None)

    # 7. Fiscal year: "FY2025"
    if _FISCAL_RE.search(combined) or normalized_period:
        period = normalized_period or ""
        # Extract year from FY2025
        year_match = re.search(r"(\d{4})", period)
        if year_match:
            year = year_match.group(1)
            return ("duration", year, year, None)
        return ("duration", None, None, None)

    # 8. period_kind from adapter (instant/duration)
    if period_kind == "instant":
        return ("point", None, None, None)
    if period_kind == "duration":
        return ("duration", None, None, None)

    # 9. Non-temporal: if the cell is in column 0 (metric label column)
    # or has no numeric value and no date-like text
    if not cell_text.strip():
        return ("non_temporal", None, None, None)

    # 10. Unknown: cannot determine
    return ("unknown", None, None, None)


def _extract_bucket_label(header_path: list[str], cell_text: str) -> str | None:
    """Extract the bucket label from header or cell text."""
    combined = " ".join(str(h) for h in header_path) + " " + cell_text
    combined = combined.strip()
    if combined:
        return combined[:200]
    return None


def _extract_segment_label(header_path: list[str]) -> str | None:
    """Extract the segment label from header path."""
    for h in header_path:
        if _SEGMENT_RE.search(str(h)):
            return str(h).strip()[:200]
    return None


def _extract_category_label(header_path: list[str]) -> str | None:
    """Extract the category label from header path."""
    for h in header_path:
        if _CATEGORY_RE.search(str(h)):
            return str(h).strip()[:200]
    return None


def build_axis_bindings(
    cells: list[dict[str, Any]],
    table_fragment_id: str,
) -> list[SemanticAxisBinding]:
    """Build SemanticAxisBinding for all cells in a table.

    The temporal kind is determined per-column (all cells in the same
    column share the same temporal classification), then refined per-cell
    for period/bucket/segment/category labels.
    """
    # First pass: classify each unique column by examining all its cells
    column_temporal: dict[int, tuple[str, str | None, str | None, str | None]] = {}

    cells_by_col: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        col = int(cell.get("column_index") or 0)
        cells_by_col.setdefault(col, []).append(cell)

    for col, col_cells in cells_by_col.items():
        # Aggregate header_path and text across all cells in this column
        all_headers: list[str] = []
        all_text: list[str] = []
        normalized_period = None
        period_kind = None
        for c in col_cells:
            hp = c.get("header_path") or []
            all_headers.extend(str(h) for h in hp)
            all_text.append(str(c.get("resolved_text") or ""))
            if c.get("normalized_period"):
                normalized_period = c.get("normalized_period")
            if c.get("period_kind"):
                period_kind = c.get("period_kind")

        combined_text = " ".join(all_text)

        temporal_kind, p_start, p_end, comp_role = _classify_column_temporal(
            all_headers, normalized_period, period_kind, combined_text
        )
        column_temporal[col] = (temporal_kind, p_start, p_end, comp_role)

    # Second pass: create SemanticAxisBinding per cell
    results: list[SemanticAxisBinding] = []
    for cell in cells:
        cell_id = str(cell.get("cell_id") or "")
        row_id = str(cell.get("row_id") or "")
        col = int(cell.get("column_index") or 0)
        header_path = cell.get("header_path") or []
        cell_text = str(cell.get("resolved_text") or "")

        temporal_kind, p_start, p_end, comp_role = column_temporal.get(
            col, ("unknown", None, None, None)
        )

        # Refine per-cell labels
        bucket_label = None
        segment_label = None
        category_label = None
        normalized_period = cell.get("normalized_period")

        if temporal_kind == "bucket":
            bucket_label = _extract_bucket_label(header_path, cell_text)
        elif temporal_kind == "segment":
            segment_label = _extract_segment_label(header_path) or cell_text[:200]
        elif temporal_kind == "category":
            category_label = _extract_category_label(header_path) or cell_text[:200]

        # For point/duration, try to extract per-cell dates
        if temporal_kind == "point":
            dates = _extract_dates(
                " ".join(str(h) for h in header_path) + " " + cell_text
            )
            if dates:
                p_start = dates[0]
                p_end = dates[0]
        elif temporal_kind == "duration":
            dates = _extract_dates(
                " ".join(str(h) for h in header_path) + " " + cell_text
            )
            if dates:
                p_end = dates[-1]
                if len(dates) > 1:
                    p_start = dates[0]

        results.append(
            SemanticAxisBinding(
                cell_id=cell_id,
                row_id=row_id,
                table_fragment_id=table_fragment_id,
                column_index=col,
                temporal_kind=temporal_kind,
                period_start=p_start,
                period_end=p_end,
                normalized_period=normalized_period,
                comparison_role=comp_role,
                bucket_label=bucket_label,
                segment_label=segment_label,
                category_label=category_label,
            )
        )

    return results
