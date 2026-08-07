"""Gate 02 R3 structure integrity checks.

Validates Page/Table/Row/Cell completeness, identity uniqueness,
BBox validity, and foreign key integrity.
"""

from __future__ import annotations

from typing import Any


def check_page_integrity(pages: list[dict[str, Any]], expected_pages: int) -> dict[str, Any]:
    """Check page-level integrity."""
    page_keys = [(p["document_id"], p["pdf_page"]) for p in pages]
    unique_pages = set(page_keys)
    return {
        "page_records": len(pages),
        "expected_pages": expected_pages,
        "missing_page_records": max(0, expected_pages - len(unique_pages)),
        "duplicate_page_records": len(pages) - len(unique_pages),
        "passed": len(pages) == expected_pages and len(unique_pages) == len(pages),
    }


def check_identity_integrity(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Check Table/Row/Cell identity uniqueness and foreign keys."""
    table_ids: list[str] = []
    row_ids: list[str] = []
    cell_ids: list[str] = []

    row_to_table: dict[str, str] = {}
    cell_to_row: dict[str, str] = {}

    broken_row_to_table = 0
    broken_cell_to_row = 0

    for page in pages:
        for table in page.get("tables", []):
            t_id = table["table_fragment_id"]
            table_ids.append(t_id)
            for row in table.get("rows", []):
                r_id = row["row_id"]
                row_ids.append(r_id)
                row_to_table[r_id] = t_id
            for cell in table.get("cells", []):
                c_id = cell["cell_id"]
                cell_ids.append(c_id)
                cell_to_row[c_id] = cell.get("row_id", "")

    # Check foreign keys
    valid_table_ids = set(table_ids)
    valid_row_ids = set(row_ids)
    for r_id, t_id in row_to_table.items():
        if t_id not in valid_table_ids:
            broken_row_to_table += 1
    for c_id, r_id in cell_to_row.items():
        if r_id not in valid_row_ids:
            broken_cell_to_row += 1

    return {
        "duplicate_table_id": len(table_ids) - len(set(table_ids)),
        "duplicate_row_id": len(row_ids) - len(set(row_ids)),
        "duplicate_cell_id": len(cell_ids) - len(set(cell_ids)),
        "row_to_table_missing": broken_row_to_table,
        "cell_to_row_missing": broken_cell_to_row,
        "passed": (
            len(table_ids) == len(set(table_ids))
            and len(row_ids) == len(set(row_ids))
            and len(cell_ids) == len(set(cell_ids))
            and broken_row_to_table == 0
            and broken_cell_to_row == 0
        ),
    }


def _is_valid_bbox(bbox: list[float] | None) -> bool:
    """Check if bbox has 4 finite numbers with positive area."""
    if not bbox or len(bbox) != 4:
        return False
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return False
    if any(v != v for v in (x0, y0, x1, y1)):  # NaN check
        return False
    return x1 > x0 and y1 > y0


def check_bbox_integrity(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Check BBox validity for tables, rows, and cells."""
    invalid_table_bbox = 0
    invalid_row_bbox = 0
    invalid_cell_bbox = 0
    out_of_page_bbox = 0

    for page in pages:
        page_w = float(page.get("page_width") or 0)
        page_h = float(page.get("page_height") or 0)
        for table in page.get("tables", []):
            if not _is_valid_bbox(table.get("table_bbox")):
                invalid_table_bbox += 1
            else:
                bbox = table["table_bbox"]
                if page_w > 0 and page_h > 0:
                    if bbox[0] < -5 or bbox[1] < -5 or bbox[2] > page_w + 5 or bbox[3] > page_h + 5:
                        out_of_page_bbox += 1
            for row in table.get("rows", []):
                if not _is_valid_bbox(row.get("row_bbox")):
                    invalid_row_bbox += 1
            for cell in table.get("cells", []):
                if not _is_valid_bbox(cell.get("cell_bbox")):
                    invalid_cell_bbox += 1

    return {
        "invalid_table_bbox": invalid_table_bbox,
        "invalid_row_bbox": invalid_row_bbox,
        "invalid_cell_bbox": invalid_cell_bbox,
        "out_of_page_bbox": out_of_page_bbox,
        "passed": (
            invalid_table_bbox == 0
            and invalid_row_bbox == 0
            and out_of_page_bbox == 0
        ),
    }


def check_text_integrity(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Check text/numeric integrity.

    ``numeric_cell_native_loss`` counts cells that contain numeric content
    (digits in raw_text) but have NO resolved text at all — i.e. both
    native alignment and MinerU fallback failed.  Cells that fell back to
    MinerU table text are NOT counted as native loss; they are reported
    separately as ``native_alignment_failed``.
    """
    import re

    numeric_cells_native_loss = 0
    native_alignment_failed = 0
    invalid_numeric_parse = 0
    unresolved_cells = 0

    for page in pages:
        for table in page.get("tables", []):
            for cell in table.get("cells", []):
                source = cell.get("text_source", "")
                if source == "unresolved":
                    unresolved_cells += 1

                raw_text = str(cell.get("raw_text") or "")
                resolved = str(cell.get("resolved_text") or "")
                if re.search(r"\d", raw_text):
                    if not resolved:
                        numeric_cells_native_loss += 1
                    elif not cell.get("native_text"):
                        native_alignment_failed += 1

                parsed = cell.get("parsed_numeric") or []
                for p in parsed:
                    if not p.get("normalized"):
                        invalid_numeric_parse += 1

    return {
        "numeric_cell_native_loss": numeric_cells_native_loss,
        "native_alignment_failed": native_alignment_failed,
        "invalid_numeric_parse": invalid_numeric_parse,
        "unresolved_cells": unresolved_cells,
    }
