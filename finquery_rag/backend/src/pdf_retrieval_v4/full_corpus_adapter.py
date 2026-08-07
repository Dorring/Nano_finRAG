"""Gate 02 R3 full-corpus unified structured adapter.

Converts MinerU Hybrid High output for all 8 frozen benchmark PDFs into a
unified Document → Page → Table Fragment → Row → Cell structure with
PyMuPDF native word alignment.

This module ONLY builds structure.  It does NOT:
  - Build header graphs or metric hierarchies
  - Build evidence units or candidate views
  - Build indexes (BM25 / Dense)
  - Run retrieval, RRF, reranker, or answer generation
  - Read questions, gold, governance, or expected values
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.adapter_identity import (  # noqa: E402
    build_cell_signature,
    build_row_signature,
    build_table_signature,
    cell_id,
    normalize_bbox,
    row_id,
    table_fragment_id,
)
from src.pdf_retrieval_v4.native_alignment import (  # noqa: E402
    column_bands,
    extract_middle_tables,
    inside,
    match_native_words,
    page_words,
    resolve_pdf,
    union_bbox,
)
from src.pdf_retrieval_v4.table_html_parser import (  # noqa: E402
    extract_numeric_values,
    norm_text,
    period_from_text,
    period_kind,
    tokenize_text,
)

SCALE_PATTERN = (
    r"(?:in|dollars in|amounts in)\s+(?:millions?|thousands?|billions?)"
    r"|\b(?:millions?|thousands?|billions?)\b"
)


def _find_middle_json(doc_output_dir: Path) -> Path | None:
    """Find the middle.json for a document in its R2 output directory."""
    matches = sorted(doc_output_dir.rglob("*_middle.json"))
    return matches[0] if matches else None


def _find_content_list_json(doc_output_dir: Path) -> Path | None:
    """Find content_list.json for a document."""
    matches = sorted(doc_output_dir.rglob("*_content_list.json"))
    return matches[0] if matches else None


def _count_text_blocks(content_path: Path, page_idx: int) -> int:
    """Count non-empty text blocks for a page from content_list.json."""
    if not content_path or not content_path.is_file():
        return 0
    try:
        data = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(data, list):
        return 0
    count = 0
    for block in data:
        if not isinstance(block, dict):
            continue
        if block.get("page_idx") != page_idx:
            continue
        block_type = str(block.get("type") or "")
        if block_type in ("text", "title", "discarded") and block.get("text"):
            count += 1
    return count


def _page_scale_candidates(content_path: Path, page_idx: int) -> list[str]:
    """Extract scale keywords from content_list text blocks on a page."""
    if not content_path or not content_path.is_file():
        return []
    try:
        data = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    text_parts: list[str] = []
    for block in data:
        if not isinstance(block, dict):
            continue
        if block.get("page_idx") != page_idx:
            continue
        block_type = str(block.get("type") or "")
        if block_type in ("text", "title", "discarded") and block.get("text"):
            text_parts.append(str(block["text"]))
    combined = " ".join(text_parts)
    return sorted(set(re.findall(SCALE_PATTERN, combined, re.I)))


def _page_dimensions(pdf_path: Path, page_number: int) -> tuple[float, float]:
    """Get page width and height from PDF."""
    try:
        import fitz

        document = fitz.open(pdf_path)
        page = document[page_number - 1]
        rect = page.rect
        return float(rect.width), float(rect.height)
    except Exception:
        return 0.0, 0.0


def build_table_fragment(
    table: dict[str, Any],
    document_id: str,
    pdf_page: int,
    table_index: int,
    words: list[dict[str, Any]],
    page_scale_candidates: list[str] | None = None,
) -> dict[str, Any]:
    """Build a single table fragment with rows and cells.

    Reuses the R1 alignment logic but uses the R3 identity scheme where
    cell_id depends on row_id (not directly on table_id).
    """
    parsed = table["parsed"]
    bbox = table["bbox"] or [0.0, 0.0, 0.0, 0.0]
    grid = parsed["grid"]
    header_texts = [cell["raw_text"] for cell in parsed["cells"] if cell.get("header")]

    # Determine header rows (first 8 rows with period text or keywords)
    header_rows: list[int] = []
    for row_index, row in enumerate(grid[: min(8, len(grid))]):
        text = " ".join(cell["raw_text"] for cell in row if cell)
        if period_from_text(text) or re.search(
            r"\b(?:year|quarter|month|as of|ended)\b", text, re.I
        ):
            header_rows.append(row_index)
    if not header_rows and grid:
        header_rows = [0]

    header_by_col: defaultdict[int, list[str]] = defaultdict(list)
    period_by_col: dict[int, str] = {}
    for row_index in header_rows:
        for col_index, cell in enumerate(grid[row_index]):
            if not cell:
                continue
            text = str(cell["raw_text"])
            if text and text not in header_by_col[col_index]:
                header_by_col[col_index].append(text)
            period = period_from_text(text)
            if period:
                for col in range(cell["grid_col"], cell["grid_col"] + cell["colspan"]):
                    period_by_col[col] = period

    table_words = [word for word in words if inside(word, bbox, 0.5)]
    local_words = [word for word in words if inside(word, bbox, 80.0)]
    used: set[int] = set()
    bands = column_bands(bbox, parsed["column_count"])

    # First pass: build cells without IDs (need row_id which needs table_id)
    adapter_cells: list[dict[str, Any]] = []
    for cell in parsed["cells"]:
        row_index = int(cell["grid_row"])
        col_index = int(cell["grid_col"])
        native_words = match_native_words(
            cell["raw_text"], table_words, row_index, parsed["row_count"], bbox, used
        )
        cell_bbox = union_bbox(native_words)
        if cell_bbox is None:
            span = bands[col_index : col_index + int(cell["colspan"])]
            if span:
                row_h = (bbox[3] - bbox[1]) / max(parsed["row_count"], 1)
                cell_bbox = [
                    span[0][0],
                    bbox[1] + row_index * row_h,
                    span[-1][2],
                    bbox[1] + (row_index + int(cell["rowspan"])) * row_h,
                ]
            else:
                cell_bbox = None

        header_path: list[str] = []
        for header_row in header_rows:
            if header_row >= len(grid):
                continue
            header_cell = (
                grid[header_row][min(col_index, len(grid[header_row]) - 1)]
                if grid[header_row]
                else None
            )
            if (
                header_cell
                and header_cell["raw_text"]
                and header_cell["raw_text"] not in header_path
            ):
                header_path.append(header_cell["raw_text"])

        period = period_by_col.get(col_index)
        if period is None:
            for col in range(col_index, col_index + int(cell["colspan"])):
                period = period_by_col.get(col)
                if period:
                    break

        native_parts: list[str] = []
        for word in sorted(
            native_words, key=lambda item: (item["bbox"][1], item["bbox"][0])
        ):
            if word.get("_numeric_continuation") and native_parts:
                native_parts[-1] += word["text"]
            else:
                native_parts.append(word["text"])
        native_text = " ".join(native_parts)
        resolved_text = native_text or cell["raw_text"]

        adapter_cells.append(
            {
                "row_index": row_index,
                "column_index": col_index,
                "rowspan": cell["rowspan"],
                "colspan": cell["colspan"],
                "raw_text": cell["raw_text"],
                "normalized_text": norm_text(cell["raw_text"]),
                "header_path": header_path,
                "normalized_period": period,
                "period_kind": period_kind(" ".join(header_path)),
                "cell_bbox": cell_bbox,
                "mineru_text": cell["raw_text"],
                "native_words": native_words,
                "native_text": native_text,
                "resolved_text": resolved_text,
                "text_source": "pymupdf_native"
                if native_words
                else "mineru_table_text",
                "alignment_confidence": round(
                    min(
                        1.0,
                        len(native_words)
                        / max(1, len(tokenize_text(cell["raw_text"]))),
                    ),
                    4,
                ),
                "parsed_numeric": extract_numeric_values(
                    native_text or cell["raw_text"]
                ),
                "scale_candidates": sorted(
                    set(re.findall(SCALE_PATTERN, " ".join(header_path), re.I))
                ),
            }
        )

    # Compute table_fragment_id
    t_signature = build_table_signature(adapter_cells)
    t_id = table_fragment_id(document_id, pdf_page, bbox, t_signature)

    # Build rows and assign row_ids
    row_height = (bbox[3] - bbox[1]) / max(parsed["row_count"], 1)
    rows: list[dict[str, Any]] = []
    for row_index in range(parsed["row_count"]):
        row_cells = [c for c in adapter_cells if c["row_index"] == row_index]
        row_bbox = union_bbox(
            [{"bbox": c["cell_bbox"]} for c in row_cells if c.get("cell_bbox")]
        )
        if row_bbox is None:
            # Fallback: estimate from table bbox and row index
            row_bbox = [
                bbox[0],
                bbox[1] + row_index * row_height,
                bbox[2],
                bbox[1] + (row_index + 1) * row_height,
            ]
        row_sig = build_row_signature(row_cells)
        r_id = row_id(t_id, row_index, row_sig)

        # Now compute cell_ids (depend on row_id per R3 spec)
        for cell in row_cells:
            c_sig = build_cell_signature(
                cell["raw_text"], cell["rowspan"], cell["colspan"]
            )
            cell["cell_id"] = cell_id(r_id, cell["column_index"], c_sig)
            cell["table_fragment_id"] = t_id
            cell["row_id"] = r_id

        rows.append(
            {
                "row_id": r_id,
                "row_index": row_index,
                "row_bbox": row_bbox,
                "raw_text": row_sig,
                "resolved_text": row_sig,
                "cell_ids": [
                    c["cell_id"]
                    for c in sorted(row_cells, key=lambda item: item["column_index"])
                ],
                "metric_text": row_cells[0]["resolved_text"] if row_cells else "",
            }
        )

    table_text = " ".join(str(c.get("raw_text") or "") for c in parsed["cells"])
    native_local_text = " ".join(w["text"] for w in local_words)
    table_scale_candidates = sorted(
        set(
            re.findall(SCALE_PATTERN, table_text + " " + native_local_text, re.I)
            + (page_scale_candidates or [])
        )
    )

    return {
        "table_fragment_id": t_id,
        "document_id": document_id,
        "pdf_page": pdf_page,
        "table_index": table_index,
        "table_bbox": bbox,
        "normalized_bbox": normalize_bbox(bbox),
        "row_count": parsed["row_count"],
        "column_count": parsed["column_count"],
        "parser_backend": "mineru_hybrid_high",
        "mineru_table_html": table.get("html", ""),
        "rows": rows,
        "cells": adapter_cells,
        "header_texts": header_texts,
        "periods": sorted(
            {
                c["normalized_period"]
                for c in adapter_cells
                if c.get("normalized_period")
            }
        ),
        "scale_candidates": table_scale_candidates,
        "source_lineage": {
            "document_id": document_id,
            "pdf_page": pdf_page,
            "table_bbox": bbox,
        },
    }


def build_page_record(
    document_id: str,
    pdf_page: int,
    page_index: int,
    pdf_path: Path | None,
    page_tables: list[dict[str, Any]],
    content_path: Path | None,
) -> dict[str, Any]:
    """Build a page-level record."""
    page_width, page_height = (0.0, 0.0)
    native_text_present = False
    if pdf_path:
        page_width, page_height = _page_dimensions(pdf_path, pdf_page)
        try:
            words = page_words(pdf_path, pdf_page)
            native_text_present = len(words) > 0
        except Exception:
            pass

    text_block_count = (
        _count_text_blocks(content_path, page_index) if content_path else 0
    )

    return {
        "document_id": document_id,
        "pdf_page": pdf_page,
        "page_index": page_index,
        "page_width": page_width,
        "page_height": page_height,
        "mineru_page_present": True,
        "native_text_present": native_text_present,
        "text_block_count": text_block_count,
        "table_fragment_ids": [t["table_fragment_id"] for t in page_tables],
        "tables": page_tables,
    }


def run_full_corpus_adapter(
    documents: list[dict[str, Any]],
    mineru_output_root: Path,
    pdf_dir: Path,
    shared_root: Path,
) -> list[dict[str, Any]]:
    """Run the adapter on all documents.

    Args:
        documents: List of document records from frozen corpus manifest,
                   sorted by document_id.
        mineru_output_root: Root directory with per-document MinerU output.
        pdf_dir: Directory containing the frozen PDF files.
        shared_root: Shared nanochat root for PDF path resolution.

    Returns:
        List of page records (one per page per document).
    """
    all_pages: list[dict[str, Any]] = []

    for doc in documents:
        doc_id = str(doc["document_id"])
        doc_output = mineru_output_root / doc_id
        if not doc_output.is_dir():
            continue

        middle_path = _find_middle_json(doc_output)
        content_path = _find_content_list_json(doc_output)
        if middle_path is None:
            continue

        pages_tables = extract_middle_tables(middle_path)
        pdf_path = resolve_pdf(
            str(doc.get("source_path") or doc.get("pdf_filename") or f"{doc_id}.pdf"),
            pdf_dir,
            shared_root,
        )

        # Determine total page count for this document
        total_pages = int(doc.get("page_count") or 0)

        # Build page records for all pages
        page_indices = set(pages_tables.keys())
        # Also include pages that have no tables but exist in the document
        if total_pages > 0:
            page_indices.update(range(total_pages))

        # Track scale candidates from the previous page within this document.
        # Used as nearby_page_scale_candidate when the current page has none.
        last_doc_scale_candidates: list[str] = []

        for page_index in sorted(page_indices):
            pdf_page = page_index + 1
            tables_data = pages_tables.get(page_index, [])

            # Get PyMuPDF words for this page
            words: list[dict[str, Any]] = []
            if pdf_path:
                try:
                    words = page_words(pdf_path, pdf_page)
                except Exception:
                    pass

            # Build table fragments
            page_scale_cands = (
                _page_scale_candidates(content_path, page_index) if content_path else []
            )
            page_tables: list[dict[str, Any]] = []
            for table_index, table in enumerate(tables_data):
                if not table.get("bbox"):
                    continue
                page_tables.append(
                    build_table_fragment(
                        table,
                        doc_id,
                        pdf_page,
                        table_index,
                        words,
                        page_scale_candidates=page_scale_cands,
                    )
                )

            # Propagate scale candidates across all tables on the same page.
            # Scale keywords like "In millions" often appear in only one table
            # (e.g. a note table) but apply to all tables on the page.
            if page_tables:
                all_scale = sorted(
                    set(sc for t in page_tables for sc in t.get("scale_candidates", []))
                )
                if all_scale:
                    for t in page_tables:
                        t["scale_candidates"] = all_scale
                    # Update last seen scale candidates for nearby-page fallback.
                    # Only update when we actually found scale candidates, so
                    # pages with no scale don't overwrite the running value.
                    last_doc_scale_candidates = all_scale

            # If tables on this page still have no scale candidates, fall back
            # to the immediately preceding page's scale candidates as
            # nearby_page_scale_candidate (per spec section 7).
            if (
                page_tables
                and not page_tables[0].get("scale_candidates")
                and last_doc_scale_candidates
            ):
                for t in page_tables:
                    t["scale_candidates"] = sorted(last_doc_scale_candidates)

            page_record = build_page_record(
                doc_id,
                pdf_page,
                page_index,
                pdf_path,
                page_tables,
                content_path,
            )
            all_pages.append(page_record)

    return all_pages


def collect_structure_metrics(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect aggregate structure metrics from all pages."""
    table_count = 0
    row_count = 0
    cell_count = 0
    native_aligned = 0
    mineru_fallback = 0
    ocr_fallback = 0
    unresolved = 0
    numeric_cells = 0
    numeric_parsed = 0
    scale_candidate_count = 0

    seen_tables: set[str] = set()
    seen_rows: set[str] = set()
    seen_cells: set[str] = set()

    pages_with_native = 0
    pages_with_tables = 0

    for page in pages:
        if page.get("native_text_present"):
            pages_with_native += 1
        if page.get("table_fragment_ids"):
            pages_with_tables += 1
        for table in page.get("tables", []):
            table_count += 1
            seen_tables.add(table["table_fragment_id"])
            row_count += len(table["rows"])
            seen_rows.update(r["row_id"] for r in table["rows"])
            cell_count += len(table["cells"])
            seen_cells.update(c["cell_id"] for c in table["cells"])
            scale_candidate_count += len(table.get("scale_candidates", []))
            for cell in table["cells"]:
                source = cell.get("text_source", "")
                if source == "pymupdf_native":
                    native_aligned += 1
                elif source == "mineru_table_text":
                    mineru_fallback += 1
                elif source == "mineru_ocr":
                    ocr_fallback += 1
                else:
                    unresolved += 1
                if cell.get("parsed_numeric"):
                    numeric_cells += 1
                    numeric_parsed += len(cell["parsed_numeric"])

    return {
        "page_count": len(pages),
        "pages_with_native_text": pages_with_native,
        "pages_with_tables": pages_with_tables,
        "table_count": table_count,
        "row_count": row_count,
        "cell_count": cell_count,
        "native_aligned_cell_count": native_aligned,
        "mineru_text_fallback_count": mineru_fallback,
        "ocr_fallback_count": ocr_fallback,
        "unresolved_cell_count": unresolved,
        "numeric_cell_count": numeric_cells,
        "numeric_parse_success_count": numeric_parsed,
        "scale_candidate_count": scale_candidate_count,
        "duplicate_table_id_count": table_count - len(seen_tables),
        "duplicate_row_id_count": row_count - len(seen_rows),
        "duplicate_cell_id_count": cell_count - len(seen_cells),
        "table_identity_hash": _hash_sorted(seen_tables),
        "row_identity_hash": _hash_sorted(seen_rows),
        "cell_identity_hash": _hash_sorted(seen_cells),
    }


def collect_document_metrics(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect per-document structure metrics."""
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        by_doc[page["document_id"]].append(page)

    results: list[dict[str, Any]] = []
    for doc_id in sorted(by_doc.keys()):
        doc_pages = by_doc[doc_id]
        metrics = collect_structure_metrics(doc_pages)
        metrics["document_id"] = doc_id
        results.append(metrics)
    return results


def _hash_sorted(items: set[str]) -> str:
    """Hash a sorted set of identity strings."""
    import hashlib

    return hashlib.sha256(
        json.dumps(sorted(items), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
