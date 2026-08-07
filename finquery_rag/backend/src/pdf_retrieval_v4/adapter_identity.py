"""Gate 02 R3 stable identity scheme for Table/Row/Cell.

Identity is entirely source-based (document_id + page + geometry + text).
No oracle, evaluation, or review metadata is included.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.pdf_retrieval_v4.table_html_parser import norm_text


def _stable_hash(payload: list[Any]) -> str:
    """Deterministic SHA-256 for a JSON-serialisable list."""
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalize_bbox(bbox: list[float] | None) -> list[float]:
    """Round bbox to 2 decimal places for stability."""
    if not bbox or len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [round(float(v), 2) for v in bbox]


def table_fragment_id(
    document_id: str,
    pdf_page: int,
    table_bbox: list[float],
    table_signature: list[Any],
) -> str:
    """Compute table_fragment_id per R3 spec.

    sha256(document_id + pdf_page + normalized_bbox + normalized_table_signature)
    """
    payload = [
        document_id,
        int(pdf_page),
        normalize_bbox(table_bbox),
        table_signature,
    ]
    return "table:" + _stable_hash(payload)


def row_id(
    table_fragment_id: str,
    row_index: int,
    row_signature: str,
) -> str:
    """Compute row_id per R3 spec.

    sha256(table_fragment_id + row_index + normalized_row_signature)
    """
    payload = [table_fragment_id, int(row_index), norm_text(row_signature)]
    return "row:" + _stable_hash(payload)


def cell_id(
    row_id: str,
    column_index: int,
    cell_signature: str,
) -> str:
    """Compute cell_id per R3 spec.

    sha256(row_id + column_index + normalized_cell_signature)
    """
    payload = [row_id, int(column_index), norm_text(cell_signature)]
    return "cell:" + _stable_hash(payload)


def build_table_signature(cells: list[dict[str, Any]]) -> list[Any]:
    """Build normalized table signature from adapter cells."""
    return [
        [c["raw_text"], c["row_index"], c["column_index"]]
        for c in sorted(cells, key=lambda c: (c["row_index"], c["column_index"]))
    ]


def build_row_signature(cells: list[dict[str, Any]]) -> str:
    """Build normalized row text from cells in column order."""
    return " | ".join(
        c["resolved_text"]
        for c in sorted(cells, key=lambda c: c["column_index"])
    )


def build_cell_signature(raw_text: str, rowspan: int, colspan: int) -> str:
    """Build normalized cell signature."""
    return f"{raw_text}|{rowspan}|{colspan}"
