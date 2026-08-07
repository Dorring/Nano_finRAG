from __future__ import annotations

import html
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluation.run_pdf_v4_gate_01_r1 import normalize_financial_numeric_text


class _TableHTMLParser(HTMLParser):
    """Parse MinerU HTML table output into source rows of cell dicts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, Any]]] = []
        self._row: list[dict[str, Any]] | None = None
        self._cell: dict[str, Any] | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"td", "th"} and self._row is not None:
            attrs_dict = dict(attrs)
            self._cell = {
                "raw_text": "",
                "rowspan": max(1, int(attrs_dict.get("rowspan") or 1)),
                "colspan": max(1, int(attrs_dict.get("colspan") or 1)),
                "header": tag.lower() == "th",
            }
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._cell["raw_text"] = re.sub(r"\s+", " ", " ".join(self._buffer)).strip()
            self._row.append(self._cell)
            self._cell = None
            self._buffer = []
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def parse_table_html(table_html: str) -> dict[str, Any]:
    """Parse HTML table markup into a grid expanding rowspan/colspan.

    Returns a dict with:
      - ``grid``: 2D list (rows x columns) of cell records or ``None``.
      - ``cells``: unique cell records (one per source cell).
      - ``row_count``: number of grid rows.
      - ``column_count``: grid width.

    Each cell record has: ``source_row``, ``source_col``, ``grid_row``,
    ``grid_col``, ``rowspan``, ``colspan``, ``raw_text``, ``header``.
    """
    parser = _TableHTMLParser()
    parser.feed(table_html)
    source_rows = parser.rows
    grid: list[list[dict[str, Any] | None]] = []
    cell_records: list[dict[str, Any]] = []
    for row_index, source_row in enumerate(source_rows):
        while len(grid) <= row_index:
            grid.append([])
        col = 0
        for source_cell in source_row:
            while col < len(grid[row_index]) and grid[row_index][col] is not None:
                col += 1
            record = {
                "source_row": row_index,
                "source_col": col,
                "rowspan": int(source_cell["rowspan"]),
                "colspan": int(source_cell["colspan"]),
                "raw_text": source_cell["raw_text"],
                "header": bool(source_cell["header"]),
            }
            cell_records.append(record)
            for rr in range(row_index, row_index + record["rowspan"]):
                while len(grid) <= rr:
                    grid.append([])
                while len(grid[rr]) < col + record["colspan"]:
                    grid[rr].append(None)
                for cc in range(col, col + record["colspan"]):
                    if grid[rr][cc] is None:
                        grid[rr][cc] = record
            col += record["colspan"]
    width = max((len(row) for row in grid), default=0)
    for row in grid:
        row.extend([None] * (width - len(row)))
    unique_records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row_index, row in enumerate(grid):
        for col_index, record in enumerate(row):
            if record is None:
                continue
            ident = id(record)
            if ident not in seen:
                record["grid_row"] = row_index
                record["grid_col"] = col_index
                unique_records.append(record)
                seen.add(ident)
    return {
        "grid": grid,
        "cells": unique_records,
        "row_count": len(grid),
        "column_count": width,
    }


def norm_text(value: Any) -> str:
    """HTML-unescape, strip tags, normalize whitespace, and lowercase."""
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def compact_text(value: Any) -> str:
    """``norm_text`` then remove all non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]+", "", norm_text(value))


def tokenize_text(value: Any) -> list[str]:
    """Extract word and number tokens from ``value``."""
    value = html.unescape(str(value or "")).lower().replace("−", "-")
    return re.findall(r"[a-z]+|\(?[-+]?\d[\d,]*(?:\.\d+)?\)?%?", value)


def period_from_text(text: str) -> str | None:
    """Extract an ``FY{year}`` period token from ``text`` if present."""
    match = re.search(r"\b(?:fy|fiscal\s+year\s*)?(19|20)\d{2}\b", text, re.I)
    if match:
        year = re.search(r"(?:19|20)\d{2}", match.group(0))
        return f"FY{year.group(0)}" if year else None
    match = re.search(
        r"\b(?:jan...|dec...)\.?\s+\d{1,2},?\s+(19|20)\d{2}\b", text, re.I
    )
    if not match:
        return None
    year_match = re.search(r"(?:19|20)\d{2}", match.group(0))
    return f"FY{year_match.group(0)}" if year_match else None


def period_kind(text: str) -> str | None:
    """Classify a period phrase as ``"instant"`` or ``"duration"``."""
    lowered = text.lower()
    if "as of" in lowered:
        return "instant"
    if "ended" in lowered or "year" in lowered:
        return "duration"
    return None


def _fix_mineru_currency_position(text: str) -> str:
    """Fix MinerU HTML artifacts where currency symbols are mid-number.

    MinerU sometimes outputs ``"281,72$ 4"`` instead of ``"$ 281,724"``.
    This function moves the currency symbol to the front and joins the
    split digit groups.  Only applies when a currency symbol is directly
    between digits (with optional comma/period/space), not at word
    boundaries.
    """
    # Pattern: digits+commas, then currency symbol, optional space, then digits
    # e.g. "281,72$ 4" -> "$ 281,724"
    fixed = re.sub(
        r"(\d[\d,]*?)([$€£¥₹₽₩])\s*(\d)",
        r"\2 \1\3",
        html.unescape(text or ""),
    )
    return fixed


def extract_numeric_values(text: str) -> list[dict[str, Any]]:
    """Extract numeric values from ``text`` using a financial-aware regex.

    Each returned dict has ``raw``, ``normalized``, and ``percent`` keys.
    """
    values: list[dict[str, Any]] = []
    cleaned = _fix_mineru_currency_position(text)
    grouped_number = r"(?:\d{1,3}(?:\s\d{3})+(?:\.\d+)?|\d[\d,]*(?:\.\d+)?)"
    pattern = rf"(?<![A-Za-z0-9])(?:\(\s*(?:[$€£¥₹₽₩]?\s*)?[-+−]?\s*{grouped_number}\s*\)|(?:[$€£¥₹₽₩]?\s*)?[-+−]?\s*{grouped_number}%?)"
    for match in re.finditer(pattern, cleaned):
        token = normalize_financial_numeric_text(match.group(0))
        if token["valid"]:
            values.append(
                {
                    "raw": token["raw"],
                    "normalized": token["normalized"],
                    "percent": token["percent"],
                }
            )
    return values
