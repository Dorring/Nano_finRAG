"""Deterministic financial-table structure helpers for PDF SR-V2."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from hashlib import sha256
import re
from typing import Any


YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
SCALE_RE = re.compile(r"(?i)(?:\$|dollars?|amounts?)?\s*in\s+(thousands?|millions?|billions?)")
STATEMENT_MARKERS = (
    "statements of operations",
    "statements of income",
    "statements of earnings",
    "balance sheets",
    "statements of cash flows",
    "statements of stockholders",
    "statements of shareholders",
)


def normalize_text(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split())


def parse_number(value: str) -> Decimal | None:
    text = normalize_text(value).replace(",", "").replace("$", "").strip()
    if text in {"", "-", "—", "N/A"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("() ")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?%?", text):
        return None
    percentage = text.endswith("%")
    try:
        parsed = Decimal(text.removesuffix("%"))
    except InvalidOperation:
        return None
    if negative:
        parsed = -abs(parsed)
    return parsed if not percentage else parsed


def resolve_period_headers(matrix: list[list[str]], width: int) -> list[str | None]:
    """Resolve a year per column only when its header path is unambiguous."""
    resolved: list[str | None] = [None] * width
    for row in matrix[:8]:
        years = [(index, YEAR_RE.findall(cell)) for index, cell in enumerate(row[:width])]
        explicit = [(index, matches[-1]) for index, matches in years if matches]
        if len(explicit) < 2:
            continue
        for index, year in explicit:
            resolved[index] = f"FY{year}"
        # Currency spacer columns immediately before an explicit year share it.
        for index, year in explicit:
            if index and (
                normalize_text(row[index - 1]) == "$"
                or (index > 1 and normalize_text(row[index - 1]) == "")
            ):
                resolved[index - 1] = f"FY{year}"
        break
    return resolved


def row_label(row: list[str]) -> str | None:
    labels = []
    for cell in row:
        text = normalize_text(cell)
        if parse_number(text) is not None or YEAR_RE.fullmatch(text):
            break
        if text not in {"", "$"}:
            labels.append(text)
    label = normalize_text(" ".join(labels))
    return label or None


def extract_scale(text: str) -> tuple[str | None, str | None]:
    match = SCALE_RE.search(text)
    if not match:
        return None, None
    raw = normalize_text(match.group(0))
    unit = match.group(1).casefold().rstrip("s")
    return raw, unit


def statement_from_lines(lines: list[str]) -> str | None:
    for line in reversed(lines):
        normalized = normalize_text(line)
        if any(marker in normalized.casefold() for marker in STATEMENT_MARKERS):
            return normalized[:240]
    return None


def stable_identity(namespace: str, *parts: Any) -> str:
    payload = "|".join([namespace, *[str(part) for part in parts]])
    return f"{namespace}:{sha256(payload.encode('utf-8')).hexdigest()}"
