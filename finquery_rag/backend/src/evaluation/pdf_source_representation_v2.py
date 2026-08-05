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
LINEAGE_PATTERNS = (
    ("primary_financial_statement", "income_statement", re.compile(r"(?i)(?:statements?|schedules?)\s+(?:of\s+)?(?:operations|income|earnings|comprehensive income)")),
    ("primary_financial_statement", "balance_sheet", re.compile(r"(?i)(?:consolidated\s+)?(?:balance sheets?|statements? of financial position)")),
    ("primary_financial_statement", "cash_flow_statement", re.compile(r"(?i)statements? of cash flows?")),
    ("primary_financial_statement", "equity_statement", re.compile(r"(?i)statements? of (?:stockholders|shareholders|changes in equity)")),
    ("segment_section", None, re.compile(r"(?i)segment information")),
    ("financial_schedule", None, re.compile(r"(?i)(?:debt maturities|schedule of revenues?)")),
    ("note_section", None, re.compile(r"(?i)^\s*note\s+\d+\b")),
    ("other_financial_section", None, re.compile(r"(?i)consolidated financial statements")),
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


def resolve_period_headers_v2(matrix: list[list[str]], width: int) -> list[dict[str, Any] | None]:
    """Resolve multi-row matrix headers without using query or Gold fields."""
    paths: list[list[str]] = [[] for _ in range(width)]
    years_by_column: list[set[str]] = [set() for _ in range(width)]
    global_header_parts: list[str] = []
    for row in matrix[:12]:
        nonempty = [(index, normalize_text(cell)) for index, cell in enumerate(row[:width]) if normalize_text(cell)]
        if not nonempty:
            continue
        row_text = normalize_text(" ".join(value for _, value in nonempty))
        if re.search(r"(?i)(years?|months?)\s+ended|as\s+of|fiscal", row_text):
            global_header_parts.append(row_text)
        for index, value in nonempty:
            matches = YEAR_RE.findall(value)
            if matches:
                if re.fullmatch(r"(?i)(?:FY|Fiscal\s*)?(?:19|20)\d{2}", value):
                    years_by_column[index].update(matches)
                    paths[index].append(value)
                else:
                    global_header_parts.append(value)
            elif not any(parse_number(token) is not None for token in value.split()):
                paths[index].append(value)
    explicit_years = {year for values in years_by_column for year in values}
    explicit_years.update(YEAR_RE.findall(" ".join(global_header_parts)))
    resolved: list[dict[str, Any] | None] = [None] * width
    for index, years in enumerate(years_by_column):
        if len(years) != 1:
            continue
        year = next(iter(years))
        parent = global_header_parts[-1] if global_header_parts else None
        header_path = tuple(dict.fromkeys([part for part in (parent, *paths[index]) if part]))
        joined = " ".join(header_path).casefold()
        kind = "duration" if "ended" in joined else "instant" if "as of" in joined else "fiscal_year"
        resolved[index] = {
            "header_path": header_path,
            "normalized_period": f"FY{year}",
            "period_kind": kind,
            "resolution_method": "matrix_multilevel",
        }
    # A single global period is safe only for a table with one numeric value column.
    if not any(resolved) and len(explicit_years) == 1:
        numeric_columns = {
            index
            for row in matrix
            for index, cell in enumerate(row[:width])
            if parse_number(cell) is not None and not YEAR_RE.fullmatch(normalize_text(cell))
        }
        if len(numeric_columns) == 1:
            index = next(iter(numeric_columns))
            year = next(iter(explicit_years))
            resolved[index] = {
                "header_path": tuple(global_header_parts or [f"FY{year}"]),
                "normalized_period": f"FY{year}",
                "period_kind": "duration" if any("ended" in part.casefold() for part in global_header_parts) else "fiscal_year",
                "resolution_method": "single_period_single_numeric_column",
            }
    # Explicit currency spacer columns inherit only from an adjacent resolved value column.
    for index in range(width - 1):
        if resolved[index] is None and resolved[index + 1] is not None:
            values = [normalize_text(row[index]) for row in matrix[:12] if index < len(row)]
            if any(value == "$" for value in values):
                resolved[index] = {**resolved[index + 1], "resolution_method": "currency_spacer_propagation"}
    return resolved


def resolve_lineage(lines: list[str]) -> dict[str, str | None] | None:
    """Classify the nearest deterministic statement or section heading."""
    for line in reversed(lines):
        normalized = normalize_text(line)
        for lineage_type, statement_type, pattern in LINEAGE_PATTERNS:
            if pattern.search(normalized):
                return {
                    "lineage_title": normalized[:240],
                    "lineage_type": lineage_type,
                    "statement_type": statement_type,
                }
    return None


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
