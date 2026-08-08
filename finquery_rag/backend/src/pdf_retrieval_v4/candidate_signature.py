"""Gate 05 R5 — Candidate Signature builder.

Extracts normalized features from a Production Candidate record (from
``view-pairs.jsonl``) into a ``CandidateSignature`` for bridge matching.

No Question / Gold / Expected-Value data is read.
"""

from __future__ import annotations

import re
from typing import Any

from src.pdf_retrieval_v4.candidate_bridge_models import (
    CandidateSignature,
)

# ---------------------------------------------------------------------------
# Numeric normalization (match-only — no approximate correction)
# ---------------------------------------------------------------------------

# Matches currency-prefixed numbers: $1,234 / $ 1,234 / €1.234,56
_CURRENCY_NUM_RE = re.compile(
    r"(?:[$€£¥]\s*)?"  # optional currency symbol + space
    r"\(?"  # optional opening paren (negative)
    r"[-−]?"  # optional minus (ASCII or Unicode)
    r"[\d,]+(?:\.\d+)?"  # digits with optional commas and decimal
    r"\)?"  # optional closing paren
)

# Matches percentages: 12.5%
_PERCENT_RE = re.compile(r"[-−]?[\d,]+(?:\.\d+)?%")

# Matches FY-like period tokens: FY2025, FY 2025, Q1 2025, 2025, 2024-2025
_PERIOD_RE = re.compile(
    r"(?:FY\s*\d{4}|Q[1-4]\s*\d{4}|\b\d{4}(?:\s*[-–]\s*\d{4})?\b)",
    re.IGNORECASE,
)

# Unicode minus normalization
_UNICODE_MINUS = "\u2212\u2010\u2011\u2012\u2013\u2014"


def _normalize_unicode_minus(text: str) -> str:
    """Replace Unicode minus/dash variants with ASCII hyphen."""
    result = text
    for ch in _UNICODE_MINUS:
        result = result.replace(ch, "-")
    return result


def _normalize_number(raw: str) -> str:
    """Normalize a numeric token: remove currency, commas; handle parens.

    $1,234    → 1234
    (1,234)   → -1234
    12.5%     → 12.5%
    Unicode minus → ASCII minus
    """
    s = _normalize_unicode_minus(raw).strip()
    # Remove currency symbols and percentage signs
    s = re.sub(r"[$€£¥\s%]", "", s)
    # Handle parenthesized negatives
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1]
    # Remove commas
    s = s.replace(",", "")
    # Strip any remaining non-numeric prefix
    s = re.sub(r"^[^\d.-]+", "", s)
    if is_negative and not s.startswith("-"):
        s = "-" + s
    return s


def extract_numeric_multiset(text: str) -> tuple[str, ...]:
    """Extract and normalize all numeric tokens from text.

    Returns a sorted tuple of normalized number strings.

    Handles:
    - Currency: $1,234 -> 1234
    - Parenthesized: (1,234) -> -1234
    - Percentages: 12.5% -> 12.5%
    - Scale suffixes: $14.8T -> 14.8, 212.6B -> 212.6
    - Footnote refs: (1) -> filtered out
    """
    numbers: list[str] = []

    # First extract percentages (before general numbers to preserve %)
    for m in _PERCENT_RE.finditer(text):
        numbers.append(_normalize_number(m.group()))

    # Pattern for numbers with scale suffixes: $14.8T, 212.6B, 5.2M, 3K
    _SCALE_NUM_RE = re.compile(
        r"(?:[$€£¥]\s*)?"
        r"\(?"
        r"[-−]?"
        r"[\d,]+(?:\.\d+)?"
        r"\)"
        r"\s*([TBMK])\b",
        re.IGNORECASE,
    )

    # Extract scale-suffixed numbers first (and remove them from text)
    scale_positions = set()
    for m in _SCALE_NUM_RE.finditer(text):
        scale_positions.add(m.start())
        clean = _normalize_number(m.group())
        # Strip the scale suffix letter
        clean = re.sub(r"[TBMKtbmk]$", "", clean).strip()
        if clean:
            numbers.append(clean)

    # Then extract currency/regular numbers (skip scale-suffixed ones)
    for m in _CURRENCY_NUM_RE.finditer(text):
        # Skip if this match overlaps with a scale-suffixed number
        if any(abs(m.start() - sp) < 10 for sp in scale_positions):
            continue
        token = m.group()
        # Skip if it's a year (4-digit, 1900-2099)
        clean = _normalize_number(token)
        if re.fullmatch(r"\d{4}", clean) and 1900 <= int(clean) <= 2099:
            continue
        # Skip single-digit numbers in parentheses (likely footnote refs)
        if re.fullmatch(r"-?[1-9]", clean):
            continue
        # Skip if it's "0" (often a placeholder)
        if clean == "0":
            continue
        numbers.append(clean)

    # Deduplicate and sort
    return tuple(sorted(set(numbers)))


def extract_period_tokens(text: str) -> tuple[str, ...]:
    """Extract period tokens (FY2025, Q1 2025, 2025, etc.) from text."""
    periods: list[str] = []
    for m in _PERIOD_RE.finditer(text):
        # Normalize: FY 2025 → FY2025, Q1 2025 → Q1 2025
        token = re.sub(r"\s+", "", m.group().upper().replace("FY", "FY"))
        # Standardize: remove internal spaces
        token = m.group().upper().replace(" ", "")
        periods.append(token)
    return tuple(sorted(set(periods)))


def extract_text_tokens(text: str) -> tuple[str, ...]:
    """Extract significant text tokens (words) for text matching.

    Filters out pure numbers and very short tokens.
    """
    # Remove the retrieval_text header lines (Document:, Page:, Block Type:, Source:)
    # and extract the "Source:" portion
    source_match = re.search(r"Source:\s*\n(.*)", text, re.DOTALL)
    if source_match:
        text = source_match.group(1)

    # Tokenize: split on non-alphanumeric (keep underscores)
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text)

    # Filter: lowercase, deduplicate, remove very common stopwords
    _STOP = frozenset(
        {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "any",
            "can",
            "her",
            "was",
            "one",
            "our",
            "out",
            "has",
            "have",
            "had",
            "his",
            "how",
            "its",
            "may",
            "new",
            "now",
            "old",
            "see",
            "way",
            "who",
            "did",
            "got",
            "let",
            "say",
            "she",
            "too",
            "use",
        }
    )
    tokens = {t.lower() for t in raw_tokens if len(t) >= 3 and t.lower() not in _STOP}
    return tuple(sorted(tokens))


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip header."""
    # Extract source portion if it's a retrieval_text
    source_match = re.search(r"Source:\s*\n(.*)", text, re.DOTALL)
    if source_match:
        text = source_match.group(1)

    # Normalize whitespace and lowercase
    text = _normalize_unicode_minus(text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def extract_block_type(retrieval_text: str) -> str:
    """Extract block type from retrieval_text header.

    The retrieval_text contains a line like:
        Block Type: table_row
    """
    m = re.search(r"Block Type:\s*(\S+)", retrieval_text)
    if m:
        return m.group(1)
    return "unknown"


def extract_raw_content(retrieval_text: str) -> str:
    """Extract the raw source content from retrieval_text.

    The retrieval_text has a header section followed by 'Source:\n<content>'.
    """
    source_match = re.search(r"Source:\s*\n(.*)", retrieval_text, re.DOTALL)
    if source_match:
        return source_match.group(1).strip()
    # Fallback: return the whole text
    return retrieval_text.strip()


# ---------------------------------------------------------------------------
# Candidate Signature Builder
# ---------------------------------------------------------------------------


def build_candidate_signature(candidate_record: dict[str, Any]) -> CandidateSignature:
    """Build a CandidateSignature from a view-pairs.jsonl record.

    The record structure:
    {
        "candidate_key": "...",
        "document_id": "...",
        "pdf_page": 26,
        "raw_view": {
            "retrieval_text": "...",
            ...
        },
        "structured_view": {...} or null,
        "row_ids": [...],
        "logical_table_ids": [...],
        "metric_paths": [...],
        "periods": [...],
        "bridge_grade": "raw_only" | "A1" | "A2" | "A3",
    }
    """
    candidate_key = str(candidate_record.get("candidate_key") or "")
    document_id = str(candidate_record.get("document_id") or "")
    pdf_page = int(candidate_record.get("pdf_page") or 0)

    raw_view = candidate_record.get("raw_view") or {}
    retrieval_text = str(raw_view.get("retrieval_text") or "")

    block_type = extract_block_type(retrieval_text)
    raw_content = extract_raw_content(retrieval_text)

    text_tokens = extract_text_tokens(retrieval_text)
    numeric_multiset = extract_numeric_multiset(raw_content)
    period_tokens = extract_period_tokens(retrieval_text)
    normalized_text = normalize_text(retrieval_text)

    # Extract existing structural mapping (if any)
    existing_row_ids = tuple(candidate_record.get("row_ids") or [])
    existing_logical_table_ids = tuple(candidate_record.get("logical_table_ids") or [])
    existing_metric_paths = tuple(candidate_record.get("metric_paths") or [])
    existing_bridge_grade = str(candidate_record.get("bridge_grade") or "raw_only")

    # Also pull from structured_view if present
    structured_view = candidate_record.get("structured_view")
    if structured_view and isinstance(structured_view, dict):
        if not existing_row_ids:
            existing_row_ids = tuple(structured_view.get("row_ids") or [])
        if not existing_logical_table_ids:
            existing_logical_table_ids = tuple(
                structured_view.get("logical_table_ids") or []
            )
        if not existing_metric_paths:
            existing_metric_paths = tuple(structured_view.get("metric_paths") or [])
        sv_grade = str(structured_view.get("bridge_grade") or "")
        if sv_grade and existing_bridge_grade == "raw_only":
            existing_bridge_grade = sv_grade

    return CandidateSignature(
        candidate_key=candidate_key,
        document_id=document_id,
        pdf_page=pdf_page,
        block_type=block_type,
        raw_content=raw_content,
        text_tokens=text_tokens,
        numeric_multiset=numeric_multiset,
        period_tokens=period_tokens,
        normalized_text=normalized_text,
        existing_row_ids=existing_row_ids,
        existing_logical_table_ids=existing_logical_table_ids,
        existing_metric_paths=existing_metric_paths,
        existing_bridge_grade=existing_bridge_grade,
    )
