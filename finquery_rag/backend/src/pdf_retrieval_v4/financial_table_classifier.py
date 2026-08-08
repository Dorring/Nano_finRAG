"""Gate 03 R2 Pass A — Financial Table Classification.

Classifies each physical table fragment into a ``statement_type`` using
ONLY document-internal signals: table title/caption, row label patterns,
header text, and scale keywords.  No question / gold / company-specific
rules are used.

Statement types (see ``STATEMENT_TYPES`` in ``semantic_graph_models``):
  income_statement, balance_sheet, cash_flow, segment_table,
  maturity_table, aging_table, share_table, operating_metric_table,
  other_financial_table, unknown
"""

from __future__ import annotations

import re
from typing import Any

from src.pdf_retrieval_v4.semantic_graph_models import LogicalTable, STATEMENT_TYPES
from src.pdf_retrieval_v4.table_html_parser import norm_text

# ---------------------------------------------------------------------------
# Keyword banks (lowercase, matched via substring on normalized text)
# ---------------------------------------------------------------------------

_INCOME_KEYWORDS = (
    "revenue",
    "net sales",
    "cost of sales",
    "cost of revenue",
    "gross margin",
    "gross profit",
    "operating expenses",
    "operating income",
    "net income",
    "earnings per share",
    "diluted",
    "income before",
    "provision for income tax",
    "income tax",
    "loss before",
    "interest expense",
    "interest income",
    "other income",
    "total operating",
)

_BALANCE_SHEET_KEYWORDS = (
    "total assets",
    "total liabilities",
    "total stockholders",
    "total shareholders",
    "cash and cash equivalents",
    "short-term investments",
    "accounts receivable",
    "accounts payable",
    "inventories",
    "inventory",
    "long-term debt",
    "current assets",
    "current liabilities",
    "non-current",
    "property, plant",
    "goodwill",
    "intangible assets",
    "retained earnings",
    "accumulated",
    "total equity",
    "bonds payable",
    "deposits",  # bank balance sheet
    "loans",  # bank balance sheet
)

_CASH_FLOW_KEYWORDS = (
    "net cash",
    "cash flow",
    "operating activities",
    "investing activities",
    "financing activities",
    "capital expenditures",
    "proceeds from",
    "payments for",
    "depreciation and amortization",
    "stock-based compensation",
    "repurchase",
    "dividends paid",
    "acquisition of",
    "purchases of",
    "sales of",
    "effect of exchange rate",
)

_SEGMENT_KEYWORDS = (
    "segment",
    "geographic",
    "americas",
    "emea",
    "apac",
    "europe",
    "greater china",
    "japan",
    "rest of",
    "by region",
    "by geography",
    "disaggregation of revenue",
    "external revenue",
)

_MATURITY_KEYWORDS = (
    "maturity",
    "due in",
    "less than one year",
    "one to three year",
    "one to five year",
    "more than five year",
    "over five year",
    "carrying amount",
    "fair value",
    "contractual maturities",
)

_AGING_KEYWORDS = (
    "aging",
    "aged",
    "current",
    "30 days",
    "60 days",
    "90 days",
    "past due",
    "allowance for",
    "credit losses",
    "impaired",
    "non-performing",
)

_SHARE_KEYWORDS = (
    "shares outstanding",
    "shares issued",
    "treasury shares",
    "common stock",
    "preferred stock",
    "par value",
    "share repurchase",
    "dilutive",
    "weighted-average shares",
    "basic shares",
    "earnings per share",
)

_OPERATING_KEYWORDS = (
    "operating metric",
    "key metric",
    "performance metric",
    "active users",
    "subscribers",
    "average revenue per user",
    "arpu",
    "customers",
    "stores",
    "employees",
    "headcount",
    "square feet",
    "shipments",
    "production",
    "utilization",
)


def _classify_by_keywords(
    text_blob: str,
) -> str:
    """Score keyword banks against a normalized text blob and return best type."""
    normed = norm_text(text_blob)
    scores: dict[str, int] = {
        "income_statement": 0,
        "balance_sheet": 0,
        "cash_flow": 0,
        "segment_table": 0,
        "maturity_table": 0,
        "aging_table": 0,
        "share_table": 0,
        "operating_metric_table": 0,
        "other_financial_table": 0,
    }

    banks: list[tuple[str, tuple[str, ...]]] = [
        ("income_statement", _INCOME_KEYWORDS),
        ("balance_sheet", _BALANCE_SHEET_KEYWORDS),
        ("cash_flow", _CASH_FLOW_KEYWORDS),
        ("segment_table", _SEGMENT_KEYWORDS),
        ("maturity_table", _MATURITY_KEYWORDS),
        ("aging_table", _AGING_KEYWORDS),
        ("share_table", _SHARE_KEYWORDS),
        ("operating_metric_table", _OPERATING_KEYWORDS),
    ]

    for stmt_type, keywords in banks:
        for kw in keywords:
            if kw in normed:
                scores[stmt_type] += 1

    best_type = max(scores, key=lambda k: scores[k])
    if scores[best_type] == 0:
        return "unknown"
    return best_type


_TITLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("income_statement", re.compile(r"statement[s]? of operations", re.IGNORECASE)),
    ("income_statement", re.compile(r"statement[s]? of income", re.IGNORECASE)),
    ("income_statement", re.compile(r"statement[s]? of earnings", re.IGNORECASE)),
    (
        "income_statement",
        re.compile(r"consolidated statement[s]? of operation", re.IGNORECASE),
    ),
    ("balance_sheet", re.compile(r"balance sheet", re.IGNORECASE)),
    (
        "balance_sheet",
        re.compile(r"statement[s]? of financial position", re.IGNORECASE),
    ),
    ("cash_flow", re.compile(r"statement[s]? of cash flow", re.IGNORECASE)),
    ("cash_flow", re.compile(r"cash flow statement", re.IGNORECASE)),
    ("segment_table", re.compile(r"disaggregation of (revenue|sales)", re.IGNORECASE)),
    (
        "segment_table",
        re.compile(r"revenue by (segment|region|geography)", re.IGNORECASE),
    ),
    (
        "segment_table",
        re.compile(r"geographic (revenue|sales|information)", re.IGNORECASE),
    ),
    ("maturity_table", re.compile(r"contractual maturit", re.IGNORECASE)),
    ("maturity_table", re.compile(r"maturity schedule", re.IGNORECASE)),
    ("aging_table", re.compile(r"aging (of|schedule)", re.IGNORECASE)),
    ("aging_table", re.compile(r"allowance for (credit|doubtful)", re.IGNORECASE)),
    ("share_table", re.compile(r"shares outstanding", re.IGNORECASE)),
    (
        "share_table",
        re.compile(
            r"statement[s]? of (shareholders|stockholders).? equity", re.IGNORECASE
        ),
    ),
    (
        "share_table",
        re.compile(r"changes in (shareholders|stockholders).? equity", re.IGNORECASE),
    ),
    (
        "operating_metric_table",
        re.compile(r"(key|operating|performance) metric", re.IGNORECASE),
    ),
]


def _classify_by_title(title: str) -> str | None:
    """Match table title against known statement-title patterns."""
    for stmt_type, pattern in _TITLE_PATTERNS:
        if pattern.search(title):
            return stmt_type
    return None


def _extract_table_title(table: dict[str, Any]) -> str:
    """Best-effort extraction of a table title/caption.

    Sources (in priority order):
      1. ``header_texts`` if populated
      2. First row whose cells are all header-like (short, no numerics)
      3. First row metric_text
    """
    header_texts = table.get("header_texts") or []
    if header_texts:
        return " ".join(str(h) for h in header_texts)[:300]

    rows = table.get("rows") or []
    if rows:
        # Use first row's metric_text as a fallback title hint
        first_metric = str(rows[0].get("metric_text") or "")
        return first_metric[:300]
    return ""


def classify_table(table: dict[str, Any]) -> LogicalTable:
    """Classify a single physical table fragment into a LogicalTable.

    Parameters
    ----------
    table
        A table fragment dict from adapter-predictions (build_table_fragment
        output).
    """
    table_fragment_id = str(table.get("table_fragment_id") or "")
    document_id = str(table.get("document_id") or "")
    pdf_page = int(table.get("pdf_page") or 0)
    table_index = int(table.get("table_index") or 0)
    row_count = int(table.get("row_count") or 0)
    column_count = int(table.get("column_count") or 0)
    scale_candidates = tuple(str(s) for s in (table.get("scale_candidates") or []))

    title = _extract_table_title(table)

    # 1. Try title-based classification first (most reliable)
    statement_type = _classify_by_title(title) if title else None

    # 2. Fall back to row-pattern keyword scoring
    if statement_type is None:
        row_texts: list[str] = []
        for row in table.get("rows") or []:
            metric_text = str(row.get("metric_text") or "")
            resolved = str(row.get("resolved_text") or "")
            row_texts.append(metric_text + " " + resolved)
        text_blob = title + " " + " ".join(row_texts)
        statement_type = _classify_by_keywords(text_blob)

    if statement_type not in STATEMENT_TYPES:
        statement_type = "unknown"

    source_traceback = {
        "table_fragment_id": table_fragment_id,
        "document_id": document_id,
        "pdf_page": pdf_page,
        "table_bbox": table.get("table_bbox") or [],
    }

    return LogicalTable(
        table_fragment_id=table_fragment_id,
        document_id=document_id,
        pdf_page=pdf_page,
        table_index=table_index,
        statement_type=statement_type,
        table_title=title,
        row_count=row_count,
        column_count=column_count,
        scale_candidates=scale_candidates,
        source_traceback=source_traceback,
    )
