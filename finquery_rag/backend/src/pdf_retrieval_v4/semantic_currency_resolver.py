"""Gate 03 R2 Pass D — Semantic Currency Resolver.

Currency is resolved SEPARATELY from scale (never merged).

  "$"           → currency_symbol = "$", currency_code = None (unresolved)
  "U.S. dollars" → currency_symbol = "$", currency_code = "USD" (resolved)
  "Amounts in U.S. dollars" → currency_code = "USD"

Default: symbol-only without explicit report context does NOT bind to
a currency code.
"""

from __future__ import annotations

import re
from typing import Any

from src.pdf_retrieval_v4.semantic_graph_models import CurrencyResolution

# ---------------------------------------------------------------------------
# Symbol → code mapping (only used when report context is explicit)
# ---------------------------------------------------------------------------

_SYMBOL_MAP: dict[str, str] = {
    "$": "USD",
    "us$": "USD",
    "u.s.$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "CNY",
    "rmb": "CNY",
}

# Explicit currency code patterns
_CODE_RE = re.compile(r"\b(USD|EUR|GBP|JPY|CNY|RMB|CAD|AUD|CHF)\b")
# "Amounts in U.S. dollars" / "Amounts in Euros" etc.
_EXPLICIT_CURRENCY_RE = re.compile(
    r"(u\.?s\.?\s*dollars?|euro[s]?|british\s*pound[s]?|japanese\s*yen|"
    r"chinese\s*yuan|canadian\s*dollar[s]?|australian\s*dollar[s]?)",
    re.IGNORECASE,
)

_EXPLICIT_CODE_MAP: dict[str, str] = {
    "u.s.dollar": "USD",
    "usdollar": "USD",
    "u.s.dollars": "USD",
    "usdollars": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "euro": "EUR",
    "euros": "EUR",
    "britishpound": "GBP",
    "britishpounds": "GBP",
    "japaneseyen": "JPY",
    "chineseyuan": "CNY",
    "canadiandollar": "CAD",
    "canadiandollars": "CAD",
    "australiandollar": "AUD",
    "australiandollars": "AUD",
}


def _normalize_explicit(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def resolve_table_currency(
    table: dict[str, Any],
    table_fragment_id: str,
    page_context: str | None = None,
) -> CurrencyResolution:
    """Resolve currency for a single table.

    Parameters
    ----------
    table
        Table fragment dict from adapter-predictions.
    table_fragment_id
        The table's fragment id.
    page_context
        Additional text context from the page (for explicit currency
        declarations like "Amounts in U.S. dollars").
    """
    # Collect all text from the table
    all_texts: list[str] = []
    for cell in table.get("cells") or []:
        all_texts.append(str(cell.get("resolved_text") or ""))
        all_texts.append(str(cell.get("raw_text") or ""))
    for row in table.get("rows") or []:
        all_texts.append(str(row.get("resolved_text") or ""))
    for ht in table.get("header_texts") or []:
        all_texts.append(str(ht))
    if page_context:
        all_texts.append(page_context)

    combined = " ".join(all_texts)

    # 1. Check for explicit currency code
    code_match = _CODE_RE.search(combined)
    if code_match:
        code = code_match.group(1)
        symbol = "$" if code in ("USD",) else None
        return CurrencyResolution(
            table_fragment_id=table_fragment_id,
            currency_symbol=symbol,
            currency_code=code,
            currency_source="explicit_code",
            currency_status="resolved",
        )

    # 2. Check for explicit currency declaration ("U.S. dollars", "Euros")
    explicit_match = _EXPLICIT_CURRENCY_RE.search(combined)
    if explicit_match:
        normalized = _normalize_explicit(explicit_match.group(1))
        code = _EXPLICIT_CODE_MAP.get(normalized)
        if code:
            symbol = "$" if code == "USD" else None
            return CurrencyResolution(
                table_fragment_id=table_fragment_id,
                currency_symbol=symbol,
                currency_code=code,
                currency_source="explicit_declaration",
                currency_status="resolved",
            )

    # 3. Check for currency symbol only (e.g. "$")
    # Only bind symbol, NOT code — requires explicit report context
    for symbol in ("$", "€", "£", "¥"):
        if symbol in combined:
            return CurrencyResolution(
                table_fragment_id=table_fragment_id,
                currency_symbol=symbol,
                currency_code=None,
                currency_source="symbol_only",
                currency_status="unresolved",
            )

    # 4. No currency signal
    return CurrencyResolution(
        table_fragment_id=table_fragment_id,
        currency_symbol=None,
        currency_code=None,
        currency_source=None,
        currency_status="unresolved",
    )
