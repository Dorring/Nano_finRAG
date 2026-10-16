"""Dynamic multi-source routing for financial query planning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from rag_v2.contracts.plan import RequiredSlot, SupervisorPlan


class DataSourceType(str, Enum):
    """Target financial data source tier."""

    STRUCTURED_STORE = "STRUCTURED_STORE"           # Canonical iXBRL / SQL fact store (e.g. top-line consolidated facts)
    PRIMARY_STATEMENTS = "PRIMARY_STATEMENTS"       # Face of financial statements (Operations, Balance Sheet, Cash Flow)
    FOOTNOTE_DISCLOSURES = "FOOTNOTE_DISCLOSURES"   # Footnote notes & segment disclosures (Note 19 Segments, Note 12 Debt)
    NARRATIVE_DISCLOSURES = "NARRATIVE_DISCLOSURES" # Item 7 MD&A, Item 1A Risk Factors, Qualitative reasoning
    MARKET_API = "MARKET_API"                       # External real-time quotes, PE ratio, FX exchange rate


@dataclass(frozen=True)
class RouteDecision:
    """Routing determination for a slot or query."""

    target_source: DataSourceType
    section_hint: str | None = None
    confidence: float = 1.0
    reason: str = ""

    def to_dict(self) -> dict[str, str | float | None]:
        return {
            "target_source": self.target_source.value,
            "section_hint": self.section_hint,
            "confidence": self.confidence,
            "reason": self.reason,
        }


# Rule banks for financial intent and metric patterns
_MARKET_API_PATTERNS = re.compile(
    r"\b(p/e|pe ratio|stock price|share price|market cap|market capitalization|exchange rate|spot rate|real-time|fx rate|quote)\b",
    re.IGNORECASE,
)

_NARRATIVE_PATTERNS = re.compile(
    r"\b(md&a|risk factors|business overview|why did|reason for|reasons behind|explain the|management discussion|qualitative|outlook|trend analysis|headwinds|tailwinds)\b",
    re.IGNORECASE,
)

_FOOTNOTE_PATTERNS = re.compile(
    r"\b(segment|geographic|division|product line|services revenue|cloud revenue|operating lease|lease liability|debt maturity|senior notes|contractual obligation|restricted cash|share-based|stock compensation|fair value|income taxes|pension|derivative|note \d+|footnote|disaggregation)\b",
    re.IGNORECASE,
)

_PRIMARY_STATEMENT_PATTERNS = re.compile(
    r"\b(balance sheet|statement of operations|income statement|statement of financial position|cash flow statement|statement of cash flows)\b",
    re.IGNORECASE,
)

_NON_GAAP_PATTERNS = re.compile(
    r"\b(non-gaap|adjusted|adj\b|ebitda|经调整|非gaap|free cash flow|fcf)\b",
    re.IGNORECASE,
)

_STANDARD_STRUCTURED_METRICS = frozenset(
    {
        "total net sales",
        "total revenue",
        "revenue",
        "revenues",
        "net sales",
        "gross profit",
        "operating income",
        "operating profit",
        "net income",
        "total assets",
        "current assets",
        "total liabilities",
        "current liabilities",
        "stockholders' equity",
        "shareholders' equity",
        "total equity",
        "earnings per share",
        "diluted eps",
        "basic eps",
        "cost of sales",
        "cost of revenue",
        "research and development",
        "selling, general and administrative",
        "operating cash flows",
        "operating cash flow",
    }
)


class SourceRouter:
    """Stateless router that determines optimal data source and section hints."""

    _custom_structured_metrics: set[str] = set()
    _custom_footnote_keywords: set[str] = set()
    _custom_market_keywords: set[str] = set()

    @classmethod
    def register_structured_metric(cls, metric: str) -> None:
        """Register custom canonical structured financial metric."""
        cls._custom_structured_metrics.add(metric.strip().lower())

    @classmethod
    def register_footnote_keyword(cls, keyword: str) -> None:
        """Register custom keyword identifying footnote disclosures."""
        cls._custom_footnote_keywords.add(keyword.strip().lower())

    @classmethod
    def register_market_keyword(cls, keyword: str) -> None:
        """Register custom keyword identifying real-time market API metrics."""
        cls._custom_market_keywords.add(keyword.strip().lower())

    @classmethod
    def route_query(cls, query: str) -> RouteDecision:
        """Route based on user query phrasing."""
        query_norm = query.strip().lower()

        if _MARKET_API_PATTERNS.search(query_norm) or any(k in query_norm for k in cls._custom_market_keywords):
            return RouteDecision(
                target_source=DataSourceType.MARKET_API,
                section_hint="Market Data Provider",
                confidence=0.95,
                reason="Query explicitly targets market-traded or real-time metrics",
            )

        if _NON_GAAP_PATTERNS.search(query_norm):
            return RouteDecision(
                target_source=DataSourceType.NARRATIVE_DISCLOSURES,
                section_hint="Item 7. Non-GAAP Financial Measures Reconciliation",
                confidence=0.92,
                reason="Query explicitly targets Non-GAAP or adjusted financial measures",
            )

        if _NARRATIVE_PATTERNS.search(query_norm):
            return RouteDecision(
                target_source=DataSourceType.NARRATIVE_DISCLOSURES,
                section_hint="Item 7. Management's Discussion and Analysis",
                confidence=0.90,
                reason="Query requests qualitative operational drivers or explanations",
            )

        if _FOOTNOTE_PATTERNS.search(query_norm):
            return RouteDecision(
                target_source=DataSourceType.FOOTNOTE_DISCLOSURES,
                section_hint="Item 8. Note Disclosures",
                confidence=0.88,
                reason="Query references segment or footnote-level disaggregated disclosures",
            )

        if _PRIMARY_STATEMENT_PATTERNS.search(query_norm):
            return RouteDecision(
                target_source=DataSourceType.PRIMARY_STATEMENTS,
                section_hint="Item 8. Consolidated Financial Statements",
                confidence=0.85,
                reason="Query explicitly targets primary financial statement line items",
            )

        # Default to structured store / primary financials
        return RouteDecision(
            target_source=DataSourceType.STRUCTURED_STORE,
            section_hint="Item 8. Consolidated Financial Statements",
            confidence=0.80,
            reason="Standard top-line financial metric query",
        )

    @classmethod
    def route_slot(cls, slot: RequiredSlot, query: str = "") -> RouteDecision:
        """Route a specific RequiredSlot using metric name, scope, and query context."""
        metric_norm = slot.metric.strip().lower()
        scope_norm = (slot.scope or "").strip().lower()

        # 1. Check if the slot metric itself targets market API
        if _MARKET_API_PATTERNS.search(metric_norm) or any(k in metric_norm for k in cls._custom_market_keywords):
            return RouteDecision(
                target_source=DataSourceType.MARKET_API,
                section_hint="Market Data Provider",
                confidence=0.95,
                reason="Slot metric targets real-time market metrics",
            )

        # 2. Check if metric is explicitly footnote/segment
        if _FOOTNOTE_PATTERNS.search(metric_norm) or any(k in metric_norm for k in cls._custom_footnote_keywords) or "segment" in scope_norm:
            note_match = re.search(r"note\s+(\d+)", query or "", re.IGNORECASE) or re.search(r"note\s+(\d+)", metric_norm, re.IGNORECASE)
            if note_match:
                hint = f"Item 8. Note {note_match.group(1)}"
            elif "segment" in metric_norm or "segment" in scope_norm:
                hint = "Item 8. Note Disclosures - Segment Information"
            elif "debt" in metric_norm or "borrowing" in metric_norm:
                hint = "Item 8. Note Disclosures - Debt"
            elif "lease" in metric_norm:
                hint = "Item 8. Note Disclosures - Leases"
            else:
                hint = "Item 8. Note Disclosures"

            return RouteDecision(
                target_source=DataSourceType.FOOTNOTE_DISCLOSURES,
                section_hint=hint,
                confidence=0.92,
                reason=f"Metric '{slot.metric}' or scope '{slot.scope}' is footnote/segment-specific",
            )

        # 3. Standard canonical metrics route to STRUCTURED_STORE or PRIMARY_STATEMENTS
        if metric_norm in _STANDARD_STRUCTURED_METRICS or metric_norm in cls._custom_structured_metrics:
            # Differentiate statement type for section_hint
            if any(term in metric_norm for term in ("asset", "liabilit", "equity", "debt", "cash and cash", "marketable securities", "payable", "receivable", "inventory", "working capital", "retained earnings", "goodwill", "intangible", "property, plant", "ppe")):
                section = "Item 8. Consolidated Balance Sheets"
            elif any(term in metric_norm for term in ("cash flow", "operating activities", "investing activities", "financing activities", "capital expenditure", "capex", "dividends paid")):
                section = "Item 8. Consolidated Statements of Cash Flows"
            else:
                section = "Item 8. Consolidated Statements of Operations"


            return RouteDecision(
                target_source=DataSourceType.STRUCTURED_STORE,
                section_hint=section,
                confidence=0.95,
                reason=f"Metric '{slot.metric}' matches canonical primary statement line item",
            )

        # 4. Check if slot or query asks for narrative explanation for qualitative items
        if _NARRATIVE_PATTERNS.search(metric_norm) or (query and _NARRATIVE_PATTERNS.search(query.lower())):
            return RouteDecision(
                target_source=DataSourceType.NARRATIVE_DISCLOSURES,
                section_hint="Item 7. Management's Discussion and Analysis",
                confidence=0.85,
                reason="Query or metric asks for MD&A qualitative discussion",
            )

        # 5. Query-level fallback for market API if slot was otherwise unclassified
        if query and _MARKET_API_PATTERNS.search(query.lower()):
            return RouteDecision(
                target_source=DataSourceType.MARKET_API,
                section_hint="Market Data Provider",
                confidence=0.90,
                reason="Query targets real-time market metrics",
            )

        # 6. Default fallback
        return RouteDecision(
            target_source=DataSourceType.PRIMARY_STATEMENTS,
            section_hint="Item 8. Consolidated Financial Statements",
            confidence=0.75,
            reason=f"General primary financial statement lookup for '{slot.metric}'",
        )


    @classmethod
    def annotate_plan(cls, plan: SupervisorPlan, query: str = "") -> SupervisorPlan:
        """Return a new SupervisorPlan where each RequiredSlot has source_preference and section_hint set."""
        annotated_slots: list[RequiredSlot] = []
        for slot in plan.required_slots:
            decision = cls.route_slot(slot, query)
            new_slot = RequiredSlot(
                slot_id=slot.slot_id,
                metric=slot.metric,
                period=slot.period,
                role=slot.role,
                value_type=slot.value_type,
                unit=slot.unit,
                entity=slot.entity,
                entity_id=slot.entity_id,
                scope=slot.scope,
                scope_id=slot.scope_id,
                source_preference=decision.target_source.value,
                section_hint=decision.section_hint,
            )
            annotated_slots.append(new_slot)

        return SupervisorPlan(
            intent=plan.intent,
            required_slots=tuple(annotated_slots),
            operation=plan.operation,
            next_action=plan.next_action,
            execution_dag=plan.execution_dag,
        )
