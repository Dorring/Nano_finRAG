"""Multi-source retrieval adapter integrating SQL wide table, Market API, and PDF retrieval."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import replace
from enum import IntEnum
from typing import Any

from rag_v2.adaptive.adaptive_contracts import (
    AdaptiveRAGStateV1,
    ReplanActionV1,
)
from rag_v2.routing.source_router import DataSourceType, SourceRouter
from .market_api import MarketDataProvider, SandboxMarketProvider
from .sql_fact_store import SqlFactStore


class SourceAuthority(IntEnum):
    """Authority priority for conflicting financial evidence packets."""

    AUDITED_PRIMARY_STATEMENT = 100
    AUDITED_FOOTNOTE = 90
    STRUCTURED_IXBRL_SQL = 80
    MARKET_API_REALTIME = 70
    UNAUDITED_INTERIM = 50
    UNKNOWN = 10


def determine_source_authority(packet: Mapping[str, Any]) -> SourceAuthority:
    """Determine authority level of an evidence packet."""
    source = str(packet.get("source", "")).upper()
    doc_id = str(packet.get("document_id", "")).upper()
    citation = str(packet.get("citation_id", "")).upper()
    is_interim = any(k in str(packet.get("period", "")).upper() for k in ("Q1", "Q2", "Q3", "Q4", "MONTHS ENDED"))

    if is_interim:
        return SourceAuthority.UNAUDITED_INTERIM

    if "STATEMENT" in citation or "STATEMENT" in doc_id or "OPERATIONS" in citation or "BALANCE SHEET" in citation or "CASH FLOW" in citation:
        return SourceAuthority.AUDITED_PRIMARY_STATEMENT
    if "NOTE" in citation or "FOOTNOTE" in citation or "NOTE" in doc_id:
        return SourceAuthority.AUDITED_FOOTNOTE
    if "SQL" in source or "IXBRL" in source or "SQL_FACT_STORE" in source:
        return SourceAuthority.STRUCTURED_IXBRL_SQL
    if "MARKET" in source:
        return SourceAuthority.MARKET_API_REALTIME

    return SourceAuthority.UNKNOWN


def arbitrate_candidate_packets(
    candidates: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Arbitrate candidate packets claiming the same slot/coordinate based on source authority.

    If multiple packets claim the same slot_id or semantic coordinate, selects the one
    with highest SourceAuthority score and attaches arbitration metadata.
    """
    candidates_list = list(candidates)
    if not candidates_list:
        return ()

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    ungrouped: list[dict[str, Any]] = []

    for c in candidates_list:
        c_dict = dict(c)
        slots = c.get("slot_ids") or c.get("slots") or [""]
        slot_id = str(slots[0]) if slots else ""
        entity = str(c.get("entity_id") or c.get("entity") or "").lower()
        metric = str(c.get("metric", "")).lower()
        period = str(c.get("period", "")).lower()

        key = (slot_id, entity, metric, period)
        if slot_id and metric:
            grouped.setdefault(key, []).append(c_dict)
        else:
            ungrouped.append(c_dict)

    arbitrated: list[dict[str, Any]] = list(ungrouped)

    for key, packets in grouped.items():
        if len(packets) == 1:
            arbitrated.append(packets[0])
            continue

        # Sort by authority score descending, preferring restated audited columns
        sorted_packets = sorted(
            packets,
            key=lambda p: (
                determine_source_authority(p).value,
                1 if p.get("is_restated") else 0,
            ),
            reverse=True,
        )
        winner = dict(sorted_packets[0])
        suppressed = sorted_packets[1:]
        winner["arbitration_record"] = {
            "selected_source": winner.get("source"),
            "selected_authority": determine_source_authority(winner).name,
            "suppressed_count": len(suppressed),
            "suppressed_sources": [p.get("source") for p in suppressed],
        }
        arbitrated.append(winner)

    return tuple(arbitrated)



class MultiSourceRetrievalAdapter:
    """Unified retrieval capability port routing across heterogeneous financial data stores."""

    def __init__(
        self,
        *,
        sql_store: SqlFactStore | None = None,
        market_provider: MarketDataProvider | None = None,
        pdf_retriever: Any | None = None,
        enable_sql_fallback: bool = True,
    ) -> None:
        self.sql_store = sql_store
        self.market_provider = market_provider
        self.pdf_retriever = pdf_retriever
        self.enable_sql_fallback = enable_sql_fallback
        self.last_dispatch_stats: dict[str, int] = {
            "sql_hits": 0,
            "sql_misses": 0,
            "market_hits": 0,
            "market_misses": 0,
            "pdf_calls": 0,
        }

    def retrieve(
        self,
        action: ReplanActionV1,
        state: AdaptiveRAGStateV1,
    ) -> tuple[Mapping[str, Any], ...]:
        """Dispatch retrieval action across SQL wide table, Market API, and PDF retriever."""
        # 1. Resolve slots targeted by this action
        slot_map = {str(s.get("slot_id")): s for s in state.required_slots}
        target_slot_ids = tuple(str(x) for x in action.target_slots)
        if target_slot_ids:
            target_slots = [slot_map[sid] for sid in target_slot_ids if sid in slot_map]
        else:
            target_slots = list(state.required_slots)

        candidates: list[Mapping[str, Any]] = []
        sql_slots: list[dict[str, Any]] = []
        market_slots: list[dict[str, Any]] = []
        pdf_slots: list[dict[str, Any]] = []

        # Reset per-call dispatch counters
        self.last_dispatch_stats = {
            "sql_hits": 0,
            "sql_misses": 0,
            "market_hits": 0,
            "market_misses": 0,
            "pdf_calls": 0,
        }

        # 2. Partition slots by source preference
        for slot in target_slots:
            slot_dict = dict(slot)
            pref = slot_dict.get("source_preference") or action.constraints.get("source_preference")
            if not pref:
                # Dynamic route determination if slot lacks explicit preference
                decision = SourceRouter.route_slot(
                    # Mock a temporary slot-like object if needed
                    type("SlotProxy", (), {
                        "metric": slot_dict.get("metric", ""),
                        "scope": slot_dict.get("scope"),
                    })(),
                    query=action.query,
                )
                pref = decision.target_source.value
                slot_dict["source_preference"] = pref
                if decision.section_hint:
                    slot_dict.setdefault("section_hint", decision.section_hint)

            if pref == DataSourceType.STRUCTURED_STORE.value:
                sql_slots.append(slot_dict)
            elif pref == DataSourceType.MARKET_API.value:
                market_slots.append(slot_dict)
            else:
                pdf_slots.append(slot_dict)

        # 3. Query SQL Fact Store
        if sql_slots:
            for s in sql_slots:
                slot_id = str(s.get("slot_id", ""))
                metric = str(s.get("metric", ""))
                period = str(s.get("period", ""))
                entity = str(s.get("entity", ""))
                scope = str(s.get("scope", "consolidated"))

                hit = False
                if self.sql_store is not None:
                    fact = self.sql_store.query_fact(entity=entity, metric=metric, period=period, scope=scope)
                    if fact is not None:
                        packet = self.sql_store.as_evidence_packet(fact, slot_id=slot_id)
                        candidates.append(packet)
                        self.last_dispatch_stats["sql_hits"] += 1
                        hit = True

                if not hit:
                    self.last_dispatch_stats["sql_misses"] += 1
                    if self.enable_sql_fallback:
                        # Fallback to PDF retriever for this slot
                        pdf_slots.append(s)

        # 4. Query Market Data Provider
        if market_slots:
            for s in market_slots:
                slot_id = str(s.get("slot_id", ""))
                metric = str(s.get("metric", ""))
                entity = str(s.get("entity", ""))

                hit = False
                if self.market_provider is not None:
                    quote = self.market_provider.get_quote(entity=entity, metric=metric)
                    if quote is not None:
                        packet = self.market_provider.as_evidence_packet(quote, slot_id=slot_id)
                        candidates.append(packet)
                        self.last_dispatch_stats["market_hits"] += 1
                        hit = True

                if not hit:
                    self.last_dispatch_stats["market_misses"] += 1

        # 5. Delegate narrative / footnote / primary statements / fallback to PDF retriever
        if pdf_slots and self.pdf_retriever is not None:
            self.last_dispatch_stats["pdf_calls"] += 1
            # Adjust sub-action target slots to only those routed to PDF
            sub_target_slots = tuple(str(s.get("slot_id", "")) for s in pdf_slots)
            sub_action = replace(action, target_slots=sub_target_slots)
            pdf_packets = self.pdf_retriever.retrieve(sub_action, state)
            if pdf_packets:
                candidates.extend(pdf_packets)

        # 6. Arbitrate multiple candidates claiming the same slot/coordinate
        return arbitrate_candidate_packets(candidates)

    async def retrieve_async(
        self,
        action: ReplanActionV1,
        state: AdaptiveRAGStateV1,
    ) -> tuple[Mapping[str, Any], ...]:
        """Asynchronously dispatch retrieval action concurrently in a non-blocking thread."""
        return await asyncio.to_thread(self.retrieve, action, state)

