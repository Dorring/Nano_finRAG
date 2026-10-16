"""Multi-source retrieval adapter integrating SQL wide table, Market API, and PDF retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from rag_v2.adaptive.adaptive_contracts import (
    AdaptiveRAGStateV1,
    ReplanActionV1,
)
from rag_v2.routing.source_router import DataSourceType, SourceRouter
from .market_api import MarketDataProvider, SandboxMarketProvider
from .sql_fact_store import SqlFactStore


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

        return tuple(candidates)
