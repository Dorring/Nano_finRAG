"""Conversation Context Manager Service.

Coordinates:
- State retrieval and isolation
- Context relevance filtering
- Context budget management
- Contextual query resolution (Qwen3.6-Flash / Fast Path)
- State transitions and topic switch updates
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from .bailian_client import BailianClient
from .context_budget import ContextBudgetManager
from .contracts import (
    ConversationResolution,
    DialogueState,
    DialogueTurn,
    ReasonCode,
)
from .relevance_filter import ContextRelevanceFilter
from .resolver import KNOWN_ENTITIES, ContextualQueryResolver
from .store import ConversationStateStore, InMemoryConversationStore


class ConversationContextManager:
    """Central orchestrator for the Conversation Context Layer."""

    def __init__(
        self,
        store: ConversationStateStore | None = None,
        resolver: ContextualQueryResolver | None = None,
        relevance_filter: ContextRelevanceFilter | None = None,
        budget_manager: ContextBudgetManager | None = None,
    ) -> None:
        self.store = store or InMemoryConversationStore()
        self.resolver = resolver or ContextualQueryResolver()
        self.relevance_filter = relevance_filter or ContextRelevanceFilter()
        self.budget_manager = budget_manager or ContextBudgetManager()

    def process_user_turn(
        self,
        conversation_id: str,
        user_query: str,
    ) -> ConversationResolution:
        """Processes a user turn, resolving multi-turn context and updating state."""
        state = self.store.get_state(conversation_id)
        if not state:
            state = DialogueState(conversation_id=conversation_id)

        # 1. Filter relevant history turns
        relevant_turns = self.relevance_filter.filter_turns(user_query, state, state.recent_turns)

        # 2. Budget and prepare context
        budgeted_turns, compressed_history, est_tokens = self.budget_manager.prepare_context(
            user_query, state, relevant_turns
        )

        # 3. Resolve contextual query
        resolution = self.resolver.resolve(user_query, state, budgeted_turns)

        # 4. If supported and resolved, update dialogue state
        if resolution.supported and not resolution.clarification_required:
            self._update_state_on_resolution(state, user_query, resolution)
            self.store.save_state(state)

        return resolution

    def record_assistant_turn(
        self,
        conversation_id: str,
        assistant_response: str,
        referenced_evidence_ids: list[str] | None = None,
    ) -> None:
        """Records assistant turn provenance metadata.
        
        Trust Boundary:
        Stores evidence IDs as provenance metadata only; NEVER converts raw text to facts.
        """
        state = self.store.get_state(conversation_id)
        if state and state.recent_turns:
            last_turn = state.recent_turns[-1]
            last_turn.assistant_response = assistant_response
            if referenced_evidence_ids:
                last_turn.referenced_evidence_ids = list(referenced_evidence_ids)
            self.store.save_state(state)

    def _extract_semantic_entities(self, query: str) -> tuple[str | None, list[str], str | None]:
        """Extracts entity, metrics list, and period from query text."""
        q_lower = query.lower()
        entity = None
        for cand in KNOWN_ENTITIES:
            if re.search(rf"\b{re.escape(cand)}\b", q_lower):
                entity = cand.capitalize() if len(cand) > 4 else cand.upper()
                break
                
        metrics = []
        for m in ["automotive gross margin", "operating margin", "operating income", "net income", "gross margin", "free cash flow", "capital expenditures", "capital expenditure", "revenue", "billings", "eps"]:
            if m in q_lower:
                metrics.append("Capital Expenditures" if "capital expenditure" in m else m.title())
                
        period = None
        p_match = re.search(r"\b(fy\s*20\d\d|20\d\d|q[1-4]\s*20\d\d|q[1-4])\b", q_lower)
        if p_match:
            period = p_match.group(1).upper()
            
        return entity, metrics, period

    def _update_state_on_resolution(
        self,
        state: DialogueState,
        raw_user_query: str,
        res: ConversationResolution,
    ) -> None:
        """Updates dialogue state following a successful query resolution."""
        state.turn_count += 1
        turn_id = f"turn_{state.turn_count}"
        
        parsed_e, parsed_m_list, parsed_p = self._extract_semantic_entities(res.standalone_query or raw_user_query)
        
        ent = res.resolved_entity or parsed_e
        per = res.resolved_period or parsed_p
        
        # Check if multiple metrics were requested in the query
        if len(parsed_m_list) > 1:
            state.active_topic = "MULTIPLE_METRICS_" + "_".join(parsed_m_list)
            state.active_metric = None
        elif parsed_m_list:
            state.active_metric = parsed_m_list[0]
            state.active_topic = f"{ent}_{parsed_m_list[0]}" if ent else state.active_topic
        elif res.resolved_metric:
            state.active_metric = res.resolved_metric
            state.active_topic = f"{ent}_{res.resolved_metric}" if ent else state.active_topic

        # Create new turn record
        turn = DialogueTurn(
            turn_id=turn_id,
            user_query=raw_user_query,
            standalone_query=res.standalone_query,
            timestamp=time.time(),
            topic=state.active_topic,
        )
        state.recent_turns.append(turn)

        # Update entity, metric, period tracking
        if res.topic_switch or (ent and state.active_entity and ent != state.active_entity):
            state.comparison_entity = state.active_entity
            state.comparison_metric = state.active_metric
            state.comparison_period = state.active_period
            
            state.active_entity = ent
            state.active_period = per
        else:
            if ent:
                state.active_entity = ent
            if per:
                if state.active_period and per != state.active_period:
                    state.comparison_period = state.active_period
                state.active_period = per

        state.last_resolved_query = res.standalone_query

        # Update compressed summary if turn threshold reached
        compressed = self.budget_manager.update_compressed_history(state, state.recent_turns)
        if compressed:
            state.compressed_history = compressed
