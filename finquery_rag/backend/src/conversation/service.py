"""Conversation Context Manager Service.

Coordinates:
- State retrieval and isolation
- Context relevance filtering
- Context budget management
- Contextual query resolution (Qwen3.6-Flash / Fast Path)
- State transitions and topic switch updates
"""

from __future__ import annotations

import re
import time
from typing import Any

from .context_budget import ContextBudgetManager
from .contracts import (
    ConversationResolution,
    DialogueState,
    DialogueTurn,
)
from .relevance_filter import ContextRelevanceFilter
from .resolver import KNOWN_ENTITIES, ContextualQueryResolver
from .store import ConversationStateStore, InMemoryConversationStore

ALL_METRIC_NAMES = [
    "automotive gross margin",
    "capital expenditures",
    "capital expenditure",
    "operating income",
    "operating margin",
    "free cash flow",
    "gross margin",
    "net income",
    "billings",
    "revenue",
    "eps",
]


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
        history_turns: list[DialogueTurn] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> ConversationResolution:
        """Process one turn with optional transient prior raw history.

        Shadow callers supply SessionManager history loaded before the current
        user message is committed. It is never persisted by this component.
        Existing callers that omit it retain the historical behavior.
        """
        state = self.store.get_state(conversation_id)
        if not state:
            state = DialogueState(conversation_id=conversation_id)

        context_turns = state.recent_turns if history_turns is None else list(history_turns)

        # 1. Filter relevant history turns
        relevant_turns = self.relevance_filter.filter_turns(user_query, state, context_turns)

        # 2. Budget and prepare context
        budgeted_turns, compressed_history, est_tokens = self.budget_manager.prepare_context(
            user_query, state, relevant_turns
        )
        if diagnostics is not None:
            diagnostics.update({
                "raw_history_turn_count": len(context_turns),
                "relevant_turn_count": len(relevant_turns),
                "selected_turn_count": len(budgeted_turns),
                "dropped_turn_count": max(0, len(context_turns) - len(budgeted_turns)),
                "estimated_context_tokens": est_tokens,
                "compressed_history_tokens": self.budget_manager.count_tokens(
                    compressed_history or ""
                ),
            })

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
        assistant_response: str | None,
        referenced_evidence_ids: list[str] | None = None,
    ) -> None:
        """Record assistant provenance without promoting text to a fact.

        Shadow callers pass assistant_response=None: SessionManager owns the
        raw assistant message; this store receives structured provenance only.
        """
        state = self.store.get_state(conversation_id)
        if not state:
            return
        if state.recent_turns and assistant_response is not None:
            last_turn = state.recent_turns[-1]
            last_turn.assistant_response = assistant_response
            if referenced_evidence_ids:
                last_turn.referenced_evidence_ids = list(referenced_evidence_ids)
        if referenced_evidence_ids:
            existing = list(state.referenced_evidence_ids)
            state.referenced_evidence_ids = existing + [
                value for value in referenced_evidence_ids
                if value not in existing
            ]
        self.store.save_state(state)

    def _extract_semantic_entities(self, query: str) -> tuple[str | None, list[str], str | None]:
        """Extracts entity, metrics list, and period from query text without sub-phrase overlap."""
        q_lower = query.lower()
        entity = None
        for cand in KNOWN_ENTITIES:
            if re.search(rf"\b{re.escape(cand)}\b", q_lower):
                entity = cand.capitalize() if len(cand) > 4 else cand.upper()
                break
                
        metrics = []
        q_temp = q_lower
        # Check longest metric names first, replacing matched substrings to prevent sub-phrase duplicates
        for m in ALL_METRIC_NAMES:
            if m in q_temp:
                norm_m = "Capital Expenditures" if "capital expenditure" in m else m.title()
                metrics.append(norm_m)
                q_temp = q_temp.replace(m, " " * len(m))
                
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
        
        # Check if genuinely distinct multiple metrics were requested (e.g. "Revenue AND Operating Margin")
        if len(parsed_m_list) > 1 and (" and " in raw_user_query.lower() or " & " in raw_user_query.lower() or "compare" in raw_user_query.lower()):
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
