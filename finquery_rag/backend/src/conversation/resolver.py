"""Contextual Query Resolver.

Reconstructs multi-turn conversational queries into standalone financial queries
using Bailian Qwen3.6-Flash and deterministic Fast Path rules.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .bailian_client import BailianClient
from .contracts import (
    ConversationResolution,
    DialogueState,
    DialogueTurn,
    ReasonCode,
)

# Common indicators of contextual dependency
DEPENDENCY_MARKERS = [
    r"\b(it|this|that|these|those|they|their|its)\b",
    r"\b(the\s+company|the\s+firm|the\s+issuer)\b",
    r"\b(what\s+about|how\s+about|and\s+for|what\s+of)\b",
    r"\b(previous|prior|preceding|following|next)\s+(year|quarter|period|fiscal)\b",
    r"\b(last\s+year|last\s+quarter|same\s+period|yoy|qoq)\b",
    r"\b(increase|decrease|grow|growth|change|difference|higher|lower|margin)\b",
    r"\b(compare|compared|versus|vs\.?)\b",
    r"\b(both|neither|all\s+three|the\s+latter|the\s+former)\b",
]

# Non-financial out-of-scope markers
OUT_OF_SCOPE_MARKERS = [
    r"\b(recommend\s+a\s+movie|weather\s+in|write\s+a\s+poem|tell\s+a\s+joke|translate)\b",
    r"\b(who\s+won|play\s+music|recipe\s+for|capital\s+of)\b",
]


class ContextualQueryResolver:
    """Resolves conversational ellipsis and references into standalone queries."""

    def __init__(self, client: BailianClient | None = None) -> None:
        self.client = client or BailianClient()

    def is_self_contained_fast_path(self, query: str) -> bool:
        """Determines if a query is self-contained and can bypass LLM resolution."""
        q_lower = query.lower().strip()
        
        # Check for dependency indicators
        for pattern in DEPENDENCY_MARKERS:
            if re.search(pattern, q_lower):
                return False
                
        # If query is short (< 4 words) and lacks specific company/metric, it likely needs context
        words = q_lower.split()
        if len(words) < 4 and not any(term in q_lower for term in ["revenue", "income", "margin", "cash", "asset", "debt"]):
            return False
            
        return True

    def resolve(
        self,
        current_query: str,
        dialogue_state: DialogueState | None = None,
        filtered_turns: list[DialogueTurn] | None = None,
    ) -> ConversationResolution:
        """Resolves current user query against dialogue state and recent turns."""
        turns = filtered_turns if filtered_turns is not None else (dialogue_state.recent_turns if dialogue_state else [])
        
        # 1. Out of Scope Check
        q_lower = current_query.lower()
        for pat in OUT_OF_SCOPE_MARKERS:
            if re.search(pat, q_lower):
                return ConversationResolution(
                    supported=False,
                    requires_context=False,
                    standalone_query=current_query,
                    reason_codes=[ReasonCode.OUT_OF_SCOPE],
                )

        # 2. Fast Path: First turn or self-contained query
        if not dialogue_state or dialogue_state.turn_count == 0 or len(turns) == 0:
            return ConversationResolution(
                supported=True,
                requires_context=False,
                standalone_query=current_query,
                reason_codes=[ReasonCode.NO_CONTEXT_REQUIRED],
            )

        if self.is_self_contained_fast_path(current_query):
            # Check if this self-contained query introduces a topic switch
            topic_switched = False
            if dialogue_state.active_entity and not re.search(rf"\b{re.escape(dialogue_state.active_entity.lower())}\b", q_lower):
                topic_switched = True
                
            codes = [ReasonCode.NO_CONTEXT_REQUIRED, ReasonCode.RESOLVER_BYPASS]
            if topic_switched:
                codes.append(ReasonCode.TOPIC_SWITCH)
                
            return ConversationResolution(
                supported=True,
                requires_context=False,
                standalone_query=current_query,
                topic_switch=topic_switched,
                reason_codes=codes,
            )

        # 3. Contextual Resolution Path via Qwen3.6-Flash or Deterministic Rule Engine
        return self._resolve_with_context(current_query, dialogue_state, turns)

    def _resolve_with_context(
        self,
        current_query: str,
        dialogue_state: DialogueState,
        turns: list[DialogueTurn],
    ) -> ConversationResolution:
        """Invokes Bailian Qwen3.6-Flash or deterministic resolution logic."""
        # Try LLM invocation if client has API Key
        llm_resolution = self._call_llm_resolver(current_query, dialogue_state, turns)
        if llm_resolution is not None:
            return llm_resolution

        # Deterministic Rule-Based Resolution (Offline / Fallback)
        return self._deterministic_fallback_resolve(current_query, dialogue_state, turns)

    def _call_llm_resolver(
        self,
        current_query: str,
        dialogue_state: DialogueState,
        turns: list[DialogueTurn],
    ) -> ConversationResolution | None:
        """Constructs prompt and calls Qwen3.6-Flash for structured resolution."""
        if not self.client.api_key:
            return None

        # Build context prompt
        recent_summary = []
        for i, t in enumerate(turns[-4:]):
            recent_summary.append(f"Turn {i+1} User: {t.user_query}")
            if t.standalone_query != t.user_query:
                recent_summary.append(f"Turn {i+1} Resolved Query: {t.standalone_query}")

        state_summary = {
            "active_entity": dialogue_state.active_entity,
            "active_metric": dialogue_state.active_metric,
            "active_period": dialogue_state.active_period,
            "active_scope": dialogue_state.active_scope,
            "active_topic": dialogue_state.active_topic,
        }

        system_prompt = (
            "You are the Conversation Context Layer for a Trusted Financial RAG system.\n"
            "Your ONLY task is to reconstruct the user's current query into a complete, standalone financial question by resolving pronouns, ellipses, and relative time expressions.\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. DO NOT answer the question. DO NOT perform calculations. DO NOT invent numbers.\n"
            "2. Precedence: User's CURRENT query > Explicitly referenced turn > Dialogue State > History.\n"
            "3. If the user explicitly mentions an entity or metric, it OVERRIDES past dialogue state.\n"
            "4. If there is genuine ambiguity (e.g. user asks 'what about last year?' after discussing multiple metrics), set ambiguity_detected=true and clarification_required=true.\n"
            "5. Return ONLY a valid JSON object matching the requested schema."
        )

        user_prompt = json.dumps({
            "current_user_query": current_query,
            "dialogue_state": state_summary,
            "recent_turns": recent_summary,
            "compressed_history": dialogue_state.compressed_history,
        }, ensure_ascii=False, indent=2)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw_output = self.client.chat_completion(messages, response_format={"type": "json_object"})
        if not raw_output:
            return None

        try:
            data = json.loads(raw_output)
            return ConversationResolution(
                supported=data.get("supported", True),
                requires_context=data.get("requires_context", True),
                standalone_query=data.get("standalone_query", current_query),
                resolved_entity=data.get("resolved_entity"),
                resolved_metric=data.get("resolved_metric"),
                resolved_period=data.get("resolved_period"),
                resolved_scope=data.get("resolved_scope"),
                inherited_fields=data.get("inherited_fields", []),
                explicit_fields=data.get("explicit_fields", []),
                topic_switch=data.get("topic_switch", False),
                ambiguity_detected=data.get("ambiguity_detected", False),
                clarification_required=data.get("clarification_required", False),
                clarification_question=data.get("clarification_question"),
                clarification_options=data.get("clarification_options", []),
                relevant_turn_ids=data.get("relevant_turn_ids", []),
                confidence=float(data.get("confidence", 1.0)),
                reason_codes=data.get("reason_codes", [ReasonCode.REFERENCE_RESOLVED]),
            )
        except Exception:
            return None

    def _deterministic_fallback_resolve(
        self,
        current_query: str,
        state: DialogueState,
        turns: list[DialogueTurn],
    ) -> ConversationResolution:
        """Deterministic resolver for local execution and offline testing."""
        q_lower = current_query.lower().strip()
        reason_codes = []
        inherited_fields = []
        explicit_fields = []
        
        # Entity extraction from current query
        entity = None
        for cand in ["apple", "aapl", "microsoft", "msft", "tesla", "tsla", "google", "googl", "amazon", "amzn", "coca-cola", "ko", "oracle", "orcl"]:
            if re.search(rf"\b{re.escape(cand)}\b", q_lower):
                entity = cand.upper()
                explicit_fields.append("entity")
                break
                
        # Metric extraction from current query
        metric = None
        for m in ["revenue", "operating margin", "operating income", "net income", "gross margin", "free cash flow", "capital expenditures", "billings", "eps"]:
            if m in q_lower:
                metric = m.title()
                explicit_fields.append("metric")
                break
                
        # Period extraction from current query
        period = None
        period_match = re.search(r"\b(fy\s*20\d\d|20\d\d|q[1-4]\s*20\d\d|q[1-4])\b", q_lower)
        if period_match:
            period = period_match.group(1).upper().replace(" ", "")
            explicit_fields.append("period")

        # 1. Ambiguity Detection
        # If previous state had multiple metrics and user asks "What about 2023?" without specifying metric
        if state.active_topic and "MULTIPLE_METRICS" in state.active_topic and not metric:
            return ConversationResolution(
                supported=True,
                requires_context=True,
                standalone_query="",
                ambiguity_detected=True,
                clarification_required=True,
                clarification_question="Which metric would you like to check for that period?",
                clarification_options=["Revenue", "Operating Margin", "Both"],
                reason_codes=[ReasonCode.AMBIGUOUS_METRIC],
            )

        # 2. Topic Switch / Entity Switch: "What about Microsoft?"
        if entity and entity != (state.active_entity or "").upper():
            topic_switched = True
            active_m = metric or state.active_metric or "Revenue"
            active_p = period or state.active_period or "FY2024"
            if not metric and state.active_metric:
                inherited_fields.append("metric")
            if not period and state.active_period:
                inherited_fields.append("period")
                
            standalone = f"What was {entity} {active_p} {active_m}?"
            return ConversationResolution(
                supported=True,
                requires_context=True,
                standalone_query=standalone,
                resolved_entity=entity,
                resolved_metric=active_m,
                resolved_period=active_p,
                inherited_fields=inherited_fields,
                explicit_fields=explicit_fields,
                topic_switch=topic_switched,
                reason_codes=[ReasonCode.TOPIC_SWITCH, ReasonCode.ENTITY_INHERITED if not entity else ReasonCode.EXPLICIT_QUERY_OVERRIDE],
            )

        # 3. Relative Period Resolution: "What about the previous year?" / "last year?"
        is_prev_year = bool(re.search(r"\b(previous|prior|preceding|last)\s+(year|period|fy)\b", q_lower))
        if is_prev_year and state.active_period:
            base_p = state.active_period
            year_match = re.search(r"20\d\d", base_p)
            if year_match:
                prev_year = str(int(year_match.group(0)) - 1)
                resolved_p = base_p.replace(year_match.group(0), prev_year)
                active_e = entity or state.active_entity or "Apple"
                active_m = metric or state.active_metric or "Revenue"
                
                inherited_fields.extend(["entity", "metric"])
                standalone = f"What was {active_e} {resolved_p} {active_m}?"
                return ConversationResolution(
                    supported=True,
                    requires_context=True,
                    standalone_query=standalone,
                    resolved_entity=active_e,
                    resolved_metric=active_m,
                    resolved_period=resolved_p,
                    inherited_fields=inherited_fields,
                    explicit_fields=explicit_fields,
                    reason_codes=[ReasonCode.RELATIVE_PERIOD_RESOLVED, ReasonCode.PERIOD_INHERITED],
                )

        # 4. Cross-turn Calculation: "How much did it grow?" / "What is the growth?"
        is_calc = bool(re.search(r"\b(grow|growth|increase|change|difference)\b", q_lower))
        if is_calc and state.active_entity and state.active_metric:
            e = entity or state.active_entity
            m = metric or state.active_metric
            p1 = state.comparison_period or "FY2023"
            p2 = state.active_period or "FY2024"
            standalone = f"Calculate the change in {e} {m} from {p1} to {p2}."
            return ConversationResolution(
                supported=True,
                requires_context=True,
                standalone_query=standalone,
                resolved_entity=e,
                resolved_metric=m,
                resolved_period=f"{p1}_TO_{p2}",
                inherited_fields=["entity", "metric", "period"],
                reason_codes=[ReasonCode.CROSS_TURN_CALCULATION_RESOLVED, ReasonCode.REFERENCE_RESOLVED],
            )

        # 5. General Ellipsis / Metric Inheritance
        active_e = entity or state.active_entity or "Apple"
        active_m = metric or state.active_metric or "Revenue"
        active_p = period or state.active_period or "FY2024"
        
        if not entity and state.active_entity:
            inherited_fields.append("entity")
        if not metric and state.active_metric:
            inherited_fields.append("metric")
        if not period and state.active_period:
            inherited_fields.append("period")
            
        standalone = f"What was {active_e} {active_p} {active_m}?"
        return ConversationResolution(
            supported=True,
            requires_context=bool(inherited_fields),
            standalone_query=standalone,
            resolved_entity=active_e,
            resolved_metric=active_m,
            resolved_period=active_p,
            inherited_fields=inherited_fields,
            explicit_fields=explicit_fields,
            reason_codes=[ReasonCode.ELLIPSIS_RESOLVED if inherited_fields else ReasonCode.NO_CONTEXT_REQUIRED],
        )
