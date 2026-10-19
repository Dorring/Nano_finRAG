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
    PendingClarification,
    ReasonCode,
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

        # 3. Resolve contextual query. A short answer to a pending
        # clarification is handled at this lifecycle boundary so the
        # historical resolver algorithm remains unchanged.
        resolution = self._resolve_pending_choice(user_query, state)
        if resolution is None:
            resolution = self.resolver.resolve(user_query, state, budgeted_turns)

        # 4. Every supported user request is one logical conversation turn.
        # Clarification is a real user turn, but its semantic fields remain
        # pending rather than being guessed into active state.
        if resolution.supported:
            state.turn_count += 1
            if resolution.clarification_required:
                self._set_pending_clarification(state, user_query, resolution)
            else:
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
        if not entity:
            m_comp = re.search(r"([\u4e00-\u9fff]{2,10}(?:集团|公司|科技|股份|控股|汽车|重工|生物|医药|银行|证券|保险|百货|物流))", query)
            if m_comp:
                entity = m_comp.group(1)
                
        metrics = []
        q_temp = q_lower
        # Check longest metric names first, replacing matched substrings to prevent sub-phrase duplicates
        for m in ALL_METRIC_NAMES:
            if m in q_temp:
                norm_m = "Capital Expenditures" if "capital expenditure" in m else m.title()
                metrics.append(norm_m)
                q_temp = q_temp.replace(m, " " * len(m))
                
        period = None
        p_match = re.search(r"\b(fy\s*20\d\d|20\d\d|q[1-4]\s*20\d\d|q[1-4])\b|(\d{2})年度?|20(\d{2})年?", q_lower)
        if p_match:
            if p_match.group(1):
                period = p_match.group(1).upper()
            elif p_match.group(2):
                period = f"FY20{p_match.group(2)}"
            elif p_match.group(3):
                period = f"FY20{p_match.group(3)}"
            
        return entity, metrics, period

    def _resolve_pending_choice(
        self,
        user_query: str,
        state: DialogueState,
    ) -> ConversationResolution | None:
        """Resolve a semantic choice without changing the resolver algorithm."""
        pending = state.pending_clarification
        if pending is None and state.last_clarification is not None:
            # Detect backtracking intent to switch or re-select from previous clarification
            clean_q = user_query.strip().lower()
            backtrack_keywords = [
                "换成", "改成", "换为", "切换", "不对", "其实是", "重新选",
                "看另外", "选另外", "换选项", "选第", "看第", "查第",
                "switch to", "change to", "instead", "rather", "option", "choice",
            ]
            is_backtrack = (
                clean_q.isdigit()
                or any(kw in clean_q for kw in backtrack_keywords)
                or bool(re.search(r"(?:第|选项)\s*[1-9一二三四五六七八九]", user_query))
                or bool(re.search(r"\b(?:first|second|third|fourth|1st|2nd|3rd|4th)\b", clean_q))
            )
            if is_backtrack:
                pending = state.last_clarification

        if pending is None:
            return None
        entity, metrics, period = self._extract_semantic_entities(user_query)
        # If user explicitly specifies a different entity, route to full resolver
        if entity and pending.entity and entity.upper() != pending.entity.upper():
            return None

        # If user directly answers the pending clarification with period or metric:
        if not entity and (period or metrics):
            selected_metric = metrics[0] if metrics else (pending.topic or state.active_metric or "Revenue")
            target_entity = pending.entity or state.active_entity
            target_period = period or pending.period or state.active_period or "FY2025"
            if target_entity:
                standalone = f"What was {target_entity} {target_period} {selected_metric}?"
                inherited = []
                if not entity:
                    inherited.append("entity")
                if not metrics:
                    inherited.append("metric")
                if not period:
                    inherited.append("period")
                return ConversationResolution(
                    supported=True,
                    requires_context=True,
                    standalone_query=standalone,
                    resolved_entity=target_entity,
                    resolved_metric=selected_metric,
                    resolved_period=target_period,
                    inherited_fields=inherited,
                    explicit_fields=["period"] if period else ["metric"],
                    reason_codes=[
                        ReasonCode.PENDING_CLARIFICATION_RESOLVED,
                        ReasonCode.REFERENCE_RESOLVED,
                    ],
                )
        selected_metric = None
        if metrics:
            candidates = {
                value.strip().lower() for value in pending.candidates
            }
            if metrics[0].lower() in candidates:
                selected_metric = metrics[0]
        if selected_metric is None and user_query.strip().lower() == "both":
            if any(value.strip().lower() == "both" for value in pending.candidates):
                selected_metric = "Revenue and Operating Margin"
        if selected_metric is None:
            coord_res = self._resolve_coordinate_choice(user_query, pending, state)
            if coord_res is not None:
                return coord_res
            clean_q = user_query.strip().lower()
            if clean_q.isdigit() and pending.candidates:
                val_idx = int(clean_q)
                if val_idx < 1 or val_idx > len(pending.candidates):
                    return ConversationResolution(
                        supported=True,
                        requires_context=True,
                        standalone_query="",
                        ambiguity_detected=True,
                        clarification_required=True,
                        clarification_question=(
                            f"您输入的序号 [{clean_q}] 超出候选范围（当前共 {len(pending.candidates)} 个选项，"
                            f"有效序号为 1 ~ {len(pending.candidates)}）。请重新输入有效序号、具体数值或所在行特征关键词。"
                        ),
                        clarification_options=list(pending.candidates),
                        reason_codes=[ReasonCode.PENDING_CLARIFICATION_UNRESOLVED],
                    )
            return None
        entity = pending.entity or state.active_entity
        period = pending.period or state.active_period
        if not entity or not period:
            # A pending choice without enough semantic coordinates must not
            # invent an entity or period to make the financial query run.
            return None
        standalone = f"What was {entity} {period} {selected_metric}?"
        return ConversationResolution(
            supported=True,
            requires_context=True,
            standalone_query=standalone,
            resolved_entity=entity,
            resolved_metric=selected_metric,
            resolved_period=period,
            inherited_fields=["entity", "period"],
            explicit_fields=["metric"],
            reason_codes=[
                ReasonCode.PENDING_CLARIFICATION_RESOLVED,
                ReasonCode.REFERENCE_RESOLVED,
            ],
        )

    @staticmethod
    def _extract_candidate_value(cand_str: str) -> str:
        m = re.search(r"数值\s+([^\s(]+|\([^)]+\))", cand_str)
        if m:
            return m.group(1).strip()
        parts = cand_str.split()
        return parts[0].strip() if parts else cand_str.strip()

    @staticmethod
    def _extract_candidate_page(cand_str: str) -> str | None:
        m = re.search(r"第\s*(\d+)\s*页", cand_str)
        if m:
            return m.group(1)
        m_en = re.search(r"\bpage\s*(\d+)\b", cand_str, re.IGNORECASE)
        if m_en:
            return m_en.group(1)
        return None

    @staticmethod
    def _extract_candidate_context(cand_str: str) -> str:
        m = re.search(r"\[上下文:\s*(.*?)\]", cand_str)
        if m:
            return m.group(1).strip()
        return ""

    def _resolve_coordinate_choice(
        self,
        user_query: str,
        pending: PendingClarification,
        state: DialogueState,
    ) -> ConversationResolution | None:
        if not pending.candidates:
            return None

        candidates = pending.candidates
        matched_idx: int | None = None
        clean_q = user_query.strip().lower()

        # 1. Match by pure digit index: e.g. "1", "2", "3"
        if clean_q.isdigit():
            val_idx = int(clean_q)
            if 1 <= val_idx <= len(candidates):
                matched_idx = val_idx - 1

        # 2. Match by English ordinal words: first, second, third, 1st, 2nd, etc.
        if matched_idx is None:
            en_word_map = {
                "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
            }
            for word, idx_val in en_word_map.items():
                if re.search(rf"\b{word}\b", clean_q):
                    if 1 <= idx_val <= len(candidates):
                        matched_idx = idx_val - 1
                        break

        # 3. Match by English patterns: "option 1", "choice 2", "item 3"
        if matched_idx is None:
            m_en = re.search(r"\b(?:option|choice|item|selection|number)\s*([1-9])\b", clean_q)
            if m_en:
                val_idx = int(m_en.group(1))
                if 1 <= val_idx <= len(candidates):
                    matched_idx = val_idx - 1

        # 4. Match by Chinese patterns like "第1项", "第一项", "选项 1", "选项1", "第 1 个"
        if matched_idx is None:
            cn_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
            m_cn = re.search(r"(?:第|选项)\s*([1-9一二三四五六七八九])\s*(?:项|个)?", user_query)
            if m_cn:
                raw_token = m_cn.group(1)
                val_idx = cn_map.get(raw_token) if raw_token in cn_map else int(raw_token)
                if 1 <= val_idx <= len(candidates):
                    matched_idx = val_idx - 1

        # 5. Match by candidate value
        if matched_idx is None:
            # Sort candidate values by length of digits descending to prevent substring collision (e.g. 10 matching 100)
            cand_vals: list[tuple[int, str, str, bool]] = []
            for idx, cand in enumerate(candidates):
                val = self._extract_candidate_value(cand)
                if not val:
                    continue
                digits = re.sub(r"[^\d]", "", val)
                is_neg = "(" in val or "-" in val
                cand_vals.append((idx, val, digits, is_neg))

            cand_vals.sort(key=lambda x: len(x[2]), reverse=True)

            is_user_neg = "-" in clean_q or "(" in clean_q

            for idx, val, digits, is_neg in cand_vals:
                # Exact match against user input
                if val.lower() == clean_q:
                    matched_idx = idx
                    break

                # If user input specified negative, skip positive candidates
                if is_user_neg and not is_neg:
                    continue

                # Match full number with digit boundaries
                if digits and re.search(rf"(?<!\d){re.escape(digits)}(?!\d)", clean_q):
                    matched_idx = idx
                    break

        # 6. Match by page specification: e.g. "第43页那项", "第43页", "page 43"
        if matched_idx is None:
            m_page = re.search(r"(?:第\s*(\d+)\s*页|\bpage\s*(\d+)\b)", clean_q)
            if m_page:
                target_page = m_page.group(1) or m_page.group(2)
                page_matches = [
                    idx for idx, cand in enumerate(candidates)
                    if self._extract_candidate_page(cand) == target_page
                ]
                if len(page_matches) == 1:
                    matched_idx = page_matches[0]

        # 7. Match by candidate context keywords / row content
        if matched_idx is None:
            ctx_scores: list[tuple[int, int]] = []
            for idx, cand in enumerate(candidates):
                ctx = self._extract_candidate_context(cand).lower()
                if not ctx:
                    continue
                cells = [c.strip() for c in ctx.split("|") if c.strip()]
                score = 0
                for cell in cells:
                    cell_clean = re.sub(r"[^\w\d]", "", cell)
                    if len(cell_clean) >= 2 and cell_clean in clean_q:
                        score += len(cell_clean)
                    elif len(cell) >= 2 and cell.lower() in clean_q:
                        score += len(cell)
                if score > 0:
                    ctx_scores.append((idx, score))

            if ctx_scores:
                ctx_scores.sort(key=lambda x: x[1], reverse=True)
                if len(ctx_scores) == 1 or ctx_scores[0][1] > ctx_scores[1][1]:
                    matched_idx = ctx_scores[0][0]

        if matched_idx is None:
            return None

        selected_val = self._extract_candidate_value(candidates[matched_idx])
        entity = pending.entity or state.active_entity
        period = pending.period or state.active_period
        metric = pending.topic or state.active_metric

        if not entity or not period or not metric:
            return None

        standalone = f"According to {entity}'s {period} filing, what was the reported {metric}? Target value for {metric}: {selected_val}."
        return ConversationResolution(
            supported=True,
            requires_context=True,
            standalone_query=standalone,
            resolved_entity=entity,
            resolved_metric=metric,
            resolved_period=period,
            inherited_fields=["entity", "period", "metric"],
            explicit_fields=["coordinate_value"],
            reason_codes=[
                ReasonCode.PENDING_CLARIFICATION_RESOLVED,
                ReasonCode.REFERENCE_RESOLVED,
            ],
        )

    @staticmethod
    def _previous_period(period: str | None) -> str | None:
        if not period:
            return period
        match = re.search(r"20\d\d", period)
        if not match:
            return period
        previous = str(int(match.group(0)) - 1)
        return period[: match.start()] + previous + period[match.end() :]

    def _set_pending_clarification(
        self,
        state: DialogueState,
        raw_user_query: str,
        res: ConversationResolution,
    ) -> None:
        """Persist only semantic ambiguity coordinates for a later answer."""
        codes = [
            str(getattr(code, "value", code))
            for code in (res.reason_codes or [])
        ]
        unresolved_fields = []
        for code, field_name in (
            (ReasonCode.AMBIGUOUS_METRIC, "metric"),
            (ReasonCode.AMBIGUOUS_ENTITY, "entity"),
            (ReasonCode.AMBIGUOUS_PERIOD, "period"),
        ):
            if code in codes:
                unresolved_fields.append(field_name)
        if not unresolved_fields:
            unresolved_fields = ["reference"]

        period = state.active_period
        if re.search(
            r"\b(previous|prior|preceding|last)\s+(year|period|fy)\b",
            raw_user_query.lower(),
        ):
            period = self._previous_period(period)

        parsed_e, parsed_m_list, parsed_p = self._extract_semantic_entities(res.standalone_query or raw_user_query)
        ent = res.resolved_entity or parsed_e
        if ent:
            if state.active_entity and ent.upper() != state.active_entity.upper():
                state.comparison_entity = state.active_entity
                state.active_entity = ent
            elif not state.active_entity:
                state.active_entity = ent

        state.pending_clarification = PendingClarification(
            reason_codes=codes,
            candidates=list(res.clarification_options or []),
            unresolved_fields=unresolved_fields,
            source_turn_id=f"turn_{state.turn_count}",
            entity=state.active_entity,
            period=period,
            topic=state.active_topic,
        )

    def _update_state_on_resolution(
        self,
        state: DialogueState,
        raw_user_query: str,
        res: ConversationResolution,
    ) -> None:
        """Updates dialogue state following a successful query resolution."""
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
        if state.pending_clarification:
            state.last_clarification = state.pending_clarification
        state.pending_clarification = None
        if turn_id not in state.referenced_turn_ids:
            state.referenced_turn_ids.append(turn_id)

        # Update compressed summary if turn threshold reached
        compressed = self.budget_manager.update_compressed_history(state, state.recent_turns)
        if compressed:
            state.compressed_history = compressed
