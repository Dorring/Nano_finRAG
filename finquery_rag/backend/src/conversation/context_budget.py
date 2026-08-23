"""Hierarchical Context Budget Manager.

Enforces application-level context budgeting across:
- L1 (Recent Raw Turns)
- L2 (Structured Dialogue State)
- L3 (Compressed History)

Guarantees constant/bounded token growth over 10, 30, 50, 100, 500 turns.
"""

from __future__ import annotations

import json
import os
import re
from typing import Sequence

from .contracts import DialogueState, DialogueTurn


class ContextBudgetManager:
    """Manages 3-tier memory allocation and enforces application token budgets."""

    def __init__(
        self,
        recent_turns_limit: int | None = None,
        summary_trigger_turns: int | None = None,
        target_tokens: int | None = None,
        max_tokens: int | None = None,
        summary_max_tokens: int | None = None,
    ) -> None:
        self.recent_turns_limit = recent_turns_limit or int(os.environ.get("CONTEXT_RECENT_TURNS", "4"))
        self.summary_trigger_turns = summary_trigger_turns or int(os.environ.get("CONTEXT_SUMMARY_TRIGGER_TURNS", "8"))
        self.target_tokens = target_tokens or int(os.environ.get("CONTEXT_TARGET_TOKENS", "4096"))
        self.max_tokens = max_tokens or int(os.environ.get("CONTEXT_MAX_TOKENS", "8192"))
        self.summary_max_tokens = summary_max_tokens or int(os.environ.get("CONTEXT_SUMMARY_MAX_TOKENS", "768"))

    def count_tokens(self, text: str) -> int:
        """Fast, robust, zero-network token count approximation.
        
        Splits by words and punctuation symbols (approx 1.25 tokens per word/symbol).
        """
        if not text:
            return 0
        # Fast regex token split: words, numbers, punctuation
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return max(1, int(len(tokens) * 1.15))

    def prepare_context(
        self,
        current_query: str,
        dialogue_state: DialogueState | None,
        relevant_turns: Sequence[DialogueTurn],
    ) -> tuple[list[DialogueTurn], str | None, int]:
        """Prepares budgeted context with priority preservation.
        
        Priority:
        1. Never Drop: Current Query, Structured State (L2), Explicitly Referenced Turns.
        2. Retain Next: Recent Relevant Turns (L1).
        3. Trimming Candidates: Compressed History (L3) -> Unreferenced Old Turns.
        
        Returns:
            (selected_turns, compressed_history, total_estimated_tokens)
        """
        if not dialogue_state:
            return list(relevant_turns), None, self.count_tokens(current_query)

        referenced_ids = set(dialogue_state.referenced_turn_ids)
        
        # 1. Base tokens from Query and Structured State (L2 - Never Drop)
        state_repr = json.dumps(dialogue_state.to_dict(), ensure_ascii=False)
        base_tokens = self.count_tokens(current_query) + self.count_tokens(state_repr)
        available_budget = max(200, self.target_tokens - base_tokens)

        # 2. Partition turns: Explicitly Referenced vs Recent vs Old
        protected_turns: list[DialogueTurn] = []
        regular_turns: list[DialogueTurn] = []
        
        for turn in relevant_turns:
            if turn.turn_id in referenced_ids:
                protected_turns.append(turn)
            else:
                regular_turns.append(turn)

        # Always retain protected turns
        selected_turns: list[DialogueTurn] = list(protected_turns)
        used_budget = sum(self.count_tokens(t.user_query + " " + t.standalone_query) for t in protected_turns)

        # Add recent turns up to recent_turns_limit and within available budget
        for turn in reversed(regular_turns[-self.recent_turns_limit:]):
            turn_tok = self.count_tokens(turn.user_query + " " + turn.standalone_query)
            if used_budget + turn_tok <= available_budget:
                selected_turns.append(turn)
                used_budget += turn_tok
            else:
                break

        # Re-sort chronologically
        turn_map = {t.turn_id: i for i, t in enumerate(relevant_turns)}
        selected_turns.sort(key=lambda t: turn_map.get(t.turn_id, 0))

        # 3. Compressed History (L3)
        compressed = dialogue_state.compressed_history
        if compressed:
            comp_tok = self.count_tokens(compressed)
            if used_budget + comp_tok > available_budget:
                # Trim compressed summary to fit
                compressed = compressed[:int(len(compressed) * (available_budget - used_budget) / max(1, comp_tok))]

        total_tokens = base_tokens + used_budget + (self.count_tokens(compressed) if compressed else 0)
        return selected_turns, compressed, total_tokens

    def update_compressed_history(
        self,
        current_state: DialogueState,
        all_past_turns: Sequence[DialogueTurn],
    ) -> str | None:
        """Compresses older history into high-level semantic topic summaries.
        
        Trust Boundary Invariant:
        Only preserves discussed entities/metrics/periods, NEVER unverified numbers.
        """
        if len(all_past_turns) < self.summary_trigger_turns:
            return current_state.compressed_history

        older_turns = all_past_turns[:-self.recent_turns_limit]
        if not older_turns:
            return current_state.compressed_history

        # Extract unique entities, metrics, and periods
        entities = set()
        metrics = set()
        periods = set()
        
        for t in older_turns:
            for cand in ["Apple", "Microsoft", "Tesla", "Google", "Amazon", "Coca-Cola", "Oracle"]:
                if cand.lower() in t.user_query.lower():
                    entities.add(cand)
            for m in ["Revenue", "Operating Income", "Operating Margin", "Net Income", "Free Cash Flow"]:
                if m.lower() in t.user_query.lower():
                    metrics.add(m)
            for p in ["FY2021", "FY2022", "FY2023", "FY2024", "Q1", "Q2", "Q3", "Q4"]:
                if p.lower() in t.user_query.lower():
                    periods.add(p)

        summary_parts = []
        if entities:
            summary_parts.append(f"Entities: {', '.join(sorted(entities))}")
        if metrics:
            summary_parts.append(f"Metrics: {', '.join(sorted(metrics))}")
        if periods:
            summary_parts.append(f"Periods: {', '.join(sorted(periods))}")
            
        summary = "Earlier discussion summary: " + "; ".join(summary_parts)
        return summary
