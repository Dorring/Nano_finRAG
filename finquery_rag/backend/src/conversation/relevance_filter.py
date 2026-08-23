"""Context Relevance Filter.

Performs dynamic noise reduction on multi-turn dialogue history
by evaluating multi-factor relevance scores against current query,
dialogue state, and recent turns.
"""

from __future__ import annotations

import re
from typing import Sequence

from .contracts import DialogueState, DialogueTurn


class ContextRelevanceFilter:
    """Filters dialogue history to retain only relevant context turns."""

    def filter_turns(
        self,
        current_query: str,
        dialogue_state: DialogueState | None,
        turns: Sequence[DialogueTurn],
    ) -> list[DialogueTurn]:
        """Filters turns based on relevance signals from query, state, and history."""
        if not turns:
            return []

        q_lower = current_query.lower()
        active_e = (dialogue_state.active_entity.lower() if dialogue_state and dialogue_state.active_entity else "")
        active_m = (dialogue_state.active_metric.lower() if dialogue_state and dialogue_state.active_metric else "")
        active_topic = (dialogue_state.active_topic.lower() if dialogue_state and dialogue_state.active_topic else "")
        referenced_ids = set(dialogue_state.referenced_turn_ids if dialogue_state else [])

        # Detect if current query explicitly introduces a new entity (Topic Switch)
        new_entity_detected = False
        for cand in ["apple", "aapl", "microsoft", "msft", "tesla", "tsla", "google", "googl", "amazon", "amzn", "coca-cola", "ko", "oracle", "orcl"]:
            if re.search(rf"\b{re.escape(cand)}\b", q_lower):
                if active_e and cand != active_e:
                    new_entity_detected = True
                    break

        scored_turns: list[tuple[float, DialogueTurn]] = []
        total_turns = len(turns)

        for i, turn in enumerate(turns):
            turn_q = turn.user_query.lower()
            score = 0.0

            # 1. Explicit Reference Match (Highest Priority)
            if turn.turn_id in referenced_ids:
                score += 10.0

            # 2. Recency Scoring (Recent turns get base boost)
            recency_rank = total_turns - 1 - i
            if recency_rank == 0:
                score += 4.0
            elif recency_rank == 1:
                score += 3.0
            elif recency_rank == 2:
                score += 2.0
            elif recency_rank < 5:
                score += 1.0

            # 3. Entity & Metric Overlap
            if active_e and active_e in turn_q:
                score += 3.0
            if active_m and active_m in turn_q:
                score += 2.5
            if active_topic and turn.topic and active_topic in turn.topic.lower():
                score += 2.0

            # 4. Chit-chat / Noise Penalty
            if len(turn_q.split()) < 3 and any(w in turn_q for w in ["thanks", "thank you", "ok", "hello", "hi", "good"]):
                score -= 5.0

            # 5. Topic Switch Penalty (Penalize older turns when new entity is introduced)
            if new_entity_detected and active_e and active_e in turn_q:
                score -= 6.0

            if score > 0:
                scored_turns.append((score, turn))

        # Sort by score descending, keep top 4-6 most relevant
        scored_turns.sort(key=lambda x: x[0], reverse=True)
        selected = [t for _, t in scored_turns[:4]]
        
        # Re-sort chronologically by original order in turns
        selected_ids = {t.turn_id for t in selected}
        return [t for t in turns if t.turn_id in selected_ids]
