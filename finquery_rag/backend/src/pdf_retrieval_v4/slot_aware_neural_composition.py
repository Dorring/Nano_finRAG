"""Preregistered slot-aware neural Top5 composition (not scored in R3.2)."""

from __future__ import annotations

from typing import Any


def compose_slot_aware_top5(
    main_ranking: list[dict[str, Any]],
    slot_rankings: dict[str, list[dict[str, Any]]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if top_k != 5:
        raise ValueError("slot_aware_top_k_must_equal_5")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for slot_id, ranking in slot_rankings.items():
        if not ranking:
            continue
        candidate = ranking[0]
        key = candidate["candidate_key"]
        if key not in seen:
            selected.append({**candidate, "selection_source": f"slot:{slot_id}"})
            seen.add(key)
    for candidate in main_ranking:
        if len(selected) >= top_k:
            break
        key = candidate["candidate_key"]
        if key not in seen:
            selected.append({**candidate, "selection_source": "main_residual"})
            seen.add(key)
    return [{**item, "final_rank": rank} for rank, item in enumerate(selected, 1)]
