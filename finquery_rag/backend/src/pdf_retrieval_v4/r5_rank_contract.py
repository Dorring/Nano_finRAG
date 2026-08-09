"""Pure rank-contract helpers for Gate 08 R5.1 diagnostics."""

from __future__ import annotations


def cutoff_label(rank: int | None) -> str:
    if rank is None:
        return "absent"
    if rank <= 10:
        return "top10"
    if rank <= 20:
        return "11_to_20"
    if rank <= 40:
        return "21_to_40"
    if rank <= 50:
        return "41_to_50"
    return "beyond_50"


def classify_rank_migration(old_rank: int | None, new_rank: int | None) -> str:
    if old_rank is None and new_rank is None:
        return "still_missed"
    if old_rank is None:
        return f"new_entry_{cutoff_label(new_rank)}"
    if new_rank is None:
        return "dropped_out_top40" if old_rank <= 40 else "dropped_out_top50"
    if old_rank > 50 and new_rank <= 50:
        return f"new_entry_{cutoff_label(new_rank)}"
    if old_rank <= 50 < new_rank:
        return "dropped_out_top40" if old_rank <= 40 else "dropped_out_top50"
    if new_rank < old_rank:
        return "improved_within_top50"
    if new_rank > old_rank:
        return "worsened_within_top50"
    return "unchanged_within_top50"


def recovered_to_cutoff(old_rank: int | None, new_rank: int | None, cutoff: int) -> bool:
    return (old_rank is None or old_rank > cutoff) and new_rank is not None and new_rank <= cutoff
