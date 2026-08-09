from __future__ import annotations

from src.pdf_retrieval_v4.r5_rank_contract import classify_rank_migration, cutoff_label, recovered_to_cutoff
from src.pdf_retrieval_v4.slot_aware_candidate_composer import FINAL_POOL_K, RRF_K, SLOT_CANDIDATE_HORIZON, SLOT_MIN_BUDGET, compose_slot_candidates


def test_rank_cutoff_contract() -> None:
    assert cutoff_label(50) == "41_to_50"
    assert cutoff_label(51) == "beyond_50"
    assert classify_rank_migration(None, 99) == "new_entry_beyond_50"
    assert classify_rank_migration(None, 50) == "new_entry_41_to_50"
    assert classify_rank_migration(None, 51) == "new_entry_beyond_50"
    assert recovered_to_cutoff(None, 40, 40)
    assert not recovered_to_cutoff(None, 41, 40)
    assert recovered_to_cutoff(None, 50, 50)


def test_slot_composer_contract() -> None:
    rankings = {
        "a": [{"candidate_key": f"c{i}"} for i in range(40)],
        "b": [{"candidate_key": f"c{i}"} for i in range(5, 45)],
    }
    pool, audit = compose_slot_candidates(rankings)
    assert RRF_K == 60
    assert FINAL_POOL_K == SLOT_CANDIDATE_HORIZON == 40
    assert SLOT_MIN_BUDGET == 10
    assert len(pool) == 40
    assert len({item["candidate_key"] for item in pool}) == 40
    assert all(value >= 10 for value in audit["slot_coverage"].values())
    assert any(set(item["supporting_slots"]) == {"a", "b"} for item in pool)


def test_slot_composer_deterministic() -> None:
    rankings = {"a": [{"candidate_key": f"a{i}"} for i in range(40)], "b": [{"candidate_key": f"b{i}"} for i in range(40)]}
    first, _ = compose_slot_candidates(rankings)
    second, _ = compose_slot_candidates(rankings)
    assert first == second
