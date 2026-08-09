from __future__ import annotations

import pytest

from src.pdf_retrieval_v4.bounded_candidate_selector import (
    CANDIDATE_BUDGET,
    RRF_K,
    SLOT_CANDIDATE_HORIZON,
    SLOT_MIN_BUDGET,
    build_raw_family,
    build_structured_family,
    fuse_ranked_families,
    select_multi_slot_top50,
    select_single_slot_top50,
)


def hits(prefix: str, count: int) -> list[dict[str, object]]:
    return [{"candidate_key": f"{prefix}{rank}", "rank": rank} for rank in range(1, count + 1)]


def test_r7_pool_position_not_used_as_rank() -> None:
    fused = fuse_ranked_families(
        {"lane": [{"candidate_key": "a", "pool_position": 1, "rank": 9}]}
    )
    assert fused[0]["lane_ranks"] == {"lane": 9}


def test_production_raw_source_rank_reconstructed() -> None:
    family = build_raw_family(
        [{"candidate_key": "a", "stage_rank": 7}],
        [{"candidate_key": "b", "rank": 1}],
    )
    by_key = {item["candidate_key"]: item for item in family}
    assert by_key["a"]["lane_ranks"]["production_raw"] == 7


def test_families_each_have_one_top_level_vote() -> None:
    raw = build_raw_family(hits("raw", 2), [])
    structured = build_structured_family(hits("structured", 2), hits("metric", 2))
    selected = select_single_slot_top50(raw, structured)
    assert set(selected[0]["lane_ranks"]) <= {"raw_family", "structured_family"}


def test_metric_lane_does_not_add_top_level_vote() -> None:
    structured = build_structured_family([], [{"candidate_key": "metric", "rank": 1}])
    selected = select_single_slot_top50([], structured)
    assert selected[0]["lane_ranks"] == {"structured_family": 1}


def test_cross_family_candidate_dedup() -> None:
    selected = select_single_slot_top50(
        [{"candidate_key": "same", "rank": 1}],
        [{"candidate_key": "same", "rank": 1}],
    )
    assert len(selected) == 1
    assert selected[0]["lane_ranks"] == {"raw_family": 1, "structured_family": 1}


def test_single_slot_output_at_most_50() -> None:
    assert len(select_single_slot_top50(hits("raw", 70), hits("s", 70))) == 50


def test_multislot_output_at_most_50_and_minimum_coverage() -> None:
    pool, audit = select_multi_slot_top50(
        {"left": hits("left", 50), "right": hits("right", 50)}, hits("main", 50)
    )
    assert len(pool) == 50
    assert audit["slot_coverage"] == {"left": 10, "right": 10}
    assert audit["minimum_coverage_available"] is True


def test_multislot_shared_candidate_counts_for_both_slots() -> None:
    shared = [{"candidate_key": f"shared{i}", "rank": i} for i in range(1, 11)]
    _, audit = select_multi_slot_top50({"left": shared, "right": shared}, [])
    assert audit["slot_coverage"] == {"left": 10, "right": 10}


def test_selector_deterministic_candidate_key_tie_break() -> None:
    first = select_single_slot_top50(hits("z", 1), hits("a", 1))
    second = select_single_slot_top50(hits("z", 1), hits("a", 1))
    assert first == second
    assert [item["candidate_key"] for item in first] == ["a1", "z1"]


def test_fixed_contract_constants() -> None:
    assert RRF_K == 60
    assert CANDIDATE_BUDGET == 50
    assert SLOT_CANDIDATE_HORIZON == 50
    assert SLOT_MIN_BUDGET == 10


def test_rrf_k_cannot_be_scanned() -> None:
    with pytest.raises(ValueError, match="rrf_k_must_equal_60"):
        fuse_ranked_families({"lane": hits("x", 1)}, rrf_k=30)
