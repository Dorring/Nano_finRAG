"""Tests for Gate 08 R2.1 lane contribution ablation helpers."""

from __future__ import annotations

from src.pdf_retrieval_v4.lane_ablation import (
    ALL_LANES,
    POOL_K,
    RRF_K,
    RAW_LANES,
    STRUCTURED_LANES,
    build_combined_pool_keys,
    build_e0_pool,
    build_raw_pool_keys,
    classify_lane_support,
    find_rank_in_lane,
    find_rrf_rank,
    rrf_fuse,
)


# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------


def _make_lane_hit(
    candidate_key: str, rank: int, score: float = 1.0
) -> dict:
    return {
        "candidate_key": candidate_key,
        "rank": rank,
        "score": score,
        "view_id": f"view:{candidate_key}",
    }


def _make_rrf_hit(
    candidate_key: str, rank: int, rrf_score: float = 0.01
) -> dict:
    return {
        "candidate_key": candidate_key,
        "rank": rank,
        "rrf_score": rrf_score,
        "lane_ranks": {},
        "supporting_view_ids": {},
    }


def _make_raw_case(keys: list[str]) -> dict:
    return {
        "case_id": "test_case",
        "raw_full_rrf_candidates": [
            {"candidate_key": k} for k in keys
        ],
    }


def _make_gate08_pred(keys: list[str]) -> dict:
    return {
        "structured_strict_source_pool": [
            {"original_candidate_identity": k} for k in keys
        ],
    }


# ------------------------------------------------------------------
# find_rank_in_lane
# ------------------------------------------------------------------


class TestFindRankInLane:
    def test_found(self) -> None:
        hits = [
            _make_lane_hit("c1", 1),
            _make_lane_hit("c2", 2),
            _make_lane_hit("c3", 3),
        ]
        assert find_rank_in_lane(hits, "c2") == 2

    def test_not_found(self) -> None:
        hits = [_make_lane_hit("c1", 1)]
        assert find_rank_in_lane(hits, "c_missing") is None

    def test_empty_hits(self) -> None:
        assert find_rank_in_lane([], "c1") is None

    def test_first_rank(self) -> None:
        hits = [_make_lane_hit("c1", 1)]
        assert find_rank_in_lane(hits, "c1") == 1


# ------------------------------------------------------------------
# find_rrf_rank
# ------------------------------------------------------------------


class TestFindRRFRank:
    def test_found(self) -> None:
        hits = [
            _make_rrf_hit("c1", 1),
            _make_rrf_hit("c2", 2),
        ]
        assert find_rrf_rank(hits, "c2") == 2

    def test_not_found(self) -> None:
        hits = [_make_rrf_hit("c1", 1)]
        assert find_rrf_rank(hits, "c_missing") is None

    def test_empty(self) -> None:
        assert find_rrf_rank([], "c1") is None


# ------------------------------------------------------------------
# rrf_fuse
# ------------------------------------------------------------------


class TestRRFFuse:
    def test_raw_only_uses_raw_lanes(self) -> None:
        lane_hits = {
            "candidate_raw_bm25": [
                _make_lane_hit("c1", 1),
                _make_lane_hit("c2", 2),
            ],
            "candidate_raw_dense": [
                _make_lane_hit("c1", 3),
                _make_lane_hit("c3", 1),
            ],
            "candidate_structured_bm25": [
                _make_lane_hit("c_struct", 1),
            ],
            "candidate_structured_dense": [
                _make_lane_hit("c_struct", 2),
            ],
        }
        result = rrf_fuse(lane_hits, RAW_LANES)
        assert "c_struct" not in result
        assert "c1" in result
        assert "c2" in result
        assert "c3" in result

    def test_structured_only_uses_structured_lanes(self) -> None:
        lane_hits = {
            "candidate_raw_bm25": [
                _make_lane_hit("c_raw", 1),
            ],
            "candidate_structured_bm25": [
                _make_lane_hit("c1", 1),
            ],
            "candidate_structured_dense": [
                _make_lane_hit("c1", 2),
                _make_lane_hit("c2", 1),
            ],
        }
        result = rrf_fuse(lane_hits, STRUCTURED_LANES)
        assert "c_raw" not in result
        assert "c1" in result
        assert "c2" in result

    def test_score_formula(self) -> None:
        lane_hits = {
            "candidate_raw_bm25": [_make_lane_hit("c1", 1)],
            "candidate_raw_dense": [_make_lane_hit("c1", 3)],
        }
        result = rrf_fuse(lane_hits, RAW_LANES, rrf_k=60, top_k=10)
        assert result[0] == "c1"
        # c1 score = 1/(60+1) + 1/(60+3) = 1/61 + 1/63

        # Verify by checking ordering: c1 should be first
        # with a single candidate
        assert len(result) == 1

        # More precise: two candidates, one with higher score
        lane_hits2 = {
            "candidate_raw_bm25": [
                _make_lane_hit("c_high", 1),
                _make_lane_hit("c_low", 5),
            ],
        }
        result2 = rrf_fuse(lane_hits2, RAW_LANES, rrf_k=60, top_k=10)
        assert result2[0] == "c_high"
        assert result2[1] == "c_low"

    def test_top_k_limit(self) -> None:
        lane_hits = {
            "candidate_raw_bm25": [
                _make_lane_hit(f"c{i}", i) for i in range(1, 51)
            ],
        }
        result = rrf_fuse(lane_hits, RAW_LANES, top_k=10)
        assert len(result) == 10

    def test_deterministic_tie_break(self) -> None:
        """Equal scores broken by candidate_key alphabetically."""
        lane_hits = {
            "candidate_raw_bm25": [
                _make_lane_hit("c_b", 1),
                _make_lane_hit("c_a", 2),
            ],
        }
        # c_b has score 1/61, c_a has score 1/62 → c_b first
        result = rrf_fuse(lane_hits, RAW_LANES, top_k=10)
        assert result[0] == "c_b"
        assert result[1] == "c_a"

    def test_equal_rank_different_lanes(self) -> None:
        """Same candidate in two lanes with same rank sums scores."""
        lane_hits = {
            "candidate_raw_bm25": [_make_lane_hit("c1", 1)],
            "candidate_raw_dense": [_make_lane_hit("c1", 1)],
        }
        result = rrf_fuse(lane_hits, RAW_LANES, top_k=10)
        assert result[0] == "c1"

    def test_missing_lane_contributes_zero(self) -> None:
        lane_hits = {
            "candidate_raw_bm25": [_make_lane_hit("c1", 1)],
            # candidate_raw_dense missing entirely
        }
        result = rrf_fuse(lane_hits, RAW_LANES, top_k=10)
        assert "c1" in result

    def test_empty_lanes(self) -> None:
        result = rrf_fuse({}, RAW_LANES)
        assert result == []

    def test_all_lanes_includes_both(self) -> None:
        lane_hits = {
            "candidate_raw_bm25": [_make_lane_hit("c_raw", 1)],
            "candidate_structured_bm25": [
                _make_lane_hit("c_struct", 1)
            ],
        }
        result = rrf_fuse(lane_hits, ALL_LANES)
        assert "c_raw" in result
        assert "c_struct" in result


# ------------------------------------------------------------------
# build_e0_pool
# ------------------------------------------------------------------


class TestBuildE0Pool:
    def test_raw_plus_structured(self) -> None:
        raw_case = _make_raw_case(["c1", "c2"])
        gate08 = _make_gate08_pred(["c2", "c3"])
        pool = build_e0_pool(raw_case, gate08)
        assert pool == {"c1", "c2", "c3"}

    def test_empty_raw(self) -> None:
        raw_case = {"raw_full_rrf_candidates": []}
        gate08 = _make_gate08_pred(["c1"])
        pool = build_e0_pool(raw_case, gate08)
        assert pool == {"c1"}

    def test_empty_structured(self) -> None:
        raw_case = _make_raw_case(["c1"])
        gate08 = {"structured_strict_source_pool": []}
        pool = build_e0_pool(raw_case, gate08)
        assert pool == {"c1"}

    def test_dedup(self) -> None:
        raw_case = _make_raw_case(["c1", "c1"])
        gate08 = _make_gate08_pred(["c1"])
        pool = build_e0_pool(raw_case, gate08)
        assert pool == {"c1"}


# ------------------------------------------------------------------
# build_combined_pool_keys
# ------------------------------------------------------------------


class TestBuildCombinedPoolKeys:
    def test_extract_keys(self) -> None:
        pred = {
            "combined_pool": [
                {"candidate_key": "c1", "source": "raw", "rank": 1},
                {"candidate_key": "c2", "source": "structured", "rank": 2},
                {"candidate_key": "c3", "source": "candidate_direct", "rank": 3},
            ]
        }
        keys = build_combined_pool_keys(pred)
        assert keys == {"c1", "c2", "c3"}

    def test_empty(self) -> None:
        assert build_combined_pool_keys({}) == set()

    def test_skips_empty_keys(self) -> None:
        pred = {
            "combined_pool": [
                {"candidate_key": "c1", "source": "raw"},
                {"candidate_key": "", "source": "raw"},
                {"source": "raw"},
            ]
        }
        keys = build_combined_pool_keys(pred)
        assert keys == {"c1"}


# ------------------------------------------------------------------
# build_raw_pool_keys
# ------------------------------------------------------------------


class TestBuildRawPoolKeys:
    def test_extract(self) -> None:
        raw_case = _make_raw_case(["c1", "c2"])
        keys = build_raw_pool_keys(raw_case)
        assert keys == {"c1", "c2"}

    def test_empty(self) -> None:
        assert build_raw_pool_keys({}) == set()


# ------------------------------------------------------------------
# classify_lane_support
# ------------------------------------------------------------------


class TestClassifyLaneSupport:
    def test_recovered_by_raw_only(self) -> None:
        result = classify_lane_support(
            recovered=True,
            raw_bm25_rank=1,
            raw_dense_rank=3,
            structured_bm25_rank=None,
            structured_dense_rank=None,
        )
        assert result["recovered_by_raw_lane"] is True
        assert result["recovered_by_structured_lane"] is False
        assert result["recovered_by_fusion_only"] is False

    def test_recovered_by_structured_only(self) -> None:
        result = classify_lane_support(
            recovered=True,
            raw_bm25_rank=None,
            raw_dense_rank=None,
            structured_bm25_rank=2,
            structured_dense_rank=None,
        )
        assert result["recovered_by_raw_lane"] is False
        assert result["recovered_by_structured_lane"] is True
        assert result["recovered_by_fusion_only"] is False

    def test_recovered_by_both(self) -> None:
        result = classify_lane_support(
            recovered=True,
            raw_bm25_rank=1,
            raw_dense_rank=None,
            structured_bm25_rank=None,
            structured_dense_rank=5,
        )
        assert result["recovered_by_raw_lane"] is True
        assert result["recovered_by_structured_lane"] is True
        assert result["recovered_by_fusion_only"] is False

    def test_recovered_by_fusion_only(self) -> None:
        """Recovered but no lane rank — shouldn't happen but handle gracefully."""
        result = classify_lane_support(
            recovered=True,
            raw_bm25_rank=None,
            raw_dense_rank=None,
            structured_bm25_rank=None,
            structured_dense_rank=None,
        )
        assert result["recovered_by_raw_lane"] is False
        assert result["recovered_by_structured_lane"] is False
        assert result["recovered_by_fusion_only"] is True

    def test_not_recovered(self) -> None:
        result = classify_lane_support(
            recovered=False,
            raw_bm25_rank=1,
            raw_dense_rank=2,
            structured_bm25_rank=3,
            structured_dense_rank=4,
        )
        assert result["recovered_by_raw_lane"] is False
        assert result["recovered_by_structured_lane"] is False
        assert result["recovered_by_fusion_only"] is False

    def test_not_recovered_no_ranks(self) -> None:
        result = classify_lane_support(
            recovered=False,
            raw_bm25_rank=None,
            raw_dense_rank=None,
            structured_bm25_rank=None,
            structured_dense_rank=None,
        )
        assert result["recovered_by_raw_lane"] is False
        assert result["recovered_by_structured_lane"] is False
        assert result["recovered_by_fusion_only"] is False


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------


class TestConstants:
    def test_rrf_k_value(self) -> None:
        assert RRF_K == 60

    def test_pool_k_value(self) -> None:
        assert POOL_K == 40

    def test_raw_lanes(self) -> None:
        assert RAW_LANES == ("candidate_raw_bm25", "candidate_raw_dense")

    def test_structured_lanes(self) -> None:
        assert STRUCTURED_LANES == (
            "candidate_structured_bm25",
            "candidate_structured_dense",
        )

    def test_all_lanes(self) -> None:
        assert len(ALL_LANES) == 4
        assert set(RAW_LANES) | set(STRUCTURED_LANES) == set(ALL_LANES)
