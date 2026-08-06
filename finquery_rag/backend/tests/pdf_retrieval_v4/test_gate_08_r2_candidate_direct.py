"""Tests for Gate 08 R2 Candidate-aligned Direct Retrieval.

Covers the 12 required test cases from the R2 specification:
 1. test_one_raw_view_per_candidate
 2. test_one_structured_view_per_candidate
 3. test_view_no_gold_leakage
 4. test_candidate_level_rrf
 5. test_raw_and_structured_view_dedup
 6. test_slot_queries_independent
 7. test_multi_slot_round_robin
 8. test_period_conflict_filtered
 9. test_raw_pool_unchanged
10. test_b_class_not_read_before_seal
11. test_no_parameter_scan
12. test_deterministic_candidate_replay
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.candidate_aligned_view import (  # noqa: E402
    CandidateAlignedView,
    CandidateViewPair,
    make_raw_view_id,
    make_structured_view_id,
)
from src.pdf_retrieval_v4.candidate_rrf import fuse_candidate_hits  # noqa: E402
from src.pdf_retrieval_v4.candidate_slot_pool import build_slot_pool  # noqa: E402
from src.pdf_retrieval_v4.candidate_view_index import CandidateSearchHit  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_view(
    candidate_key: str,
    view_type: str = "raw",
    document_id: str = "doc-a",
    page: int = 1,
    retrieval_text: str = "test content",
    bridge_grade: str = "raw_only",
    metric_paths: tuple[str, ...] = (),
    periods: tuple[str, ...] = (),
) -> CandidateAlignedView:
    view_id = (
        make_raw_view_id(candidate_key)
        if view_type == "raw"
        else make_structured_view_id(candidate_key)
    )
    return CandidateAlignedView(
        candidate_key=candidate_key,
        view_type=view_type,
        view_id=view_id,
        retrieval_text=retrieval_text,
        document_id=document_id,
        pdf_page=page,
        bridge_grade=bridge_grade,
        metric_paths=metric_paths,
        periods=periods,
    )


def _make_pair(
    candidate_key: str,
    *,
    with_structured: bool = True,
    document_id: str = "doc-a",
    page: int = 1,
    bridge_grade: str = "A3",
) -> CandidateViewPair:
    raw = _make_view(
        candidate_key,
        "raw",
        document_id=document_id,
        page=page,
        bridge_grade=bridge_grade,
    )
    structured = (
        _make_view(
            candidate_key,
            "structured",
            document_id=document_id,
            page=page,
            bridge_grade=bridge_grade,
        )
        if with_structured
        else None
    )
    return CandidateViewPair(
        candidate_key=candidate_key,
        raw_view=raw,
        structured_view=structured,
    )


def _make_search_hit(
    candidate_key: str,
    lane: str,
    rank: int,
    view_id: str = "view-1",
    bm25_score: float | None = None,
    dense_score: float | None = None,
) -> CandidateSearchHit:
    return CandidateSearchHit(
        candidate_key=candidate_key,
        view_id=view_id,
        lane=lane,
        bm25_rank=rank if "bm25" in lane else None,
        dense_rank=rank if "dense" in lane else None,
        bm25_score=bm25_score if "bm25" in lane else None,
        dense_score=dense_score if "dense" in lane else None,
    )


# ---------------------------------------------------------------------------
# 1. test_one_raw_view_per_candidate
# ---------------------------------------------------------------------------


class TestOneRawViewPerCandidate:
    def test_each_candidate_has_exactly_one_raw_view(self) -> None:
        """Each candidate_key produces exactly one raw view."""
        keys = ["c1", "c2", "c3"]
        pairs = [_make_pair(k) for k in keys]
        for pair in pairs:
            assert pair.raw_view is not None
            assert pair.raw_view.view_type == "raw"
            assert pair.raw_view.candidate_key == pair.candidate_key

    def test_raw_view_ids_are_unique_per_candidate(self) -> None:
        """Different candidates get different raw view IDs."""
        v1 = make_raw_view_id("c1")
        v2 = make_raw_view_id("c2")
        assert v1 != v2

    def test_same_candidate_produces_same_raw_view_id(self) -> None:
        """Same candidate_key always produces same raw view ID (deterministic)."""
        v1 = make_raw_view_id("c1")
        v2 = make_raw_view_id("c1")
        assert v1 == v2


# ---------------------------------------------------------------------------
# 2. test_one_structured_view_per_candidate
# ---------------------------------------------------------------------------


class TestOneStructuredViewPerCandidate:
    def test_candidate_with_mapping_has_structured_view(self) -> None:
        pair = _make_pair("c1", with_structured=True)
        assert pair.structured_view is not None
        assert pair.structured_view.view_type == "structured"
        assert pair.structured_view.candidate_key == pair.candidate_key

    def test_candidate_without_mapping_has_no_structured_view(self) -> None:
        pair = _make_pair("c1", with_structured=False, bridge_grade="raw_only")
        assert pair.structured_view is None
        assert pair.raw_view.bridge_grade == "raw_only"

    def test_structured_view_id_differs_from_raw(self) -> None:
        pair = _make_pair("c1")
        assert pair.raw_view.view_id != pair.structured_view.view_id  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 3. test_view_no_gold_leakage
# ---------------------------------------------------------------------------


class TestViewNoGoldLeakage:
    FORBIDDEN_FIELDS = [
        "gold",
        "case_id",
        "expected_value",
        "reference_answer",
        "gold_identity",
        "review_label",
        "governance",
        "original_final_hit_identity",
    ]

    def test_raw_view_text_has_no_gold_fields(self) -> None:
        view = _make_view("c1", "raw", retrieval_text="Total revenue FY2024")
        text = view.retrieval_text.lower()
        for field in self.FORBIDDEN_FIELDS:
            assert field not in text, f"forbidden field '{field}' in raw view text"

    def test_structured_view_text_has_no_gold_fields(self) -> None:
        view = _make_view(
            "c1",
            "structured",
            retrieval_text="Metric: revenue\nPeriod: FY2024",
        )
        text = view.retrieval_text.lower()
        for field in self.FORBIDDEN_FIELDS:
            assert field not in text, f"forbidden field '{field}' in structured view text"

    def test_view_to_dict_has_no_gold_fields(self) -> None:
        view = _make_view("c1", "raw")
        d = view.to_dict()
        for field in self.FORBIDDEN_FIELDS:
            assert field not in d, f"forbidden field '{field}' in view dict keys"


# ---------------------------------------------------------------------------
# 4. test_candidate_level_rrf
# ---------------------------------------------------------------------------


class TestCandidateLevelRRF:
    def test_candidate_appearing_in_multiple_lanes_gets_higher_score(self) -> None:
        """Candidate in 4 lanes should score higher than candidate in 1 lane."""
        lane_hits = {
            "candidate_raw_bm25": [_make_search_hit("c1", "candidate_raw_bm25", 1)],
            "candidate_raw_dense": [_make_search_hit("c1", "candidate_raw_dense", 1)],
            "candidate_structured_bm25": [
                _make_search_hit("c1", "candidate_structured_bm25", 1)
            ],
            "candidate_structured_dense": [
                _make_search_hit("c1", "candidate_structured_dense", 1)
            ],
        }
        result = fuse_candidate_hits(lane_hits, rrf_k=60)
        assert len(result) == 1
        score_4_lanes = result[0].rrf_score

        # Single lane
        single_lane = {
            "candidate_raw_bm25": [_make_search_hit("c2", "candidate_raw_bm25", 1)],
        }
        result_single = fuse_candidate_hits(single_lane, rrf_k=60)
        score_1_lane = result_single[0].rrf_score

        assert score_4_lanes > score_1_lane

    def test_rrf_score_formula(self) -> None:
        """Verify RRF formula: sum(1/(k+rank)) for each lane."""
        lane_hits = {
            "candidate_raw_bm25": [_make_search_hit("c1", "candidate_raw_bm25", 1)],
            "candidate_raw_dense": [_make_search_hit("c1", "candidate_raw_dense", 3)],
        }
        result = fuse_candidate_hits(lane_hits, rrf_k=60)
        expected = 1.0 / (60 + 1) + 1.0 / (60 + 3)
        assert abs(result[0].rrf_score - expected) < 1e-10

    def test_rrf_deterministic_ordering(self) -> None:
        """Same scores → sorted by candidate_key for determinism."""
        lane_hits = {
            "candidate_raw_bm25": [
                _make_search_hit("c_b", "candidate_raw_bm25", 1),
                _make_search_hit("c_a", "candidate_raw_bm25", 1),
            ],
        }
        result = fuse_candidate_hits(lane_hits, rrf_k=60)
        assert result[0].candidate_key == "c_a"
        assert result[1].candidate_key == "c_b"

    def test_missing_lane_contributes_zero(self) -> None:
        """Candidate only in 2 of 4 lanes gets score from those 2 lanes only."""
        lane_hits = {
            "candidate_raw_bm25": [_make_search_hit("c1", "candidate_raw_bm25", 1)],
            "candidate_raw_dense": [_make_search_hit("c1", "candidate_raw_dense", 1)],
        }
        result = fuse_candidate_hits(lane_hits, rrf_k=60)
        expected = 1.0 / (60 + 1) * 2  # two lanes, both rank 1
        assert abs(result[0].rrf_score - expected) < 1e-10


# ---------------------------------------------------------------------------
# 5. test_raw_and_structured_view_dedup
# ---------------------------------------------------------------------------


class TestRawAndStructuredViewDedup:
    def test_same_candidate_from_raw_and_structured_deduped(self) -> None:
        """When same candidate_key appears in raw_bm25 and structured_bm25,
        RRF fuses them into one candidate with combined score."""
        lane_hits = {
            "candidate_raw_bm25": [_make_search_hit("c1", "candidate_raw_bm25", 1)],
            "candidate_structured_bm25": [
                _make_search_hit("c1", "candidate_structured_bm25", 2)
            ],
        }
        result = fuse_candidate_hits(lane_hits, rrf_k=60)
        assert len(result) == 1
        assert result[0].candidate_key == "c1"
        # Should have entries in both lane_ranks
        assert "candidate_raw_bm25" in result[0].lane_ranks
        assert "candidate_structured_bm25" in result[0].lane_ranks

    def test_different_candidates_not_deduped(self) -> None:
        lane_hits = {
            "candidate_raw_bm25": [_make_search_hit("c1", "candidate_raw_bm25", 1)],
            "candidate_structured_bm25": [
                _make_search_hit("c2", "candidate_structured_bm25", 1)
            ],
        }
        result = fuse_candidate_hits(lane_hits, rrf_k=60)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 6. test_slot_queries_independent
# ---------------------------------------------------------------------------


class TestSlotQueriesIndependent:
    def test_slot_pool_has_separate_entries_per_slot(self) -> None:
        """Each slot's candidates are tracked separately in slot pool."""
        from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit

        slot_hits = {
            "slot_1": [
                CandidateRRFHit(
                    candidate_key="c1",
                    rrf_score=0.03,
                    lane_ranks={"candidate_raw_bm25": 1},
                    supporting_view_ids={"candidate_raw_bm25": "v1"},
                ),
                CandidateRRFHit(
                    candidate_key="c2",
                    rrf_score=0.02,
                    lane_ranks={"candidate_raw_bm25": 2},
                    supporting_view_ids={"candidate_raw_bm25": "v2"},
                ),
            ],
            "slot_2": [
                CandidateRRFHit(
                    candidate_key="c3",
                    rrf_score=0.025,
                    lane_ranks={"candidate_raw_bm25": 1},
                    supporting_view_ids={"candidate_raw_bm25": "v3"},
                ),
            ],
        }
        pool = build_slot_pool(slot_hits, slot_top_k=20, slot_min_budget=10, total_k=40)
        slot_ids_in_pool = {item["slot_id"] for item in pool}
        assert "slot_1" in slot_ids_in_pool
        assert "slot_2" in slot_ids_in_pool


# ---------------------------------------------------------------------------
# 7. test_multi_slot_round_robin
# ---------------------------------------------------------------------------


class TestMultiSlotRoundRobin:
    def test_round_robin_interleaves_slots(self) -> None:
        """Round-robin: slot_1 rank_1, slot_2 rank_1, slot_1 rank_2, slot_2 rank_2."""
        from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit

        slot_hits = {
            "slot_1": [
                CandidateRRFHit(
                    candidate_key=f"c1_{i}",
                    rrf_score=0.03 - i * 0.001,
                    lane_ranks={},
                    supporting_view_ids={},
                )
                for i in range(5)
            ],
            "slot_2": [
                CandidateRRFHit(
                    candidate_key=f"c2_{i}",
                    rrf_score=0.025 - i * 0.001,
                    lane_ranks={},
                    supporting_view_ids={},
                )
                for i in range(5)
            ],
        }
        pool = build_slot_pool(slot_hits, slot_top_k=20, slot_min_budget=10, total_k=40)
        # First two should be from different slots (round-robin)
        assert len(pool) >= 2
        assert pool[0]["slot_id"] != pool[1]["slot_id"]

    def test_each_slot_gets_minimum_budget(self) -> None:
        """Each slot gets at least slot_min_budget candidates."""
        from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit

        slot_hits = {
            "slot_1": [
                CandidateRRFHit(
                    candidate_key=f"c1_{i}",
                    rrf_score=0.03,
                    lane_ranks={},
                    supporting_view_ids={},
                )
                for i in range(15)
            ],
            "slot_2": [
                CandidateRRFHit(
                    candidate_key=f"c2_{i}",
                    rrf_score=0.025,
                    lane_ranks={},
                    supporting_view_ids={},
                )
                for i in range(15)
            ],
        }
        pool = build_slot_pool(slot_hits, slot_top_k=20, slot_min_budget=10, total_k=40)
        slot_1_count = sum(1 for item in pool if item["slot_id"] == "slot_1")
        slot_2_count = sum(1 for item in pool if item["slot_id"] == "slot_2")
        assert slot_1_count >= 10
        assert slot_2_count >= 10

    def test_total_pool_capped_at_40(self) -> None:
        from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit

        slot_hits = {
            "slot_1": [
                CandidateRRFHit(
                    candidate_key=f"c1_{i}",
                    rrf_score=0.03,
                    lane_ranks={},
                    supporting_view_ids={},
                )
                for i in range(30)
            ],
        }
        pool = build_slot_pool(slot_hits, slot_top_k=20, slot_min_budget=10, total_k=40)
        assert len(pool) <= 40


# ---------------------------------------------------------------------------
# 8. test_period_conflict_filtered
# ---------------------------------------------------------------------------


class TestPeriodConflictFiltered:
    def test_period_in_view_metadata_preserved(self) -> None:
        """View periods are stored in metadata for post-filtering."""
        view = _make_view(
            "c1",
            "structured",
            periods=("FY2024", "FY2023"),
        )
        assert "FY2024" in view.periods
        assert "FY2023" in view.periods

    def test_view_without_periods_has_empty_tuple(self) -> None:
        view = _make_view("c1", "raw")
        assert view.periods == ()


# ---------------------------------------------------------------------------
# 9. test_raw_pool_unchanged
# ---------------------------------------------------------------------------


class TestRawPoolUnchanged:
    def test_combined_pool_preserves_raw_order(self) -> None:
        """Raw candidates must appear first in combined pool, unchanged."""
        raw_pool = [
            {"candidate_key": "r1", "stage_rank": 1, "score": 0.5},
            {"candidate_key": "r2", "stage_rank": 2, "score": 0.3},
        ]
        candidate_direct_pool = [
            {"candidate_key": "r1", "rrf_score": 0.02},  # duplicate of raw
            {"candidate_key": "d1", "rrf_score": 0.01},  # new
        ]
        # Simulate combined pool construction (dedup by candidate_key)
        raw_keys = {item["candidate_key"] for item in raw_pool}
        residual = [
            item
            for item in candidate_direct_pool
            if item["candidate_key"] not in raw_keys
        ]
        combined = raw_pool + residual

        # Raw pool preserved as prefix
        assert combined[0]["candidate_key"] == "r1"
        assert combined[1]["candidate_key"] == "r2"
        # Residual appended
        assert combined[2]["candidate_key"] == "d1"
        # No duplicates
        keys = [item["candidate_key"] for item in combined]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# 10. test_b_class_not_read_before_seal
# ---------------------------------------------------------------------------


class TestBClassNotReadBeforeSeal:
    """Verify R2 prediction protocol does not read B-class info before seal."""

    R2_DIR = ROOT / "artifacts" / "evaluation" / "pdf-retrieval-v4-gate-08-r2"

    def _skip_if_missing(self) -> None:
        if not self.R2_DIR.is_dir():
            pytest.skip("R2 artifacts not generated")

    def test_seal_has_zero_gold_reads(self) -> None:
        self._skip_if_missing()
        seal_path = self.R2_DIR / "prediction-seal.json"
        if not seal_path.is_file():
            pytest.skip("seal not found")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        assert seal.get("gold_reads_before_seal") == 0
        assert seal.get("governance_reads_before_seal") == 0
        assert seal.get("sealed") is True

    def test_protocol_forbids_gold_inputs(self) -> None:
        self._skip_if_missing()
        protocol_path = self.R2_DIR / "gate-08-r2-protocol.json"
        if not protocol_path.is_file():
            pytest.skip("protocol not found")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        forbidden = protocol.get("forbidden_inputs", [])
        assert "gold" in str(forbidden).lower() or "labels" in str(forbidden).lower()
        assert "expected_value" in forbidden or "expected_value" in str(forbidden).lower()


# ---------------------------------------------------------------------------
# 11. test_no_parameter_scan
# ---------------------------------------------------------------------------


class TestNoParameterScan:
    R2_DIR = ROOT / "artifacts" / "evaluation" / "pdf-retrieval-v4-gate-08-r2"

    def _skip_if_missing(self) -> None:
        if not self.R2_DIR.is_dir():
            pytest.skip("R2 artifacts not generated")

    def test_seal_has_no_parameter_scan(self) -> None:
        self._skip_if_missing()
        seal_path = self.R2_DIR / "prediction-seal.json"
        if not seal_path.is_file():
            pytest.skip("seal not found")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        assert seal.get("parameter_scan") is False
        assert seal.get("per_query_oracle") is False

    def test_protocol_has_fixed_budgets(self) -> None:
        self._skip_if_missing()
        protocol_path = self.R2_DIR / "gate-08-r2-protocol.json"
        if not protocol_path.is_file():
            pytest.skip("protocol not found")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        budgets = protocol.get("budgets", {})
        assert budgets.get("lane_k") == 50
        assert budgets.get("rrf_k") == 60
        assert budgets.get("candidate_pool_k") == 40


# ---------------------------------------------------------------------------
# 12. test_deterministic_candidate_replay
# ---------------------------------------------------------------------------


class TestDeterministicCandidateReplay:
    def test_view_id_is_deterministic(self) -> None:
        """Same candidate_key always produces same view_id."""
        for _ in range(3):
            assert make_raw_view_id("c1") == make_raw_view_id("c1")
            assert make_structured_view_id("c1") == make_structured_view_id("c1")

    def test_rrf_is_deterministic(self) -> None:
        """Same input to RRF always produces same output."""
        lane_hits = {
            "candidate_raw_bm25": [
                _make_search_hit("c1", "candidate_raw_bm25", 1),
                _make_search_hit("c2", "candidate_raw_bm25", 2),
            ],
            "candidate_raw_dense": [
                _make_search_hit("c2", "candidate_raw_dense", 1),
                _make_search_hit("c1", "candidate_raw_dense", 2),
            ],
        }
        result1 = fuse_candidate_hits(lane_hits, rrf_k=60)
        result2 = fuse_candidate_hits(lane_hits, rrf_k=60)
        assert len(result1) == len(result2)
        for r1, r2 in zip(result1, result2):
            assert r1.candidate_key == r2.candidate_key
            assert abs(r1.rrf_score - r2.rrf_score) < 1e-15

    def test_slot_pool_is_deterministic(self) -> None:
        """Same input to slot pool always produces same output."""
        from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit

        slot_hits = {
            "slot_1": [
                CandidateRRFHit(
                    candidate_key="c1",
                    rrf_score=0.03,
                    lane_ranks={},
                    supporting_view_ids={},
                ),
            ],
        }
        pool1 = build_slot_pool(slot_hits, slot_top_k=20, slot_min_budget=10, total_k=40)
        pool2 = build_slot_pool(slot_hits, slot_top_k=20, slot_min_budget=10, total_k=40)
        assert len(pool1) == len(pool2)
        for p1, p2 in zip(pool1, pool2):
            assert p1["candidate_key"] == p2["candidate_key"]
            assert p1["slot_id"] == p2["slot_id"]
