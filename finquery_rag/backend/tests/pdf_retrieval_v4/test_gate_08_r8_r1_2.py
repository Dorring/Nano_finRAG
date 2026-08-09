from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from src.pdf_retrieval_v4.bounded_candidate_selector import (
    CANDIDATE_BUDGET,
    SLOT_CANDIDATE_HORIZON,
    SLOT_MIN_BUDGET,
)
from src.pdf_retrieval_v4.support_invariant_candidate_selector import (
    RRF_K,
    build_raw_family_v2,
    build_structured_family_v2,
    fuse_main_families_v2,
    rank_support_invariant_family,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r8-r1-2"
PRED = OUT / "support-invariant-predictions.jsonl.gz"


def hit(key: str, rank: int) -> dict[str, object]:
    return {"candidate_key": key, "rank": rank}


def test_single_rank10_beats_double_rank40() -> None:
    ranked = rank_support_invariant_family(
        {"a": [hit("strong", 10), hit("weak", 40)], "b": [hit("weak", 40)]}
    )
    assert [item["candidate_key"] for item in ranked] == ["strong", "weak"]


def test_single_rank1_beats_multiweak_consensus() -> None:
    ranked = rank_support_invariant_family(
        {
            "a": [hit("strong", 1), hit("weak", 30)],
            "b": [hit("weak", 30)],
            "c": [hit("weak", 30)],
        }
    )
    assert ranked[0]["candidate_key"] == "strong"
    assert ranked[1]["support_count"] == 3


def test_equal_best_rank_uses_second_best_only() -> None:
    ranked = rank_support_invariant_family(
        {"a": [hit("single", 10), hit("double", 10)], "b": [hit("double", 15)]}
    )
    assert [item["candidate_key"] for item in ranked] == ["double", "single"]
    assert ranked[0]["second_best_rank"] == 15
    assert ranked[1]["second_best_rank"] is None


def test_support_count_not_added_to_primary_score() -> None:
    ranked = rank_support_invariant_family(
        {"a": [hit("single", 10), hit("double", 10)], "b": [hit("double", 40)]}
    )
    assert ranked[0]["best_rank_score"] == ranked[1]["best_rank_score"]


def test_family_v2_helpers_use_best_rank() -> None:
    raw = build_raw_family_v2([hit("x", 8)], [hit("x", 30)])
    structured = build_structured_family_v2([hit("x", 12)], [hit("x", 2)])
    top = fuse_main_families_v2(raw, structured)
    assert raw[0]["best_rank"] == 8
    assert structured[0]["best_rank"] == 2
    assert top[0]["best_rank"] == 1


def test_candidate_dedup_across_lanes() -> None:
    ranked = rank_support_invariant_family({"a": [hit("x", 1)], "b": [hit("x", 2)]})
    assert len(ranked) == 1
    assert ranked[0]["lane_ranks"] == {"a": 1, "b": 2}


def test_rrf_constant_cannot_be_scanned() -> None:
    assert RRF_K == 60
    with pytest.raises(ValueError, match="rrf_k_must_equal_60"):
        rank_support_invariant_family({"a": [hit("x", 1)]}, rrf_k=30)


def test_multislot_budget_contract_is_unchanged() -> None:
    assert CANDIDATE_BUDGET == 50
    assert SLOT_CANDIDATE_HORIZON == 50
    assert SLOT_MIN_BUDGET == 10
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["multi_slot_residual_fusion"] == "r8_r1_sum_rrf_exact"


def test_h0_exact_unified_binding_parity() -> None:
    metrics = json.loads((OUT / "full-system-metrics.json").read_text())
    parity = json.loads((OUT / "baseline-parity.json").read_text())
    assert metrics["h0"]["recall_at_50"] == "55/80"
    assert parity["expected_post_seal_score"]["raw_retained"] == "22/24"
    assert metrics["h0"]["multi_complete_at_50"] == "9/16"
    assert metrics["h0"]["calculation_complete_at_50"] == "7/11"


def test_h1_passes_support_invariant_contract() -> None:
    acceptance = json.loads((OUT / "acceptance.json").read_text())
    assert acceptance["decision"] == "support_count_invariant_fusion_passed"
    assert acceptance["metrics"]["h1"]["recall_at_50"] == "57/80"
    assert acceptance["metrics"]["raw_regression"] == 1


def test_prediction_budget_exact_50_and_seal_hash() -> None:
    with gzip.open(PRED, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    assert len(records) == 72
    assert all(len(item["h1_bounded_candidate_ranking"]) == 50 for item in records)
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    assert hashlib.sha256(PRED.read_bytes()).hexdigest() == seal["prediction_sha256"]


def test_prediction_protocol_has_no_runtime_mutation() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    for field in (
        "bm25_searches",
        "dense_searches",
        "embedding_calls",
        "index_reads",
        "index_builds",
        "bridge_runs",
        "semantic_graph_runs",
        "query_plan_changes",
        "query_rebuilds",
        "gold_reads_before_seal",
        "governance_reads_before_seal",
    ):
        assert protocol[field] == 0
    assert protocol["weight_scan"] is False
    assert protocol["quota_scan"] is False
    assert protocol["topk_scan"] is False
