"""Gate 08 R4 lane-preserving fusion tests."""

from __future__ import annotations

import json
from pathlib import Path

from src.pdf_retrieval_v4.lane_preserving_fusion import (
    fuse_candidate_families,
    fuse_multi_slot_families,
    fuse_single_slot_families,
)

ROOT = Path(__file__).resolve().parents[2]
R4 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r4"


def _ranked(prefix: str, count: int) -> list[dict]:
    return [{"candidate_key": f"{prefix}{i}", "rank": i} for i in range(1, count + 1)]


def _json(name: str) -> dict:
    return json.loads((R4 / name).read_text())


def test_structured_top20_protected() -> None:
    result = fuse_single_slot_families(_ranked("r", 50), _ranked("s", 50))
    assert [x["candidate_key"] for x in result[:20]] == [f"s{i}" for i in range(1, 21)]
    assert all(x["protected_structured"] for x in result[:20])


def test_budget_and_cross_family_dedup() -> None:
    raw = _ranked("x", 50)
    structured = _ranked("x", 50)
    result = fuse_candidate_families(raw, structured, protected_structured_k=20)
    assert len(result) == 40
    assert len({x["candidate_key"] for x in result}) == 40
    assert result[0]["family_support"] == ["raw", "structured"]


def test_single_slot_deterministic() -> None:
    raw, structured = _ranked("r", 50), _ranked("s", 50)
    assert fuse_single_slot_families(raw, structured) == fuse_single_slot_families(raw, structured)


def test_multi_slot_each_has_structured_opportunity() -> None:
    ranking = {slot: {"raw": {"fused": _ranked(f"r{slot}", 50)}, "structured": {"fused": _ranked(f"s{slot}", 50)}} for slot in ("a", "b")}
    pool, traces = fuse_multi_slot_families(ranking)
    assert len(pool) <= 40
    assert all(traces[slot][0]["protected_structured"] for slot in traces)
    assert {x["slot_id"] for x in pool} == {"a", "b"}


def test_prediction_seal_zero_search() -> None:
    seal = _json("prediction-seal.json")
    assert seal["sealed"] is True
    assert seal["prediction_count"] == 72
    assert seal["raw_e0_prefix_exact_cases"] == 72
    for key in ("bm25_searches", "dense_searches", "embedding_calls", "index_reads", "index_builds", "gold_reads_before_seal"):
        assert seal[key] == 0
    assert seal["parameter_scan"] is False
    assert seal["quota_scan"] is False


def test_scored_contract() -> None:
    acceptance = _json("acceptance.json")
    metrics = acceptance["metrics"]
    assert metrics["f0"] == "52/80"
    assert metrics["family_union"] == "58/80"
    assert metrics["gross_fusion_loss"] == 8
    assert metrics["fusion_synergy_gain"] == 2
    assert metrics["f1"] == "52/80"
    assert metrics["f2"] == "54/80"
    assert metrics["gross_fusion_loss_recovered"] == "1/8"
    assert metrics["synergy_gold_retained"] == "1/2"
    assert metrics["contract_gross_loss_net_gain"] == 0
    assert metrics["new_outside_union_synergy_gain"] == 2
    assert metrics["observed_score_delta"] == 2
    assert acceptance["raw_gold_retained"] == "31/31"
    assert acceptance["decision"] == "lane_preserving_fusion_insufficient"
