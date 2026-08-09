from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R2A = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r8-r2a"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r8-r2a-1"


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def test_audit_exactly_78_deep_present_and_21_misses() -> None:
    acceptance = load("acceptance.json")
    assert acceptance["deep_present"] == "78/80"
    assert acceptance["compression_gap"] == 21
    assert acceptance["first_failure_classified"] == "21/21"
    assert load("deep-present-top50-misses.json")["count"] == 21


def test_atomic_lane_rank_is_preserved() -> None:
    records = load("compression-lineage.json")["records"]
    for record in records:
        ranks = record["atomic_lane_ranks"]
        expected = min(ranks.values()) if ranks else None
        assert record["best_atomic_lane_rank"] == expected
        if expected is not None:
            assert ranks[record["best_atomic_lane_name"]] == expected


def test_family_best_rank_is_distinct_from_ordinal_rank() -> None:
    misses = load("deep-present-top50-misses.json")["records"]
    assert any(
        record["structured_family_best_rank"]
        != record["structured_family_ordinal_rank"]
        for record in misses
        if record["structured_family_best_rank"] is not None
    )


def test_top_level_ordinal_rank_is_traced() -> None:
    misses = load("deep-present-top50-misses.json")["records"]
    assert all("main_top_level_ordinal_rank" in record for record in misses)


def test_multislot_rank_lineage_is_complete() -> None:
    records = [
        item
        for item in load("compression-lineage.json")["records"]
        if item["is_multi_slot"]
    ]
    assert records
    assert all(item["slot_lineage"] for item in records)
    assert all("minimum_coverage_selected" in item for item in records)
    assert all("main_residual_rank" in item for item in records)


def test_provenance_diagnostic_uses_best_rank_and_second_only_tiebreak() -> None:
    trace = load("provenance-diagnostic.json")["trace"]
    for record in trace.values():
        ranking = record["main_priority_ranking"]
        tuples = [
            (
                item["top_priority_rank"],
                item["second_priority_rank"] or 10**9,
                item["candidate_key"],
            )
            for item in ranking
        ]
        assert tuples == sorted(tuples)


def test_provenance_top50_ceiling_and_top100_accessibility() -> None:
    acceptance = load("acceptance.json")
    assert acceptance["provenance_diagnostic_at50"] == "59/80"
    assert acceptance["rerank_input_accessibility_at100"] == "73/80"
    assert acceptance["decision"] == "top50_heuristic_compression_ceiling_reached"
    assert acceptance["next_gate"] == "bounded_top100_rerank_input"


def test_diagnostic_protects_raw_multi_and_calculation() -> None:
    diagnostic = load("provenance-diagnostic.json")
    assert diagnostic["raw_retained"] == "23/24"
    assert diagnostic["multi_evidence"] == "11/16"
    assert diagnostic["calculation"] == "8/11"


def test_no_runtime_operations_or_prediction_mutation() -> None:
    protocol = load("protocol.json")
    for field in (
        "prediction_reruns",
        "retriever_reruns",
        "bm25_searches",
        "dense_searches",
        "embedding_calls",
        "index_reads",
        "index_builds",
        "bridge_changes",
        "query_changes",
        "fusion_changes",
        "selector_changes",
        "reranker_calls",
        "calculator_calls",
        "generator_calls",
    ):
        assert protocol[field] == 0


def test_r2a_prediction_sha_is_immutable() -> None:
    prediction = R2A / "deep-supply-predictions.jsonl.gz"
    expected = json.loads((R2A / "prediction-seal.json").read_text())[
        "prediction_sha256"
    ]
    assert hashlib.sha256(prediction.read_bytes()).hexdigest() == expected
    assert load("acceptance.json")["r2a_prediction_sha256"] == expected


def test_bridge_and_embedding_changes_are_closed() -> None:
    acceptance = load("acceptance.json")
    assert acceptance["bridge_recovery_needed"] is False
    assert acceptance["embedding_change_allowed"] is False
    assert acceptance["reranker_allowed"] is False
