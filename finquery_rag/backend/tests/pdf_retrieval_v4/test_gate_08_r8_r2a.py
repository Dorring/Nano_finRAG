from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from src.pdf_retrieval_v4.deep_candidate_supply import (
    RRF_K,
    SUPPLY_LANE_K,
    retrieve_deep_supply,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r8-r2a"
PRED = OUT / "deep-supply-predictions.jsonl.gz"


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def test_supply_lane_k_exact_200() -> None:
    assert SUPPLY_LANE_K == 200
    with pytest.raises(ValueError, match="supply_lane_k_must_equal_200"):
        retrieve_deep_supply(None, None, general_query="x", field_queries={}, document_scope=set(), lane_k=100)  # type: ignore[arg-type]


def test_candidate_and_slot_budgets_are_frozen() -> None:
    protocol = load("protocol.json")
    assert protocol["candidate_budget"] == 50
    assert protocol["slot_composition_horizon"] == 50
    assert protocol["slot_min_budget"] == 10
    assert protocol["rrf_k"] == RRF_K == 60


def test_embedding_model_and_revision_are_exact() -> None:
    protocol = load("protocol.json")
    assert protocol["embedding_model"] == "all-MiniLM-L6-v2"
    assert protocol["embedding_revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def test_query_and_index_contracts_are_hashed() -> None:
    integrity = load("input-integrity.json")
    hashes = integrity["input_hashes"]
    assert hashes["query_plans"]
    assert hashes["query_builder"]
    assert hashes["field_query_builder"]
    assert hashes["candidate_index"]
    assert hashes["field_index"]
    assert hashes["support_invariant_selector"]


def test_bridge_and_semantic_graph_are_not_run() -> None:
    protocol = load("protocol.json")
    assert protocol["bridge_runs"] == 0
    assert protocol["semantic_graph_runs"] == 0
    assert protocol["index_builds"] == 0


def test_no_scan_or_gold_before_seal() -> None:
    protocol = load("protocol.json")
    assert protocol["gold_reads_before_seal"] == 0
    assert protocol["governance_reads_before_seal"] == 0
    assert protocol["parameter_scan"] is False
    assert protocol["topk_scan"] is False
    assert protocol["weight_scan"] is False
    assert protocol["model_scan"] is False


def test_search_and_embedding_accounting_is_real() -> None:
    counts = load("search-counts.json")
    assert counts["logical_queries"] == 108
    assert counts["bm25_searches"] == 648
    assert counts["dense_searches"] == 216
    assert counts["total_searches"] == 864
    assert counts["logical_embedding_requests"] == 216
    assert counts["unique_embedding_computations"] == 105
    assert counts["embedding_cache_hits"] == 111


def test_prediction_has_exact_top50_and_seal_hash() -> None:
    with gzip.open(PRED, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    assert len(records) == 72
    assert all(len(record["bounded_candidate_top50"]) == 50 for record in records)
    assert hashlib.sha256(PRED.read_bytes()).hexdigest() == load("prediction-seal.json")["prediction_sha256"]


def test_deep_supply_recovers_but_compression_is_insufficient() -> None:
    acceptance = load("acceptance.json")
    metrics = acceptance["metrics"]
    assert metrics["deep_supply_presence"] == "78/80"
    assert metrics["bounded_recall_at_50"] == "57/80"
    assert acceptance["decision"] == "deep_supply_recovered_but_top50_compression_insufficient"
    assert acceptance["reranker_allowed"] is False


def test_raw_multi_and_calculation_do_not_regress() -> None:
    metrics = load("acceptance.json")["metrics"]
    assert metrics["raw_retained"] == "23/24"
    assert metrics["multi_evidence_complete_at_50"] == "10/16"
    assert metrics["calculation_complete_at_50"] == "8/11"


def test_raw_production_ranking_declared_immutable() -> None:
    assert load("input-integrity.json")["raw_production_ranking_immutable"] is True
