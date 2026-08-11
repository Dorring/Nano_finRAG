"""Focused, CPU-safe checks for the NF-OPT-24 sealed admission artifacts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[2]
EVAL = BACKEND / "artifacts/evaluation"
OUT = EVAL / "nf-opt-24-r0-deep-supply-top100-admission"
DEEP = EVAL / "pdf-retrieval-v4-gate-08-r8-r2a/deep-supply-predictions.jsonl.gz"
TOP100 = EVAL / "pdf-retrieval-v4-gate-08-r8-r2a-2/bounded-top100-predictions.jsonl.gz"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_frozen_contract_and_decision():
    decision = read_json(OUT / "decision.json")
    reranker = read_json(OUT / "frozen-reranker-contract.json")
    seal = read_json(OUT / "sada-v1-prediction-seal.json")
    assert decision["retrieval_rerun"] is False
    assert decision["training"] is False
    assert decision["model_execution"] is True
    assert decision["model_revision"] == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    assert reranker["reranker_contract_match"] is True
    assert decision["gold_reads_before_sada_prediction_seal"] == 0
    assert seal["gold_reads_before_prediction_seal"] == 0
    assert decision["production_switch_allowed"] is False


def test_exact_deep_supply_and_top100_identity():
    deep = read_gz(DEEP)
    top = read_gz(TOP100)
    manifest = read_json(OUT / "deep-supply-manifest.json")
    assert len(deep) == len(top) == 72
    manifest_by_case = {row["case_id"]: row for row in manifest["queries"]}
    for source in deep:
        keys = source["deep_supply_candidate_keys"]
        assert len(keys) == len(set(keys))
        assert set(keys) == set(item["candidate_key"] for item in manifest_by_case[source["case_id"]]["candidate_ranks"])
    for source in top:
        candidates = source["candidates"]
        assert len(candidates) == 100
        assert len({item["candidate_key"] for item in candidates}) == 100


def test_sada_is_top100_subset_and_sealed():
    deep = {row["case_id"]: set(row["deep_supply_candidate_keys"]) for row in read_gz(DEEP)}
    sada = read_gz(OUT / "sada-v1-top100-predictions.jsonl.gz")
    seal = read_json(OUT / "sada-v1-prediction-seal.json")
    assert len(sada) == 72
    assert seal["queries"] == 72
    assert seal["pairs_scored"] == 66033
    for row in sada:
        keys = [item["candidate_key"] for item in row["ranked_candidates"]]
        assert len(keys) == 100
        assert len(keys) == len(set(keys))
        assert set(keys) <= deep[row["case_id"]]
        assert [item["post_rerank_rank"] for item in row["ranked_candidates"]] == list(range(1, 101))


def test_serialization_contract_and_prediction_metrics():
    contract = read_json(OUT / "frozen-statement-aware-contract.json")
    seal = read_json(OUT / "serialization-seal.json")
    curve = read_json(OUT / "strict-recall-curve.json")
    assert contract["statement_aware_contract_reused"] is True
    assert contract["nf23_serialization_overlap"]["mismatches"] == 0
    assert seal["gold_reads_before_prediction_seal"] == 0
    assert curve["sada"]["@100"]["hits"] == 78
    assert curve["current_top100"]["@100"]["hits"] == 68


def test_runtime_has_no_resource_or_scoring_failures():
    runtime = read_json(OUT / "runtime-capacity.json")
    assert runtime["selected_gpu_ids"] == [1, 2, 7]
    assert runtime["total_pairs"] == 66033
    assert runtime["query_level_deterministic_sharding"] is True
    assert all(worker["oom"] == 0 for worker in runtime["workers"])
    assert all(worker["nonfinite"] == 0 for worker in runtime["workers"])
    assert all(worker["truncated"] == 0 for worker in runtime["workers"])


def test_lost_sources_and_retention_invariants():
    loss = read_json(OUT / "lost-top100-gold-audit.json")
    recovery = read_json(OUT / "lost-10-recovery.json")
    retention = read_json(OUT / "existing-68-retention.json")
    movement = read_json(OUT / "top100-movement.json")
    assert loss["lost_count"] == 10
    assert recovery["recovered_count"] == 10
    assert retention["current_gold_bindings"] == 68
    assert retention["retained"] == 68
    assert retention["dropped"] == 0
    assert movement["rescued"] == 10
    assert movement["damaged"] == 0

