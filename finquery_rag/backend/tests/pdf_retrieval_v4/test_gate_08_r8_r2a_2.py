from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from src.pdf_retrieval_v4.bounded_rerank_input_selector import (
    RERANK_INPUT_BUDGET,
    SLOT_COMPOSITION_HORIZON,
    SLOT_MIN_BUDGET,
    build_priority_ranking,
    select_multi_slot_top100,
    select_single_slot_top100,
)

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "artifacts").exists():
    ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/evaluation"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r2a-2"
R2A = BASE / "pdf-retrieval-v4-gate-08-r8-r2a"
EXPECTED_R2A_SHA = "63dd2f91f078d6101e564c06d174e5772be11b82ba91e8a2c7416d9512dc6ee9"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records() -> list[dict]:
    with gzip.open(OUT / "bounded-top100-predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_input_exact_r2a_prediction_sha() -> None:
    assert sha(R2A / "deep-supply-predictions.jsonl.gz") == EXPECTED_R2A_SHA
    assert json.loads((OUT / "prediction-seal.json").read_text())["input_hashes"]["r2a_prediction"] == EXPECTED_R2A_SHA


def test_priority_uses_family_best_rank_not_ordinal_rank() -> None:
    ranked = build_priority_ranking(
        [{"candidate_key": "gold", "best_rank": 2, "rank": 80}],
        [{"candidate_key": "other", "best_rank": 3, "rank": 1}],
    )
    assert [item["candidate_key"] for item in ranked] == ["gold", "other"]


def test_second_priority_rank_is_tiebreak_only() -> None:
    ranked = build_priority_ranking(
        [
            {"candidate_key": "single", "best_rank": 4, "rank": 99},
            {"candidate_key": "supported", "best_rank": 5, "rank": 1},
        ],
        [{"candidate_key": "supported", "best_rank": 5, "rank": 2}],
    )
    assert ranked[0]["candidate_key"] == "single"
    tied = build_priority_ranking(
        [{"candidate_key": "a", "best_rank": 4}, {"candidate_key": "b", "best_rank": 4}],
        [{"candidate_key": "b", "best_rank": 7}],
    )
    assert tied[0]["candidate_key"] == "b"


def test_single_slot_budget_and_contiguous_ranks() -> None:
    selected = select_single_slot_top100(
        [{"candidate_key": f"c{i}", "rank": i + 1} for i in range(130)]
    )
    assert len(selected) == 100
    assert [item["final_candidate_rank"] for item in selected] == list(range(1, 101))


def test_multislot_budget_dedup_and_minimum_coverage() -> None:
    slot_a = [{"candidate_key": f"a{i}", "rank": i + 1} for i in range(100)]
    slot_b = [{"candidate_key": f"b{i}", "rank": i + 1} for i in range(100)]
    main = [{"candidate_key": f"m{i}", "rank": i + 1} for i in range(100)]
    selected, audit = select_multi_slot_top100({"a": slot_a, "b": slot_b}, main)
    keys = [item["candidate_key"] for item in selected]
    assert len(keys) == len(set(keys)) == 100
    assert audit["slot_coverage"]["a"] >= 10
    assert audit["slot_coverage"]["b"] >= 10


def test_frozen_gate_parameters_and_zero_operation_contract() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (RERANK_INPUT_BUDGET, SLOT_COMPOSITION_HORIZON, SLOT_MIN_BUDGET) == (100, 100, 10)
    for field in (
        "bm25_searches", "dense_searches", "embedding_calls", "index_reads", "index_builds",
        "bridge_runs", "semantic_graph_runs", "reranker_calls", "calculator_calls",
        "generator_calls", "gold_reads_before_seal", "governance_reads_before_seal",
    ):
        assert protocol[field] == 0
    for field in ("parameter_scan", "quota_scan", "topk_scan", "weight_scan"):
        assert protocol[field] is False


def test_predictions_are_bounded_ranked_and_deep_supply_only() -> None:
    with gzip.open(R2A / "deep-supply-predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
        deep = {item["case_id"]: set(item["deep_supply_candidate_keys"]) for item in map(json.loads, handle)}
    records = load_records()
    assert len(records) == 72
    for record in records:
        candidates = record["candidates"]
        keys = [item["candidate_key"] for item in candidates]
        assert len(keys) == len(set(keys)) == 100
        assert set(keys) <= deep[record["case_id"]]
        assert [item["final_candidate_rank"] for item in candidates] == list(range(1, 101))


def test_formal_acceptance_and_protections() -> None:
    acceptance = json.loads((OUT / "acceptance.json").read_text())
    assert acceptance["decision"] == "bounded_top100_rerank_input_passed"
    assert acceptance["reranker_allowed"] is True
    assert acceptance["metrics"]["recall_at_100"] == "68/80"
    assert acceptance["metrics"]["raw_regression"] <= 1
    assert acceptance["metrics"]["multi_evidence_complete_at_100"] == "12/16"
    assert acceptance["metrics"]["calculation_complete_at_100"] == "9/11"
