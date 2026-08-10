from __future__ import annotations

import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "artifacts/evaluation/t2-ragbench-02a-fusion-audit"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_t2_02a_contract_is_sealed_without_retrieval_mutation() -> None:
    contract = read_json(AUDIT / "contract-audit.json")
    seal = read_json(AUDIT / "prediction-seal.json")
    assert contract["status"] == "passed"
    assert contract["manifest"]["query_count"] == 23088
    assert contract["manifest"]["corpus_count"] == 7318
    assert contract["protocol"]["dense_effective_max_seq_length"] == 256
    assert "no_query_or_document_instruction_used" in contract["observations"]
    assert seal["sealed"] is True
    assert seal["rank_movement_count"] == 23088
    assert seal["retrieval_runs"] == 0
    assert seal["reranker_calls"] == 0
    assert seal["embedding_calls"] == 0
    assert seal["parameter_scan"] is False


def test_t2_02a_complementarity_partition_and_transfer_counts() -> None:
    data = read_json(AUDIT / "complementarity.json")
    overall = data["overall"]
    for k in ("5", "10", "20", "50", "100"):
        values = overall[k]
        assert (
            values["bm25_and_dense"]
            + values["bm25_only"]
            + values["dense_only"]
            + values["neither"]
            == 23088
        )
    assert overall["5"]["bm25_only"] == 10472
    assert overall["5"]["dense_only"] == 642
    assert overall["5"]["bm25_hit_rrf_miss"] == 6766
    assert overall["5"]["bm25_miss_rrf_hit"] == 1129
    assert overall["100"]["union_oracle"] == 22314


def test_t2_02a_decision_is_evidence_based() -> None:
    decision = read_json(AUDIT / "decision.json")
    assert decision["published_denominator"] == 23088
    assert decision["decisions"] == [
        "dense_branch_model_quality_insufficient",
        "fusion_negative_transfer_confirmed",
    ]
    assert decision["dense_equal_fusion_rejected"] is True
    assert decision["next_gate"] == "t2_02b_dense_rescue_decision"


def test_t2_02a_rank_movement_has_one_row_per_query() -> None:
    with gzip.open(AUDIT / "rank-movement.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 23088
    assert len({row["query_id"] for row in rows}) == 23088
    assert {row["subset"] for row in rows} == {"FinQA", "ConvFinQA", "TAT-DQA"}
    assert all(set(row["bm25_hit"]) == {"5", "10", "20", "50", "100"} for row in rows)

