from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r8-r0"


def test_depth_pools_are_exact_prefixes() -> None:
    with gzip.open(OUT / "candidate-depth-predictions.jsonl.gz", "rt") as handle:
        for line in handle:
            item = json.loads(line)
            assert item["candidate_pool_10"] == item["candidate_pool_20"][:10]
            assert item["candidate_pool_20"] == item["candidate_pool_40"][:20]
            assert item["candidate_pool_40"] == item["candidate_pool_50"][:40]


def test_r0_records_zero_retrieval_and_no_reranker() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["bm25_reruns"] == 0
    assert protocol["dense_reruns"] == 0
    assert protocol["embedding_reruns"] == 0
    assert protocol["reranker_calls"] == 0
    assert protocol["gold_reads_before_seal"] == 0


def test_historical_at40_contract_mismatch_is_explicit() -> None:
    acceptance = json.loads((OUT / "acceptance.json").read_text())
    assert acceptance["metrics"]["recall_at_40"] == "20/80"
    assert acceptance["metrics"]["recall_at_50"] == "23/80"
    assert acceptance["historical_r7_at40_contract"] == "cutoff_mismatch_detected"
    assert acceptance["reranker_allowed"] is False


def test_sealed_pool_has_no_missing_top50() -> None:
    integrity = json.loads((OUT / "input-integrity.json").read_text())
    assert integrity["sealed_pool_length"]["count_below_50"] == 0
