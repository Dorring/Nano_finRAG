from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "evaluation"
MODULE_PATH = SCRIPT_DIR / "t2_05_qwen3_strong_cross_encoder.py"
ARTIFACT = ROOT / "artifacts/evaluation/t2-ragbench-05-qwen3-strong-reranker"
RETRIEVAL = ROOT / "artifacts/evaluation/t2-ragbench-01-standard-retrieval"

sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location("t2_05", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def read_json(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_frozen_contract_and_method_hash() -> None:
    contract = read_json("frozen-contract.json")
    assert contract["model_id"] == module.EXPECTED_MODEL_ID
    assert contract["model_revision"] == module.MODEL_REVISION
    assert contract["candidate_depth"] == 50
    assert contract["batch_size"] == 1
    assert contract["instruction_sha256"] == module.sha256_text(module.INSTRUCTION)
    assert contract["method_hash"] == module.METHOD_HASH
    assert contract["feature_seal"] == module.FEATURE_SEAL
    assert contract["fresh_blind_test"] is False


def test_shard_manifest_is_contiguous_and_complete() -> None:
    manifest = read_json("shard-manifest.json")
    assert manifest["query_count"] == 2291
    assert manifest["candidate_depth"] == 50
    assert manifest["formal_pairs"] == 114550
    shards = manifest["shards"]
    assert shards[0]["start"] == 0
    assert shards[-1]["end"] == 2291
    assert all(left["end"] == right["start"] for left, right in zip(shards, shards[1:]))
    assert sum(shard["end"] - shard["start"] for shard in shards) == 2291


def test_runtime_equivalence_is_accepted() -> None:
    equivalence = read_json("runtime-equivalence.json")
    assert equivalence["gold_reads"] == 0
    assert equivalence["top50_ordered_agreement"] == 1.0
    assert equivalence["multi_gpu_runtime_accepted"] is True


def test_prediction_seal_completeness_and_gold_preseal() -> None:
    seal = read_json("prediction-seal.json")
    manifest = read_json("prediction-manifest.json")
    assert seal["sealed"] is True
    assert seal["prediction_count"] == 2291
    assert seal["pair_count"] == 114550
    assert seal["gold_reads_before_seal"] == 0
    assert seal["candidate_mutation"] == 0
    assert manifest["gold_reads_before_seal"] == 0
    assert manifest["query_count"] == 2291
    assert manifest["pair_count"] == 114550


def test_candidate_universe_unchanged() -> None:
    frozen: dict[str, list[str]] = {}
    with gzip.open(RETRIEVAL / "bm25-predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id.startswith(("finqa_test_", "tatqa_test_")):
                frozen[query_id] = [item["context_id"] for item in row["ranked_contexts"][:50]]
    with gzip.open(ARTIFACT / "predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
        assert len(rows) == 2291
        assert all(
            len(row["ranked_contexts"]) == 50
            and set(item["context_id"] for item in row["ranked_contexts"])
            == set(frozen[row["query_id"]])
            for row in rows
        )


def test_metrics_and_r50_invariant() -> None:
    metrics = read_json("metrics.json")
    bm25 = metrics["bm25"]
    qwen = metrics["qwen3_reranker_4b"]
    assert bm25["count"] == qwen["count"] == 2291
    assert qwen["hits"]["50"] == bm25["hits"]["50"]
    decision = read_json("decision.json")
    assert decision["candidate_mutation"] == 0
    assert decision["recall_at_50_invariant"] is True
    assert decision["qwen_prediction_preseal_gold_reads"] == 0
