from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/evaluation/t2-ragbench-03-qwen3-cross-encoder"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_t2_03_r0_passes_before_full_prediction() -> None:
    probe = read_json(OUTPUT / "r0-runtime-probe.json")
    assert probe["gate"] == "T2-03R0"
    assert probe["decision"] == "reranker_runtime_probe_passed"
    assert probe["query_count"] == 256
    assert probe["expected_pairs"] == 256 * 50
    assert probe["pairs_processed"] == probe["expected_pairs"]
    assert probe["candidate_depth"] == 50
    assert probe["gold_reads_before_seal"] == 0
    assert probe["candidate_identity_mutation"] == 0
    assert probe["model_revision"] == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    assert probe["runtime_errors"] == []


def test_t2_03_r0_input_and_instruction_contracts_are_sealed() -> None:
    inputs = read_json(OUTPUT / "r0-input-manifest.json")
    instruction = read_json(OUTPUT / "instruction-contract.json")
    runtime = read_json(OUTPUT / "runtime-contract.json")
    assert inputs["query_count"] == 256
    assert inputs["candidate_depth"] == 50
    assert instruction["per_query_instruction"] is False
    assert instruction["instruction_sha256"] == "c9525627a439d8beb14de046eadcaea0d7696010223501f41bb99763d035c77d"
    assert runtime["model_id"] == "Qwen/Qwen3-Reranker-4B"
    assert runtime["attention_implementation"] == "flash_attention_2"
    assert runtime["dtype"] == "bfloat16"
    assert runtime["max_length"] == 8192


def test_t2_03_full_prediction_was_not_started_after_r0_block() -> None:
    assert not (OUTPUT / "predictions.jsonl.gz").exists()
    assert not (OUTPUT / "prediction-seal.json").exists()

