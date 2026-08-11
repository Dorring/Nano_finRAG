"""Focused contract tests for NF-OPT-21 R1 sealed selector artifacts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "evaluation" / "nf-opt-21-r1-top10-listwise-selector"


def load_json(name: str):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def load_rows(name: str):
    with gzip.open(ARTIFACTS / name, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_selector_contract_is_frozen():
    contract = load_json("selector-contract.json")
    assert contract["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert contract["revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert contract["candidate_depth"] == 10
    assert contract["one_call_per_query"] is True
    assert contract["temperature"] == 0.0
    assert contract["seed"] == 0
    assert len((ARTIFACTS / "prompt-sha256.txt").read_text().strip()) == 64


def test_prediction_count_and_selection_contract():
    rows = load_rows("predictions.jsonl.gz")
    assert len(rows) == 72
    for row in rows:
        assert len(row["input_candidate_ids"]) == 10
        assert len(row["selected_ids"]) == 5
        assert len(set(row["selected_ids"])) == 5
        assert set(row["selected_ids"]) <= set(row["input_candidate_ids"])


def test_no_candidate_beyond_qwen_top10():
    for row in load_rows("predictions.jsonl.gz"):
        assert all(candidate in row["input_candidate_ids"] for candidate in row["selected_ids"])


def test_prediction_sealed_before_gold_and_validity():
    seal = load_json("prediction-seal.json")
    validity = load_json("output-validity.json")
    assert seal["sealed"] is True
    assert seal["rows"] == 72
    assert seal["candidate_depth"] == 10
    assert seal["gold_reads_during_prediction"] == 0
    assert validity["queries"] == 72
    assert validity["final_invalid"] == 0
    assert validity["fallback_to_qwen"] == 72


def test_top10_supply_and_baselines_unchanged():
    metrics = load_json("strict-metrics.json")
    assert metrics["qwen_top10_supply"] == 60
    assert metrics["qwen"]["@5"]["hits"] == 43
    assert metrics["lrrf_v1"]["@5"]["hits"] == 46
    assert metrics["listwise"]["@10"]["hits"] == 60


def test_shadow_decision_and_no_production_switch():
    decision = load_json("decision.json")
    assert decision["evaluation_role"] == "development_shadow_listwise_selection"
    assert decision["fresh_blind_evaluation"] is False
    assert decision["retrieval_rerun"] is False
    assert decision["training"] is False
    assert decision["model_calls"] == 72
    assert decision["listwise_selector_effective"] is False
    assert decision["production_switch_allowed"] is False
    assert decision["next_gate"] == "internal_retrieval_method_freeze"


def test_runtime_source_gold_blind_before_seal():
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluation" / "run_nf_opt_21_r1_top10_listwise_selector.py"
    source = script.read_text(encoding="utf-8").split("# Post-seal only", 1)[0]
    # Paths may be declared before the seal, but no Gold/diagnostic artifact may
    # be read until the explicitly marked post-seal section.
    assert "read_jsonl(strict_path)" not in source
    assert "load_targets(targets_path)" not in source
    assert "read_gzip_jsonl(registry_path)" not in source
    assert "failure-taxonomy" not in source
    assert "C1" not in source
