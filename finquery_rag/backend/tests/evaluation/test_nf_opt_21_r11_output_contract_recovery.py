"""Focused contract tests for NF-OPT-21 R1.1 output recovery."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "evaluation" / "nf-opt-21-r11-listwise-output-contract-recovery"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluation" / "run_nf_opt_21_r11_output_contract_recovery.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r11_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(name: str):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def test_stage_a_raw_audit_and_threshold():
    audit = read_json("raw-output-failure-taxonomy.json")
    assert audit["recoverable_total"] == 0
    assert audit["unrecoverable"] == 72
    assert sum(audit["counts"].values()) == 72
    decision = read_json("decision.json")
    assert decision["stage_a_raw_output_available"] is True
    assert decision["stage_b_executed"] is True


def test_tolerant_parser_never_invents_or_autofills():
    module = load_module()
    allowed = {f"candidate:v1:{i:064x}" for i in range(10)}
    exact = '{"selected_ids":[' + ",".join(json.dumps(x) for x in sorted(allowed)[:5]) + ']}'
    assert module.classify_and_recover(exact, allowed)["recoverable"] is True
    fenced = "```json\n" + exact + "\n```"
    assert module.classify_and_recover(fenced, allowed)["recoverable"] is True
    embedded = "Result:\n" + exact + "\nDone"
    assert module.classify_and_recover(embedded, allowed)["recoverable"] is True
    explicit = "Selected: " + ", ".join(sorted(allowed)[:5])
    assert module.classify_and_recover(explicit, allowed)["recoverable"] is True
    fewer = "Selected: " + ", ".join(sorted(allowed)[:4])
    assert module.classify_and_recover(fewer, allowed)["recoverable"] is False
    duplicate = "Selected: " + ", ".join([sorted(allowed)[0]] * 2 + sorted(allowed)[1:4])
    assert module.classify_and_recover(duplicate, allowed)["recoverable"] is False
    outside = "Selected: " + ", ".join(sorted(allowed)[:4] + ["candidate:v1:" + "f" * 64])
    assert module.classify_and_recover(outside, allowed)["recoverable"] is False


def test_stage_b_structured_output_contract_and_runtime():
    contract = read_json("stage-b-output-contract.json")
    runtime = read_json("stage-b-runtime.json")
    assert contract["executed"] is True
    assert contract["no_repair_prompt"] is True
    assert contract["candidate_enum"] == "per_query_frozen_top10"
    assert runtime["model_calls"] == 72
    assert runtime["structured_output_valid"] == 72
    assert runtime["fallback_count"] == 0
    assert runtime["gold_reads_during_prediction"] == 0


def test_stage_b_predictions_are_exactly_five_top10_ids():
    rows = [json.loads(line) for line in gzip.open(ARTIFACTS / "stage-b-predictions.jsonl.gz", "rt", encoding="utf-8")]
    assert len(rows) == 72
    for row in rows:
        assert len(row["selected_ids"]) == 5
        assert len(set(row["selected_ids"])) == 5
        assert set(row["selected_ids"]).issubset(set(row["input_candidate_ids"]))
        assert row["structured_output_valid"] is True


def test_frozen_identity_prompt_and_seal_contract():
    frozen = read_json("frozen-contract.json")
    decision = read_json("decision.json")
    seal = read_json("stage-b-prediction-seal.json")
    assert frozen["semantic_prompt_unchanged"] is True
    assert frozen["semantic_prompt_sha"] == "01ab63296cf5b3581281eb5c0c55dd81be75f6bbc8c8bbeac5101a2f5151b645"
    assert frozen["model_revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert seal["rows"] == 72
    assert seal["gold_reads_during_prediction"] == 0
    assert decision["production_switch_allowed"] is False
    assert decision["mandatory_method_freeze"] is True


def test_final_decision_is_frozen_method_stop():
    decision = read_json("decision.json")
    assert decision["listwise_strict_r5_hits"] == 33
    assert decision["listwise_rescued_vs_qwen"] == 7
    assert decision["listwise_damaged_vs_qwen"] == 17
    assert decision["listwise_selector_effective"] is False
    assert decision["selected_internal_shadow_method"] == "lrrf_v1"
    assert decision["next_gate"] == "internal_retrieval_method_freeze"
