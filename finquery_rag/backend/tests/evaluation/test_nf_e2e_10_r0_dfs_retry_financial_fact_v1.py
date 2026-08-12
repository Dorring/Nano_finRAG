"""Focused NF-E2E-10 R0 DFS retry contract tests."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/evaluation/nf-e2e-10-r0-dfs-retry-financial-fact-v1"
FACT_SHA = "7a253b443962c5f372dd897c49c057a19b553e92314faadc31eefc82b27b54eb"


def read_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def read_jsonl_gz(name: str) -> list[dict]:
    with gzip.open(ARTIFACTS / name, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_financial_fact_contract_and_execution_guards() -> None:
    contract = read_json("frozen-financial-fact-contract.json")
    assert contract["financial_fact_contract_sha"] == FACT_SHA
    assert contract["facts"] == 169
    assert contract["provenance_complete_facts"] == 169
    assert contract["query_level_full_provenance"] == 39
    assert contract["relation_integrity_failures"] == 0
    assert contract["fabricated_cross_candidate_facts"] == 0
    assert contract["financial_fact_rematerialized"] is False
    assert contract["model_calls"] == contract["retrieval_calls"] == contract["reranker_calls"] == 0
    assert contract["pdf_reparse"] is False
    assert contract["production_switch_allowed"] is False


def test_query_signal_contract_is_existing_and_complete() -> None:
    coverage = read_json("query-signal-coverage.json")
    assert coverage["denominator"] == 46
    assert coverage["document_scope_available"] == 46
    assert coverage["metric_available"] == 46
    assert coverage["period_available"] == 46
    assert coverage["metric_period_available"] == 46
    assert coverage["gold_reads"] == 0
    assert coverage["query_extractor_changed"] is False
    assert coverage["taxonomy"] == {"QA0_complete_metric_period_scope": 46}
    rows = read_json("existing-query-signals.json")["rows"]
    assert len(rows) == 46
    assert all(row["candidate_access"] is False and row["gold_access"] is False for row in rows)


def test_dfs_policy_is_sealed_and_has_no_tiebreak_or_fallback() -> None:
    contract = read_json("dfs-v1-contract.json")
    seal_text = (ARTIFACTS / "dfs-v1-policy.sha256").read_text(encoding="utf-8").strip()
    assert contract["enabled"] is True
    assert contract["policy_locked"] is True
    assert contract["policy_sha256"] == seal_text
    assert contract["ready_requires_exactly_one"] is True
    assert contract["rank_tie_break"] is False
    assert contract["can_use_gold"] is False
    assert contract["can_use_reference_answer"] is False
    assert contract["can_use_expected_value"] is False
    assert contract["can_use_old_answer"] is False
    assert contract["can_use_reranker_score"] is False
    assert contract["can_search_top5"] is False
    policy = (ARTIFACTS / "dfs-v1-policy.txt").read_text(encoding="utf-8").lower()
    assert "exact normalized" in policy
    assert "rank" in policy and "tie-break" in policy
    assert "semantic" in policy


def test_predictions_are_gold_blind_and_fail_closed() -> None:
    predictions = read_jsonl_gz("dfs-v1-financial-fact-predictions.jsonl.gz")
    assert len(predictions) == 46
    statuses = {status: sum(row["selector_status"] == status for row in predictions) for status in ("ready", "missing", "ambiguous", "unavailable")}
    assert statuses == {"ready": 1, "missing": 31, "ambiguous": 7, "unavailable": 7}
    assert all(row["gold_access"] is False for row in predictions)
    assert all(row["answer_value_reverse_lookup"] is False for row in predictions)
    assert all(row["rank_tie_break"] is False for row in predictions)
    assert all(row["selected_fact_id"] is None for row in predictions if row["selector_status"] != "ready")
    ready = [row for row in predictions if row["selector_status"] == "ready"]
    assert len(ready) == 1
    row = ready[0]
    assert row["matching_fact_count"] == 1
    assert row["selected_fact"]["provenance_complete"] is True
    assert row["answer_derivation"] == "selected_fact.parsed_numeric_value"
    assert row["citation_derivation"] == "selected_fact.physical_source_id"
    assert row["claim_support_ids"] == [row["selected_fact"]["physical_source_id"]]


def test_prediction_seal_precedes_gold_and_is_complete() -> None:
    seal = read_json("dfs-v1-prediction-seal.json")
    assert seal["complete"] is True
    assert seal["case_count"] == 46
    assert seal["gold_reads_before_prediction_seal"] == 0
    assert seal["reference_answer_reads_before_prediction_seal"] == 0
    assert seal["historical_wrong_source_reads_before_prediction_seal"] == 0
    assert seal["model_calls"] == seal["retrieval_calls"] == seal["reranker_calls"] == 0
    assert seal["financial_fact_rematerialized"] is False
    assert sha256(ARTIFACTS / "dfs-v1-financial-fact-predictions.jsonl.gz") == seal["prediction_sha256"]


def test_selection_funnel_and_safety_gate() -> None:
    funnel = read_json("selection-funnel.json")
    assert funnel["deterministic_fact"] == 46
    assert funnel["financial_fact_available"] == 39
    assert funnel["query_metric_signal_available"] == 46
    assert funnel["query_period_signal_available"] == 46
    assert funnel["metric_matched"] == 11
    assert funnel["period_matched"] == 8
    assert funnel["unique_fact_tuple"] == funnel["dfs_ready"] == 1
    metrics = read_json("selection-metrics.json")
    assert metrics["exact_selected_fact_provenance"] == 1
    assert metrics["false_source_binding"] == 0
    assert read_json("wrong-source-safety.json")["false_source_binding"] == 0
    assert read_json("decision.json")["deterministic_fact_selection_recovery_effective"] is False


def test_full_replay_route_isolation_and_no_runtime_execution() -> None:
    replay = read_json("full-e2e-replay.json")
    assert replay["stage_executed"] is True
    assert replay["output_seal"]["complete"] is True
    assert replay["output_seal"]["case_count"] == 72
    assert replay["output_seal"]["dfs_invocations"] == {
        "deterministic_fact": 46,
        "deterministic_calculation": 0,
        "safe_response": 0,
    }
    assert replay["output_seal"]["model_calls"] == 0
    assert replay["output_seal"]["retrieval_calls"] == 0
    assert replay["output_seal"]["reranker_calls"] == 0
    assert replay["output_seal"]["gold_reads_during_execution"] == 0
    assert replay["old_numeric_window_fallback"] is False
    assert replay["answer_value_reverse_lookup"] is False


def test_calculation_and_no_answer_contracts_are_reported_separately() -> None:
    calc = read_json("calculation-preservation.json")
    assert calc["post_dfs"]["binder_ready"] == 5
    assert calc["post_dfs"]["runtime_ready"] == 5
    assert calc["post_dfs"]["executed"] == 5
    assert calc["post_dfs"]["calculator_strict_correct"] == 5
    assert calc["post_dfs"]["final_numeric_correct"] == 5
    assert calc["post_dfs"]["period_correct"] == 5
    assert calc["post_dfs"]["unit_correct"] == 5
    assert calc["post_dfs"]["false_binding"] == 0
    assert calc["post_dfs"]["false_execution"] == 0
    assert calc["post_dfs"]["executed_incorrect"] == 0
    no_answer = read_json("no-answer-preservation.json")
    assert no_answer["dfs_invocations"]["safe_response"] == 0
    assert no_answer["post_dfs"]["correct_safe_response"] >= no_answer["baseline"]["correct_safe_response"]
    assert no_answer["post_dfs"]["false_answer_release"] <= no_answer["baseline"]["false_answer_release"]


def test_decision_and_production_guardrail() -> None:
    decision = read_json("decision.json")
    assert decision["production_switch_allowed"] is False
    assert decision["next_gate"] == "query_fact_alignment_review"
    assert decision["dominant_residual_bottleneck"] == "query_fact_alignment"
    assert decision["false_source_binding"] == 0
    assert decision["calculation_preserved"] is True
