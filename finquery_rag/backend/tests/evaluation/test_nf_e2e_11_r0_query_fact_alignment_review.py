"""Focused NF-E2E-11 R0 contract tests."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/evaluation/nf-e2e-11-r0-query-fact-alignment-review"


def load(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_nf11_frozen_execution_and_fact_contract() -> None:
    contract = load("frozen-input-contract.json")
    assert contract["financial_fact_contract_sha"] == "7a253b443962c5f372dd897c49c057a19b553e92314faadc31eefc82b27b54eb"
    assert contract["financial_fact_count"] == 169
    assert contract["model_calls"] == 0
    assert contract["retrieval_calls"] == 0
    assert contract["reranker_calls"] == 0
    assert contract["pdf_reparse"] is False
    assert contract["dfs_execution"] is False
    assert contract["financial_fact_modified"] is False
    assert contract["query_extractor_modified"] is False
    assert contract["e2e_replay"] is False
    assert contract["top5_ids_unchanged"] is True
    assert contract["top5_order_unchanged"] is True
    assert contract["gold_used_before_diagnostic_seal"] is False


def test_nf11_canonical_namespace_is_audit_only() -> None:
    comparison = load("canonical-namespace-comparison.json")
    assert comparison["shared_normalizer"] is False
    assert comparison["shared_alias_contract"] is False
    assert comparison["shared_canonical_namespace"] is False
    assert comparison["shared_metric_id_system"] is False
    groups = load("proposed-metric-equivalence-groups.json")["groups"]
    assert groups
    assert all(group["apply_now"] is False for group in groups)


def test_nf11_mismatch_denominators_and_conservative_projection() -> None:
    metric = load("metric-mismatch-taxonomy.json")
    assert metric["denominator"] == 28
    assert sum(metric["counts"].values()) == 28
    assert metric["canonical_recoverable"] == 5
    period = load("period-mismatch-taxonomy.json")
    assert period["denominator"] == 3
    assert period["counts"]["QP5_financial_fact_period_wrong"] == 3
    assert period["period_canonical_recoverable"] == 0
    ambiguity = load("ambiguity-taxonomy.json")
    assert ambiguity["denominator"] == 7
    assert ambiguity["counts"]["AM4_same_metric_across_multiple_statements"] == 7
    assert ambiguity["dedup_recoverable"] == 0
    projection = load("projected-recoverability.json")
    assert projection["current_dfs_ready"] == 1
    assert projection["projected_dfs_ready_after_canonical_only_recovery"] == 6
    assert projection["projected_provenance_safe_ready"] == 5
    assert projection["double_count_guard"] is True


def test_nf11_ready_wrong_and_decision() -> None:
    ready_wrong = load("ready-wrong-root-cause.json")
    assert ready_wrong["question_id"] == "v_fy2025_001"
    assert ready_wrong["primary_root_cause"] == "RW6_answer_format_contract_mismatch"
    assert ready_wrong["financial_fact_semantic_defect_supported"] is False
    assert ready_wrong["gold_used_for_root_cause"] is False
    decision = load("decision.json")
    assert decision["query_fact_alignment_recovery_warranted"] is False
    assert decision["next_gate"] == "end_to_end_method_freeze"
    assert decision["production_switch_allowed"] is False
    assert decision["e2e_replay"] is False


def test_nf11_contains_no_new_e2e_metrics() -> None:
    names = {path.name for path in ARTIFACT.iterdir()}
    assert "full-e2e-replay.json" not in names
    assert "grounded-replay.json" not in names
    assert "citation-replay.json" not in names
    contract = load("frozen-input-contract.json")
    reference = contract["nf10_frozen_reference"]
    assert reference == {"grounded": 3, "citation_full_recall": 10, "answerable_released": 12, "wrong_source": 0}
