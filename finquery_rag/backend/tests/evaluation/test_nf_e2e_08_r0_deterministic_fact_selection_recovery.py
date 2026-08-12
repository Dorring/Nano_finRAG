"""Focused NF-E2E-08 R0 contract tests."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/evaluation/nf-e2e-08-r0-deterministic-fact-selection-recovery"


def read_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def read_jsonl_gz(name: str) -> list[dict]:
    with gzip.open(ARTIFACTS / name, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_contract_and_execution_guards() -> None:
    contract = read_json("frozen-e2e-contract.json")
    assert contract["selected_internal_shadow_method"] == "sada_statement_aware_v1"
    assert contract["sada_top100"] == {"hits": 78, "total": 80, "recall": 97.5}
    assert contract["context"] == {"top_k": 5, "token_budget": 1100}
    assert contract["nf_opt_26_manifest_sha256"] == (
        "70048502ec918ae6ee56246a788da42129df3b073c2be8682e14f97e409e7c80"
    )
    assert contract["model_calls"] == 0
    assert contract["retrieval_calls"] == 0
    assert contract["reranker_calls"] == 0
    assert contract["production_switch_allowed"] is False


def test_current_fact_audit_is_fail_closed() -> None:
    current = read_json("current-deterministic-fact-contract.json")
    audit = read_json("deterministic-fact-runtime-audit.json")
    assert audit["denominator"] == 46
    assert audit["counts"] == {
        "FS4_structured_fields_incomplete": 43,
        "FS6_no_machine_readable_fact_candidate": 3,
    }
    assert current["observer_selection_state_serialized_in_sealed_artifact"] is False
    assert current["answer_value_structured_in_sealed_artifact"] is False
    assert all(not row["selected_candidate_known"] for row in audit["rows"])
    assert all(not row["selected_fact_known"] for row in audit["rows"])
    assert all(not row["exact_provenance_known"] for row in audit["rows"])


def test_inventory_does_not_promote_serialized_value_to_typed_fact() -> None:
    inventory = read_jsonl_gz("deterministic-fact-candidate-inventory.jsonl.gz")
    assert len(inventory) == 72 * 5
    assert all(row["parsed_numeric_value"] is None for row in inventory)
    assert all(row["raw_value"] is None for row in inventory)
    assert all(row["full_machine_readable_provenance"] is False for row in inventory)
    assert sum(row["machine_readable_metric"] for row in inventory) == 191
    assert sum(row["machine_readable_period"] for row in inventory) == 179


def test_pre_dfs_gate_blocks_selector_before_gold() -> None:
    funnel = read_json("pre-dfs-fact-provenance-funnel.json")
    feasibility = read_json("dfs-v1-feasibility-decision.json")
    assert funnel["deterministic_fact"] == 46
    assert funnel["structured_fact_available"] == 43
    assert funnel["metric_resolvable"] == 43
    assert funnel["period_resolvable"] == 39
    assert funnel["metric_period_fact_available"] == 0
    assert funnel["full_machine_readable_provenance_available"] == 0
    assert funnel["unique_fact_tuple_possible"] == 0
    assert feasibility["minimum_required_for_dfs_v1"] == 15
    assert feasibility["full_machine_readable_fact_provenance_available"] == 0
    assert feasibility["dfs_v1_allowed"] is False
    assert feasibility["decision"] == "structured_fact_representation_insufficient"
    assert feasibility["gold_reads"] == 0


def test_disabled_dfs_has_no_predictions_or_route_invocations() -> None:
    decision = read_json("decision.json")
    dfs = read_json("dfs-v1-contract.json")
    seal = read_json("dfs-v1-prediction-seal.json")
    replay = read_json("full-shadow-replay.json")
    metrics = read_json("dfs-v1-selection-metrics.json")
    assert decision["dfs_v1_allowed"] is False
    assert decision["dfs_v1_executed"] is False
    assert decision["next_gate"] == "structured_fact_representation_review"
    assert decision["production_switch_allowed"] is False
    assert dfs["allowed"] is False
    assert dfs["executed"] is False
    assert dfs["can_use_gold"] is False
    assert dfs["can_use_expected_value"] is False
    assert dfs["rank_tie_break"] is False
    assert seal["executed"] is False
    assert seal["gold_reads_before_prediction_seal"] == 0
    assert read_jsonl_gz("dfs-v1-predictions.jsonl.gz") == []
    assert replay["stage_d_executed"] is False
    assert replay["dfs_invocations"] == {
        "deterministic_fact": 0,
        "deterministic_calculation": 0,
        "safe_response": 0,
    }
    assert metrics["stage_b_executed"] is False


def test_baselines_and_safety_are_not_rewritten() -> None:
    decision = read_json("decision.json")
    calc = read_json("calculation-preservation.json")
    no_answer = read_json("no-answer-preservation.json")
    safety = read_json("safety-analysis.json")
    assert decision["baseline_grounded_pass"] == 3
    assert decision["baseline_citation_full_recall"] == 23
    assert decision["baseline_answerable_released"] == 55
    assert decision["post_grounded_pass"] is None
    assert decision["post_citation_full_recall"] is None
    assert decision["post_answerable_released"] is None
    assert calc["baseline"]["binder_ready"] == 5
    assert calc["baseline"]["calculator_strict_correct"] == 5
    assert calc["baseline"]["final_numeric_correct"] == 5
    assert calc["baseline"]["citation_valid"] == 3
    assert no_answer["baseline"] == {"correct_safe_response": 5, "false_answer_release": 3}
    assert no_answer["path_preserved"] is True
    assert safety["false_source_binding"] == 0
    assert safety["false_execution"] == 0
    assert safety["executed_incorrect"] == 0
    assert safety["production_switch_allowed"] is False


def test_policy_digest_is_explicitly_disabled() -> None:
    digest = (ARTIFACTS / "dfs-v1-policy.sha256").read_text(encoding="utf-8").strip()
    assert digest == sha256(ARTIFACTS / "dfs-v1-policy.txt")
    contract = read_json("dfs-v1-contract.json")
    assert contract["allowed"] is False
    assert contract["executed"] is False
