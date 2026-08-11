"""Focused contract tests for NF-E2E-06 R0.

The gate is an offline audit over sealed NF-E2E-05 outputs.  These tests keep
the no-model/no-new-evidence invariants executable without importing or
starting any runtime service.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = (
    ROOT
    / "artifacts"
    / "evaluation"
    / "nf-e2e-06-r0-citation-binding-recovery"
)


def read_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def read_jsonl_gz(name: str) -> list[dict]:
    with gzip.open(ARTIFACTS / name, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_nf06_frozen_contract_and_offline_execution() -> None:
    contract = read_json("frozen-e2e-contract.json")
    assert contract["selected_internal_shadow_method"] == "sada_statement_aware_v1"
    assert contract["sada_top100"] == "78/80"
    assert contract["context"] == {"top_k": 5, "token_budget": 1100}
    assert contract["nf_opt_26_manifest_sha256"] == (
        "70048502ec918ae6ee56246a788da42129df3b073c2be8682e14f97e409e7c80"
    )
    assert contract["model_execution"] is False
    assert contract["production_switch_allowed"] is False


def test_nf06_routing_is_frozen_deterministic_and_explains_zero_calls() -> None:
    audit = read_json("deterministic-routing-audit.json")
    assert audit["denominator"] == 72
    assert audit["route_counts"] == {
        "deterministic_calculation": 11,
        "deterministic_fact": 46,
        "safe_response": 15,
    }
    assert audit["llm_required_routes"] == 0
    assert audit["llm_bypassed_routes"] == 72
    assert audit["nf_e2e_05_model_calls"] == 0
    assert audit["model_execution"] is False


def test_nf06_answerable_lineage_has_valid_denominator_and_no_invented_identity() -> None:
    lineage = read_json("deterministic-citation-lineage.json")
    first_loss = read_json("citation-first-loss-analysis.json")
    supported = read_json("supported-uncited-lineage.json")
    assert lineage["denominator"] == 64
    assert len(lineage["rows"]) == 64
    assert first_loss["denominator"] == 64
    assert sum(first_loss["counts"].values()) == 64
    assert supported["denominator"] == 51
    assert supported["counts"]["support_identity_known_upstream"] == 51
    assert supported["counts"]["identity_available_at_answer_builder_input"] == 51
    assert supported["counts"]["identity_dropped_by_response_construction"] == 0


def test_nf06_cba_is_fail_closed_when_no_contract_defect_is_supported() -> None:
    stage_a = read_json("citation-binding-decision.json")
    cba = read_json("cba-v1-contract.json")
    mapping = read_json("cba-v1-mapping-manifest.json")
    assert stage_a["deterministic_citation_contract_defect_supported"] is False
    assert stage_a["cba_v1_allowed"] is False
    assert cba["executed"] is False
    assert cba["can_use_gold"] is False
    assert cba["can_search_new_evidence"] is False
    assert cba["can_invent_citation"] is False
    assert mapping["citations_added_total"] == 0
    assert mapping["citations_added_from_known_support_identity"] == 0
    assert mapping["citations_added_without_known_support_identity"] == 0
    assert len(mapping["rows"]) == 72


def test_nf06_response_and_safety_invariants() -> None:
    seal = read_json("response-seal.json")
    invariant = read_json("answer-text-invariance.json")
    safety = read_json("safety-analysis.json")
    assert seal["complete"] is True
    assert seal["case_count"] == 72
    assert seal["model_execution"] is False
    assert seal["gold_reads_during_reconstruction"] == 0
    assert invariant["answer_text_byte_identical"] is True
    assert invariant["no_answer_byte_identical"] is True
    assert safety["citations_added_without_known_support_identity"] == 0
    assert safety["false_binding"] == 0
    assert safety["false_execution"] == 0
    assert safety["executed_incorrect"] == 0
    assert safety["answerable_release_invariant"] is True


def test_nf06_metrics_and_decision_remain_frozen_after_noop_reconstruction() -> None:
    metrics = read_json("citation-metrics.json")
    claims = read_json("claim-citation-metrics.json")
    calc = read_json("calculation-preservation.json")
    no_answer = read_json("no-answer-preservation.json")
    decision = read_json("decision.json")
    assert metrics["baseline"]["full_recall"] == 23
    assert metrics["post_cba"]["full_recall"] == 23
    assert metrics["citations_added_total"] == 0
    assert claims["supported_uncited_baseline"] == 51
    assert claims["supported_uncited_post"] == 51
    assert calc["calculator_response_byte_equivalent"] == 5
    assert calc["final_numeric_correct"] == 5
    assert calc["final_period_correct"] == 5
    assert calc["final_unit_correct"] == 5
    assert calc["citation_valid"] == 3
    assert no_answer["correct_safe_response"] == 5
    assert no_answer["false_answer_release"] == 3
    assert decision["post_grounded_pass"] == 3
    assert decision["post_citation_full_recall"] == 23
    assert decision["citation_binding_recovery_effective"] is False
    assert decision["dominant_bottleneck_after_recovery"] == "claim_grounding"
    assert decision["next_gate"] == "claim_grounding_recovery"
    assert decision["production_switch_allowed"] is False


def test_nf06_reconstructed_response_text_is_byte_identical() -> None:
    rows = read_jsonl_gz("reconstructed-responses.jsonl.gz")
    assert len(rows) == 72
    source = ROOT / "artifacts" / "evaluation" / "nf-e2e-05-r0-generation-grounding-recovery" / "raw-generation-outputs.jsonl.gz"
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        baseline = {row["question_id"]: row for row in map(json.loads, handle) if row}
    for row in rows:
        old = baseline[row["question_id"]]
        assert row.get("raw_answer") == old.get("raw_answer")
        assert row.get("released_answer") == old.get("released_answer")

