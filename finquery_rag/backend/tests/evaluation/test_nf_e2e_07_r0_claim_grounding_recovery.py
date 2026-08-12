"""Focused NF-E2E-07 R0 contract tests."""
from __future__ import annotations

import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/evaluation/nf-e2e-07-r0-claim-grounding-recovery"


def read_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def read_jsonl_gz(name: str) -> list[dict]:
    with gzip.open(ARTIFACTS / name, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_frozen_contract_is_read_only() -> None:
    contract = read_json("frozen-e2e-contract.json")
    assert contract["selected_internal_shadow_method"] == "sada_statement_aware_v1"
    assert contract["sada_top100"] == "78/80"
    assert contract["context"] == {"top_k": 5, "token_budget": 1100}
    assert contract["nf_opt_26_manifest_sha256"] == (
        "70048502ec918ae6ee56246a788da42129df3b073c2be8682e14f97e409e7c80"
    )
    assert contract["model_execution"] is False
    assert contract["retrieval_calls"] == 0
    assert contract["production_switch_allowed"] is False


def test_claim_audit_uses_exact_lineage_only() -> None:
    provenance = read_json("deterministic-claim-provenance.json")
    supported = read_json("supported-uncited-provenance.json")
    assert provenance["claims_audited"] == 64
    assert provenance["claims_exact_lineage"] == 5
    assert sum(provenance["counts"].values()) == 64
    assert provenance["counts"] == {
        "CG0_exact_support_already_available": 5,
        "CG2_support_set_incomplete": 6,
        "CG4_answer_derived_without_traceable_evidence": 40,
        "CG5_claim_exceeds_selected_evidence": 3,
        "CG7_not_applicable": 10,
    }
    assert supported["denominator"] == 51
    assert supported["counts"]["exact_derivation_provenance_available"] == 0
    assert supported["counts"]["recoverable_by_contract_only"] == 0
    assert supported["counts"]["support_set_incomplete"] == 6
    assert supported["counts"]["no_traceable_derivation"] == 35
    assert supported["counts"]["wrong_or_unresolved_provenance"] == 35
    assert supported["counts"]["claim_exceeds_evidence"] == 0


def test_deterministic_fact_and_wrong_source_audits_fail_closed() -> None:
    fact = read_json("deterministic-fact-derivation-audit.json")
    wrong = read_json("wrong-source-root-cause.json")
    assert fact["denominator"] == 46
    assert fact["counts"] == {"DF2_value_known_but_source_not_unique": 46}
    assert all(row["selection_trace_present"] is False for row in fact["rows"])
    assert wrong["denominator"] == 7
    assert wrong["counts"] == {
        "WS0_wrong_evidence_used_to_derive_answer": 0,
        "WS1_correct_value_but_wrong_provenance_attached": 0,
        "WS2_no_unique_derivation_source": 7,
        "WS3_multiple_plausible_sources_unresolved": 0,
        "WS4_claim_exceeds_selected_source": 0,
        "WS5_other": 0,
    }
    assert wrong["wrong_source_fixed_by_contract"] == 0
    assert wrong["wrong_source_unchanged"] == 7


def test_calculation_support_is_exact_but_not_a_claim_binding_defect() -> None:
    calc = read_json("calculation-support-set-audit.json")
    assert calc["denominator"] == 5
    assert calc["complete_support_sets"] == 5
    assert calc["all_operand_physical_sources_known"] == 5
    assert calc["citation_full_recall_satisfiable"] == 3
    for case_id in ("ko_fy2025_006", "nvda_fy2025_006"):
        row = calc["special_cases"][case_id]
        assert row["complete_support_set"] is True
        assert row["all_operand_physical_sources_known"] is True
        assert row["citation_full_recall_satisfiable"] is False


def test_cgba_is_not_executed_below_frozen_threshold() -> None:
    stage_a = read_json("claim-grounding-decision.json")
    contract = read_json("cgba-v1-contract.json")
    manifest = read_json("cgba-v1-manifest.json")
    seal = read_json("response-seal.json")
    assert stage_a["recoverable_by_contract_only_threshold"] == 8
    assert stage_a["recoverable_by_contract_only"] == 0
    assert stage_a["claim_grounding_contract_defect_supported"] is False
    assert stage_a["cgba_v1_allowed"] is False
    assert contract["executed"] is False
    assert contract["can_search_context"] is False
    assert contract["can_use_reranker_scores"] is False
    assert contract["can_use_gold"] is False
    assert contract["can_add_support_without_exact_lineage"] is False
    assert manifest["executed"] is False
    assert manifest["claims_support_added_without_exact_lineage"] == 0
    assert manifest["citations_added_without_exact_claim_support"] == 0
    assert seal["stage_b_executed"] is False
    assert seal["response_reconstruction"] is False
    assert seal["case_count"] == 0
    assert seal["model_calls"] == 0
    assert seal["retrieval_calls"] == 0
    assert seal["gold_reads_during_reconstruction"] == 0
    assert read_jsonl_gz("reconstructed-responses.jsonl.gz") == []


def test_safety_and_baseline_metrics_are_preserved_without_post_run() -> None:
    decision = read_json("decision.json")
    integrity = read_json("provenance-integrity.json")
    calc = read_json("calculation-preservation.json")
    metrics = read_json("claim-grounding-metrics.json")
    citation = read_json("citation-metrics.json")
    no_answer = read_json("no-answer-preservation.json")
    safety = read_json("safety-analysis.json")
    assert decision["baseline_grounded_pass"] == 3
    assert decision["baseline_citation_full_recall"] == 23
    assert decision["post_grounded_pass"] is None
    assert decision["post_citation_full_recall"] is None
    assert decision["claim_grounding_recovery_effective"] is False
    assert decision["dominant_bottleneck_after_recovery"] == "deterministic_fact_selection"
    assert decision["next_gate"] == "deterministic_fact_selection_recovery"
    assert decision["production_switch_allowed"] is False
    assert all(value == 0 for key, value in integrity.items() if key.endswith("added") or key.endswith("lineage") or key.endswith("support"))
    assert calc["baseline"]["calculator_strict_correct"] == 5
    assert calc["baseline"]["final_numeric_correct"] == 5
    assert calc["baseline"]["final_period_correct"] == 5
    assert calc["baseline"]["final_unit_correct"] == 5
    assert calc["baseline"]["citation_valid"] == 3
    assert no_answer["baseline"] == {"correct_safe_response": 5, "false_answer_release": 3}
    assert metrics["baseline"]["grounded_pass"] == 3
    assert citation["baseline"]["citation_full_recall"] == 23
    assert safety["false_binding"] == 0
    assert safety["false_execution"] == 0
    assert safety["executed_incorrect"] == 0
    assert safety["formal_result_invalid"] is False

