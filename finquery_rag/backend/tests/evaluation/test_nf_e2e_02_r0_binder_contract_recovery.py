"""CPU-safe contract checks for NF-E2E-02 R0."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[2]
SCRIPT = BACKEND / "scripts/evaluation/run_nf_e2e_02_r0_binder_contract_recovery.py"
OUT = BACKEND / "artifacts/evaluation/nf-e2e-02-r0-binder-contract-recovery"
SPEC = importlib.util.spec_from_file_location("nf_e2e_02_r0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_gate_is_shadow_only_and_contracts_are_frozen():
    assert MODULE.BASE_COMMIT == "bc6f9abce1d9b4339940ecbbac6fbd7b00fe6c1a"
    assert MODULE.MODEL_EXECUTION is False
    assert MODULE.CALCULATOR_EXECUTION is False
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Gold is intentionally first opened" in source
    assert "_bind_r53" in source
    assert "execute_plan" in source
    assert "retrieval" in source.lower()


def test_applicability_uses_existing_query_contract():
    if not OUT.exists():
        return
    audit = read_json(OUT / "binder-applicability-audit.json")
    assert audit["binder_applicability"] == {
        "required": 11,
        "optional": 0,
        "not_applicable": 61,
        "unknown": 0,
    }
    assert audit["zero_over_72_valid_capability_denominator"] is False
    assert audit["current_shadow_router_on_required_cases"] == {"document_qa": 11}
    assert audit["current_shadow_binding_invocation"] == {"binding_not_invoked": 11}


def test_frozen_retrieval_and_context_contract():
    if not OUT.exists():
        return
    retrieval = read_json(OUT / "frozen-retrieval-contract.json")
    assert retrieval["manifest_sha256"] == MODULE.NF26_SHA
    assert retrieval["selected_method"] == "sada_statement_aware_v1"
    assert retrieval["sada_top100_hits"] == 78
    assert retrieval["context_top_k"] == 5
    assert retrieval["context_token_budget"] == 1100
    assert retrieval["retrieval_tuning"] is False


def test_historical_entrypoint_and_contract_diff_are_explicit():
    if not OUT.exists():
        return
    historical = read_json(OUT / "historical-binder-contract.json")
    diff = read_json(OUT / "binder-contract-diff.json")
    assert historical["binder_entrypoint"].endswith("::_bind_r53")
    assert historical["calculator_handoff"].endswith("execute_plan(CalculationPlan)")
    assert diff["schema_mismatch"] is True
    assert diff["invocation_defect"] is True
    statuses = {item["field"]: item["status"] for item in diff["fields"]}
    assert statuses["parsed_numeric_value"] == "dropped"
    assert statuses["logical_table_id"] == "present_but_not_mapped"
    assert statuses["cell_id / cell_ids"] == "dropped"


def test_bica_preserves_identity_and_cannot_add_or_reorder_evidence():
    if not OUT.exists():
        return
    contract = read_json(OUT / "bica-v1-contract.json")
    mapping = read_json(OUT / "bica-v1-mapping-manifest.json")
    assert contract["executed"] is True
    assert contract["schema_mapping_only"] is True
    assert contract["preserves_candidate_order"] is True
    assert contract["adds_candidates"] is False
    assert contract["drops_candidates"] is False
    assert mapping["identity_preserved"] is True
    assert mapping["order_preserved"] is True
    assert mapping["added"] == 0
    assert mapping["dropped"] == 0
    for row in mapping["cases"]:
        assert row["input_candidate_order"] == row["output_candidate_order"]
        assert row["added_candidates"] == 0
        assert row["dropped_candidates"] == 0


def test_binder_failure_denominator_and_bica_recovery():
    if not OUT.exists():
        return
    taxonomy = read_json(OUT / "binder-failure-taxonomy.json")
    assert taxonomy["primary_blocker_counts"]["B0_not_invoked"] == 11
    assert sum(taxonomy["primary_blocker_counts"].values()) == 11
    funnel = read_json(OUT / "calculation-funnel.json")
    assert funnel["retrieval_all_slots"] == "6/11"
    assert funnel["binder_ready"] == "5/11"
    assert funnel["runtime_ready"] == "5/11"
    assert funnel["executed"] == "5/11"
    assert funnel["strict_correct"] == "5/11"
    assert funnel["false_binding"] == 0
    assert funnel["false_execution"] == 0
    assert funnel["executed_incorrect"] == 0


def test_prediction_is_sealed_before_gold_and_safety_is_explicit():
    if not OUT.exists():
        return
    calc = read_json(OUT / "calculation-shadow-results.json")
    seal = read_json(OUT / "calculation-shadow-results.seal.json")
    safety = read_json(OUT / "safety-analysis.json")
    decision = read_json(OUT / "decision.json")
    assert calc["gold_reads_before_seal"] == 0
    assert calc["sealed"] is True
    assert seal["gold_reads_before_seal"] == 0
    assert seal["sealed"] is True
    assert safety["false_binding"] == 0
    assert safety["false_execution"] == 0
    assert safety["executed_incorrect"] == 0
    assert decision["binder_contract_recovery_effective"] is True
    assert decision["bica_v1_executed"] is True
    assert decision["production_switch_allowed"] is False
    assert decision["next_gate"] == "end_to_end_rag_replay_after_binder_recovery"
