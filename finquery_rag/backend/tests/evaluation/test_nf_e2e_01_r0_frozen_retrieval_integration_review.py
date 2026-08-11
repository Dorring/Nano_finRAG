"""CPU-safe contract checks for NF-E2E-01 R0."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[2]
SCRIPT = BACKEND / "scripts/evaluation/run_nf_e2e_01_r0_frozen_retrieval_integration_review.py"
OUT = BACKEND / "artifacts/evaluation/nf-e2e-01-r0-frozen-retrieval-integration-review"
SPEC = importlib.util.spec_from_file_location("nf_e2e_01_r0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_gate_is_shadow_only_and_never_tunes_retrieval():
    assert MODULE.BASE_COMMIT == "6072ce275227d795a817347e7e954d6c456637b5"
    assert MODULE.FLAGS == {
        "model_execution": False,
        "retrieval_rerun": False,
        "admission_rerun": False,
        "training": False,
        "parameter_tuning": False,
        "production_switch_allowed": False,
    }
    source = SCRIPT.read_text(encoding="utf-8")
    assert "gold_aware" in source
    assert "query_aware_filtering" in source
    assert MODULE.MODEL_EXECUTION is False
    assert MODULE.RETRIEVAL_RERUN is False
    assert "Shadow Retrieval Adapter V1" in source


def test_frozen_manifest_and_method_are_verified():
    if not OUT.exists():
        return
    decision = read_json(OUT / "decision.json")
    retrieval = read_json(OUT / "frozen-retrieval-contract.json")
    assert decision["retrieval_method_frozen"] is True
    assert decision["sada_top100_hits"] == 78
    assert retrieval["selected_internal_shadow_method"] == "sada_statement_aware_v1"
    assert retrieval["sada_top100"]["hits"] == 78
    assert retrieval["manifest_sha256"] == MODULE.NF26_MANIFEST_SHA256


def test_adapter_is_identity_preserving_and_context_budget_is_unchanged():
    if not OUT.exists():
        return
    adapter = read_json(OUT / "shadow-retrieval-adapter-contract.json")
    budget = read_json(OUT / "context-budget-contract.json")
    assert adapter["status"] == "ready"
    assert adapter["schema_mapping_only"] is True
    assert adapter["reorders"] is False
    assert adapter["adds_candidates"] is False
    assert adapter["drops_candidates"] is False
    assert len(adapter["cases"]) == 72
    assert all(row["input_count"] == 100 for row in adapter["cases"])
    assert all(row["adapter_output_count"] == 100 for row in adapter["cases"])
    assert all(row["output_context_count"] == 5 for row in adapter["cases"])
    assert all(row["candidate_identity_1_to_1"] for row in adapter["cases"])
    assert budget["candidates_entering_context"] == 5
    assert budget["token_budget"] == 1100


def test_stage_outputs_are_sealed_before_gold_and_replay_is_fail_closed_when_unavailable():
    if not OUT.exists():
        return
    seal = read_json(OUT / "e2e-output-seal.json")
    decision = read_json(OUT / "decision.json")
    assert seal["gold_reads_before_seal"] == 0
    assert seal["case_count"] == 72
    assert decision["production_switch_allowed"] is False
    if not seal["complete"]:
        assert decision["shadow_replay_executed"] is False
        assert decision["shadow_replay_reason"]


def test_identity_and_structured_contracts_are_explicit():
    if not OUT.exists():
        return
    identity = read_json(OUT / "evidence-identity-continuity.json")
    fields = read_json(OUT / "structured-field-consumption.json")
    assert identity["identity_loss"] == 0
    assert identity["continuity"]["candidate_id"]["preserved"] is True
    assert "row_label/metric" in fields["available_and_consumed"]
    assert "period/value bindings" in fields["available_and_consumed"]


def test_no_production_mutation_contract():
    if not OUT.exists():
        return
    integration = read_json(OUT / "integration-map.json")
    assert integration["contracts_unchanged"]["binder"] is True
    assert integration["contracts_unchanged"]["calculator"] is True
    assert integration["contracts_unchanged"]["validator"] is True
    assert integration["contracts_unchanged"]["repair_max_attempts"] == 1
    assert integration["contracts_unchanged"]["production"] is False
