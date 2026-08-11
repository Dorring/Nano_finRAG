"""CPU-safe contract checks for NF-E2E-04 R0."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[2]
SCRIPT = BACKEND / "scripts/evaluation/run_nf_e2e_04_r0_grounding_contract_recovery.py"
OUT = BACKEND / "artifacts/evaluation/nf-e2e-04-r0-grounding-contract-recovery"
SPEC = importlib.util.spec_from_file_location("nf_e2e_04_r0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_gate_is_grounding_contract_recovery_only():
    assert MODULE.BASE_COMMIT == "ea95f7c9eead6c4c5a07a4762e4934a02f23ff83"
    source = SCRIPT.read_text(encoding="utf-8")
    assert "retrieval_tuning" in source
    assert "validator_threshold_tuning" in source
    assert "gold_reads_during_execution" in source
    assert "max_repair_attempts" in source


def test_frozen_contracts_and_grounded_definition():
    if not OUT.exists():
        return
    frozen = read_json(OUT / "frozen-e2e-contract.json")
    assert frozen["selected_method"] == "sada_statement_aware_v1"
    assert frozen["sada_top100"] == "78/80"
    assert frozen["context"] == {"top_k": 5, "token_budget": 1100}
    assert frozen["production_switch_allowed"] is False
    grounded = read_json(OUT / "grounded-pass-contract.json")
    assert grounded["definition"] == "Grounded Pass = answer_contract_correct AND citation_full_recall"
    assert grounded["unchanged"] is True


def test_stage_a_first_loss_is_explicit_and_answerable_denominator_is_64():
    if not OUT.exists():
        return
    matrix = read_json(OUT / "validator-component-matrix.json")
    assert matrix["denominator"] == 64
    blockers = read_json(OUT / "first-validator-blocker.json")
    assert blockers["denominator"] == 64
    assert blockers["counts"].get("V1_answerability") == 5
    assert blockers["counts"].get("V7_calculation") == 6
    delivery = read_json(OUT / "calculation-first-loss-analysis.json")
    assert delivery["denominator"] == 5
    assert delivery["first_loss_stage"]["C1_calculator_to_generation_input"] == 5


def test_gcca_is_schema_and_lineage_only():
    if not OUT.exists():
        return
    contract = read_json(OUT / "gcca-v1-contract.json")
    assert contract["schema_mapping_only"] is True
    assert contract["preserves_calculator_arithmetic"] is True
    assert contract["cannot_invent_citation"] is True
    assert contract["cannot_rewrite_answer"] is True
    assert contract["cannot_bypass_validator"] is True
    assert contract["gold_access"] is False


def test_calculation_only_replay_preserves_five_strict_results():
    if not OUT.exists():
        return
    result = read_json(OUT / "calculation-only-replay.json")
    assert result["denominator"] == 5
    assert result["calculator_result_byte_identical"] == 5
    assert result["result_reached_generation_input"] == 5
    assert result["final_numeric_correct"] == 5
    assert result["validator_accepted"] == 5
    assert result["false_binding"] == 0
    assert result["false_execution"] == 0
    assert result["executed_incorrect"] == 0


def test_full_outputs_sealed_before_post_seal_scoring():
    if not OUT.exists():
        return
    seal = read_json(OUT / "e2e-output-seal.json")
    assert seal["complete"] is True
    assert seal["case_count"] == 72
    assert seal["gold_reads_during_execution"] == 0
    assert seal["trace_sha256"] == seal["canonical_trace_sha256"]
    assert seal["raw_output_sha256"] == seal["canonical_raw_output_sha256"]
    assert len(list(gzip.open(OUT / "per-question-traces.jsonl.gz", "rt", encoding="utf-8"))) == 72
    assert len(list(gzip.open(OUT / "raw-e2e-outputs.jsonl.gz", "rt", encoding="utf-8"))) == 72


def test_recovery_and_safety_metrics():
    if not OUT.exists():
        return
    full = read_json(OUT / "full-replay.json")
    calc = full["calculation"]
    assert calc["binder_ready"] == "5/11"
    assert calc["runtime_ready"] == "5/11"
    assert calc["executed"] == "5/11"
    assert calc["strict_correct"] == "5/11"
    assert calc["fail_closed"] == "6/11"
    safety = read_json(OUT / "safety-analysis.json")
    assert safety["false_binding"] == 0
    assert safety["false_execution"] == 0
    assert safety["executed_incorrect"] == 0


def test_non_binder_replay_equivalence_and_no_answer_audit():
    if not OUT.exists():
        return
    replay = read_json(OUT / "generation-reproducibility-audit.json")
    assert replay["denominator"] == 61
    assert replay["input_identical"] == 61
    assert replay["raw_output_identical"] == 61
    assert replay["released_output_identical"] == 61
    assert replay["validator_outcome_identical"] == 61
    no_answer = read_json(OUT / "no-answer-false-release-audit.json")
    assert no_answer["denominator"] == 8
    assert no_answer["false_releases"] == 3


def test_decision_is_contract_recovery_not_validator_tuning():
    if not OUT.exists():
        return
    decision = read_json(OUT / "decision.json")
    assert decision["grounding_contract_defect_supported"] is True
    assert decision["validator_contract_defect_supported"] is False
    assert decision["gcca_v1_executed"] is True
    assert decision["grounding_contract_recovery_effective"] is True
    assert decision["production_switch_allowed"] is False
    assert decision["next_gate"] in {"generation_grounding_recovery", "citation_binding_recovery", "answerability_safety_recovery", "final_end_to_end_showcase"}
