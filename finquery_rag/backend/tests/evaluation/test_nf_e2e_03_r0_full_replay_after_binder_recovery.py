"""CPU-safe contract checks for NF-E2E-03 R0."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[2]
SCRIPT = BACKEND / "scripts/evaluation/run_nf_e2e_03_r0_full_replay_after_binder_recovery.py"
OUT = BACKEND / "artifacts/evaluation/nf-e2e-03-r0-full-replay-after-binder-recovery"
SPEC = importlib.util.spec_from_file_location("nf_e2e_03_r0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_gate_is_pure_shadow_replay():
    assert MODULE.BASE_COMMIT == "e2ca9814c4ed4d18c9d3c059efe45dc3635d3524"
    source = SCRIPT.read_text(encoding="utf-8")
    assert "retrieval_tuning" in source
    assert "binder_tuning" in source
    assert "calculator_tuning" in source
    assert "gold_reads_during_execution" in source
    assert "--rescore-sealed" in source


def test_preflight_and_frozen_retrieval_contract():
    if not OUT.exists():
        return
    preflight = read_json(OUT / "preflight.json")
    assert all(preflight["checks"].values())
    assert preflight["selected_method"] == "sada_statement_aware_v1"
    assert preflight["sada_top100_hits"] == 78
    assert preflight["binder_required"] == 11
    assert preflight["binder_not_applicable"] == 61
    retrieval = read_json(OUT / "frozen-retrieval-contract.json")
    assert retrieval["manifest_sha256"] == MODULE.NF26_SHA
    assert retrieval["sada_top100"] == "78/80"
    assert retrieval["retrieval_rerun"] is False


def test_bica_and_applicability_are_frozen():
    if not OUT.exists():
        return
    bica = read_json(OUT / "frozen-bica-contract.json")
    assert bica["entrypoint"] == "_bind_r53"
    assert bica["applicability"] == {"required": 11, "not_applicable": 61}
    assert bica["unchanged"] is True


def test_full_output_is_sealed_before_scoring():
    if not OUT.exists():
        return
    seal = read_json(OUT / "e2e-output-seal.json")
    assert seal["complete"] is True
    assert seal["case_count"] == 72
    assert seal["gold_reads_during_execution"] == 0
    assert seal["retrieval_rerun"] is False
    assert seal["binder_recovery"] is True
    assert len(list(gzip.open(OUT / "per-question-traces.jsonl.gz", "rt", encoding="utf-8"))) == 72
    assert len(list(gzip.open(OUT / "raw-e2e-outputs.jsonl.gz", "rt", encoding="utf-8"))) == 72


def test_context_and_non_applicable_routing_are_unchanged():
    if not OUT.exists():
        return
    manifest = read_json(OUT / "shadow-input-manifest.json")
    assert manifest["context_top_k"] == 5
    assert manifest["context_token_budget"] == 1100
    assert manifest["binder_required"] == 11
    assert manifest["binder_not_applicable"] == 61
    routing = read_json(OUT / "non-applicable-routing-safety.json")
    assert routing["denominator"] == 61
    assert routing["binder_invocation"] == 0
    assert routing["calculator_attempted"] == 0
    assert routing["routing_regression"] is False


def test_calculation_funnel_uses_applicable_denominator_and_keeps_safety_zero():
    if not OUT.exists():
        return
    funnel = read_json(OUT / "calculation-funnel.json")
    assert funnel["retrieval_all_slots"] == "6/11"
    assert funnel["binder_ready"] == "5/11"
    assert funnel["runtime_ready"] == "5/11"
    assert funnel["executed"] == "5/11"
    assert funnel["strict_correct"] == "5/11"
    assert funnel["fail_closed"] == "6/11"
    assert funnel["false_binding"] == 0
    assert funnel["false_execution"] == 0
    assert funnel["executed_incorrect"] == 0
    residual = read_json(OUT / "residual-calculation-failures.json")
    assert residual["B8_multiple_operand_tuple_ambiguous"] == 2
    assert residual["B9_required_operand_not_in_context"] == 4


def test_calculator_correctness_is_separate_from_answer_contract():
    if not OUT.exists():
        return
    analysis = read_json(OUT / "calculation-answer-analysis.json")
    assert analysis["calculator_correct_result"] == "5/11"
    assert analysis["calculator_executed"] == 5
    assert "final_numeric_correct" in analysis
    assert "answer_level_executed_incorrect" in analysis


def test_decision_marks_binder_resolved_and_next_downstream_gate():
    if not OUT.exists():
        return
    decision = read_json(OUT / "decision.json")
    assert decision["binder_bottleneck_resolved"] is True
    assert decision["post_bica_binder_ready"] == 5
    assert decision["post_bica_calculation_runtime_ready"] == 5
    assert decision["post_bica_calculation_strict_correct"] == 5
    assert decision["false_binding"] == 0
    assert decision["false_execution"] == 0
    assert decision["executed_incorrect"] == 0
    assert decision["end_to_end_replay_effective"] == "partial"
    assert decision["next_gate"] == "grounding_citation_recovery"
    assert decision["production_switch_allowed"] is False


def test_no_answer_safety_and_validator_contract_are_explicit():
    if not OUT.exists():
        return
    no_answer = read_json(OUT / "no-answer-analysis.json")
    assert no_answer["denominator"] == 8
    assert no_answer["correct_safe_response"] == 5
    assert no_answer["incorrect_answer_release"] == 3
    validator = read_json(OUT / "frozen-validator-contract.json")
    assert validator["repair_max_attempts"] == 1
    assert validator["unchanged"] is True
