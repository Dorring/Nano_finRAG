from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "artifacts/evaluation"


def _read(name: str, artifact: str) -> dict:
    return json.loads((EVAL / name / artifact).read_text(encoding="utf-8"))


def test_c0_calculator_contract_is_frozen() -> None:
    metrics = _read("pdf-retrieval-v4-gate-10-c0", "c0-metrics.json")
    acceptance = _read("pdf-retrieval-v4-gate-10-c0", "acceptance.json")
    assert metrics["calculation_total"] == 11
    assert metrics["calculator_invocations"] == 3
    assert metrics["blocked_before_calculator"] == 8
    assert metrics["admitted_strict_correct"] == 3
    assert metrics["admitted_strict_accuracy"] == "3/3"
    assert metrics["false_execution"] == 0
    assert metrics["executed_incorrect"] == 0
    assert acceptance["calculator_contract_frozen"] is True


def test_r5_3_and_final_admission_counts_are_explicit() -> None:
    r53 = _read("pdf-retrieval-v4-gate-09-r5-3", "acceptance.json")
    final = _read("financial-calculation-final-showcase", "final-metrics.json")
    assert r53["calculation_runtime_ready"] == "4/11"
    assert r53["calculation_runtime_ambiguous"] == "4/11"
    assert r53["calculation_undercovered"] == "3/11"
    assert final["calculation_admitted"] == 4
    assert final["calculation_admission"] == "4/11"
    assert final["admitted_strict_accuracy"] == "4/4"
    assert final["end_to_end_strict_success"] == "4/11"


def test_final_showcase_is_fail_closed_and_does_not_overclaim_accuracy() -> None:
    final = _read("financial-calculation-final-showcase", "final-metrics.json")
    claim_registry = _read("financial-calculation-final-showcase", "claim-registry.json")
    assert final["false_execution"] == 0
    assert final["executed_incorrect"] == 0
    assert final["blocked_before_calculator"] == 7
    assert final["decision"] == "final_calculation_showcase_coverage_insufficient"
    assert final["calculation_admission"] in {"4/11", "5/11", "6/11"}
    assert "100%" in claim_registry["disclaimer"]
    assert any(
        claim["claim"] == "Calculation admission coverage"
        and claim["value"] == "4/11"
        for claim in claim_registry["claims"]
    )


def test_final_prediction_seal_has_no_preseal_gold_reads() -> None:
    seal = _read("financial-calculation-final-showcase", "prediction-seal.json")
    assert seal["sealed"] is True
    assert seal["prediction_count"] == 11
    assert seal["calculator_invocations"] == 4
    assert seal["blocked_before_calculator"] == 7
    assert seal["gold_reads_before_seal"] == 0
    assert seal["reference_answer_reads_before_seal"] == 0
    assert seal["expected_value_reads_before_seal"] == 0
    assert seal["retrieval_runs"] == 0
    assert seal["reranker_calls"] == 0

