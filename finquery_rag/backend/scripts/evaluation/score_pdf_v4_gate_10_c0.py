#!/usr/bin/env python3
"""Post-seal scorer for Gate10 C0 calculator shadow predictions."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL = ROOT / "artifacts/evaluation"
OUT = EVAL / "pdf-retrieval-v4-gate-10-c0"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
QUERY_PLAN = EVAL / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def dec(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def normalise_operation(value: str | None) -> str | None:
    if value == "ratio":
        return "percentage_share"
    return value


def _gold_cases() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with LABELS.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("calculation"):
                    result[str(row["case_id"])] = row
    return result


def _strict_result_check(
    prediction: dict[str, Any], gold: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    calculation = gold.get("calculation") or {}
    expected = gold.get("expected_answer") or {}
    result = prediction.get("calculator_result") or {}
    pred_operation = normalise_operation(result.get("operation"))
    expected_operation = normalise_operation(
        calculation.get("operation") or plan.get("operation")
    )
    operation_correct = pred_operation == expected_operation

    expected_by_period = {
        str(item.get("period")): dec(item.get("value"))
        for item in calculation.get("operands") or []
    }
    predicted_operands = result.get("operands") or []
    operand_checks: list[dict[str, Any]] = []
    for operand in predicted_operands:
        name = str(operand.get("name") or "")
        slot_period = next(
            (
                slot.get("period")
                for slot in plan.get("operand_slots") or []
                if slot.get("role") == name
            ),
            None,
        )
        expected_value = expected_by_period.get(str(slot_period)) if slot_period else None
        actual_value = dec(operand.get("value"))
        operand_checks.append(
            {
                "operand_name": name,
                "period": slot_period,
                "predicted_value": str(actual_value) if actual_value is not None else None,
                "expected_value": str(expected_value) if expected_value is not None else None,
                "correct": actual_value is not None
                and expected_value is not None
                and actual_value == expected_value,
            }
        )
    operand_correctness = bool(operand_checks) and all(
        item["correct"] for item in operand_checks
    ) and len(operand_checks) == len(expected_by_period)

    actual_value = dec(result.get("value"))
    expected_value = dec(expected.get("canonical_value"))
    tolerance = dec(expected.get("tolerance")) or Decimal("0")
    compare_value = actual_value
    if actual_value is not None and expected.get("unit") == "percentage":
        compare_value = actual_value * Decimal("100")
    numeric_result_correct = (
        compare_value is not None
        and expected_value is not None
        and abs(compare_value - expected_value) <= tolerance
    )
    strict_correct = operation_correct and operand_correctness and numeric_result_correct
    return {
        "operation_correct": operation_correct,
        "operand_correct": operand_correctness,
        "numeric_result_correct": numeric_result_correct,
        "strict_correct": strict_correct,
        "predicted_operation": pred_operation,
        "expected_operation": expected_operation,
        "predicted_value": str(actual_value) if actual_value is not None else None,
        "comparison_value": str(compare_value) if compare_value is not None else None,
        "expected_value": str(expected_value) if expected_value is not None else None,
        "tolerance": str(tolerance),
        "operand_checks": operand_checks,
    }


def main() -> int:
    seal_path = OUT / "prediction-seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("sealed") or seal.get("gold_reads_before_seal") != 0:
        raise RuntimeError("c0_prediction_seal_invalid")
    prediction_path = OUT / "calculator-shadow-predictions.jsonl.gz"
    if sha256(prediction_path) != seal["output_sha256"]["predictions"]:
        raise RuntimeError("c0_prediction_mutation")

    # Gold is intentionally opened only after the sealed prediction hash is verified.
    gold = _gold_cases()
    plan_payload = json.loads(QUERY_PLAN.read_text(encoding="utf-8"))
    plans = {
        str(row["case_id"]): row["plan"]
        for row in plan_payload["plans"]
        if row["plan"].get("task_type") == "calculation_multi_operand"
    }
    predictions = {str(row["case_id"]): row for row in read_jsonl_gz(prediction_path)}
    if set(predictions) != set(plans) or len(predictions) != 11:
        raise RuntimeError("c0_prediction_case_contract")

    case_results: list[dict[str, Any]] = []
    false_execution_cases: list[str] = []
    executed_incorrect_cases: list[str] = []
    strict_correct_count = 0
    executed_count = 0
    for case_id in sorted(plans):
        prediction = predictions[case_id]
        invoked = bool(prediction.get("calculator_invoked"))
        should_invoke = prediction.get("binding_status") == "deterministic_ready"
        false_execution = invoked and not should_invoke
        if false_execution:
            false_execution_cases.append(case_id)
        check = (
            _strict_result_check(prediction, gold[case_id], plans[case_id])
            if invoked
            else {
                "operation_correct": None,
                "operand_correct": None,
                "numeric_result_correct": None,
                "strict_correct": False,
                "operand_checks": [],
            }
        )
        if invoked:
            executed_count += 1
            if check["strict_correct"]:
                strict_correct_count += 1
            else:
                executed_incorrect_cases.append(case_id)
        case_results.append(
            {
                "case_id": case_id,
                "binding_status": prediction.get("binding_status"),
                "calculator_invoked": invoked,
                "blocked_before_calculator": bool(
                    prediction.get("blocked_before_calculator")
                ),
                "false_execution": false_execution,
                **check,
            }
        )

    blocked_count = sum(not bool(row["calculator_invoked"]) for row in case_results)
    metrics = {
        "gate": "pdf_retrieval_v4_gate_10_c0",
        "calculation_total": 11,
        "calculator_invocations": executed_count,
        "blocked_before_calculator": blocked_count,
        "admission_coverage": f"{executed_count}/11",
        "admission_coverage_pct": round(executed_count / 11 * 100, 4),
        "admitted_strict_correct": strict_correct_count,
        "admitted_strict_accuracy": f"{strict_correct_count}/{executed_count}",
        "admitted_strict_accuracy_pct": round(
            strict_correct_count / executed_count * 100, 4
        )
        if executed_count
        else 0.0,
        "end_to_end_strict_success": f"{strict_correct_count}/11",
        "end_to_end_strict_success_pct": round(strict_correct_count / 11 * 100, 4),
        "false_execution": len(false_execution_cases),
        "executed_incorrect": len(executed_incorrect_cases),
        "operand_correctness": sum(
            bool(row.get("operand_correct")) for row in case_results if row["calculator_invoked"]
        ),
        "operation_correctness": sum(
            bool(row.get("operation_correct")) for row in case_results if row["calculator_invoked"]
        ),
        "numeric_result_correctness": sum(
            bool(row.get("numeric_result_correct"))
            for row in case_results
            if row["calculator_invoked"]
        ),
        "strict_source_contract": "post_seal_only",
        "gold_reads_before_seal": 0,
    }
    decision = (
        "deterministic_calculator_execution_contract_validated"
        if executed_count == 3
        and strict_correct_count == 3
        and not false_execution_cases
        and not executed_incorrect_cases
        else "calculator_execution_contract_blocked"
    )
    write_json(OUT / "c0-metrics.json", metrics)
    write_json(OUT / "case-results.json", {"cases": case_results})
    write_json(
        OUT / "operand-correctness.json",
        {"cases": [{"case_id": row["case_id"], "operand_checks": row["operand_checks"]} for row in case_results]},
    )
    write_json(
        OUT / "operation-breakdown.json",
        {"by_operation": dict(Counter(row["expected_operation"] for row in case_results if row["calculator_invoked"]))},
    )
    write_json(
        OUT / "false-execution-audit.json",
        {
            "false_execution": len(false_execution_cases),
            "cases": false_execution_cases,
            "executed_incorrect": len(executed_incorrect_cases),
            "executed_incorrect_cases": executed_incorrect_cases,
        },
    )
    write_json(
        OUT / "acceptance.json",
        {
            **metrics,
            "decision": decision,
            "calculator_contract_frozen": decision
            == "deterministic_calculator_execution_contract_validated",
        },
    )
    write_json(
        OUT / "next-gate.json",
        {
            "decision": decision,
            "next_gate": "remaining_operand_tuple_discriminator"
            if decision == "deterministic_calculator_execution_contract_validated"
            else "calculator_adapter_or_operation_unit_fix",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

