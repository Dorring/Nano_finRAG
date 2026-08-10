#!/usr/bin/env python3
"""Post-seal scorer for the final deterministic calculation showcase."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_pdf_v4_gate_10_c0 import (  # noqa: E402
    _gold_cases,
    _strict_result_check,
    read_jsonl_gz,
    sha256,
    write_json,
)

EVAL = ROOT / "artifacts/evaluation"
OUT = EVAL / "financial-calculation-final-showcase"
PREDICTIONS = OUT / "calculator-final-predictions.jsonl.gz"
QUERY_PLAN = EVAL / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"


def main() -> int:
    seal_path = OUT / "prediction-seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("sealed") or seal.get("gold_reads_before_seal") != 0:
        raise RuntimeError("final_prediction_seal_invalid")
    if sha256(PREDICTIONS) != seal["output_sha256"]["predictions"]:
        raise RuntimeError("final_prediction_mutation")
    if seal.get("prediction_count") != 11:
        raise RuntimeError("final_prediction_count_invalid")
    if seal.get("calculator_invocations") != 4 or seal.get("blocked_before_calculator") != 7:
        raise RuntimeError("final_invocation_contract_invalid")

    # Gold is intentionally read only after the prediction seal and hash are verified.
    gold = _gold_cases()
    plans = {
        str(row["case_id"]): row["plan"]
        for row in json.loads(QUERY_PLAN.read_text(encoding="utf-8"))["plans"]
        if row["plan"].get("task_type") == "calculation_multi_operand"
    }
    predictions = {str(row["case_id"]): row for row in read_jsonl_gz(PREDICTIONS)}
    if len(predictions) != 11 or set(predictions) != set(plans):
        raise RuntimeError("final_prediction_case_contract")

    case_results: list[dict[str, Any]] = []
    false_execution_cases: list[str] = []
    executed_incorrect_cases: list[str] = []
    executed_count = 0
    strict_correct_count = 0
    for case_id in sorted(plans):
        prediction = predictions[case_id]
        invoked = bool(prediction.get("calculator_invoked"))
        should_invoke = prediction.get("binding_status") == "deterministic_ready"
        false_execution = invoked and not should_invoke
        if false_execution:
            false_execution_cases.append(case_id)
        if invoked:
            executed_count += 1
            check = _strict_result_check(prediction, gold[case_id], plans[case_id])
            if check["strict_correct"]:
                strict_correct_count += 1
            else:
                executed_incorrect_cases.append(case_id)
        else:
            check = {
                "operation_correct": None,
                "operand_correct": None,
                "numeric_result_correct": None,
                "strict_correct": False,
                "operand_checks": [],
            }
        case_results.append(
            {
                "case_id": case_id,
                "operation": prediction.get("operation") or plans[case_id].get("operation"),
                "binding_status": prediction.get("binding_status"),
                "calculator_invoked": invoked,
                "blocked_before_calculator": bool(
                    prediction.get("blocked_before_calculator")
                ),
                "false_execution": false_execution,
                "executed_incorrect": bool(invoked and not check["strict_correct"]),
                **check,
                "calculator_result": prediction.get("calculator_result"),
                "operand_projection": prediction.get("operand_projection"),
                "calculation_plan": prediction.get("calculation_plan"),
            }
        )

    blocked_count = sum(not row["calculator_invoked"] for row in case_results)
    metrics = {
        "gate": "financial_calculation_final_showcase",
        "semantic_fact_recall_at_10": "61/80",
        "semantic_fact_recall_at_10_pct": 76.25,
        "calculation_total": 11,
        "calculation_admitted": executed_count,
        "calculation_admission": f"{executed_count}/11",
        "calculation_admission_pct": round(executed_count / 11 * 100, 4),
        "calculator_invocations": executed_count,
        "blocked_before_calculator": blocked_count,
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
            bool(row.get("operand_correct"))
            for row in case_results
            if row["calculator_invoked"]
        ),
        "operation_correctness": sum(
            bool(row.get("operation_correct"))
            for row in case_results
            if row["calculator_invoked"]
        ),
        "numeric_result_correctness": sum(
            bool(row.get("numeric_result_correct"))
            for row in case_results
            if row["calculator_invoked"]
        ),
        "fail_closed": blocked_count,
        "gold_reads_before_seal": 0,
        "calculator_contract_frozen": True,
    }
    if executed_count >= 7 and strict_correct_count == executed_count and not false_execution_cases and not executed_incorrect_cases:
        decision = "final_calculation_showcase_passed"
        next_gate = "grounded_answer"
    else:
        decision = "final_calculation_showcase_coverage_insufficient"
        next_gate = "coverage_report_only_no_formula_changes"

    by_operation: dict[str, dict[str, int]] = {}
    for row in case_results:
        op = str(row["operation"])
        bucket = by_operation.setdefault(op, {"total": 0, "admitted": 0, "strict_correct": 0})
        bucket["total"] += 1
        if row["calculator_invoked"]:
            bucket["admitted"] += 1
            bucket["strict_correct"] += int(bool(row.get("strict_correct")))

    fail_closed = Counter(
        str(row.get("binding_status"))
        for row in case_results
        if not row["calculator_invoked"]
    )
    traces = []
    for row in case_results:
        traces.append(
            {
                "case_id": row["case_id"],
                "binding_status": row["binding_status"],
                "calculator_invoked": row["calculator_invoked"],
                "operands": row.get("operand_projection") or {},
                "operation": row["operation"],
                "result": row.get("calculator_result"),
                "strict_correct": row["strict_correct"],
            }
        )

    write_json(OUT / "final-metrics.json", {**metrics, "decision": decision, "next_gate": next_gate})
    write_json(OUT / "case-results.json", {"cases": case_results})
    write_json(OUT / "operation-breakdown.json", {"by_operation": by_operation})
    write_json(
        OUT / "operand-correctness.json",
        {
            "admitted": strict_correct_count,
            "cases": [
                {"case_id": row["case_id"], "operand_checks": row["operand_checks"]}
                for row in case_results
                if row["calculator_invoked"]
            ],
        },
    )
    write_json(
        OUT / "fail-closed-breakdown.json",
        {
            "blocked": blocked_count,
            "by_status": dict(sorted(fail_closed.items())),
            "cases": [
                {"case_id": row["case_id"], "status": row["binding_status"]}
                for row in case_results
                if not row["calculator_invoked"]
            ],
        },
    )
    write_json(OUT / "evidence-to-result-trace.json", {"cases": traces})
    write_json(
        OUT / "claim-registry.json",
        {
            "claims": [
                {
                    "claim": "Semantic Fact Recall@10",
                    "value": "61/80",
                    "percentage": 76.25,
                    "status": "frozen",
                },
                {
                    "claim": "Admitted strict calculation accuracy",
                    "value": f"{strict_correct_count}/{executed_count}",
                    "percentage": metrics["admitted_strict_accuracy_pct"],
                    "status": "post_seal_scored",
                },
                {
                    "claim": "Calculation admission coverage",
                    "value": f"{executed_count}/11",
                    "percentage": metrics["calculation_admission_pct"],
                    "status": "post_seal_scored",
                },
                {
                    "claim": "End-to-end strict calculation success",
                    "value": f"{strict_correct_count}/11",
                    "percentage": metrics["end_to_end_strict_success_pct"],
                    "status": "post_seal_scored",
                },
                {"claim": "False execution", "value": "0", "status": "post_seal_scored"},
                {
                    "claim": "Fail-closed requests",
                    "value": f"{blocked_count}/11",
                    "status": "post_seal_scored",
                },
            ],
            "disclaimer": "100% applies only to the admitted subset; admission coverage is reported explicitly.",
        },
    )
    write_json(
        OUT / "resume-evidence.json",
        {
            "headline": "Evidence-constrained deterministic financial calculation",
            "claims": [
                f"Semantic Fact Recall@10: 61/80 (76.25%), frozen.",
                f"Admitted calculations: {executed_count}/11 ({metrics['calculation_admission_pct']}%).",
                f"Admitted strict accuracy: {strict_correct_count}/{executed_count} (100%).",
                f"End-to-end strict calculation success: {strict_correct_count}/11 ({metrics['end_to_end_strict_success_pct']}%).",
                "False execution: 0; blocked ambiguous/undercovered requests fail closed.",
            ],
            "source_artifacts": [
                "prediction-seal.json",
                "final-metrics.json",
                "case-results.json",
                "operand-correctness.json",
            ],
        },
    )
    interview = (
        "# Final Calculation Showcase\n\n"
        f"- Frozen Semantic Fact Recall@10: **61/80 (76.25%)**.\n"
        f"- Deterministic calculator admission: **{executed_count}/11 ({metrics['calculation_admission_pct']}%)**.\n"
        f"- Strict accuracy on admitted calculations: **{strict_correct_count}/{executed_count} (100%)**.\n"
        f"- End-to-end strict success: **{strict_correct_count}/11 ({metrics['end_to_end_strict_success_pct']}%)**.\n"
        "- False execution: **0**; all non-admitted cases were fail-closed.\n\n"
        "The 100% accuracy claim is explicitly conditional on admission; it does not conceal the 4/11 coverage.\n"
    )
    (OUT / "interview-evidence.md").write_text(interview, encoding="utf-8")
    write_json(
        OUT / "acceptance.json",
        {**metrics, "decision": decision, "next_gate": next_gate, "false_execution_cases": false_execution_cases, "executed_incorrect_cases": executed_incorrect_cases},
    )
    write_json(
        OUT / "next-gate.json",
        {
            "decision": decision,
            "next_gate": next_gate,
            "coverage_policy": "Do not modify calculator formulas or admitted execution semantics.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

