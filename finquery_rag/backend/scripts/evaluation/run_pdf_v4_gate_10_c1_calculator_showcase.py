#!/usr/bin/env python3
"""Gate10 C1 final showcase prediction using the frozen C0 calculator adapter."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.calculation import (  # noqa: E402
    CalculationOperation,
    CalculationOperand,
    CalculationPlan,
    CalculationStatus,
)
from src.finance.calculation_executor import execute_plan  # noqa: E402
from src.finance.calculation_registry import get_operation_entry  # noqa: E402

from run_pdf_v4_gate_10_c0_calculator_shadow import (  # noqa: E402
    _blocked_result_record,
    _operation,
    _projection_operand,
    read_jsonl,
    sha256,
    write_json,
    write_jsonl_gz,
)

EVAL = ROOT / "artifacts/evaluation"
C0 = EVAL / "pdf-retrieval-v4-gate-10-c0"
R53 = EVAL / "pdf-retrieval-v4-gate-09-r5-3"
QUERY_PLAN = EVAL / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"
OUT = EVAL / "financial-calculation-final-showcase"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    c0_seal_path = C0 / "prediction-seal.json"
    c0_acceptance_path = C0 / "acceptance.json"
    c0_seal = json.loads(c0_seal_path.read_text(encoding="utf-8"))
    c0_acceptance = json.loads(c0_acceptance_path.read_text(encoding="utf-8"))
    if not c0_seal.get("sealed") or c0_seal.get("gold_reads_before_seal") != 0:
        raise RuntimeError("c0_prediction_seal_invalid")
    if c0_acceptance.get("decision") != "deterministic_calculator_execution_contract_validated":
        raise RuntimeError("c0_calculator_contract_not_frozen")
    r53_seal_path = R53 / "prediction-seal.json"
    r53_seal = json.loads(r53_seal_path.read_text(encoding="utf-8"))
    if not r53_seal.get("sealed") or r53_seal.get("gold_reads_before_seal") != 0:
        raise RuntimeError("r5_3_prediction_seal_invalid")
    projection_path = R53 / "operand-projections-r5-3.jsonl.gz"
    if sha256(projection_path) != r53_seal["output_sha256"]["projections"]:
        raise RuntimeError("r5_3_projection_mutation")
    plans = {
        str(row["case_id"]): row["plan"]
        for row in json.loads(QUERY_PLAN.read_text(encoding="utf-8"))["plans"]
        if row["plan"].get("task_type") == "calculation_multi_operand"
    }
    projections = {row["case_id"]: row for row in read_jsonl(projection_path)}
    if len(plans) != 11 or set(plans) != set(projections):
        raise RuntimeError("c1_calculation_case_contract")

    records: list[dict[str, Any]] = []
    invocations = 0
    statuses: Counter[str] = Counter()
    for case_id in sorted(plans):
        plan_data = plans[case_id]
        projection = projections[case_id]
        binding_status = str(projection.get("binding_status"))
        statuses[binding_status] += 1
        ready = binding_status == "deterministic_ready" and bool(
            projection.get("calculation_runtime_ready")
        )
        if not ready:
            records.append(
                _blocked_result_record(
                    case_id=case_id,
                    operation=projection.get("operation") or plan_data.get("operation"),
                    binding_status=binding_status,
                    reason=str(projection.get("blocked_reason") or binding_status),
                )
            )
            continue
        operation = _operation(str(plan_data.get("operation") or projection.get("operation")))
        entry = get_operation_entry(operation)
        if entry is None:
            raise RuntimeError(f"registry_entry_missing:{operation.value}")
        operand_payloads = projection.get("operands") or {}
        operands: list[CalculationOperand] = []
        for slot in plan_data.get("operand_slots") or []:
            payload = operand_payloads.get(str(slot["slot_id"]))
            if payload is None or not payload.get("deterministic"):
                raise RuntimeError(f"ready_projection_operand_missing:{case_id}:{slot['slot_id']}")
            operands.append(
                _projection_operand(
                    slot=slot,
                    payload=payload,
                    issuer=str(plan_data.get("issuer") or ""),
                )
            )
        calc_plan = CalculationPlan(
            operation=operation,
            operands=tuple(operands),
            formula_version=entry.formula_version,
            target_metric=str(
                (plan_data.get("operand_slots") or [{}])[0].get("raw_metric_phrase")
                or operation.value
            ),
            precision=4,
            label=case_id,
            status=CalculationStatus.READY,
        )
        invocations += 1
        result = execute_plan(calc_plan)
        records.append(
            {
                "case_id": case_id,
                "operation": operation.value,
                "binding_status": binding_status,
                "calculation_runtime_ready": True,
                "calculator_invoked": True,
                "blocked_before_calculator": False,
                "blocked_reason": None,
                "calculator_result": result.to_dict(),
                "operand_projection": operand_payloads,
                "calculation_plan": calc_plan.to_dict(),
            }
        )
    blocked = sum(bool(row["blocked_before_calculator"]) for row in records)
    if len(records) != 11 or invocations != 4 or blocked != 7:
        raise RuntimeError(f"c1_invocation_contract:{len(records)}:{invocations}:{blocked}")

    prediction_path = OUT / "calculator-final-predictions.jsonl.gz"
    write_jsonl_gz(prediction_path, records)
    source_paths = {
        "calculator_executor": ROOT / "src/finance/calculation_executor.py",
        "calculation_registry": ROOT / "src/finance/calculation_registry.py",
        "calculation_domain": ROOT / "src/domain/calculation.py",
    }
    write_json(
        OUT / "protocol.json",
        {
            "gate": "financial_calculation_final_showcase",
            "phase": "gate10_c1_final_calculation_showcase",
            "semantic_fact_recall_at_10": "61/80",
            "r5_3_prediction_seal_sha256": sha256(r53_seal_path),
            "r5_3_projection_sha256": sha256(projection_path),
            "c0_prediction_seal_sha256": sha256(c0_seal_path),
            "c0_calculator_contract_sha256": sha256(c0_acceptance_path),
            "query_plan_sha256": sha256(QUERY_PLAN),
            "calculator_source_sha256": {name: sha256(path) for name, path in source_paths.items()},
            "retrieval_runs": 0,
            "reranker_calls": 0,
            "embedding_calls": 0,
            "binder_runs": 0,
            "gold_reads_before_seal": 0,
            "reference_answer_reads_before_seal": 0,
            "expected_value_reads_before_seal": 0,
            "calculator_invocations": invocations,
            "blocked_before_calculator": blocked,
            "binding_status_counts": dict(sorted(statuses.items())),
        },
    )
    write_json(
        OUT / "input-integrity.json",
        {
            "r5_3_prediction_seal_sha256": sha256(r53_seal_path),
            "r5_3_projection_sha256": sha256(projection_path),
            "c0_prediction_seal_sha256": sha256(c0_seal_path),
            "query_plan_sha256": sha256(QUERY_PLAN),
            "candidate_mutation": 0,
            "semantic_registry_mutation": 0,
            "gold_reads": 0,
        },
    )
    write_json(
        OUT / "prediction-manifest.json",
        {
            "prediction_sha256": sha256(prediction_path),
            "prediction_count": len(records),
            "calculator_invocations": invocations,
            "blocked_before_calculator": blocked,
        },
    )
    write_json(
        OUT / "prediction-seal.json",
        {
            "sealed": True,
            "gate": "financial_calculation_final_showcase",
            "output_sha256": {"predictions": sha256(prediction_path)},
            "input_integrity_sha256": sha256(OUT / "input-integrity.json"),
            "prediction_count": len(records),
            "calculator_invocations": invocations,
            "blocked_before_calculator": blocked,
            "gold_reads_before_seal": 0,
            "reference_answer_reads_before_seal": 0,
            "expected_value_reads_before_seal": 0,
            "retrieval_runs": 0,
            "reranker_calls": 0,
            "embedding_calls": 0,
            "candidate_mutation": 0,
            "semantic_registry_mutation": 0,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

