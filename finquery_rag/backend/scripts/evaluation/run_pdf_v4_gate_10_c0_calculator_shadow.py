#!/usr/bin/env python3
"""Gate10 C0: execute only the three frozen B2-ready calculations.

This runner is intentionally a thin adapter. It consumes the sealed B2
operand projections and frozen QueryPlan, then delegates arithmetic to the
existing calculation registry/executor. It never reads benchmark labels.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

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

EVAL = ROOT / "artifacts/evaluation"
B2 = EVAL / "pdf-retrieval-v4-gate-09-r5-2-b2"
QUERY_PLAN = EVAL / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"
OUT = EVAL / "pdf-retrieval-v4-gate-10-c0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(
                            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                        )
                        + "\n"
                    )


def _operation(value: str | None) -> CalculationOperation:
    if not value:
        raise RuntimeError("calculation_operation_missing")
    try:
        return CalculationOperation(value)
    except ValueError as exc:
        raise RuntimeError(f"unsupported_query_plan_operation:{value}") from exc


def _decimal(value: Any) -> Decimal:
    if value is None:
        raise ValueError("operand_value_missing")
    return Decimal(str(value))


def _projection_operand(
    *, slot: dict[str, Any], payload: dict[str, Any], issuer: str
) -> CalculationOperand:
    normalized = payload.get("normalized_value")
    raw = normalized if normalized is not None else payload.get("value")
    unit_context = payload.get("unit_context") or {}
    scale = unit_context.get("scale")
    unit = payload.get("measurement_kind") or "unknown"
    fact_id = str(payload.get("semantic_fact_id") or "")
    return CalculationOperand(
        name=str(slot.get("role") or slot.get("slot_id") or "operand"),
        value=_decimal(raw),
        unit=str(unit),
        scale=str(scale) if scale is not None else None,
        source_text="",
        evidence_chunk_id=fact_id,
        document_name=issuer,
        page=None,
    )


def _blocked_result_record(
    *, case_id: str, operation: str | None, binding_status: str, reason: str
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "operation": operation,
        "binding_status": binding_status,
        "calculation_runtime_ready": False,
        "calculator_invoked": False,
        "blocked_before_calculator": True,
        "blocked_reason": reason,
        "calculator_result": {
            "status": "blocked",
            "operation": operation,
            "error_code": "BINDER_BLOCKED",
            "error_message": reason,
            "operands": [],
        },
        "operand_projection": {},
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    b2_seal_path = B2 / "prediction-seal.json"
    b2_seal = json.loads(b2_seal_path.read_text(encoding="utf-8"))
    if not b2_seal.get("sealed") or b2_seal.get("gold_reads_before_seal") != 0:
        raise RuntimeError("b2_prediction_seal_invalid")

    projection_path = B2 / "operand-projections-b2.jsonl.gz"
    expected_projection_sha = b2_seal["output_sha256"]["projections_b2"]
    if sha256(projection_path) != expected_projection_sha:
        raise RuntimeError("b2_projection_mutation")
    query_plan_payload = json.loads(QUERY_PLAN.read_text(encoding="utf-8"))
    plans = {str(row["case_id"]): row["plan"] for row in query_plan_payload["plans"]}
    calc_plans = {
        case_id: plan
        for case_id, plan in plans.items()
        if plan.get("task_type") == "calculation_multi_operand"
    }
    if len(calc_plans) != 11:
        raise RuntimeError(f"calculation_case_count:{len(calc_plans)}")
    projections = {str(row["case_id"]): row for row in read_jsonl(projection_path)}

    records: list[dict[str, Any]] = []
    invocation_count = 0
    status_counts: Counter[str] = Counter()
    for case_id in sorted(calc_plans):
        plan_data = calc_plans[case_id]
        projection = projections.get(case_id)
        if projection is None:
            raise RuntimeError(f"projection_missing:{case_id}")
        binding_status = str(projection.get("binding_status"))
        status_counts[binding_status] += 1
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
        operands_payload = projection.get("operands") or {}
        operands: list[CalculationOperand] = []
        for slot in plan_data.get("operand_slots") or []:
            slot_id = str(slot["slot_id"])
            payload = operands_payload.get(slot_id)
            if payload is None or not payload.get("deterministic"):
                raise RuntimeError(f"ready_projection_operand_missing:{case_id}:{slot_id}")
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
                or plan_data.get("metric_phrases", [operation.value])[0]
            ),
            precision=4,
            label=case_id,
            status=CalculationStatus.READY,
        )
        invocation_count += 1
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
                "operand_projection": operands_payload,
                "calculation_plan": calc_plan.to_dict(),
            }
        )

    if len(records) != 11 or invocation_count != 3:
        raise RuntimeError(f"calculator_invocation_contract:{len(records)}:{invocation_count}")
    blocked_count = sum(bool(row["blocked_before_calculator"]) for row in records)
    if blocked_count != 8:
        raise RuntimeError(f"blocked_before_calculator_contract:{blocked_count}")

    predictions_path = OUT / "calculator-shadow-predictions.jsonl.gz"
    write_jsonl_gz(predictions_path, records)
    source_paths = {
        "calculator_executor": ROOT / "src/finance/calculation_executor.py",
        "calculation_registry": ROOT / "src/finance/calculation_registry.py",
        "calculation_domain": ROOT / "src/domain/calculation.py",
    }
    protocol = {
        "gate": "pdf_retrieval_v4_gate_10_c0",
        "phase": "deterministic_calculator_shadow_validation",
        "input_b2_prediction_seal_sha256": sha256(b2_seal_path),
        "input_b2_projection_sha256": sha256(projection_path),
        "query_plan_sha256": sha256(QUERY_PLAN),
        "calculator_source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "binder_mutation": 0,
        "metric_contract_mutation": 0,
        "unit_contract_mutation": 0,
        "query_plan_mutation": 0,
        "evidence_set_mutation": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "embedding_calls": 0,
        "bridge_runs": 0,
        "semantic_graph_runs": 0,
        "gold_reads_before_seal": 0,
        "reference_answer_reads_before_seal": 0,
        "expected_value_reads_before_seal": 0,
        "calculator_invocations": invocation_count,
        "blocked_before_calculator": blocked_count,
        "prediction_count": len(records),
        "binding_status_counts": dict(sorted(status_counts.items())),
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(
        OUT / "input-integrity.json",
        {
            "b2_prediction_seal_sha256": sha256(b2_seal_path),
            "b2_projection_sha256": sha256(projection_path),
            "query_plan_sha256": sha256(QUERY_PLAN),
            "case_count": len(calc_plans),
            "candidate_or_semantic_registry_mutation": 0,
            "gold_reads": 0,
        },
    )
    write_json(
        OUT / "runtime-manifest.json",
        {
            "calculator_executor": "src.finance.calculation_executor.execute_plan",
            "calculator_registry": "src.finance.calculation_registry.get_operation_entry",
            "formula_source": "frozen registry",
            "operation_source": "frozen Gate07 QueryPlan",
            "generation": 0,
        },
    )
    write_json(
        OUT / "prediction-manifest.json",
        {
            "prediction_sha256": sha256(predictions_path),
            "prediction_count": len(records),
            "calculator_invocations": invocation_count,
            "blocked_before_calculator": blocked_count,
        },
    )
    write_json(
        OUT / "prediction-seal.json",
        {
            "sealed": True,
            "gate": "pdf_retrieval_v4_gate_10_c0",
            "output_sha256": {"predictions": sha256(predictions_path)},
            "b2_prediction_seal_sha256": sha256(b2_seal_path),
            "b2_projection_sha256": sha256(projection_path),
            "query_plan_sha256": sha256(QUERY_PLAN),
            "calculator_invocations": invocation_count,
            "blocked_before_calculator": blocked_count,
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

