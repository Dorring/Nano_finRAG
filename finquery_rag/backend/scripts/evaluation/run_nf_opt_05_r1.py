"""NF-OPT-05 R1 strict Oracle operand evaluation; no retrieval or generation."""

from __future__ import annotations
import json
from pathlib import Path

from scripts.evaluation import run_nf_eval_03_r1 as r1
from scripts.evaluation import run_nf_opt_01 as opt01
from src.domain.calculation import CalculationStatus
from src.domain.evidence import EvidenceItem
from src.evaluation.nf_opt_05_r1 import (
    operand_roles,
    score_operands,
    strict_result_correct,
)
from src.finance.calculation_pipeline import CalculationPipeline
from src.finance.operation_router import route_calculation
from src.services.intent import classify_query_intent

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/financial_rag_v1/data"
OUT = ROOT / "artifacts/evaluation/nf-opt-05-r1"
NEG = ROOT / "artifacts/evaluation/nf-eval-02/negative-evidence-review-report.json"
OP_MAP = {
    "ratio": "percentage_share",
    "growth_rate": "growth_rate",
    "difference": "difference",
}


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    inputs = r1._load_inputs(
        corpus_path=ROOT / "benchmarks/financial_rag_v1/corpus.json",
        manifest_path=DATA / "golden-manifest.json",
        questions_path=DATA / "questions.golden.jsonl",
        labels_path=DATA / "labels.golden.jsonl",
        review_status_path=DATA / "review-status.golden.jsonl",
        negative_report_path=NEG,
    )
    if not all(inputs.hash_report["matches"].values()):
        raise ValueError("frozen inputs invalid")
    mapping = r1._doc_map(inputs.corpus)
    keys = [
        str(s.get("candidate_key"))
        for label in inputs.labels_by_id.values()
        for s in label.get("expected_sources", [])
        if s.get("candidate_key")
    ]
    universe, _ = opt01._load_candidate_universe(
        db_path=ROOT / "rag_bm25.db",
        corpus=inputs.corpus,
        mapping=mapping,
        tenant_id=1,
        gold_keys=keys,
    )
    by_key = {x["candidate_key"]: x for x in universe}
    records = []
    contracts = []
    pipeline = CalculationPipeline(allow_derived_document_qa=True)
    for question in inputs.questions:
        label = inputs.labels_by_id[question["case_id"]]
        calc = label.get("calculation")
        if not calc:
            continue
        operation = str(calc["operation"])
        roles = operand_roles(operation, len(calc["operands"]))
        expected = []
        for index, item in enumerate(calc["operands"]):
            source = label["expected_sources"][int(item["source_index"])]
            candidate = by_key[str(source["candidate_key"])]
            expected.append(
                {
                    "role": roles[index],
                    "value": str(item["value"]),
                    "metric": item.get("metric"),
                    "period": item.get("period"),
                    "currency": None,
                    "scale": None,
                    "evidence_chunk_id": candidate["doc_id"],
                    "candidate_key": source["candidate_key"],
                }
            )
        contracts.append(
            {
                "case_id": question["case_id"],
                "operation": operation,
                "operands": expected,
                "contract_source": "calculation.operands+expected_sources",
                "unit_scale_contract_available": False,
            }
        )
        evidence = tuple(
            EvidenceItem.from_chunk(
                {
                    "doc_id": by_key[str(s["candidate_key"])]["doc_id"],
                    "content": by_key[str(s["candidate_key"])]["content"],
                    "metadata": by_key[str(s["candidate_key"])]["metadata"],
                }
            )
            for s in label["expected_sources"]
        )
        intent = classify_query_intent(question["question"])
        decision = route_calculation(
            question["question"], intent, allow_derived_document_qa=True
        )
        result = pipeline.try_calculate(question["question"], intent, evidence)
        actual = [x.to_trace_dict() for x in result.operands]
        checks = score_operands(expected=expected, actual=actual)
        extraction_completed = bool(actual)
        execution = result.status is CalculationStatus.EXECUTED
        expected_value = (
            label["expected_answer"].get("canonical_value")
            if label.get("expected_answer")
            else None
        )
        # Existing executor emits fractional ratios, whereas benchmark labels use percentage points.
        expected_unit = "percentage" if "result_percentage" in calc else None
        actual_value = str(result.value) if result.value is not None else None
        actual_unit = result.unit
        result_correct = strict_result_correct(
            execution_completed=execution,
            actual_value=actual_value,
            expected_value=expected_value,
            actual_unit=actual_unit,
            expected_unit=expected_unit,
        )
        if not extraction_completed:
            first = "operand_extraction_empty"
        elif not checks["operand_count_correct"]:
            first = "operand_count_mismatch"
        elif not checks["operand_role_assignment_correct"]:
            first = "operand_role_mismatch"
        elif not checks["operand_value_correct"]:
            first = "operand_value_mismatch"
        elif not checks["operand_evidence_identity_correct"]:
            first = "operand_evidence_identity_mismatch"
        elif not execution:
            first = "calculation_execution_failed"
        elif not result_correct:
            first = "calculation_result_wrong"
        else:
            first = "passed"
        records.append(
            {
                "case_id": question["case_id"],
                "expected_operation": OP_MAP.get(operation, operation),
                "route_correct": bool(
                    decision.operation
                    and decision.operation.value == OP_MAP.get(operation, operation)
                ),
                "extraction_completed": extraction_completed,
                "expected_operand_count": len(expected),
                "actual_operand_count": len(actual),
                **checks,
                "operand_metric_evaluable": False,
                "operand_metric_correct": False,
                "operand_metric_not_evaluable_reason": "CalculationOperand has no metric field",
                "operand_period_evaluable": False,
                "operand_period_correct": False,
                "operand_period_not_evaluable_reason": "CalculationOperand has no period field",
                "operand_unit_scale_evaluable": False,
                "operand_unit_scale_correct": False,
                "operand_unit_scale_not_evaluable_reason": "CalculationOperand has no currency or scale field",
                "actual_operands": actual,
                "execution_completed": execution,
                "actual_result_value": actual_value,
                "actual_result_unit": actual_unit,
                "expected_result_value": expected_value,
                "expected_result_unit": expected_unit,
                "calculation_result_correct": result_correct,
                "first_failure_stage": first,
            }
        )
    counts = {
        field: sum(bool(x.get(field)) for x in records)
        for field in (
            "extraction_completed",
            "operand_count_correct",
            "operand_role_assignment_correct",
            "operand_value_correct",
            "operand_evidence_identity_correct",
            "actual_operands_have_evidence_identity",
            "execution_completed",
            "calculation_result_correct",
        )
    }
    write(
        "oracle-operand-contract.json",
        {
            "artifact_schema": "nf-opt-05-r1/v1",
            "case_count": len(contracts),
            "contracts": contracts,
        },
    )
    write(
        "oracle-operand-evaluation.json",
        {"case_count": len(records), "counts": counts, "records": records},
    )
    write(
        "oracle-result-evaluation.json",
        {
            "case_count": len(records),
            "execution_completed_count": counts["execution_completed"],
            "calculation_result_correct_count": counts["calculation_result_correct"],
            "records": [
                {
                    "case_id": x["case_id"],
                    "execution_completed": x["execution_completed"],
                    "calculation_result_correct": x["calculation_result_correct"],
                    "actual_result_value": x["actual_result_value"],
                    "expected_result_value": x["expected_result_value"],
                }
                for x in records
            ],
        },
    )
    failures = {}
    for item in records:
        failures[item["first_failure_stage"]] = (
            failures.get(item["first_failure_stage"], 0) + 1
        )
    write(
        "oracle-failure-attribution.json",
        {
            "case_count": len(records),
            "counts": failures,
            "records": [
                {
                    "case_id": x["case_id"],
                    "first_failure_stage": x["first_failure_stage"],
                }
                for x in records
            ],
        },
    )
    write(
        "nf-opt-05-r1-acceptance.json",
        {
            "artifact_schema": "nf-opt-05-r1/v1",
            "decision": "calculation_routing_validated_operand_evaluation_closed",
            "router_metrics_unchanged": True,
            "production_behavior_changed": False,
            "next_gate": "structured_calculation_operand_binding",
            "case_count": len(records),
            "model_chat_completion_requests": 0,
            "answer_generation_calls": 0,
            "input_hashes_verified": True,
        },
    )
    print(json.dumps({"counts": counts, "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
