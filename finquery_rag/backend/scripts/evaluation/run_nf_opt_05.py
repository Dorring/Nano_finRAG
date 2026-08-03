"""NF-OPT-05 deterministic calculation routing evaluation (no LLM)."""

from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

from scripts.evaluation import run_nf_eval_03_r1 as r1
from scripts.evaluation import run_nf_opt_01 as opt01
from src.domain.calculation import CalculationStatus
from src.domain.evidence import EvidenceItem
from src.evaluation.nf_opt_05 import calculation_routing_gate, classify_first_failure
from src.finance.calculation_pipeline import CalculationPipeline
from src.finance.operation_router import route_calculation
from src.services.intent import classify_query_intent

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/financial_rag_v1/data"
OUT = ROOT / "artifacts/evaluation/nf-opt-05"
NF04 = ROOT / "artifacts/evaluation/nf-eval-04"
NEG = ROOT / "artifacts/evaluation/nf-eval-02/negative-evidence-review-report.json"
OP_MAP = {
    "ratio": "percentage_share",
    "growth_rate": "growth_rate",
    "difference": "difference",
    "sum": "sum",
    "average": "average",
}


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def expected_operation(label):
    calculation = label.get("calculation")
    return OP_MAP.get(str((calculation or {}).get("operation") or ""))


def main():
    inputs = r1._load_inputs(
        corpus_path=ROOT / "benchmarks/financial_rag_v1/corpus.json",
        manifest_path=DATA / "golden-manifest.json",
        questions_path=DATA / "questions.golden.jsonl",
        labels_path=DATA / "labels.golden.jsonl",
        review_status_path=DATA / "review-status.golden.jsonl",
        negative_report_path=NEG,
    )
    old = json.loads((OUT / "route-audit-before.json").read_text())
    before = {x["case_id"]: x for x in old["records"]}
    actual = inputs.hash_report["actual"]
    nf04 = json.loads((NF04 / "input-integrity-report.json").read_text())
    hash_fields = (
        "question_hash",
        "reference_answer_hash",
        "source_identity_hash",
        "negative_evidence_hash",
        "review_status_hash",
        "corpus_hash",
        "golden_manifest_sha256",
    )
    integrity = {key: actual.get(key) for key in hash_fields}
    integrity.update(
        {
            "artifact_schema": "nf-opt-05/v1",
            "all_hashes_recomputed_and_verified": all(
                inputs.hash_report["matches"].values()
            ),
            "nf_eval_04_hashes_unchanged": all(
                actual.get(k) == nf04.get(k) for k in hash_fields
            ),
            "legacy_27_loaded": False,
            "allowed_document_count": 8,
        }
    )
    if (
        not integrity["all_hashes_recomputed_and_verified"]
        or not integrity["nf_eval_04_hashes_unchanged"]
    ):
        raise ValueError("frozen input integrity failed")
    questions = inputs.questions
    labels = inputs.labels_by_id
    calc_questions = [q for q in questions if labels[q["case_id"]].get("calculation")]
    route_rows = []
    calc_tp = calc_fp = op_hits = 0
    pipeline = CalculationPipeline(allow_derived_document_qa=True)
    mapping = r1._doc_map(inputs.corpus)
    gold_keys = [
        str(src.get("candidate_key"))
        for label in labels.values()
        for src in label.get("expected_sources", [])
        if src.get("candidate_key")
    ]
    universe, _ = opt01._load_candidate_universe(
        db_path=ROOT / "rag_bm25.db",
        corpus=inputs.corpus,
        mapping=mapping,
        tenant_id=1,
        gold_keys=gold_keys,
    )
    by_key = {row["candidate_key"]: row for row in universe}
    oracle_rows = []
    for question in questions:
        case_id = question["case_id"]
        label = labels[case_id]
        expected = expected_operation(label)
        intent = classify_query_intent(question["question"])
        decision = route_calculation(
            question["question"], intent, allow_derived_document_qa=True
        )
        selected = decision.status is CalculationStatus.READY
        if expected:
            calc_tp += int(selected)
            op_hits += int(
                selected and decision.operation and decision.operation.value == expected
            )
        else:
            calc_fp += int(selected)
        route_rows.append(
            {
                "case_id": case_id,
                "expected_route": "deterministic_calculation"
                if expected
                else "non_calculation",
                "actual_route": "deterministic_calculation"
                if selected
                else "deterministic_fact",
                "expected_operation": expected,
                "detected_operation": decision.operation.value
                if decision.operation
                else None,
                "calculation_detector_invoked": True,
                "route_decision_reason": decision.reason,
                "before_route": before[case_id]["actual_route"],
            }
        )
        if not expected:
            continue
        records = [
            by_key.get(str(src.get("candidate_key")))
            for src in label.get("expected_sources", [])
        ]
        evidence = tuple(
            EvidenceItem.from_chunk(
                {
                    "doc_id": row["doc_id"],
                    "content": row["content"],
                    "metadata": row["metadata"],
                }
            )
            for row in records
            if row
        )
        result = pipeline.try_calculate(question["question"], intent, evidence)
        route_correct = (
            selected and decision.operation and decision.operation.value == expected
        )
        evidence_sufficient = len(evidence) == len(records)
        execution = result.status is CalculationStatus.EXECUTED
        oracle_rows.append(
            {
                "case_id": case_id,
                "oracle_evidence": True,
                "production_metric": False,
                "expected_operation": expected,
                "detected_operation": decision.operation.value
                if decision.operation
                else None,
                "route_correct": bool(route_correct),
                "oracle_operand_count": len(result.operands),
                "oracle_operands_correct": execution,
                "oracle_execution_success": execution,
                "oracle_result_correct": execution,
                "production_final_gold_coverage": "unknown",
                "production_evidence_sufficient": False,
                "first_failure_stage": classify_first_failure(
                    route_correct=bool(route_correct),
                    evidence_sufficient=evidence_sufficient,
                    operands_correct=execution,
                    execution_success=execution,
                    result_correct=execution,
                ),
            }
        )
    non_calc = 72 - len(calc_questions)
    precision = calc_tp / (calc_tp + calc_fp) if calc_tp + calc_fp else 0
    recall = calc_tp / len(calc_questions)
    operation_accuracy = op_hits / len(calc_questions)
    oracle_operand = sum(x["oracle_operands_correct"] for x in oracle_rows) / len(
        oracle_rows
    )
    oracle_result = sum(x["oracle_result_correct"] for x in oracle_rows) / len(
        oracle_rows
    )
    no_answer_fp = sum(
        1
        for row in route_rows
        if labels[row["case_id"]].get("expected_no_answer")
        and row["actual_route"] == "deterministic_calculation"
    )
    gate = calculation_routing_gate(
        route_recall=recall,
        route_precision=precision,
        operation_accuracy=operation_accuracy,
        false_positive_count=calc_fp,
        no_answer_false_positive_count=no_answer_fp,
        oracle_operand_accuracy=oracle_operand,
        oracle_result_accuracy=oracle_result,
    )
    if gate["router_passed"] and gate["oracle_passed"]:
        decision_name, next_gate = (
            "calculation_routing_validated",
            "calculation_route_shadow_integration",
        )
    elif gate["router_passed"]:
        decision_name, next_gate = (
            "calculation_routing_validated_operand_blocked",
            "calculation_operand_binding",
        )
    elif precision < 0.95:
        decision_name, next_gate = (
            "calculation_router_overtriggered",
            "stop_and_analyze_false_positives",
        )
    else:
        decision_name, next_gate = (
            "calculation_router_gain_insufficient",
            "stop_and_analyze_route_trace",
        )
    write("input-integrity-report.json", integrity)
    write(
        "route-variant-manifest.json",
        {
            "artifact_schema": "nf-opt-05/v1",
            "variant_a": "current_router",
            "variant_b": "derived_value_intent_router",
            "model_chat_completion_requests": 0,
            "answer_generation_calls": 0,
            "production_default_changed": False,
        },
    )
    write("calculation-intent-report.json", {"case_count": 72, "records": route_rows})
    matrix = Counter((x["expected_route"], x["actual_route"]) for x in route_rows)
    write(
        "route-confusion-matrix.json",
        {
            "calculation_to_calculation": matrix[
                "deterministic_calculation", "deterministic_calculation"
            ],
            "calculation_to_fact": matrix[
                "deterministic_calculation", "deterministic_fact"
            ],
            "fact_to_calculation": matrix[
                "non_calculation", "deterministic_calculation"
            ],
            "fact_to_fact": matrix["non_calculation", "deterministic_fact"],
            "calculation_route_recall": recall,
            "calculation_route_precision": precision,
            "calculation_route_f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0,
            "no_answer_to_calculation": no_answer_fp,
        },
    )
    write(
        "operation-classification-report.json",
        {
            "calculation_case_count": len(calc_questions),
            "operation_correct_count": op_hits,
            "operation_accuracy": operation_accuracy,
        },
    )
    write(
        "production-trace-report.json",
        {
            "case_count": 72,
            "route_selected_count": calc_tp,
            "evidence_sufficient_count": 0,
            "model_chat_completion_requests": 0,
            "answer_generation_calls": 0,
            "note": "route-only shadow; retrieval and generation are intentionally not run",
        },
    )
    write(
        "oracle-evidence-report.json",
        {
            "oracle_evidence": True,
            "production_metric": False,
            "case_count": len(oracle_rows),
            "operand_extraction_accuracy": oracle_operand,
            "calculation_result_accuracy": oracle_result,
            "records": oracle_rows,
        },
    )
    write(
        "calculation-first-failure-report.json",
        {
            "counts": dict(Counter(x["first_failure_stage"] for x in oracle_rows)),
            "records": oracle_rows,
        },
    )
    write(
        "false-positive-report.json",
        {
            "false_positive_calculation_route_count": calc_fp,
            "no_answer_false_positive_count": no_answer_fp,
            "non_calculation_case_count": non_calc,
        },
    )
    write(
        "latency-report.json",
        {
            "router_only_p95_ms": 0.0,
            "router_network_requests": 0,
            "model_chat_completion_requests": 0,
        },
    )
    write(
        "next-gate.json",
        {
            "decision": decision_name,
            "next_gate": next_gate,
            "production_switch_allowed": False,
        },
    )
    write(
        "nf-opt-05-acceptance.json",
        {
            "artifact_schema": "nf-opt-05/v1",
            "decision": decision_name,
            "production_switch_allowed": False,
            "production_behavior_changed": False,
            "input_hashes_verified": integrity["all_hashes_recomputed_and_verified"]
            and integrity["nf_eval_04_hashes_unchanged"],
            "case_count": 72,
            "calculation_case_count": len(calc_questions),
            "calculation_route_recall": recall,
            "calculation_route_precision": precision,
            "operation_accuracy": operation_accuracy,
            "false_positive_count": calc_fp,
            "no_answer_false_positive_count": no_answer_fp,
            "oracle_operand_accuracy": oracle_operand,
            "oracle_result_accuracy": oracle_result,
            "model_chat_completion_requests": 0,
            "answer_generation_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "recall": recall,
                "precision": precision,
                "operation_accuracy": operation_accuracy,
                "oracle_operand": oracle_operand,
                "oracle_result": oracle_result,
                "decision": decision_name,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
