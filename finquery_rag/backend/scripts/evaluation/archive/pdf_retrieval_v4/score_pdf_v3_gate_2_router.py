"""Gate 2 scoring: verify the prediction seal before reading governance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GOVERNANCE = ROOT / "benchmarks/financial_rag_v1/governance/benchmark-governance.jsonl"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--governance", type=Path, default=GOVERNANCE)
    args = parser.parse_args()
    seal = json.loads((args.out_dir / "router-prediction-seal.json").read_text(encoding="utf-8"))
    if not seal["predictions_sealed"] or seal["prediction_hash"] != _sha(args.out_dir / "router-predictions.json") or seal["protocol_hash"] != _sha(args.out_dir / "router-protocol.json"):
        raise RuntimeError("router prediction seal verification failed")
    predictions = {str(item["case_id"]): item for item in json.loads((args.out_dir / "router-predictions.json").read_text(encoding="utf-8"))["predictions"]}
    governance = [json.loads(line) for line in args.governance.read_text(encoding="utf-8").splitlines() if line]
    if set(predictions) != {str(item["case_id"]) for item in governance}:
        raise RuntimeError("prediction and governance case sets differ")
    task_ok = calc_ok = multi_ok = operand_ok = operation_ok = period_ok = metric_count_ok = 0
    direct_to_calc = narrative_to_structured = no_answer_outcome = 0
    confusion = Counter()
    errors = []
    no_answer_records = []
    operand_audit = []
    period_audit = []
    metric_audit = []
    for expected in governance:
        case_id = str(expected["case_id"])
        actual = predictions[case_id]["profile"]
        expected_type = expected["query_type"]
        actual_type = actual["task_type"]
        is_no_answer = expected_type == "no_answer"
        type_match = (actual_type != "unsupported") if is_no_answer else actual_type == expected_type
        task_ok += int(type_match)
        expected_slots = list(expected["operand_slots"])
        expected_periods = {item["period"] for item in expected_slots if item.get("period")}
        actual_periods = {item["normalized_period"] for item in actual["periods"] if item.get("normalized_period")}
        expected_metric_count = len({str(item["metric"]).lower() for item in expected_slots if item.get("metric")})
        actual_metric_count = len(actual["metric_phrases"])
        operand_match = is_no_answer or actual["expected_operand_count"] == expected["minimum_evidence_count"]
        operation_match = actual["operation"] == expected["operation"]
        period_match = is_no_answer or actual_periods == expected_periods
        metric_match = is_no_answer or actual_metric_count == expected_metric_count
        operand_ok += int(operand_match)
        operation_ok += int(operation_match)
        period_ok += int(period_match)
        metric_count_ok += int(metric_match)
        if expected_type == "calculation_multi_operand":
            calc_ok += int(actual_type == "calculation_multi_operand")
        if expected["requires_multiple_sources"]:
            multi_ok += int(actual["requires_multiple_sources"])
        if expected_type in {"table_single_fact", "direct_single_fact"}:
            direct_to_calc += int(actual_type == "calculation_multi_operand")
        if expected_type == "narrative_or_note":
            narrative_to_structured += int(actual_type in {"table_single_fact", "single_metric_multi_period", "multi_metric_comparison", "calculation_multi_operand"})
        if is_no_answer:
            no_answer_outcome += int(actual_type == "no_answer")
            no_answer_records.append({"case_id": case_id, "task_type": actual_type, "answerability_check_required": actual["answerability_check_required"]})
        confusion[(expected_type, actual_type)] += 1
        error_type = None
        if not type_match:
            error_type = "calculation_missed" if expected_type == "calculation_multi_operand" else "unsupported_false_positive" if actual_type == "unsupported" else "multi_period_missed" if expected_type == "single_metric_multi_period" else "comparison_missed"
        elif not period_match:
            error_type = "period_missing" if expected_periods - actual_periods else "period_over_extracted"
        elif not metric_match:
            error_type = "metric_phrase_split_error" if actual_metric_count > expected_metric_count else "metric_phrase_merge_error"
        if error_type:
            errors.append({"case_id": case_id, "error_type": error_type, "expected_task_type": expected_type, "actual_task_type": actual_type})
        operand_audit.append({"case_id": case_id, "expected": expected["minimum_evidence_count"], "actual": actual["expected_operand_count"], "matched": operand_match})
        period_audit.append({"case_id": case_id, "expected_periods": sorted(expected_periods), "actual_periods": sorted(actual_periods), "matched": period_match})
        metric_audit.append({"case_id": case_id, "expected_count": expected_metric_count, "actual_count": actual_metric_count, "matched": metric_match})
    total = len(governance)
    calculation_total = sum(item["query_type"] == "calculation_multi_operand" for item in governance)
    multi_total = sum(bool(item["requires_multiple_sources"]) for item in governance)
    no_answer_total = sum(item["query_type"] == "no_answer" for item in governance)
    metrics = {"task_type_accuracy": _ratio(task_ok, total), "calculation_route_recall": _ratio(calc_ok, calculation_total), "multi_source_route_recall": _ratio(multi_ok, multi_total), "operand_count_accuracy": _ratio(operand_ok, total), "operation_accuracy": _ratio(operation_ok, total), "period_set_exact_match": _ratio(period_ok, total), "metric_phrase_count_accuracy": _ratio(metric_count_ok, total), "direct_fact_to_calculation_count": direct_to_calc, "narrative_to_structured_count": narrative_to_structured, "unsupported_rate": _ratio(sum(item["profile"]["task_type"] == "unsupported" for item in predictions.values()), total), "no_answer_cases_directly_classified_as_no_answer": no_answer_outcome, "no_answer_cases_answerability_check_required": _ratio(sum(item["answerability_check_required"] for item in no_answer_records), no_answer_total), "no_answer_cases_structurally_valid": _ratio(sum(item["task_type"] != "unsupported" for item in no_answer_records), no_answer_total)}
    passed = task_ok >= 69 and calc_ok == calculation_total and multi_ok >= multi_total - 1 and operand_ok >= 69 and operation_ok >= 69 and period_ok >= 69 and metric_count_ok >= 69 and direct_to_calc == 0 and narrative_to_structured == 0 and no_answer_outcome == 0
    _write(args.out_dir / "router-metrics.json", metrics)
    _write(args.out_dir / "router-confusion-matrix.json", {"cells": [{"expected": key[0], "actual": key[1], "count": value} for key, value in sorted(confusion.items())]})
    _write(args.out_dir / "router-error-attribution.json", {"error_count": len(errors), "errors": errors})
    _write(args.out_dir / "operand-count-audit.json", {"records": operand_audit})
    _write(args.out_dir / "period-extraction-audit.json", {"records": period_audit})
    _write(args.out_dir / "metric-phrase-audit.json", {"records": metric_audit})
    _write(args.out_dir / "no-answer-boundary-audit.json", {"records": no_answer_records})
    _write(args.out_dir / "acceptance.json", {"gate": "pdf_retrieval_v3_gate_2", "evaluation_type": "post_benchmark_iterative_evaluation", "gate_passed": passed, "decision": "query_profile_router_passed" if passed else "query_profile_router_blocked", "next_gate": "raw_protected_structured_lane" if passed else "stop_and_fix_router", "retrieval_calls": 0, "embedding_calls": 0, "reranker_calls": 0, "answer_generation_calls": 0, "production_index_writes": 0, "production_default_config_modified": False, "runtime_governance_reads": 0, "runtime_gold_reads": 0, "production_switch_allowed": False, "scoring_governance_reads": total})
    _write(args.out_dir / "next-gate.json", {"next_gate": "raw_protected_structured_lane" if passed else "stop_and_fix_router", "gate_3_allowed": passed})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
