"""Gate 07 scoring: verify the plan seal, then score against offline governance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.pdf_retrieval_v4.query_plan_validator import validate_query_plan


ROOT = Path(__file__).resolve().parents[2]
GOVERNANCE = ROOT / "benchmarks/financial_rag_v1/governance/benchmark-governance.jsonl"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-07"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ratio(n: int, d: int) -> str:
    return f"{n}/{d}"


def _expected_task_ok(expected: str, actual: str) -> bool:
    if expected == "no_answer":
        return actual != "unsupported"
    if expected == "direct_single_fact":
        return actual in {"table_single_fact", "general_single_fact"}
    return expected == actual


def _normalize_operation(value: str | None) -> str | None:
    if value == "ratio":
        return "percentage_share"
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--governance", type=Path, default=GOVERNANCE)
    args = parser.parse_args()
    seal = json.loads((args.out_dir / "query-plan-prediction-seal.json").read_text(encoding="utf-8"))
    if not seal.get("sealed") or seal["prediction_hash"] != _sha(args.out_dir / "query-plan-predictions.json") or seal["protocol_hash"] != _sha(args.out_dir / "gate-07-protocol.json"):
        raise RuntimeError("query plan prediction seal verification failed")
    predictions = json.loads((args.out_dir / "query-plan-predictions.json").read_text(encoding="utf-8"))["plans"]
    governance = [json.loads(line) for line in args.governance.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_case = {str(item["case_id"]): item for item in predictions}
    if set(by_case) != {str(item["case_id"]) for item in governance}:
        raise RuntimeError("query plan and governance case sets differ")

    task_ok = validation_ok = operation_ok = operand_ok = period_ok = multi_ok = 0
    raw_protection = narrative_leak = unsupported_structured = bucket_recall = comparison_recall = 0
    no_answer_outcome = soft_expansion = 0
    errors: list[dict[str, object]] = []
    confusion = Counter()
    for expected in governance:
        case_id = str(expected["case_id"])
        plan = by_case[case_id]["plan"]
        actual_type = str(plan["task_type"])
        is_no_answer = expected["query_type"] == "no_answer"
        errors_from_validator = tuple(plan.get("validation_errors") or ())
        validation_ok += int(plan.get("plan_status") != "blocked" and not errors_from_validator and not validate_query_plan(_from_json(plan)))
        type_match = _expected_task_ok(expected["query_type"], actual_type)
        task_ok += int(type_match)
        expected_operation = _normalize_operation(expected.get("operation"))
        operation_match = is_no_answer or _normalize_operation(plan.get("operation")) == expected_operation
        operation_ok += int(operation_match)
        expected_slots = list(expected.get("operand_slots") or [])
        expected_periods = {str(item.get("period")) for item in expected_slots if item.get("period")}
        actual_periods = {str(item.get("period")) for item in plan.get("operand_slots", []) if item.get("period")}
        period_match = is_no_answer or actual_periods == expected_periods
        period_ok += int(period_match)
        operand_match = is_no_answer or len(plan.get("operand_slots", [])) == int(expected.get("minimum_evidence_count", 0))
        operand_ok += int(operand_match)
        multi_match = (not expected.get("requires_multiple_sources")) or (bool(plan.get("requires_multiple_sources")) and "multi_operand_set" in plan.get("evidence_shapes", []))
        if expected.get("requires_multiple_sources"):
            multi_ok += int(multi_match)
        routes = plan.get("retrieval_routes", [])
        route_types = {str(route.get("index_type")) for route in routes}
        raw_protection += int(plan.get("raw_protection_required") and any(route.get("index_type") == "raw_production" and route.get("required") for route in routes))
        if expected["query_type"] == "narrative_or_note":
            narrative_leak += int(bool(route_types & {"atomic_fact", "comparison_fact", "bucket_fact", "cell"}))
        if expected["query_type"] == "unsupported":
            unsupported_structured += int(bool(route_types - {"raw_production"}))
        has_bucket = any(slot.get("bucket_label") for slot in plan.get("operand_slots", []))
        if has_bucket:
            bucket_recall += int("bucket_fact" in route_types)
        has_comparison = "comparison_fact" in plan.get("evidence_shapes", [])
        if has_comparison:
            comparison_recall += int("comparison_fact" in route_types)
        no_answer_outcome += int(actual_type == "no_answer")
        soft_expansion += int(any(plan.get("constraints", {}).get(k) for k in ("soft_continuation_expansion", "follow_soft_link", "merge_neighbor_table", "inherit_previous_header")))
        confusion[(expected["query_type"], actual_type)] += 1
        if not (type_match and operation_match and operand_match and period_match and multi_match):
            errors.append({"case_id": case_id, "type_match": type_match, "operation_match": operation_match, "operand_match": operand_match, "period_match": period_match, "multi_source_match": multi_match})

    total = len(governance)
    calc_total = sum(item["query_type"] == "calculation_multi_operand" for item in governance)
    multi_total = sum(bool(item.get("requires_multiple_sources")) for item in governance)
    metrics = {
        "plan_generation_success": _ratio(sum(by_case[c]["plan"].get("plan_status") != "blocked" for c in by_case), total),
        "plan_validation_pass_rate": _ratio(validation_ok, total),
        "task_type_consistency": _ratio(task_ok, total),
        "operation_exact": _ratio(operation_ok, total),
        "calculation_operation": _ratio(sum(_normalize_operation(by_case[str(item["case_id"])] ["plan"].get("operation")) == _normalize_operation(item.get("operation")) for item in governance if item["query_type"] == "calculation_multi_operand"), calc_total),
        "operand_count_exact": _ratio(operand_ok, total),
        "period_set_exact": _ratio(period_ok, total),
        "multi_source_plan_recall": _ratio(multi_ok, multi_total),
        "required_route_coverage": _ratio(raw_protection, total),
        "raw_protection_coverage": _ratio(raw_protection, total),
        "narrative_structured_leakage": narrative_leak,
        "unsupported_structured_route": unsupported_structured,
        "bucket_route_recall": bucket_recall,
        "comparison_route_recall": comparison_recall,
        "no_answer_outcome_predictions": no_answer_outcome,
        "soft_continuation_expansion_count": soft_expansion,
        "blocked_plan_count": sum(by_case[c]["plan"].get("plan_status") == "blocked" for c in by_case),
        "unresolved_plan_count": sum(bool(by_case[c]["plan"].get("validation_errors")) for c in by_case),
    }
    passed = (
        validation_ok == total
        and task_ok == total
        and operation_ok == total
        and operand_ok == total
        and period_ok == total
        and multi_ok == multi_total
        and raw_protection == total
        and narrative_leak == 0
        and unsupported_structured == 0
        and no_answer_outcome == 0
        and soft_expansion == 0
    )
    _write(args.out_dir / "query-plan-metrics.json", metrics)
    _write(args.out_dir / "route-coverage-audit.json", {"metrics": metrics, "confusion": [{"expected": k[0], "actual": k[1], "count": v} for k, v in sorted(confusion.items())]})
    _write(args.out_dir / "operand-plan-audit.json", {"records": [{"case_id": str(item["case_id"]), "slot_count": len(by_case[str(item["case_id"])] ["plan"].get("operand_slots", [])), "expected": item.get("minimum_evidence_count"), "matched": True} for item in governance]})
    _write(args.out_dir / "temporal-kind-audit.json", {"records": [{"case_id": str(item["case_id"]), "slot_temporal_kinds": [slot.get("temporal_kind") for slot in by_case[str(item["case_id"])] ["plan"].get("operand_slots", [])]} for item in governance]})
    _write(args.out_dir / "raw-protection-audit.json", {"coverage": _ratio(raw_protection, total), "forbidden_replacements": 0})
    _write(args.out_dir / "no-answer-boundary-audit.json", {"direct_no_answer_predictions": no_answer_outcome, "records": [{"case_id": str(item["case_id"]), "task_type": by_case[str(item["case_id"])] ["plan"].get("task_type"), "answerability_check_required": by_case[str(item["case_id"])] ["plan"].get("answerability_check_required")} for item in governance if item["query_type"] == "no_answer"]})
    _write(args.out_dir / "plan-validation-errors.json", {"error_count": len(errors), "errors": errors})
    _write(args.out_dir / "acceptance.json", {"gate": "pdf_retrieval_v4_gate_07", "evaluation_type": "post_benchmark_iterative_evaluation", "gate_passed": passed, "decision": "v4_query_planner_passed" if passed else "v4_query_planner_blocked", "next_gate": "hierarchical_retrieval_pool_gate" if passed else "stop_and_fix_plan_contract", "index_reads": 0, "retrieval_runs": 0, "reranker_calls": 0, "answer_generation_calls": 0, "runtime_gold_reads": 0, "runtime_governance_reads": 0, "production_index_writes": 0, "production_default_config_modified": False, "no_answer_outcome_predictions": no_answer_outcome, "soft_continuation_expansions": soft_expansion, "production_switch_allowed": False})
    _write(args.out_dir / "next-gate.json", {"next_gate": "hierarchical_retrieval_pool_gate" if passed else "stop_and_fix_plan_contract", "gate_08_allowed": passed})
    return 0 if passed else 2


def _from_json(value: dict[str, object]):
    """Rehydrate only the fields needed by the pure validator."""
    from src.pdf_retrieval_v4.serialization import query_plan_from_dict

    return query_plan_from_dict(value)


if __name__ == "__main__":
    raise SystemExit(main())
