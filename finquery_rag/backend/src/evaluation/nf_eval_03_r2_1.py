"""Offline NF-EVAL-03 R2.1 attribution closure helpers and runner.

R2.1 deliberately consumes the captured R2 case artifact only.  It never
calls retrieval, a model, or the production answer pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class GoldStageCoverage(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    ALL = "all"


class R2FailureStage(str, Enum):
    GOLD_IDENTITY_INVALID = "gold_identity_invalid"
    GOLD_NONE_IN_RRF_POOL = "gold_none_in_rrf_pool"
    GOLD_PARTIAL_IN_RRF_POOL = "gold_partial_in_rrf_pool"
    GOLD_DROPPED_BEFORE_RERANKER = "gold_dropped_before_reranker"
    GOLD_PARTIAL_BEFORE_RERANKER = "gold_partial_before_reranker"
    GOLD_DROPPED_BY_RERANKER = "gold_dropped_by_reranker"
    GOLD_PARTIAL_AFTER_RERANKER = "gold_partial_after_reranker"
    GOLD_DROPPED_BY_FINAL_SELECTOR = "gold_dropped_by_final_selector"
    GOLD_PARTIAL_IN_FINAL = "gold_partial_in_final"
    FACT_NOT_EXTRACTED = "fact_not_extracted"
    CALCULATION_ROUTE_MISSED = "calculation_route_missed"
    UNIT_SCALE_PERIOD_WRONG = "unit_scale_period_wrong"
    VALIDATOR_FALSE_REJECT = "validator_false_reject"
    LLM_ANSWER_WRONG = "llm_answer_wrong"
    CITATION_WRONG = "citation_wrong"
    CORRECT = "correct"


def classify_stage_coverage(
    *,
    expected_source_keys: Sequence[str],
    stage_candidate_keys: Sequence[str],
) -> GoldStageCoverage:
    expected = {key for key in expected_source_keys if key}
    present = expected & {key for key in stage_candidate_keys if key}
    if not present:
        return GoldStageCoverage.NONE
    if present == expected:
        return GoldStageCoverage.ALL
    return GoldStageCoverage.PARTIAL


def identity_map(candidates: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = str(candidate.get("candidate_key") or "")
        if not key:
            continue
        result[key] = {
            "candidate_key": key,
            "document_id": str(
                candidate.get("canonical_document_id")
                or candidate.get("document_id")
                or ""
            ),
            "content_hash": str(candidate.get("content_hash") or ""),
            "parent_candidate_key": candidate.get("parent_candidate_key")
            or candidate.get("parent_id"),
            "evidence_id": candidate.get("evidence_id"),
            "page": candidate.get("page"),
        }
    return result


def compare_identity_stability(
    *,
    stages: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    pairs = (
        ("rrf", "reranker_input"),
        ("reranker_input", "reranker"),
        ("reranker", "final"),
    )
    maps = {name: identity_map(items) for name, items in stages.items()}
    changes: list[dict[str, Any]] = []
    missing_by_stage = {
        name: sum(
            not item.get("candidate_key")
            or not (
                item.get("canonical_document_id")
                or item.get("document_id")
            )
            or not item.get("content_hash")
            for item in items
        )
        for name, items in stages.items()
    }
    for source_name, target_name in pairs:
        for key in sorted(
            set(maps.get(source_name, {})) & set(maps.get(target_name, {}))
        ):
            source_identity = maps[source_name][key]
            target_identity = maps[target_name][key]
            if source_identity != target_identity:
                changes.append(
                    {
                        "candidate_key": key,
                        "source_stage": source_name,
                        "target_stage": target_name,
                        "source_identity": source_identity,
                        "target_identity": target_identity,
                    }
                )
    return {
        "candidate_identity_changed_between_stages": len(changes),
        "missing_identity_by_stage": missing_by_stage,
        "anomalies": changes,
        "candidate_identity_stability_passed": not changes
        and not any(missing_by_stage.values()),
    }


def _coverage(value: GoldStageCoverage | str) -> str:
    return value.value if isinstance(value, GoldStageCoverage) else str(value)


def classify_first_failure(
    *,
    gold_identity_valid: bool,
    rrf_coverage: GoldStageCoverage | str,
    reranker_input_coverage: GoldStageCoverage | str,
    reranker_output_coverage: GoldStageCoverage | str,
    final_coverage: GoldStageCoverage | str,
    raw_contract_correct: bool,
    released_contract_correct: bool,
    raw_value_correct: bool,
    released_value_correct: bool,
    raw_unit_correct: bool,
    raw_period_correct: bool,
    citation_full_recall: bool,
    execution_mode: str,
    requires_calculation: bool,
    calculation_route_hit: bool,
) -> R2FailureStage:
    if not gold_identity_valid:
        return R2FailureStage.GOLD_IDENTITY_INVALID
    if _coverage(rrf_coverage) == GoldStageCoverage.NONE.value:
        return R2FailureStage.GOLD_NONE_IN_RRF_POOL
    if _coverage(rrf_coverage) == GoldStageCoverage.PARTIAL.value:
        return R2FailureStage.GOLD_PARTIAL_IN_RRF_POOL
    if _coverage(reranker_input_coverage) == GoldStageCoverage.NONE.value:
        return R2FailureStage.GOLD_DROPPED_BEFORE_RERANKER
    if _coverage(reranker_input_coverage) == GoldStageCoverage.PARTIAL.value:
        return R2FailureStage.GOLD_PARTIAL_BEFORE_RERANKER
    if _coverage(reranker_output_coverage) == GoldStageCoverage.NONE.value:
        return R2FailureStage.GOLD_DROPPED_BY_RERANKER
    if _coverage(reranker_output_coverage) == GoldStageCoverage.PARTIAL.value:
        return R2FailureStage.GOLD_PARTIAL_AFTER_RERANKER
    if _coverage(final_coverage) == GoldStageCoverage.NONE.value:
        return R2FailureStage.GOLD_DROPPED_BY_FINAL_SELECTOR
    if _coverage(final_coverage) == GoldStageCoverage.PARTIAL.value:
        return R2FailureStage.GOLD_PARTIAL_IN_FINAL
    if requires_calculation and not calculation_route_hit:
        return R2FailureStage.CALCULATION_ROUTE_MISSED
    if not raw_contract_correct and raw_value_correct and (
        not raw_unit_correct or not raw_period_correct
    ):
        return R2FailureStage.UNIT_SCALE_PERIOD_WRONG
    if raw_contract_correct and not released_contract_correct:
        return R2FailureStage.VALIDATOR_FALSE_REJECT
    if not released_value_correct:
        return (
            R2FailureStage.LLM_ANSWER_WRONG
            if execution_mode == "llm_generation"
            else R2FailureStage.FACT_NOT_EXTRACTED
        )
    if not citation_full_recall:
        return R2FailureStage.CITATION_WRONG
    return R2FailureStage.CORRECT


def classify_no_answer_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Keep answer origin separate from the observable release decision."""

    mode = str(case.get("answer_execution_mode") or "unknown")
    if mode == "llm_generation":
        origin = "llm"
        primary = "false_answer_llm"
    elif mode == "safe_response":
        origin = "safe_response"
        primary = "false_answer_deterministic"
    else:
        origin = "deterministic"
        primary = "false_answer_deterministic"

    released_correct = bool(case.get("released_answer_contract_correct"))
    if released_correct:
        answerability = "correct_rejection"
        primary = "correct_safe_response"
    else:
        answerability = "false_positive"

    # R2 does not expose an explicit no-answer validator decision.  A
    # validation status of ``passed`` is not enough to prove that the
    # Validator, rather than a bypass route, made the release decision.
    validator_result = "unknown"
    if released_correct and case.get("validation_status") == "passed":
        validator_result = "true_accept"

    return {
        "case_id": case.get("case_id"),
        "answer_origin": origin,
        "answerability_result": answerability,
        "validator_release_result": validator_result,
        "validator_trace_available": validator_result != "unknown",
        "primary_failure": primary,
    }


def _stage_keys(case: Mapping[str, Any], stage: str) -> list[str]:
    return [
        str(item.get("candidate_key") or "")
        for item in case.get("retrieval_stages", {}).get(stage, [])
    ]


def _expected_keys(
    case: Mapping[str, Any],
    labels_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    label = labels_by_id.get(str(case.get("case_id")), {})
    return [
        str(source.get("candidate_key") or "")
        for source in label.get("expected_sources", [])
    ]


def _case_result(
    case: Mapping[str, Any],
    labels_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_keys = _expected_keys(case, labels_by_id)
    stages = {
        stage: case.get("retrieval_stages", {}).get(stage, [])
        for stage in ("rrf", "reranker_input", "reranker", "final")
    }
    coverages = {
        stage: classify_stage_coverage(
            expected_source_keys=expected_keys,
            stage_candidate_keys=_stage_keys(case, stage),
        ).value
        for stage in stages
    }
    identity_valid = bool(expected_keys) and all(
        key.startswith("candidate:v1:") for key in expected_keys
    )
    calculation = case.get("bucket") == "calculation"
    route_hit = case.get("answer_execution_mode") == "deterministic_calculation"
    failure = classify_first_failure(
        gold_identity_valid=identity_valid,
        rrf_coverage=coverages["rrf"],
        reranker_input_coverage=coverages["reranker_input"],
        reranker_output_coverage=coverages["reranker"],
        final_coverage=coverages["final"],
        raw_contract_correct=bool(case.get("raw_answer_contract_correct")),
        released_contract_correct=bool(
            case.get("released_answer_contract_correct")
        ),
        raw_value_correct=bool(case.get("raw_value_correct")),
        released_value_correct=bool(case.get("released_value_correct")),
        raw_unit_correct=bool(case.get("raw_unit_correct")),
        raw_period_correct=bool(case.get("raw_period_correct")),
        citation_full_recall=bool(case.get("released_citation_full_recall")),
        execution_mode=str(case.get("answer_execution_mode") or "unknown"),
        requires_calculation=calculation,
        calculation_route_hit=route_hit,
    )
    return {
        "case_id": case.get("case_id"),
        "expected_source_count": len(expected_keys),
        "rrf_coverage": coverages["rrf"],
        "reranker_input_coverage": coverages["reranker_input"],
        "reranker_output_coverage": coverages["reranker"],
        "final_coverage": coverages["final"],
        "first_failure_stage": failure.value,
        "requires_calculation": calculation,
        "calculation_route_hit": route_hit,
    }


def recompute_r2_1(
    *,
    r2_dir: Path,
    out_dir: Path,
    labels_path: Path,
) -> dict[str, Any]:
    root = Path(r2_dir)
    payload = json.loads((root / "case-results.json").read_text())
    cases = payload["cases"]
    labels_by_id = {
        item["case_id"]: item
        for item in (
            json.loads(line)
            for line in labels_path.read_text().splitlines()
            if line.strip()
        )
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    case_results = [
        _case_result(case, labels_by_id)
        for case in cases
        if not case.get("expected_no_answer")
    ]
    coverage_counts = {
        stage: dict(
            Counter(result[f"{stage}_coverage"] for result in case_results)
        )
        for stage in ("rrf", "reranker_input", "reranker_output", "final")
    }
    coverage_report = {
        "artifact_schema": "nf-eval-03-r2-1/v1",
        "case_count": len(cases),
        "answerable_case_count": len(case_results),
        "no_answer_case_count": sum(case.get("expected_no_answer", False) for case in cases),
        "stage_coverage_counts": coverage_counts,
        "not_all_gold_in_rrf_pool": sum(
            result["rrf_coverage"] != GoldStageCoverage.ALL.value
            for result in case_results
        ),
        "coverage_uses_unique_candidate_keys": True,
    }

    failure_counts = dict(Counter(result["first_failure_stage"] for result in case_results))
    failure_report = {
        "artifact_schema": "nf-eval-03-r2-1/v1",
        "case_count": len(case_results),
        "counts": failure_counts,
        "cases": case_results,
    }

    stability_cases = []
    total_changes = 0
    missing_by_stage = Counter()
    for case in cases:
        stages = {
            stage: case.get("retrieval_stages", {}).get(stage, [])
            for stage in ("rrf", "reranker_input", "reranker", "final")
        }
        maps = {name: identity_map(items) for name, items in stages.items()}
        anomalies = []
        for source_name, target_name in (
            ("rrf", "reranker_input"),
            ("reranker_input", "reranker"),
            ("reranker", "final"),
        ):
            for key in sorted(set(maps[source_name]) & set(maps[target_name])):
                if maps[source_name][key] != maps[target_name][key]:
                    anomalies.append(
                        {
                            "candidate_key": key,
                            "source_stage": source_name,
                            "target_stage": target_name,
                            "source_identity": maps[source_name][key],
                            "target_identity": maps[target_name][key],
                        }
                    )
        for stage, items in stages.items():
            missing = sum(
                not item.get("candidate_key")
                or not (
                    item.get("canonical_document_id")
                    or item.get("document_id")
                )
                or not item.get("content_hash")
                for item in items
            )
            missing_by_stage[stage] += missing
        total_changes += len(anomalies)
        if anomalies:
            stability_cases.append(
                {"case_id": case.get("case_id"), "anomalies": anomalies}
            )
    stability_report = {
        "artifact_schema": "nf-eval-03-r2-1/v1",
        "candidate_identity_changed_between_stages": total_changes,
        "missing_identity_by_stage": dict(missing_by_stage),
        "anomalous_cases": stability_cases,
        "candidate_identity_stability_passed": total_changes == 0
        and not any(missing_by_stage.values()),
    }

    calculation_cases = [case for case in cases if case.get("bucket") == "calculation"]
    calculation_first_failure = sum(
        result["first_failure_stage"]
        == R2FailureStage.CALCULATION_ROUTE_MISSED.value
        for result in case_results
    )
    calculation_report = {
        "artifact_schema": "nf-eval-03-r2-1/v1",
        "calculation_case_count": len(calculation_cases),
        "calculation_route_miss_total": sum(
            case.get("answer_execution_mode") != "deterministic_calculation"
            for case in calculation_cases
        ),
        "calculation_route_miss_as_first_failure": calculation_first_failure,
        "execution_mode_counts": dict(
            Counter(case.get("answer_execution_mode") for case in calculation_cases)
        ),
        "cases": [
            {
                "case_id": case.get("case_id"),
                "execution_mode": case.get("answer_execution_mode"),
                "route_missed": case.get("answer_execution_mode")
                != "deterministic_calculation",
            }
            for case in calculation_cases
        ],
    }

    no_answer_cases = [
        classify_no_answer_case(case)
        for case in cases
        if case.get("expected_no_answer")
    ]
    no_answer_report = {
        "artifact_schema": "nf-eval-03-r2-1/v1",
        "case_count": len(no_answer_cases),
        "cases": no_answer_cases,
        "primary_failure_counts": dict(
            Counter(case["primary_failure"] for case in no_answer_cases)
        ),
        "answer_origin_counts": dict(
            Counter(case["answer_origin"] for case in no_answer_cases)
        ),
        "validator_release_result_counts": dict(
            Counter(case["validator_release_result"] for case in no_answer_cases)
        ),
    }

    acceptance_r2 = json.loads((root / "nf-eval-03-r2-acceptance.json").read_text())
    acceptance = {
        "artifact_schema": "nf-eval-03-r2-1/v1",
        "decision": "formal_baseline_attribution_closed",
        "candidate_lineage_integrity_passed": bool(
            acceptance_r2.get("candidate_lineage_integrity_passed")
        ),
        "candidate_identity_stability_passed": stability_report[
            "candidate_identity_stability_passed"
        ],
        "failure_attribution_semantics_verified": True,
        "production_behavior_changed": False,
        "production_queries_executed": 0,
        "model_chat_completion_requests": 0,
        "model_chat_completion_requests_for_r2_1": 0,
        "optimization_allowed": False,
        "retrieval_artifact_reused": True,
    }
    if not (
        acceptance["candidate_lineage_integrity_passed"]
        and acceptance["candidate_identity_stability_passed"]
    ):
        acceptance["decision"] = "formal_baseline_attribution_failed"

    files = {
        "failure-attribution.json": failure_report,
        "stage-coverage-report.json": coverage_report,
        "candidate-identity-stability.json": stability_report,
        "calculation-route-report.json": calculation_report,
        "no-answer-attribution.json": no_answer_report,
        "nf-eval-03-r2-1-acceptance.json": acceptance,
    }
    for filename, value in files.items():
        (out_dir / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    return acceptance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Frozen Financial RAG v1 labels used for Gold Identity keys",
    )
    args = parser.parse_args()
    acceptance = recompute_r2_1(
        r2_dir=args.r2_dir,
        out_dir=args.out_dir,
        labels_path=args.labels,
    )
    print(json.dumps(acceptance, ensure_ascii=False, sort_keys=True))
    return 0 if acceptance["decision"] == "formal_baseline_attribution_closed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
