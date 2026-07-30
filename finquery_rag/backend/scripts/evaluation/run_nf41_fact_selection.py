"""Run NF41 deterministic-fact attribution from the NF39 R2 local snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from src.evaluation.evaluation import load_jsonl_cases
from src.evaluation.nf40_frozen_context import load_frozen_contexts
from src.evaluation.nf41_fact_selection import (
    AnswerExecutionMode,
    SELECTOR_FAILURE_TYPES,
    build_constraints,
    classify_fact_failure,
    extract_structured_facts,
    fact_matches_expected_answer,
    fact_matches_rendered_answer,
    select_constraint_aware_fact,
    summarize_failures,
)
from src.evaluation.nf40_start_gate import require_verified_nf39_r2_inputs


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--nf40-case-attribution", required=True, type=Path)
    parser.add_argument("--nf40-raw-trace", required=True, type=Path)
    parser.add_argument("--nf40-run-manifest", required=True, type=Path)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--final-context-manifest", required=True, type=Path)
    parser.add_argument("--frozen-payload-path", required=True, type=Path)
    parser.add_argument("--expected-payload-sha256", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def _public_fact(fact) -> dict:
    return {
        "fact_id": fact.fact_id,
        "candidate_key": fact.candidate_key,
        "candidate_rank": fact.candidate_rank,
        "document_id": fact.document_id,
        "page": fact.page,
        "value": str(fact.value) if fact.value is not None else None,
        "unit": fact.unit,
        "period": fact.period,
        "extraction_source": fact.extraction_source,
        "extraction_confidence": fact.extraction_confidence,
    }


def _selection_accuracy(selected, case) -> bool:
    return bool(selected and fact_matches_expected_answer(selected, case))


def main() -> None:
    args = _args()
    require_verified_nf39_r2_inputs(
        acceptance_path=args.acceptance,
        snapshot_manifest_path=args.snapshot_manifest,
        frozen_payload_path=args.frozen_payload_path,
        expected_payload_sha256=args.expected_payload_sha256,
    )
    cases = load_jsonl_cases(args.cases)
    if len(cases) != 27:
        raise ValueError(f"NF41 requires 27 cases, got {len(cases)}")
    contexts = load_frozen_contexts(args.frozen_payload_path, args.final_context_manifest)
    if set(contexts) != {case.case_id for case in cases}:
        raise ValueError("NF41 contexts do not match cases")
    nf40_cases = {row["case_id"]: row for row in json.loads(args.nf40_case_attribution.read_text(encoding="utf-8"))["cases"]}
    private = {row["case_id"]: row for row in (json.loads(line) for line in args.nf40_raw_trace.read_text(encoding="utf-8").splitlines() if line)}
    nf40_manifest = json.loads(args.nf40_run_manifest.read_text(encoding="utf-8"))

    all_gold = [
        case for case in cases
        if nf40_cases[case.case_id]["context_coverage"] == "all_gold_in_final"
    ]
    records = []
    fact_public = []
    latencies = []
    mode_counts = Counter()
    for case in cases:
        started = time.perf_counter()
        context = contexts[case.case_id]
        facts = extract_structured_facts(context)
        raw_answer = private[case.case_id].get("raw_generation") or ""
        current = next((fact for fact in facts if fact_matches_rendered_answer(fact, raw_answer)), None)
        constraints = build_constraints(case.question)
        variant = select_constraint_aware_fact(facts, constraints)
        mode = (
            AnswerExecutionMode.SAFE_RESPONSE
            if case.expected_no_answer
            else AnswerExecutionMode.DETERMINISTIC_CALCULATION
            if nf40_cases[case.case_id].get("calculation_attempted")
            else AnswerExecutionMode.DETERMINISTIC_FACT
        )
        mode_counts[mode.value] += 1
        failure = None
        if (
            case in all_gold
            and not nf40_cases[case.case_id]["raw_answer_correct"]
            and mode in {AnswerExecutionMode.DETERMINISTIC_FACT, AnswerExecutionMode.DETERMINISTIC_CALCULATION}
        ):
            failure = classify_fact_failure(
                case=case, facts=facts, selected=current, rendered_answer=raw_answer
            ).value
        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)
        record = {
            "case_id": case.case_id,
            "answer_execution_mode": mode.value,
            "model_invocation_count": 0,
            "retrieval_call_count": 0,
            "final_context_hash": context.final_context_hash,
            "final_candidate_count": len(context.candidates),
            "structured_fact_count": len(facts),
            "gold_source_fact_count": sum(
                fact.document_id == source.filename and (source.page is None or source.page == fact.page)
                for fact in facts for source in case.expected_sources
            ),
            "current_selected_fact_id": current.fact_id if current else None,
            "current_selection_correct": _selection_accuracy(current, case),
            "constraint_selected_fact_id": variant.fact_id if variant else None,
            "constraint_selection_correct": _selection_accuracy(variant, case),
            "failure": failure,
            "fact_extraction_ms": elapsed,
        }
        records.append(record)
        fact_public.append({"case_id": case.case_id, "facts": [_public_fact(fact) for fact in facts]})

    eligible = [row for row in records if row["failure"]]
    failures = summarize_failures(eligible)
    selector_count = sum(failures[item.value] for item in SELECTOR_FAILURE_TYPES)
    selector_allowed = selector_count >= 3
    eligible_ids = {row["case_id"] for row in eligible}
    current_correct = sum(row["current_selection_correct"] for row in records if row["case_id"] in eligible_ids)
    variant_correct = sum(row["constraint_selection_correct"] for row in records if row["case_id"] in eligible_ids)
    regressions = [
        row["case_id"] for row in records
        if row["case_id"] in eligible_ids
        and row["current_selection_correct"] and not row["constraint_selection_correct"]
    ]
    comparison = {
        "all_gold_case_count": len(all_gold),
        "all_gold_raw_answer_accuracy": {"count": sum(nf40_cases[case.case_id]["raw_answer_correct"] for case in all_gold), "denominator": len(all_gold), "rate": sum(nf40_cases[case.case_id]["raw_answer_correct"] for case in all_gold) / len(all_gold) if all_gold else 1.0},
        "selection_eligible_case_count": len(eligible),
        "current_fact_selection_accuracy": {"count": current_correct, "denominator": len(eligible), "rate": current_correct / len(eligible) if eligible else 1.0},
        "constraint_aware_fact_selection_accuracy": {"count": variant_correct, "denominator": len(eligible), "rate": variant_correct / len(eligible) if eligible else 1.0},
        "improved_cases": [row["case_id"] for row in records if row["case_id"] in eligible_ids and not row["current_selection_correct"] and row["constraint_selection_correct"]],
        "regressed_cases": regressions,
        "selection_gate_passed": selector_allowed and variant_correct > current_correct and not regressions,
    }
    baseline = json.loads((args.acceptance.parent / "baseline-manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "artifact_schema": "nf41/v1",
        "case_count": len(cases),
        "all_gold_case_count": len(all_gold),
        "any_gold_case_count": sum(nf40_cases[c.case_id]["context_coverage"] in {"all_gold_in_final", "partial_gold_in_final"} for c in cases),
        "question_hash": nf40_manifest["question_hash"],
        "label_hash": nf40_manifest["label_hash"],
        "candidate_pool_hash": nf40_manifest["candidate_pool_hash"],
        "frozen_payload_hash": args.expected_payload_sha256,
        "final_contexts_hash": nf40_manifest["final_contexts_hash"],
        "model_invocation_expected": False,
        "production_fact_selector": "deterministic-answer-extractor/current",
        "production_validator_hash": nf40_manifest["validator_config_hash"],
        "nf39_baseline_schema": baseline.get("artifact_schema"),
    }
    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0
    _write(args.out_dir / "baseline-manifest.json", manifest)
    _write(args.out_dir / "answer-mode-report.json", {"model_chat_completion_requests": 0, "model_embedding_requests": 0, "retrieval_calls": 0, "modes": dict(mode_counts), "cases": [{key: row[key] for key in ("case_id", "answer_execution_mode", "model_invocation_count", "retrieval_call_count")} for row in records]})
    _write(args.out_dir / "fact-candidate-report.json", {"artifact_schema": "nf41/v1", "cases": fact_public})
    _write(args.out_dir / "fact-selection-attribution.json", {"summary": {"eligible_case_count": len(eligible), **failures, "selector_failure_count": selector_count, "selector_experiment_allowed": selector_allowed}, "cases": [row for row in records if row["failure"]]})
    _write(args.out_dir / "selector-comparison.json", comparison)
    _write(args.out_dir / "case-diff-report.json", {"cases": [{key: row[key] for key in ("case_id", "current_selected_fact_id", "constraint_selected_fact_id", "current_selection_correct", "constraint_selection_correct", "failure")} for row in records]})
    _write(args.out_dir / "latency-report.json", {"fact_extraction_p50_ms": sorted(latencies)[len(latencies)//2], "fact_extraction_p95_ms": p95, "selection_p95_ms": 0.0, "total_p95_ms": p95})
    _write(args.out_dir / "nf41-acceptance.json", {"artifact_schema": "nf41/v1", "attribution_completed": True, "selector_experiment_allowed": selector_allowed, "selection_gate_passed": comparison["selection_gate_passed"], "production_behavior_changed": False, "model_chat_completion_requests": 0, "retrieval_calls": 0})


if __name__ == "__main__":
    main()
