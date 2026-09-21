"""Observe the current deterministic pipeline over NF39 R2 frozen contexts."""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from src.evaluation.evaluation import load_jsonl_cases
from src.evaluation.nf40_attribution import classify_context_coverage
from src.evaluation.nf40_frozen_context import as_evaluation_context, load_frozen_contexts
from src.evaluation.nf40_pipeline_observer import AnswerPipelineTrace
from src.evaluation.nf40_start_gate import require_verified_nf39_r2_inputs
from src.evaluation.nf41_numeric_identity import value_matches_expected
from src.evaluation.nf41_production_attribution import (
    classify_observed_fact_failure,
    next_step_for_observed_failures,
    proxy_production_relation,
)
from src.generation.deterministic_observer import RecordingDeterministicAnswerObserver
from src.services.rag_engine import RAGEngine


class _NoModelClient:
    class chat:
        class completions:
            @staticmethod
            def create(**_kwargs):
                raise RuntimeError("NF41 R1 forbids model chat completion calls")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--final-context-manifest", required=True, type=Path)
    parser.add_argument("--frozen-payload-path", required=True, type=Path)
    parser.add_argument("--expected-payload-sha256", required=True)
    parser.add_argument("--nf40-case-attribution", required=True, type=Path)
    parser.add_argument("--nf40-run-manifest", required=True, type=Path)
    parser.add_argument("--proxy-attribution", type=Path)
    parser.add_argument("--tenant-id", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def _matches_expected(fact, case) -> bool:
    if not any(
        source.filename == fact.document_id
        and (source.page is None or source.page == fact.page)
        for source in case.expected_sources
    ):
        return False
    if case.expected_numbers:
        return value_matches_expected(fact.raw_value, case.expected_numbers)
    text = (fact.evaluation_text or "").lower()
    return bool(text and any(expected.lower() in text for expected in case.expected_answer_contains))


def _proxy_family(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.removeprefix("proxy_")
    if normalized == "correct_fact_not_extracted":
        return "production_fact_not_extracted"
    if normalized in {"wrong_candidate_selected", "wrong_metric_selected", "wrong_unit_or_scale_selected", "wrong_table_column_selected"}:
        return "production_fact_available_not_selected"
    return None


def _redacted_fact(fact) -> dict:
    return {
        "fact_id": fact.fact_id,
        "candidate_key": fact.candidate_key,
        "candidate_rank": fact.candidate_rank,
        "document_id": fact.document_id,
        "page": fact.page,
        "extraction_stage": fact.extraction_stage,
        "source_span_hash": fact.source_span_hash,
        "raw_value": fact.raw_value,
        "canonical_value": fact.canonical_value,
        "currency": fact.currency,
        "unit": fact.unit,
        "scale": fact.scale,
        "period": fact.period,
    }


async def _run(args: argparse.Namespace) -> None:
    require_verified_nf39_r2_inputs(
        acceptance_path=args.acceptance, snapshot_manifest_path=args.snapshot_manifest,
        frozen_payload_path=args.frozen_payload_path, expected_payload_sha256=args.expected_payload_sha256,
    )
    if args.tenant_id != 1:
        raise ValueError("NF41 R1 frozen scope requires tenant 1")
    cases = load_jsonl_cases(args.cases)
    contexts = load_frozen_contexts(args.frozen_payload_path, args.final_context_manifest)
    previous = {item["case_id"]: item for item in json.loads(args.nf40_case_attribution.read_text(encoding="utf-8"))["cases"]}
    proxy_cases = {}
    if args.proxy_attribution:
        proxy_cases = {
            item["case_id"]: item
            for item in json.loads(args.proxy_attribution.read_text(encoding="utf-8")).get("cases", [])
        }
    manifest = json.loads(args.nf40_run_manifest.read_text(encoding="utf-8"))
    engine = RAGEngine(_NoModelClient(), use_hybrid=False, reranker_name="none", retrieval_candidate_multiplier=1)
    records = []
    failures = Counter()
    proxy_agreement = Counter()
    for case in cases:
        frozen = contexts[case.case_id]
        coverage = classify_context_coverage(
            expected_no_answer=case.expected_no_answer, expected_source_count=len(case.expected_sources),
            matched_gold_source_count=sum(any(source.filename == item.document_id and (source.page is None or source.page == item.page) for item in frozen.candidates) for source in case.expected_sources),
        )
        trace = AnswerPipelineTrace(case_id=case.case_id, trace_id=case.case_id, context_hash=frozen.final_context_hash, context_coverage=coverage.value)
        recorder = RecordingDeterministicAnswerObserver()
        trace.deterministic_observer = recorder
        result = await engine.answer_frozen_evaluation(question=case.question, user_id=args.tenant_id, frozen_context=as_evaluation_context(frozen), evaluation_observer=trace)
        if result.get("context") != as_evaluation_context(frozen).context:
            raise ValueError(f"{case.case_id}: frozen context changed")
        correct_facts = [fact for fact in recorder.facts if _matches_expected(fact, case)]
        selected = set(recorder.selected_fact_ids)
        if case.expected_no_answer:
            failure = "not_applicable"
        elif coverage.value != "all_gold_in_final":
            failure = "not_applicable"
        else:
            failure = classify_observed_fact_failure(
                facts=recorder.facts,
                selected_fact_ids=selected,
                raw_answer_correct=previous[case.case_id]["raw_answer_correct"],
                fact_matches_gold=lambda fact: fact in correct_facts,
                extraction_attempted=bool(recorder.routes),
            ).value
        failures[failure] += 1
        raw_proxy_failure = (
            proxy_cases.get(case.case_id, {}).get("proxy_failure")
            or proxy_cases.get(case.case_id, {}).get("failure")
        )
        proxy_failure = (
            raw_proxy_failure
            if not raw_proxy_failure or raw_proxy_failure.startswith("proxy_")
            else f"proxy_{raw_proxy_failure}"
        )
        proxy_family = _proxy_family(proxy_failure)
        relation = proxy_production_relation(proxy_failure=proxy_family, production_failure=failure)
        proxy_agreement[relation] += 1
        records.append({
            "case_id": case.case_id, "context_hash": frozen.final_context_hash,
            "production_route": recorder.routes[-1] if recorder.routes else "unobserved",
            "production_extracted_fact_count": len(recorder.facts),
            "production_selected_fact_ids": recorder.selected_fact_ids,
            "production_selection_reason_codes": [list(item) for item in recorder.selection_reason_codes],
            "calculation_operation": recorder.calculation_operation,
            "operand_fact_ids": list(recorder.operand_fact_ids),
            "render_source_fact_ids": list(recorder.rendered_fact_ids),
            "production_trace_available": bool(recorder.facts),
            "production_facts": [_redacted_fact(fact) for fact in recorder.facts],
            "production_failure": failure,
            "proxy_analysis": bool(proxy_cases), "production_trace": True,
            "proxy_failure": proxy_failure,
            "proxy_comparison_family": proxy_family,
            "proxy_production_relation": relation,
        })
    _write(args.out_dir / "baseline-manifest.json", {"artifact_schema":"nf41-r1/v1", "question_hash":manifest["question_hash"], "label_hash":manifest["label_hash"], "frozen_payload_hash":args.expected_payload_sha256, "final_contexts_hash":manifest["final_contexts_hash"], "case_count":len(cases), "production_behavior_changed":False})
    _write(args.out_dir / "production-route-report.json", {"model_chat_completion_requests":0, "retrieval_calls":0, "routes":dict(Counter(item["production_route"] for item in records))})
    _write(args.out_dir / "production-fact-trace.json", {"artifact_schema":"nf41-r1/v1", "cases":records})
    _write(args.out_dir / "production-failure-attribution.json", {"summary":dict(failures), "cases":[item for item in records if item["production_failure"] not in {"correct","not_applicable"}]})
    _write(args.out_dir / "proxy-production-agreement.json", {
        "proxy_analysis": True, "production_trace": True,
        "summary": dict(proxy_agreement),
    })
    _write(args.out_dir / "case-attribution.json", {"cases":records})
    allowed, next_step = next_step_for_observed_failures(dict(failures))
    _write(args.out_dir / "nf41-r1-acceptance.json", {"artifact_schema":"nf41-r1/v1", "model_chat_completion_requests":0, "retrieval_calls":0, "production_behavior_changed":False, "nf42_direction":allowed, "next_step":next_step, "trace_insufficient_count":failures["production_trace_insufficient"]})


if __name__ == "__main__":
    asyncio.run(_run(_args()))
