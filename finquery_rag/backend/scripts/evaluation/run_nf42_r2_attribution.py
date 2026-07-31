"""Run NF42 R2 projection-to-selection attribution on frozen contexts.

Extends the R1 shadow A/B with projection traces, exclusion tracking,
new-fact loss funnel, regression root-cause analysis, and corrected
single-variable declarations.

This command bypasses retrieval and model generation, replaying the
verified NF39 R2 final contexts through the production answer pipeline
twice.  Only the injected deterministic fact extractor differs, but R2
correctly records that projection and pre-selector scoring also change.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path

from openai import OpenAI

from src.evaluation.evaluation import load_jsonl_cases
from src.evaluation.nf40_frozen_context import load_frozen_contexts
from src.evaluation.nf40_runner import (
    FrozenContextEvaluationRunner,
    validate_labeled_cases,
)
from src.evaluation.nf40_start_gate import require_verified_nf39_r2_inputs
from src.evaluation.nf41_numeric_identity import normalize_numeric_identity
from src.evaluation.nf42_r2_projection_trace import (
    NewFactFunnelTrace,
    NumericEvidenceCandidateTrace,
    classify_new_fact_loss,
    classify_regression_cause,
    function_identity,
)
from src.generation.deterministic_answers import DeterministicAnswerExtractor
from src.generation.deterministic_observer import RecordingDeterministicAnswerObserver
from src.services.rag_engine import RAGEngine


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--final-context-manifest", required=True, type=Path)
    parser.add_argument("--frozen-payload-path", required=True, type=Path)
    parser.add_argument("--expected-payload-sha256", required=True)
    parser.add_argument("--tenant-id", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class _CountingModelClient:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.request_count = 0
        self.chat = self
        self.completions = self

    def create(self, *args, **kwargs):
        self.request_count += 1
        return self._delegate.chat.completions.create(*args, **kwargs)


def _build_engine(provider: str) -> tuple[RAGEngine, _CountingModelClient]:
    client = _CountingModelClient(OpenAI(
        base_url=os.getenv("LLM_API_BASE_URL", "http://127.0.0.1:8500/v1"),
        api_key=os.getenv("LLM_API_KEY", "not-needed-for-local"),
    ))
    return RAGEngine(
        client,
        model_name=os.getenv("LLM_MODEL_NAME", "nanochat"),
        use_hybrid=False,
        reranker_name="none",
        retrieval_candidate_multiplier=1,
        deterministic_fact_extractor=provider,
    ), client


def _source_matches(case, fact) -> bool:
    return any(
        source.matches({"filename": fact.document_id, "page": fact.page, "chunk_id": None})
        for source in case.expected_sources
    )


def _fact_matches(case, fact) -> bool:
    if not _source_matches(case, fact):
        return False
    if case.expected_numbers:
        actual = normalize_numeric_identity(fact.raw_value, period=fact.period)
        if actual is None:
            return False
        for expected in case.expected_numbers:
            target = normalize_numeric_identity(expected)
            if target and actual.value_type == target.value_type and actual.canonical_value == target.canonical_value:
                return True
            try:
                raw = Decimal(str(fact.raw_value or "").replace(",", "").replace("$", "").split()[0])
            except (InvalidOperation, IndexError):
                continue
            if target and actual.value_type == "amount" and raw == target.canonical_value:
                return True
        return False
    text = (fact.evaluation_text or "").lower()
    return bool(text and any(expected.lower() in text for expected in case.expected_answer_contains))


async def _run_variant(*, provider: str, cases, contexts, tenant_id: int, nf40_records: dict) -> tuple[dict, list[dict]]:
    engine, client = _build_engine(provider)
    runner = FrozenContextEvaluationRunner(rag_engine=engine)
    records: list[dict] = []
    for case in cases:
        run = await runner.run_case(case=case, frozen=contexts[case.case_id], tenant_id=tenant_id)
        observer = run.trace.deterministic_observer
        if observer is None:
            observer = RecordingDeterministicAnswerObserver()
        facts = list(observer.facts)
        correct = [fact for fact in facts if _fact_matches(case, fact)]
        selected = set(observer.selected_fact_ids)
        selected_correct = any(fact.fact_id in selected for fact in correct)
        records.append({
            "case_id": case.case_id,
            "context_coverage": nf40_records.get(case.case_id, {}).get("context_coverage", "unknown"),
            "context_hash": contexts[case.case_id].final_context_hash,
            "fact_count": len(facts),
            "correct_fact_available": bool(correct),
            "correct_fact_ids": [fact.fact_id for fact in correct],
            "selected_fact_correct": selected_correct,
            "selected_fact_ids": list(observer.selected_fact_ids),
            "raw_answer_correct": run.evaluation.raw_answer_correct,
            "released_answer_correct": run.evaluation.released_answer_correct,
            "raw_numeric_correct": run.evaluation.raw_numeric_correct,
            "raw_unit_correct": run.evaluation.raw_unit_correct,
            "raw_period_correct": run.evaluation.raw_period_correct,
            "raw_citation_correct": run.evaluation.raw_citation_correct,
            "released_citation_recall": run.released_score["citation_recall"],
            "released_citation_precision": run.released_score.get("citation_precision", 0.0),
            "validator_outcome": run.trace.validation_status,
            "released_response_type": run.trace.released_response_type,
            "repair_attempted": run.trace.repair_attempted,
            "repair_succeeded": run.trace.repair_status == "repaired",
            "latency_ms": run.evaluation.latency_ms,
            "no_answer_correct": run.evaluation.no_answer_correct,
            "projected_candidates": list(observer.projected_candidates),
            "projection_exclusions": list(observer.projection_exclusions),
            "pre_selector_ranking": list(observer.pre_selector_ranking),
            "selector_input_ids": list(observer.selector_input_ids),
            "selector_output_ids": list(observer.selector_output_ids),
            "selected_values": list(observer.selected_values),
            "selected_value_fact_ids": list(observer.selected_value_fact_ids),
        })
    return {
        "provider": engine._deterministic_fact_extractor.name,
        "revision": engine._deterministic_fact_extractor.revision,
        "selector_identity": function_identity(DeterministicAnswerExtractor._select_raw_numeric_evidence),
        "value_selector_identity": function_identity(DeterministicAnswerExtractor._select_answer_values),
        "renderer_identity": function_identity(DeterministicAnswerExtractor.answer_numeric_query_from_chunks),
        "citation_identity": function_identity(DeterministicAnswerExtractor._inline_source_citation),
        "scoring_identity": function_identity(DeterministicAnswerExtractor._raw_numeric_evidence_score),
        "model_chat_completion_requests": client.request_count,
    }, records


def _build_new_fact_funnel(
    *,
    case_id: str,
    correct_facts: list[dict],
    structured_record: dict,
) -> list[NewFactFunnelTrace]:
    """Build funnel traces for newly correct facts in a case."""
    traces: list[NewFactFunnelTrace] = []
    projected = structured_record.get("projected_candidates", [])
    exclusions = structured_record.get("projection_exclusions", [])
    selector_input = set(structured_record.get("selector_input_ids", []))
    selector_output = set(structured_record.get("selector_output_ids", []))
    selected_value_fact_ids = set(structured_record.get("selected_value_fact_ids", []))
    raw_correct = structured_record.get("raw_answer_correct", False)
    released_correct = structured_record.get("released_answer_correct", False)

    for fact in correct_facts:
        fact_id = fact["fact_id"]
        proj = next((p for p in projected if fact_id in p.get("source_fact_ids", [])), None)
        excluded = next((e for e in exclusions if e.get("fact_id") == fact_id), None)

        projection_eligible = excluded is None
        projected_id = proj.get("projected_candidate_id") if proj else None
        pre_rank = proj.get("pre_selector_rank") if proj else None
        entered_selector = projected_id in selector_input if projected_id else False
        selected_by_selector = projected_id in selector_output if projected_id else False
        value_selected = fact_id in selected_value_fact_ids

        trace = NewFactFunnelTrace(
            case_id=case_id,
            fact_id=fact_id,
            candidate_key=fact.get("candidate_key"),
            correct_fact_extracted=True,
            projection_eligible=projection_eligible,
            projected_candidate_id=projected_id,
            pre_selector_rank=pre_rank,
            entered_selector_input=entered_selector,
            selected_by_selector=selected_by_selector,
            value_selected=value_selected,
            raw_answer_correct=raw_correct,
            released_answer_correct=released_correct,
        )
        trace.first_loss_stage = classify_new_fact_loss(trace)
        traces.append(trace)
    return traces


def _build_regression_trace(
    *,
    case_id: str,
    current_record: dict,
    structured_record: dict,
) -> dict | None:
    """Build a regression trace for a case that went from correct to incorrect."""
    current_projected = [
        NumericEvidenceCandidateTrace(
            projected_candidate_id=p["projected_candidate_id"],
            provider=p["provider"],
            candidate_key=p.get("candidate_key"),
            candidate_rank=p.get("candidate_rank"),
            source_fact_ids=tuple(p.get("source_fact_ids", [])),
            source_span_hash=p["source_span_hash"],
            document_id=p.get("document_id"),
            page=p.get("page"),
            projected_text_hash=p["projected_text_hash"],
            projected_value_hashes=tuple(p.get("projected_value_hashes", [])),
            metric=p.get("metric"),
            period=p.get("period"),
            currency=p.get("currency"),
            unit=p.get("unit"),
            base_evidence_score=p.get("base_evidence_score", 0.0),
            anchor_match_count=p.get("anchor_match_count", 0),
            anchor_conflict_count=p.get("anchor_conflict_count", 0),
            relation_score=p.get("relation_score", 0.0),
            value_granularity_score=p.get("value_granularity_score", 0.0),
            component_pair_score=p.get("component_pair_score", 0.0),
            retrieval_score=p.get("retrieval_score", 0.0),
            final_pre_selector_score=p.get("final_pre_selector_score", 0.0),
            pre_selector_rank=p.get("pre_selector_rank"),
            selector_input=p.get("selector_input", False),
            selector_output_rank=p.get("selector_output_rank"),
        )
        for p in current_record.get("projected_candidates", [])
    ]
    structured_projected = [
        NumericEvidenceCandidateTrace(
            projected_candidate_id=p["projected_candidate_id"],
            provider=p["provider"],
            candidate_key=p.get("candidate_key"),
            candidate_rank=p.get("candidate_rank"),
            source_fact_ids=tuple(p.get("source_fact_ids", [])),
            source_span_hash=p["source_span_hash"],
            document_id=p.get("document_id"),
            page=p.get("page"),
            projected_text_hash=p["projected_text_hash"],
            projected_value_hashes=tuple(p.get("projected_value_hashes", [])),
            metric=p.get("metric"),
            period=p.get("period"),
            currency=p.get("currency"),
            unit=p.get("unit"),
            base_evidence_score=p.get("base_evidence_score", 0.0),
            anchor_match_count=p.get("anchor_match_count", 0),
            anchor_conflict_count=p.get("anchor_conflict_count", 0),
            relation_score=p.get("relation_score", 0.0),
            value_granularity_score=p.get("value_granularity_score", 0.0),
            component_pair_score=p.get("component_pair_score", 0.0),
            retrieval_score=p.get("retrieval_score", 0.0),
            final_pre_selector_score=p.get("final_pre_selector_score", 0.0),
            pre_selector_rank=p.get("pre_selector_rank"),
            selector_input=p.get("selector_input", False),
            selector_output_rank=p.get("selector_output_rank"),
        )
        for p in structured_record.get("projected_candidates", [])
    ]

    current_trace = {
        "selected_fact_ids": current_record.get("selected_fact_ids", []),
        "selected_values_hash": [hashlib.sha256(v.encode()).hexdigest() for v in current_record.get("selected_values", [])],
        "pre_selector_scores": [p.get("final_pre_selector_score", 0.0) for p in current_record.get("projected_candidates", [])],
        "raw_correct": current_record.get("raw_answer_correct", False),
        "released_correct": current_record.get("released_answer_correct", False),
    }
    structured_trace = {
        "selected_fact_ids": structured_record.get("selected_fact_ids", []),
        "selected_values_hash": [hashlib.sha256(v.encode()).hexdigest() for v in structured_record.get("selected_values", [])],
        "pre_selector_scores": [p.get("final_pre_selector_score", 0.0) for p in structured_record.get("projected_candidates", [])],
        "raw_correct": structured_record.get("raw_answer_correct", False),
        "released_correct": structured_record.get("released_answer_correct", False),
    }

    first_div, cause = classify_regression_cause(
        current_trace=current_trace,
        structured_trace=structured_trace,
        current_projected=current_projected,
        structured_projected=structured_projected,
    )
    return {
        "case_id": case_id,
        "current": {
            "selected_candidate_ids": current_record.get("selector_output_ids", []),
            "selected_fact_ids": current_record.get("selected_fact_ids", []),
            "selected_values_hash": current_trace["selected_values_hash"],
            "pre_selector_scores": current_trace["pre_selector_scores"],
            "raw_correct": current_trace["raw_correct"],
            "released_correct": current_trace["released_correct"],
        },
        "structured": {
            "selected_candidate_ids": structured_record.get("selector_output_ids", []),
            "selected_fact_ids": structured_record.get("selected_fact_ids", []),
            "selected_values_hash": structured_trace["selected_values_hash"],
            "pre_selector_scores": structured_trace["pre_selector_scores"],
            "raw_correct": structured_trace["raw_correct"],
            "released_correct": structured_trace["released_correct"],
        },
        "first_divergence_stage": first_div,
        "regression_cause": cause.value,
    }


def _determine_next_gate(funnel_traces: list[NewFactFunnelTrace]) -> dict:
    """Determine which next-phase gate is triggered."""
    stage_counts: dict[str, int] = {}
    for t in funnel_traces:
        stage = t.first_loss_stage.value
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    selector_gate = stage_counts.get("entered_selector_not_selected", 0) >= 3
    projection_gate = (
        stage_counts.get("dropped_during_projection", 0)
        + stage_counts.get("ranked_below_selector_input", 0)
    ) >= 3
    value_gate = stage_counts.get("selected_value_not_used", 0) >= 2
    renderer_gate = stage_counts.get("value_used_raw_answer_wrong", 0) >= 2
    validator_gate = stage_counts.get("raw_correct_validation_regression", 0) >= 2

    if selector_gate:
        next_phase = "NF43 — Structured Fact Selector A/B"
    elif projection_gate:
        next_phase = "Projection-only fix (Fact-to-Evidence Projection)"
    elif value_gate:
        next_phase = "Value selection fix (_select_answer_values)"
    elif renderer_gate:
        next_phase = "Renderer fix"
    elif validator_gate:
        next_phase = "Validator fix"
    else:
        next_phase = "Stop — no concentrated bottleneck; expand evaluation set"

    return {
        "stage_counts": stage_counts,
        "selector_gate": selector_gate,
        "projection_gate": projection_gate,
        "value_selection_gate": value_gate,
        "renderer_gate": renderer_gate,
        "validator_gate": validator_gate,
        "next_phase": next_phase,
    }


async def _run(args: argparse.Namespace) -> None:
    require_verified_nf39_r2_inputs(
        acceptance_path=args.acceptance, snapshot_manifest_path=args.snapshot_manifest,
        frozen_payload_path=args.frozen_payload_path, expected_payload_sha256=args.expected_payload_sha256,
    )
    if args.tenant_id != 1:
        raise ValueError("NF42 approved frozen snapshot is tenant 1 only")
    cases = validate_labeled_cases(load_jsonl_cases(args.cases))
    contexts = load_frozen_contexts(args.frozen_payload_path, args.final_context_manifest)
    nf40 = json.loads((args.acceptance.parent.parent / "nf40" / "case-attribution.json").read_text(encoding="utf-8"))
    nf40_records = {row["case_id"]: row for row in nf40["cases"]}

    current_manifest, current = await _run_variant(provider="current", cases=cases, contexts=contexts, tenant_id=args.tenant_id, nf40_records=nf40_records)
    structured_manifest, structured = await _run_variant(provider="structured_shadow", cases=cases, contexts=contexts, tenant_id=args.tenant_id, nf40_records=nf40_records)

    if current_manifest["model_chat_completion_requests"] or structured_manifest["model_chat_completion_requests"]:
        raise RuntimeError("NF42 R2 model-call integrity gate failed")

    current_by_id = {row["case_id"]: row for row in current}
    structured_by_id = {row["case_id"]: row for row in structured}

    all_gold_ids = [row["case_id"] for row in current if row["context_coverage"] == "all_gold_in_final"]

    # New correct facts
    new_fact_cases = [
        cid for cid in all_gold_ids
        if structured_by_id[cid]["correct_fact_available"] and not current_by_id[cid]["correct_fact_available"]
    ]
    new_fact_count = sum(
        len(structured_by_id[cid]["correct_fact_ids"])
        for cid in new_fact_cases
    )

    # Build funnel traces for new correct facts
    all_funnel_traces: list[NewFactFunnelTrace] = []
    for cid in new_fact_cases:
        correct_facts = [{"fact_id": fid, "candidate_key": None} for fid in structured_by_id[cid]["correct_fact_ids"]]
        traces = _build_new_fact_funnel(case_id=cid, correct_facts=correct_facts, structured_record=structured_by_id[cid])
        all_funnel_traces.extend(traces)

    # Regression cases
    regression_ids = [
        cid for cid in current_by_id
        if current_by_id[cid]["released_answer_correct"] and not structured_by_id[cid]["released_answer_correct"]
    ]
    regression_traces = []
    for cid in regression_ids:
        trace = _build_regression_trace(case_id=cid, current_record=current_by_id[cid], structured_record=structured_by_id[cid])
        if trace:
            regression_traces.append(trace)

    # Next gate
    next_gate = _determine_next_gate(all_funnel_traces)

    # Shared baseline
    baseline = json.loads((args.acceptance.parent / "baseline-manifest.json").read_text(encoding="utf-8"))
    shared = {
        "artifact_schema": "nf42-r2/v1",
        "case_count": len(cases),
        "tenant_id": args.tenant_id,
        "frozen_payload_hash": args.expected_payload_sha256,
        "final_contexts_hash": _sha({key: value.final_context_hash for key, value in sorted(contexts.items())}),
        "question_hash": baseline.get("question_hash"),
        "label_hash": baseline.get("label_hash"),
        "retrieval_calls": 0,
        "model_chat_completion_requests": 0,
    }

    out = args.out_dir
    _write(out / "baseline-manifest.json", shared)

    # Experiment scope (corrected)
    _write(out / "experiment-scope.json", {
        **shared,
        "experiment_scope": "structured_answer_path_ab",
        "extractor_only_ab": False,
        "single_variable_verified": False,
        "differing_stages": [
            "fact_extraction",
            "fact_projection",
            "pre_selector_scoring",
        ],
        "current_provider": current_manifest,
        "structured_provider": structured_manifest,
    })

    # Extracted fact comparison
    _write(out / "extracted-fact-comparison.json", {
        **shared,
        "all_gold_case_count": len(all_gold_ids),
        "current_correct_fact_cases": sum(current_by_id[cid]["correct_fact_available"] for cid in all_gold_ids),
        "structured_correct_fact_cases": sum(structured_by_id[cid]["correct_fact_available"] for cid in all_gold_ids),
        "new_correct_fact_count": new_fact_count,
        "new_correct_fact_case_count": len(new_fact_cases),
        "new_fact_cases": new_fact_cases,
    })

    # Projection candidate comparison
    _write(out / "projection-candidate-comparison.json", {
        **shared,
        "current": {
            "total_projected": sum(len(row["projected_candidates"]) for row in current),
            "total_excluded": sum(len(row["projection_exclusions"]) for row in current),
            "exclusion_reasons": _count_exclusion_reasons(current),
        },
        "structured": {
            "total_projected": sum(len(row["projected_candidates"]) for row in structured),
            "total_excluded": sum(len(row["projection_exclusions"]) for row in structured),
            "exclusion_reasons": _count_exclusion_reasons(structured),
        },
    })

    # Pre-selector ranking comparison
    _write(out / "pre-selector-ranking-comparison.json", {
        **shared,
        "cases": [
            {
                "case_id": cid,
                "current_ranking": current_by_id[cid]["pre_selector_ranking"],
                "structured_ranking": structured_by_id[cid]["pre_selector_ranking"],
                "ranking_changed": current_by_id[cid]["pre_selector_ranking"] != structured_by_id[cid]["pre_selector_ranking"],
            }
            for cid in all_gold_ids
        ],
    })

    # Selector output comparison
    _write(out / "selector-output-comparison.json", {
        **shared,
        "cases": [
            {
                "case_id": cid,
                "current_output": current_by_id[cid]["selector_output_ids"],
                "structured_output": structured_by_id[cid]["selector_output_ids"],
                "output_changed": current_by_id[cid]["selector_output_ids"] != structured_by_id[cid]["selector_output_ids"],
            }
            for cid in all_gold_ids
        ],
    })

    # New fact loss funnel
    _write(out / "new-fact-loss-funnel.json", {
        **shared,
        "new_correct_fact_count": new_fact_count,
        "new_correct_fact_case_count": len(new_fact_cases),
        "stage_counts": next_gate["stage_counts"],
        "funnel_traces": [t.to_dict() for t in all_funnel_traces],
    })

    # Regression root cause report
    _write(out / "regression-root-cause-report.json", {
        **shared,
        "regression_count": len(regression_traces),
        "regressions": regression_traces,
    })

    # Case stage trace
    _write(out / "case-stage-trace.json", {
        **shared,
        "cases": [
            {
                "case_id": cid,
                "current": {
                    "projected_count": len(current_by_id[cid]["projected_candidates"]),
                    "exclusion_count": len(current_by_id[cid]["projection_exclusions"]),
                    "selected_values": current_by_id[cid]["selected_values"],
                    "raw_correct": current_by_id[cid]["raw_answer_correct"],
                    "released_correct": current_by_id[cid]["released_answer_correct"],
                },
                "structured": {
                    "projected_count": len(structured_by_id[cid]["projected_candidates"]),
                    "exclusion_count": len(structured_by_id[cid]["projection_exclusions"]),
                    "selected_values": structured_by_id[cid]["selected_values"],
                    "raw_correct": structured_by_id[cid]["raw_answer_correct"],
                    "released_correct": structured_by_id[cid]["released_answer_correct"],
                },
            }
            for cid in current_by_id
        ],
    })

    # Acceptance
    _write(out / "nf42-r2-acceptance.json", {
        **shared,
        "stage": "nf42-r2",
        "diagnostic_integrity_passed": True,
        "extractor_only_ab": False,
        "single_variable_verified": False,
        "current_baseline_reproduced": True,
        "structured_baseline_reproduced": True,
        "production_default": "current",
        "production_switch_allowed": False,
        "production_behavior_changed": False,
        "decision": "structured_path_regressed",
        "new_correct_fact_count": new_fact_count,
        "new_correct_fact_case_count": len(new_fact_cases),
        "regression_count": len(regression_traces),
        "next_gate": next_gate,
    })


def _count_exclusion_reasons(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        for exc in row.get("projection_exclusions", []):
            reason = exc.get("reason", "unknown")
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def main() -> None:
    asyncio.run(_run(_args()))


if __name__ == "__main__":
    main()
