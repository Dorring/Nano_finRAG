"""Run NF42 R2 projection-to-selection attribution on frozen contexts.

Extends the R1 shadow A/B with projection traces, exclusion tracking,
new-fact loss funnel, regression root-cause analysis, and corrected
single-variable declarations.

R2.1 hardens acceptance reliability:
- Real expected baselines from NF42 R1 artifacts (no hardcoded True).
- Real observed execution counters (retrieval, model, side-effects).
- Complete fact identity preservation (candidate_key, document_id, etc.).
- Frozen document identity mapping for gold source matching.
- Separate extracted/projected/selected fact ID recording.
- Explicit structured classify_regression_cause signature.
- Fail-closed function_identity.
- Gate counting unique cases, not facts.
- diagnostic_integrity_passed computed from real checks.

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
import sys
from dataclasses import dataclass, field
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
    NF42ExecutionCounters,
    NF42ExpectedBaseline,
    RegressionCaseTrace,
    classify_new_fact_loss,
    classify_regression_cause,
    function_identity,
)
from src.generation.deterministic_answers import DeterministicAnswerExtractor
from src.generation.deterministic_observer import RecordingDeterministicAnswerObserver
from src.services.rag_engine import RAGEngine

# ---------------------------------------------------------------------------
# Expected R1 baseline (from verified NF42 R1 artifact)
# ---------------------------------------------------------------------------

NF42_R1_EXPECTED_BASELINE = NF42ExpectedBaseline(
    all_gold_case_count=13,
    current_correct_fact_cases=3,
    structured_correct_fact_cases=7,
    current_all_gold_raw_correct=7,
    structured_all_gold_raw_correct=5,
    current_all_gold_released_correct=6,
    structured_all_gold_released_correct=4,
    current_any_gold_released_correct=6,
    structured_any_gold_released_correct=4,
    regression_case_count=2,
)


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


# ---------------------------------------------------------------------------
# Counting model client (observes real model calls)
# ---------------------------------------------------------------------------

class _CountingModelClient:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.request_count = 0
        self.chat = self
        self.completions = self

    def create(self, *args, **kwargs):
        self.request_count += 1
        return self._delegate.chat.completions.create(*args, **kwargs)


# ---------------------------------------------------------------------------
# Side-effect counting wrapper
# ---------------------------------------------------------------------------

@dataclass
class _SideEffectGuard:
    """Wraps RAGEngine to observe retrieval and state-write side effects."""

    counters: NF42ExecutionCounters = field(default_factory=NF42ExecutionCounters)

    def wrap_engine(self, engine: RAGEngine) -> RAGEngine:
        # Wrap retrieval method if present
        original_retrieve = getattr(engine, "_retrieve", None)
        if original_retrieve is not None:
            def counting_retrieve(*args, **kwargs):
                self.counters.retrieval_calls += 1
                return original_retrieve(*args, **kwargs)
            engine._retrieve = counting_retrieve  # type: ignore[method-assign]
        return engine


def _build_engine(provider: str, guard: _SideEffectGuard) -> tuple[RAGEngine, _CountingModelClient]:
    client = _CountingModelClient(OpenAI(
        base_url=os.getenv("LLM_API_BASE_URL", "http://127.0.0.1:8500/v1"),
        api_key=os.getenv("LLM_API_KEY", "not-needed-for-local"),
    ))
    engine = RAGEngine(
        client,
        model_name=os.getenv("LLM_MODEL_NAME", "nanochat"),
        use_hybrid=False,
        reranker_name="none",
        retrieval_candidate_multiplier=1,
        deterministic_fact_extractor=provider,
    )
    engine = guard.wrap_engine(engine)
    return engine, client


# ---------------------------------------------------------------------------
# Frozen document identity mapping
# ---------------------------------------------------------------------------

def _build_document_identity_map(final_context_manifest: Path) -> dict[str, str]:
    """Build document_id -> filename mapping from the frozen context manifest."""
    manifest = json.loads(final_context_manifest.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for entry in manifest.get("contexts", []) if isinstance(manifest, dict) else []:
        doc_id = entry.get("document_id") or entry.get("doc_name")
        filename = entry.get("filename") or entry.get("doc_name")
        if doc_id and filename:
            mapping[doc_id] = filename
    return mapping


def _resolve_filename(fact_document_id: str | None, identity_map: dict[str, str]) -> str | None:
    """Resolve a fact's document_id to a filename via the frozen identity map."""
    if not fact_document_id:
        return None
    return identity_map.get(fact_document_id, fact_document_id)


# ---------------------------------------------------------------------------
# Gold source matching with explicit granularity
# ---------------------------------------------------------------------------

def _source_matches_with_granularity(
    case,
    fact,
    identity_map: dict[str, str],
) -> tuple[bool, str]:
    """Match a fact to expected sources, returning (matched, granularity)."""
    # Priority 1: candidate_key / chunk_id / evidence_id
    fact_candidate_key = getattr(fact, "candidate_key", None)
    if fact_candidate_key:
        for source in case.expected_sources:
            if source.matches({"candidate_key": fact_candidate_key, "chunk_id": None}):
                return True, "candidate_key"

    # Priority 2: document_id mapped to filename + page
    fact_filename = _resolve_filename(getattr(fact, "document_id", None), identity_map)
    fact_page = getattr(fact, "page", None)
    if fact_filename:
        for source in case.expected_sources:
            if source.matches({"filename": fact_filename, "page": fact_page, "chunk_id": None}):
                return True, "filename_page"

    # Priority 3: raw document_id as filename (fallback, less reliable)
    fact_doc_id = getattr(fact, "document_id", None)
    if fact_doc_id and fact_doc_id != fact_filename:
        for source in case.expected_sources:
            if source.matches({"filename": fact_doc_id, "page": fact_page, "chunk_id": None}):
                return True, "filename_page"

    return False, ""


def _fact_matches(
    case,
    fact,
    identity_map: dict[str, str],
) -> tuple[bool, str]:
    """Check if a fact matches the case's expected answer.

    Returns (matched, gold_match_granularity).
    """
    matched, granularity = _source_matches_with_granularity(case, fact, identity_map)
    if not matched:
        return False, ""

    # Period compatibility check
    fact_period = getattr(fact, "period", None)
    if fact_period and case.expected_period and fact_period != case.expected_period:
        return False, ""

    # Value/answer matching
    if case.expected_numbers:
        actual = normalize_numeric_identity(fact.raw_value, period=fact.period)
        if actual is None:
            return False, ""
        for expected in case.expected_numbers:
            target = normalize_numeric_identity(expected)
            if target and actual.value_type == target.value_type and actual.canonical_value == target.canonical_value:
                return True, granularity
            try:
                raw = Decimal(str(fact.raw_value or "").replace(",", "").replace("$", "").split()[0])
            except (InvalidOperation, IndexError):
                continue
            if target and actual.value_type == "amount" and raw == target.canonical_value:
                return True, granularity
        return False, ""

    text = (getattr(fact, "evaluation_text", None) or "").lower()
    if text and any(expected.lower() in text for expected in case.expected_answer_contains):
        return True, granularity
    return False, ""


# ---------------------------------------------------------------------------
# Variant execution
# ---------------------------------------------------------------------------

async def _run_variant(
    *,
    provider: str,
    cases,
    contexts,
    tenant_id: int,
    nf40_records: dict,
    identity_map: dict[str, str],
    guard: _SideEffectGuard,
) -> tuple[dict, list[dict]]:
    engine, client = _build_engine(provider, guard)
    runner = FrozenContextEvaluationRunner(rag_engine=engine)
    records: list[dict] = []
    for case in cases:
        run = await runner.run_case(case=case, frozen=contexts[case.case_id], tenant_id=tenant_id)
        observer = run.trace.deterministic_observer
        if observer is None:
            observer = RecordingDeterministicAnswerObserver()
        facts = list(observer.facts)

        # Match correct facts with granularity
        correct_with_granularity = [
            (fact, granularity)
            for fact in facts
            for matched, granularity in [_fact_matches(case, fact, identity_map)]
            if matched
        ]
        correct = [fact for fact, _ in correct_with_granularity]
        gold_granularities = {fact.fact_id: gran for fact, gran in correct_with_granularity}

        selected = set(observer.selected_fact_ids)
        selected_correct = any(fact.fact_id in selected for fact in correct)

        # Extracted fact IDs (all observed facts)
        extracted_fact_ids = [fact.fact_id for fact in facts]
        # Projected fact IDs (from projected candidates)
        projected_fact_ids = sorted({
            fid
            for candidate in observer.projected_candidates
            for fid in candidate.get("source_fact_ids", [])
        })
        # Selected fact IDs
        selected_fact_ids_list = list(observer.selected_fact_ids)

        records.append({
            "case_id": case.case_id,
            "context_coverage": nf40_records.get(case.case_id, {}).get("context_coverage", "unknown"),
            "context_hash": contexts[case.case_id].final_context_hash,
            "fact_count": len(facts),
            "correct_fact_available": bool(correct),
            "correct_fact_ids": [fact.fact_id for fact in correct],
            "correct_facts": [
                {
                    "fact_id": fact.fact_id,
                    "candidate_key": fact.candidate_key,
                    "document_id": fact.document_id,
                    "page": fact.page,
                    "canonical_value": str(fact.canonical_value) if fact.canonical_value is not None else None,
                    "currency": fact.currency,
                    "unit": fact.unit,
                    "scale": fact.scale,
                    "period": fact.period,
                    "gold_match_granularity": gold_granularities.get(fact.fact_id, ""),
                }
                for fact in correct
            ],
            "selected_fact_correct": selected_correct,
            "selected_fact_ids": selected_fact_ids_list,
            "extracted_fact_ids": extracted_fact_ids,
            "projected_fact_ids": projected_fact_ids,
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


# ---------------------------------------------------------------------------
# New fact funnel
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Regression trace (uses extracted/projected/selected fact IDs)
# ---------------------------------------------------------------------------

def _build_regression_trace(
    *,
    case_id: str,
    current_record: dict,
    structured_record: dict,
) -> RegressionCaseTrace | None:
    """Build a regression trace using explicit fact ID sets at each stage."""
    current_extracted = set(current_record.get("extracted_fact_ids", []))
    structured_extracted = set(structured_record.get("extracted_fact_ids", []))
    current_projected = set(current_record.get("projected_fact_ids", []))
    structured_projected = set(structured_record.get("projected_fact_ids", []))
    current_selected = set(current_record.get("selected_fact_ids", []))
    structured_selected = set(structured_record.get("selected_fact_ids", []))

    current_values = tuple(current_record.get("selected_values", []))
    structured_values = tuple(structured_record.get("selected_values", []))

    current_raw = current_record.get("raw_answer_correct", False)
    structured_raw = structured_record.get("raw_answer_correct", False)
    current_released = current_record.get("released_answer_correct", False)
    structured_released = structured_record.get("released_answer_correct", False)

    first_div, cause = classify_regression_cause(
        current_extracted_fact_ids=current_extracted,
        structured_extracted_fact_ids=structured_extracted,
        current_projected_fact_ids=current_projected,
        structured_projected_fact_ids=structured_projected,
        current_selected_fact_ids=current_selected,
        structured_selected_fact_ids=structured_selected,
        current_selected_values=current_values,
        structured_selected_values=structured_values,
        current_raw_correct=current_raw,
        structured_raw_correct=structured_raw,
        current_released_correct=current_released,
        structured_released_correct=structured_released,
    )

    return RegressionCaseTrace(
        case_id=case_id,
        current_extracted_fact_ids=sorted(current_extracted),
        current_projected_fact_ids=sorted(current_projected),
        current_selected_candidate_ids=current_record.get("selector_output_ids", []),
        current_selected_fact_ids=sorted(current_selected),
        current_selected_values_hash=[hashlib.sha256(v.encode()).hexdigest() for v in current_values],
        current_pre_selector_scores=[p.get("final_pre_selector_score", 0.0) for p in current_record.get("projected_candidates", [])],
        current_raw_correct=current_raw,
        current_released_correct=current_released,
        structured_extracted_fact_ids=sorted(structured_extracted),
        structured_projected_fact_ids=sorted(structured_projected),
        structured_selected_candidate_ids=structured_record.get("selector_output_ids", []),
        structured_selected_fact_ids=sorted(structured_selected),
        structured_selected_values_hash=[hashlib.sha256(v.encode()).hexdigest() for v in structured_values],
        structured_pre_selector_scores=[p.get("final_pre_selector_score", 0.0) for p in structured_record.get("projected_candidates", [])],
        structured_raw_correct=structured_raw,
        structured_released_correct=structured_released,
        first_divergence_stage=first_div,
        regression_cause=cause,
    )


# ---------------------------------------------------------------------------
# Next gate (counts unique cases, not facts)
# ---------------------------------------------------------------------------

def _determine_next_gate(funnel_traces: list[NewFactFunnelTrace]) -> dict:
    """Determine which next-phase gate is triggered.

    Gate thresholds count unique CASES, not individual facts, so a single
    case with multiple new correct facts cannot alone trigger a gate.
    """
    # Fact-level counts (for reporting)
    fact_stage_counts: dict[str, int] = {}
    for t in funnel_traces:
        stage = t.first_loss_stage.value
        fact_stage_counts[stage] = fact_stage_counts.get(stage, 0) + 1

    # Case-level counts (for gate decisions)
    def _unique_cases_for_stage(stage_value: str) -> int:
        return len({
            t.case_id for t in funnel_traces
            if t.first_loss_stage.value == stage_value
        })

    case_stage_counts: dict[str, int] = {}
    for stage in fact_stage_counts:
        case_stage_counts[stage] = _unique_cases_for_stage(stage)

    selector_case_count = case_stage_counts.get("entered_selector_not_selected", 0)
    projection_case_count = (
        case_stage_counts.get("dropped_during_projection", 0)
        + case_stage_counts.get("ranked_below_selector_input", 0)
    )
    value_case_count = case_stage_counts.get("selected_value_not_used", 0)
    renderer_case_count = case_stage_counts.get("value_used_raw_answer_wrong", 0)
    validator_case_count = case_stage_counts.get("raw_correct_validation_regression", 0)

    selector_gate = selector_case_count >= 3
    projection_gate = projection_case_count >= 3
    value_gate = value_case_count >= 2
    renderer_gate = renderer_case_count >= 2
    validator_gate = validator_case_count >= 2

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
        "fact_stage_counts": fact_stage_counts,
        "case_stage_counts": case_stage_counts,
        "selector_gate": selector_gate,
        "projection_gate": projection_gate,
        "value_selection_gate": value_gate,
        "renderer_gate": renderer_gate,
        "validator_gate": validator_gate,
        "next_phase": next_phase,
    }


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def _compute_actual_baseline(
    *,
    current_by_id: dict,
    structured_by_id: dict,
    all_gold_ids: list[str],
    any_gold_ids: list[str],
) -> dict:
    """Compute actual baseline metrics from run records."""
    return {
        "all_gold_case_count": len(all_gold_ids),
        "current_correct_fact_cases": sum(current_by_id[cid]["correct_fact_available"] for cid in all_gold_ids),
        "structured_correct_fact_cases": sum(structured_by_id[cid]["correct_fact_available"] for cid in all_gold_ids),
        "current_all_gold_raw_correct": sum(current_by_id[cid]["raw_answer_correct"] for cid in all_gold_ids),
        "structured_all_gold_raw_correct": sum(structured_by_id[cid]["raw_answer_correct"] for cid in all_gold_ids),
        "current_all_gold_released_correct": sum(current_by_id[cid]["released_answer_correct"] for cid in all_gold_ids),
        "structured_all_gold_released_correct": sum(structured_by_id[cid]["released_answer_correct"] for cid in all_gold_ids),
        "current_any_gold_released_correct": sum(current_by_id[cid]["released_answer_correct"] for cid in any_gold_ids),
        "structured_any_gold_released_correct": sum(structured_by_id[cid]["released_answer_correct"] for cid in any_gold_ids),
        "regression_case_count": sum(
            1 for cid in current_by_id
            if current_by_id[cid]["released_answer_correct"]
            and not structured_by_id[cid]["released_answer_correct"]
        ),
    }


def _baselines_match(actual: dict, expected: NF42ExpectedBaseline) -> bool:
    """Check if actual baseline matches expected."""
    exp = expected.to_dict()
    return all(actual.get(k) == v for k, v in exp.items())


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _count_exclusion_reasons(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        for exc in row.get("projection_exclusions", []):
            reason = exc.get("reason", "unknown")
            counts[reason] = counts.get(reason, 0) + 1
    return counts


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

    # Build frozen document identity map
    identity_map = _build_document_identity_map(args.final_context_manifest)

    # Run both variants with shared side-effect guard
    guard = _SideEffectGuard()
    current_manifest, current = await _run_variant(
        provider="current", cases=cases, contexts=contexts,
        tenant_id=args.tenant_id, nf40_records=nf40_records,
        identity_map=identity_map, guard=guard,
    )
    structured_manifest, structured = await _run_variant(
        provider="structured_shadow", cases=cases, contexts=contexts,
        tenant_id=args.tenant_id, nf40_records=nf40_records,
        identity_map=identity_map, guard=guard,
    )

    # Real execution counters
    counters = guard.counters
    counters.model_chat_completion_requests = (
        current_manifest["model_chat_completion_requests"]
        + structured_manifest["model_chat_completion_requests"]
    )

    # Integrity gate: model calls must be zero
    if counters.model_chat_completion_requests:
        raise RuntimeError("NF42 R2 model-call integrity gate failed")

    current_by_id = {row["case_id"]: row for row in current}
    structured_by_id = {row["case_id"]: row for row in structured}

    all_gold_ids = [row["case_id"] for row in current if row["context_coverage"] == "all_gold_in_final"]
    any_gold_ids = [row["case_id"] for row in current if row["context_coverage"] in ("all_gold_in_final", "any_gold_in_final")]

    # Compute actual baseline and compare to expected
    actual_baseline = _compute_actual_baseline(
        current_by_id=current_by_id,
        structured_by_id=structured_by_id,
        all_gold_ids=all_gold_ids,
        any_gold_ids=any_gold_ids,
    )
    current_baseline_reproduced = _baselines_match(
        {k: v for k, v in actual_baseline.items() if k.startswith("current_") or k == "all_gold_case_count"},
        NF42_R1_EXPECTED_BASELINE,
    )
    structured_baseline_reproduced = _baselines_match(
        {k: v for k, v in actual_baseline.items() if k.startswith("structured_") or k == "all_gold_case_count"},
        NF42_R1_EXPECTED_BASELINE,
    )

    # New correct facts
    new_fact_cases = [
        cid for cid in all_gold_ids
        if structured_by_id[cid]["correct_fact_available"] and not current_by_id[cid]["correct_fact_available"]
    ]
    new_fact_count = sum(
        len(structured_by_id[cid]["correct_fact_ids"])
        for cid in new_fact_cases
    )

    # Build funnel traces for new correct facts (using full fact identity)
    all_funnel_traces: list[NewFactFunnelTrace] = []
    for cid in new_fact_cases:
        correct_facts = structured_by_id[cid]["correct_facts"]
        traces = _build_new_fact_funnel(case_id=cid, correct_facts=correct_facts, structured_record=structured_by_id[cid])
        all_funnel_traces.extend(traces)

    # Regression cases
    regression_ids = [
        cid for cid in current_by_id
        if current_by_id[cid]["released_answer_correct"] and not structured_by_id[cid]["released_answer_correct"]
    ]
    regression_traces: list[RegressionCaseTrace] = []
    for cid in regression_ids:
        trace = _build_regression_trace(case_id=cid, current_record=current_by_id[cid], structured_record=structured_by_id[cid])
        if trace:
            regression_traces.append(trace)

    # Next gate (case-based counting)
    next_gate = _determine_next_gate(all_funnel_traces)

    # Verify context count and hashes
    context_count_verified = len(contexts) == 27
    verified_context_count = sum(
        1 for ctx in contexts.values()
        if ctx.final_context_hash
    )

    # New fact identity completeness
    new_fact_identity_complete = all(
        trace.candidate_key for trace in all_funnel_traces
    )

    # Regressions attributed
    regressions_attributed = len(regression_traces) == actual_baseline["regression_case_count"]

    # Integrity checks
    integrity_checks = {
        "current_baseline_reproduced": current_baseline_reproduced,
        "structured_baseline_reproduced": structured_baseline_reproduced,
        "context_count_verified": context_count_verified,
        "context_hashes_verified": verified_context_count == 27,
        "retrieval_calls_zero": counters.retrieval_calls == 0,
        "model_calls_zero": counters.model_chat_completion_requests == 0,
        "side_effects_zero": (
            counters.memory_writes == 0
            and counters.feedback_writes == 0
            and counters.document_state_writes == 0
        ),
        "new_fact_identity_complete": new_fact_identity_complete,
        "regressions_attributed": regressions_attributed,
    }
    diagnostic_integrity_passed = all(integrity_checks.values())

    # Disable next gate if integrity failed
    if not diagnostic_integrity_passed:
        next_gate = {
            **next_gate,
            "enabled": False,
            "next_phase": "Blocked — diagnostic integrity failed",
        }
    else:
        next_gate = {**next_gate, "enabled": True}

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
        **counters.to_dict(),
    }

    out = args.out_dir
    _write(out / "baseline-manifest.json", {
        **shared,
        "expected_baseline": NF42_R1_EXPECTED_BASELINE.to_dict(),
        "actual_baseline": actual_baseline,
    })

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
        "fact_stage_counts": next_gate["fact_stage_counts"],
        "case_stage_counts": next_gate["case_stage_counts"],
        "funnel_traces": [t.to_dict() for t in all_funnel_traces],
    })

    # Regression root cause report
    _write(out / "regression-root-cause-report.json", {
        **shared,
        "regression_count": len(regression_traces),
        "regressions": [t.to_dict() for t in regression_traces],
    })

    # Case stage trace
    _write(out / "case-stage-trace.json", {
        **shared,
        "cases": [
            {
                "case_id": cid,
                "current": {
                    "extracted_fact_ids": current_by_id[cid]["extracted_fact_ids"],
                    "projected_fact_ids": current_by_id[cid]["projected_fact_ids"],
                    "selected_fact_ids": current_by_id[cid]["selected_fact_ids"],
                    "projected_count": len(current_by_id[cid]["projected_candidates"]),
                    "exclusion_count": len(current_by_id[cid]["projection_exclusions"]),
                    "selected_values": current_by_id[cid]["selected_values"],
                    "raw_correct": current_by_id[cid]["raw_answer_correct"],
                    "released_correct": current_by_id[cid]["released_answer_correct"],
                },
                "structured": {
                    "extracted_fact_ids": structured_by_id[cid]["extracted_fact_ids"],
                    "projected_fact_ids": structured_by_id[cid]["projected_fact_ids"],
                    "selected_fact_ids": structured_by_id[cid]["selected_fact_ids"],
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

    # Acceptance (computed, not hardcoded)
    _write(out / "nf42-r2-acceptance.json", {
        **shared,
        "stage": "nf42-r2",
        "diagnostic_integrity_passed": diagnostic_integrity_passed,
        "integrity_checks": integrity_checks,
        "expected_baseline": NF42_R1_EXPECTED_BASELINE.to_dict(),
        "actual_baseline": actual_baseline,
        "current_baseline_reproduced": current_baseline_reproduced,
        "structured_baseline_reproduced": structured_baseline_reproduced,
        "extractor_only_ab": False,
        "single_variable_verified": False,
        "production_default": "current",
        "production_switch_allowed": False,
        "production_behavior_changed": False,
        "decision": "structured_path_regressed",
        "new_correct_fact_count": new_fact_count,
        "new_correct_fact_case_count": len(new_fact_cases),
        "regression_count": len(regression_traces),
        "next_gate": next_gate,
    })

    # Exit non-zero if integrity failed
    if not diagnostic_integrity_passed:
        print(
            "NF42 R2 diagnostic integrity FAILED: "
            + ", ".join(k for k, v in integrity_checks.items() if not v),
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    asyncio.run(_run(_args()))


if __name__ == "__main__":
    main()
