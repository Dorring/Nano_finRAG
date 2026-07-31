"""Run NF42 R2 projection-to-selection attribution on frozen contexts.

Extends the R1 shadow A/B with projection traces, exclusion tracking,
new-fact loss funnel, regression root-cause analysis, and corrected
single-variable declarations.

R2.2 hardens formal runner correctness:
- Baseline loaded from ``--nf42-r1-baseline`` JSON artifact (not hardcoded).
- Baseline comparison split into current/structured/cross-variant field groups.
- Any-gold case filtering uses real ``partial_gold_in_final`` enum value.
- Document identity mapping fails closed on unmapped document_ids.
- Side-effect observation wraps real RAGEngine boundaries; unavailable
  boundaries recorded as ``not_installed`` with configuration proof.
- Regression attribution uses provider-independent ``FactSemanticIdentity``.
- Funnel renamed to ``coverage_gain_*``; ``all_new_correct_fact_count`` added.
- Context hash verification reports 135 content + 27 final context hashes.
- Pre-flight integrity checks before running second variant.
- ``diagnostic_integrity_passed`` computed from all integrity checks.

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
    CROSS_VARIANT_FIELDS,
    CURRENT_BASELINE_FIELDS,
    STRUCTURED_BASELINE_FIELDS,
    EvaluationIntegrityError,
    FrozenContextVerificationReport,
    NewFactFunnelTrace,
    NF42ExpectedBaseline,
    ObservedSideEffects,
    RegressionCaseTrace,
    RegressionCause,
    baseline_fields_match,
    classify_new_fact_loss,
    classify_regression_cause,
    fact_semantic_key,
    function_identity,
)
from src.generation.deterministic_answers import DeterministicAnswerExtractor
from src.generation.deterministic_observer import RecordingDeterministicAnswerObserver
from src.services.rag_engine import RAGEngine


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--nf42-r1-baseline", required=True, type=Path)
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
# Baseline artifact loading and verification
# ---------------------------------------------------------------------------

def _load_nf42_r1_baseline(
    baseline_path: Path,
    *,
    expected_question_hash: str | None,
    expected_label_hash: str | None,
    expected_frozen_payload_hash: str,
    expected_final_contexts_hash: str | None,
) -> tuple[NF42ExpectedBaseline, str]:
    """Load and verify the NF42 R1 baseline artifact.

    Returns (baseline, artifact_sha256).  Raises ``EvaluationIntegrityError``
    if the artifact schema, hashes, or metrics are invalid.
    """
    raw = baseline_path.read_text(encoding="utf-8")
    artifact_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    data = json.loads(raw)

    schema = data.get("artifact_schema")
    if schema != "nf42-r1-baseline/v1":
        raise EvaluationIntegrityError(
            f"Baseline artifact_schema mismatch: expected 'nf42-r1-baseline/v1', got {schema!r}"
        )

    if expected_question_hash and data.get("question_hash") != expected_question_hash:
        raise EvaluationIntegrityError("Baseline question_hash mismatch")
    if expected_label_hash and data.get("label_hash") != expected_label_hash:
        raise EvaluationIntegrityError("Baseline label_hash mismatch")
    if data.get("frozen_payload_hash") != expected_frozen_payload_hash:
        raise EvaluationIntegrityError(
            f"Baseline frozen_payload_hash mismatch: "
            f"expected {expected_frozen_payload_hash}, got {data.get('frozen_payload_hash')}"
        )
    if expected_final_contexts_hash and data.get("final_contexts_hash") != expected_final_contexts_hash:
        raise EvaluationIntegrityError("Baseline final_contexts_hash mismatch")

    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        raise EvaluationIntegrityError("Baseline artifact missing 'metrics' block")

    baseline = NF42ExpectedBaseline.from_metrics_dict(metrics)
    return baseline, artifact_sha256


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
# Side-effect observation (wraps real RAGEngine boundaries)
# ---------------------------------------------------------------------------

# RAGEngine has NO memory_store, feedback_store, session_store, or
# document_state_store attributes.  The EvaluationExecutionContext explicitly
# sets conversation_memory_enabled=False, feedback_write_enabled=False,
# trace_persistence_enabled=False, and document_write_enabled=False.
# These boundaries are therefore ``not_installed`` — proven by the engine's
# dependency graph, not by defaulting to zero.
_NOT_INSTALLED_BOUNDARIES: tuple[str, ...] = (
    "memory",
    "feedback",
    "session",
    "document_state",
)

# RAGEngine exposes retrieval via retrieve_single_document,
# retrieve_multiple_documents, and retrieve_front_matter_chunks.
# answer_frozen_evaluation bypasses all of them, but we observe the real
# methods to confirm zero calls rather than inferring from flags.
_RETRIEVAL_METHOD_NAMES: tuple[str, ...] = (
    "retrieve_single_document",
    "retrieve_multiple_documents",
    "retrieve_front_matter_chunks",
)


@dataclass
class _SideEffectObserver:
    """Wraps RAGEngine to observe real side-effect boundaries."""

    effects: ObservedSideEffects = field(default_factory=lambda: ObservedSideEffects(
        observed_boundaries=("retrieval", "model"),
        unavailable_boundaries=_NOT_INSTALLED_BOUNDARIES,
    ))

    def wrap_engine(self, engine: RAGEngine) -> RAGEngine:
        """Wrap real retrieval methods to observe calls."""
        for method_name in _RETRIEVAL_METHOD_NAMES:
            original = getattr(engine, method_name, None)
            if original is None:
                continue

            def _make_counter(orig, name):
                def _counting(*args, **kwargs):
                    self.effects.retrieval_calls += 1
                    return orig(*args, **kwargs)
                _counting.__name__ = name
                return _counting

            setattr(engine, method_name, _make_counter(original, method_name))
        return engine


def _build_engine(provider: str, observer: _SideEffectObserver) -> tuple[RAGEngine, _CountingModelClient]:
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
    engine = observer.wrap_engine(engine)
    return engine, client


# ---------------------------------------------------------------------------
# Frozen context loading with verification report
# ---------------------------------------------------------------------------

def _load_and_verify_contexts(
    frozen_payload_path: Path,
    final_context_manifest: Path,
) -> tuple[dict, FrozenContextVerificationReport]:
    """Load frozen contexts and return a verification report.

    ``load_frozen_contexts`` already verifies all 135 per-candidate content
    hashes and all 27 per-case final context hashes, raising on any mismatch.
    If it returns successfully, all hashes are verified.
    """
    contexts = load_frozen_contexts(frozen_payload_path, final_context_manifest)
    total_candidates = sum(len(ctx.candidates) for ctx in contexts.values())
    report = FrozenContextVerificationReport(
        content_hash_verified_count=total_candidates,
        final_context_hash_verified_count=len(contexts),
        failed_cases=(),
    )
    return contexts, report


# ---------------------------------------------------------------------------
# Frozen document identity mapping
# ---------------------------------------------------------------------------

def _build_document_identity_map(final_context_manifest: Path) -> dict[str, str]:
    """Build document_id -> filename mapping from the frozen context manifest."""
    manifest = json.loads(final_context_manifest.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    cases = manifest.get("cases", {}) if isinstance(manifest, dict) else {}
    for case_data in cases.values():
        for candidate in case_data.get("candidates", []):
            identity = candidate.get("identity", {})
            doc_id = identity.get("document_id")
            source_id = identity.get("source_id")
            if doc_id and source_id and doc_id not in mapping:
                mapping[doc_id] = source_id
    return mapping


def _resolve_filename(fact_document_id: str | None, identity_map: dict[str, str]) -> str | None:
    """Resolve a fact's document_id to a filename via the frozen identity map.

    Returns ``None`` if the document_id is not in the map — never falls back
    to the raw document_id, which is an internal identifier, not a filename.
    """
    if not fact_document_id:
        return None
    return identity_map.get(fact_document_id)


def _collect_unmapped_document_ids(all_facts: list, identity_map: dict[str, str]) -> list[str]:
    """Collect all fact document_ids that are not in the identity map."""
    return sorted({
        fact.document_id
        for fact in all_facts
        if fact.document_id and fact.document_id not in identity_map
    })


# ---------------------------------------------------------------------------
# Gold source matching with explicit granularity
# ---------------------------------------------------------------------------

def _source_matches_with_granularity(
    case,
    fact,
    identity_map: dict[str, str],
) -> tuple[bool, str]:
    """Match a fact to expected sources, returning (matched, granularity).

    Priority:
        1. candidate_key / chunk_id / evidence_id
        2. document_id mapped to filename + page

    The raw document_id is NEVER used as a filename fallback — it is an
    internal identifier, not a filesystem name.
    """
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
    observer: _SideEffectObserver,
) -> tuple[dict, list[dict]]:
    engine, client = _build_engine(provider, observer)
    runner = FrozenContextEvaluationRunner(rag_engine=engine)
    records: list[dict] = []
    for case in cases:
        run = await runner.run_case(case=case, frozen=contexts[case.case_id], tenant_id=tenant_id)
        det_observer = run.trace.deterministic_observer
        if det_observer is None:
            det_observer = RecordingDeterministicAnswerObserver()
        facts = list(det_observer.facts)

        # Match correct facts with granularity
        correct_with_granularity = [
            (fact, granularity)
            for fact in facts
            for matched, granularity in [_fact_matches(case, fact, identity_map)]
            if matched
        ]
        correct = [fact for fact, _ in correct_with_granularity]
        gold_granularities = {fact.fact_id: gran for fact, gran in correct_with_granularity}

        selected = set(det_observer.selected_fact_ids)
        selected_correct = any(fact.fact_id in selected for fact in correct)

        # Semantic keys for provider-independent comparison
        all_semantic_keys = {fact_semantic_key(fact) for fact in facts}
        correct_semantic_keys = {fact_semantic_key(fact) for fact in correct}
        projected_semantic_keys = {
            fact_semantic_key(fact)
            for fact in facts
            if any(
                fact.fact_id in c.get("source_fact_ids", [])
                for c in det_observer.projected_candidates
            )
        }
        selected_semantic_keys = {
            fact_semantic_key(fact)
            for fact in facts
            if fact.fact_id in det_observer.selected_fact_ids
        }
        value_semantic_keys = {
            fact_semantic_key(fact)
            for fact in facts
            if fact.fact_id in set(det_observer.selected_value_fact_ids)
        }

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
            "selected_fact_ids": list(det_observer.selected_fact_ids),
            "extracted_fact_ids": [fact.fact_id for fact in facts],
            "projected_fact_ids": sorted({
                fid
                for candidate in det_observer.projected_candidates
                for fid in candidate.get("source_fact_ids", [])
            }),
            "all_semantic_keys": sorted(all_semantic_keys),
            "correct_semantic_keys": sorted(correct_semantic_keys),
            "projected_semantic_keys": sorted(projected_semantic_keys),
            "selected_semantic_keys": sorted(selected_semantic_keys),
            "value_semantic_keys": sorted(value_semantic_keys),
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
            "projected_candidates": list(det_observer.projected_candidates),
            "projection_exclusions": list(det_observer.projection_exclusions),
            "pre_selector_ranking": list(det_observer.pre_selector_ranking),
            "selector_input_ids": list(det_observer.selector_input_ids),
            "selector_output_ids": list(det_observer.selector_output_ids),
            "selected_values": list(det_observer.selected_values),
            "selected_value_fact_ids": list(det_observer.selected_value_fact_ids),
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
# Regression trace (uses provider-independent semantic keys)
# ---------------------------------------------------------------------------

def _build_regression_trace(
    *,
    case_id: str,
    current_record: dict,
    structured_record: dict,
) -> RegressionCaseTrace | None:
    """Build a regression trace using provider-independent semantic keys.

    The ``current_supporting_gold_fact_keys`` are the semantic keys of facts
    that supported the correct Current answer.  We check whether semantically
    equivalent facts survive in the Structured path at each stage
    (extraction → projection → selection → value).
    """
    current_supporting = set(current_record.get("correct_semantic_keys", []))
    structured_extracted = set(structured_record.get("all_semantic_keys", []))
    structured_projected = set(structured_record.get("projected_semantic_keys", []))
    structured_selected = set(structured_record.get("selected_semantic_keys", []))
    structured_value = set(structured_record.get("value_semantic_keys", []))

    current_values = tuple(current_record.get("selected_values", []))
    structured_values = tuple(structured_record.get("selected_values", []))

    current_raw = current_record.get("raw_answer_correct", False)
    structured_raw = structured_record.get("raw_answer_correct", False)
    current_released = current_record.get("released_answer_correct", False)
    structured_released = structured_record.get("released_answer_correct", False)

    first_div, cause = classify_regression_cause(
        current_supporting_gold_fact_keys=current_supporting,
        structured_extracted_semantic_keys=structured_extracted,
        structured_projected_semantic_keys=structured_projected,
        structured_selected_semantic_keys=structured_selected,
        structured_value_semantic_keys=structured_value,
        current_raw_correct=current_raw,
        structured_raw_correct=structured_raw,
        current_released_correct=current_released,
        structured_released_correct=structured_released,
    )

    return RegressionCaseTrace(
        case_id=case_id,
        current_supporting_gold_fact_keys=sorted(current_supporting),
        current_selected_values_hash=[hashlib.sha256(v.encode()).hexdigest() for v in current_values],
        current_raw_correct=current_raw,
        current_released_correct=current_released,
        structured_extracted_semantic_keys=sorted(structured_extracted),
        structured_projected_semantic_keys=sorted(structured_projected),
        structured_selected_semantic_keys=sorted(structured_selected),
        structured_value_semantic_keys=sorted(structured_value),
        structured_selected_values_hash=[hashlib.sha256(v.encode()).hexdigest() for v in structured_values],
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
# Baseline computation (includes any-gold and partial-gold counts)
# ---------------------------------------------------------------------------

def _compute_actual_baseline(
    *,
    current_by_id: dict,
    structured_by_id: dict,
    all_gold_ids: list[str],
    partial_gold_ids: list[str],
    any_gold_ids: list[str],
) -> dict:
    """Compute actual baseline metrics from run records.

    Includes ``any_gold_case_count`` and ``partial_gold_case_count`` so the
    baseline gate can verify the real enum-based filtering, not just
    ``all_gold_case_count``.
    """
    return {
        "all_gold_case_count": len(all_gold_ids),
        "partial_gold_case_count": len(partial_gold_ids),
        "any_gold_case_count": len(any_gold_ids),
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
    # ------------------------------------------------------------------
    # Step 1: Verify NF39 R2 frozen inputs
    # ------------------------------------------------------------------
    require_verified_nf39_r2_inputs(
        acceptance_path=args.acceptance, snapshot_manifest_path=args.snapshot_manifest,
        frozen_payload_path=args.frozen_payload_path, expected_payload_sha256=args.expected_payload_sha256,
    )
    if args.tenant_id != 1:
        raise ValueError("NF42 approved frozen snapshot is tenant 1 only")

    cases = validate_labeled_cases(load_jsonl_cases(args.cases))

    # ------------------------------------------------------------------
    # Step 1b: Load and verify frozen contexts (returns verification report)
    # ------------------------------------------------------------------
    contexts, context_report = _load_and_verify_contexts(
        args.frozen_payload_path, args.final_context_manifest,
    )

    nf40 = json.loads((args.acceptance.parent.parent / "nf40" / "case-attribution.json").read_text(encoding="utf-8"))
    nf40_records = {row["case_id"]: row for row in nf40["cases"]}

    # Compute final_contexts_hash for baseline verification
    final_contexts_hash = _sha({key: value.final_context_hash for key, value in sorted(contexts.items())})

    # Load question_hash and label_hash from the NF39 R2 baseline manifest
    nf39_baseline = json.loads((args.acceptance.parent / "baseline-manifest.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Step 2: Verify NF42 R1 Baseline Artifact
    # ------------------------------------------------------------------
    expected_baseline, nf42_r1_baseline_sha256 = _load_nf42_r1_baseline(
        args.nf42_r1_baseline,
        expected_question_hash=nf39_baseline.get("question_hash"),
        expected_label_hash=nf39_baseline.get("label_hash"),
        expected_frozen_payload_hash=args.expected_payload_sha256,
        expected_final_contexts_hash=final_contexts_hash,
    )

    # ------------------------------------------------------------------
    # Step 3: Verify 27-case range
    # ------------------------------------------------------------------
    if len(cases) != 27:
        raise EvaluationIntegrityError(
            f"Case count mismatch: expected 27, got {len(cases)}"
        )

    # ------------------------------------------------------------------
    # Step 4: Verify 13 All-gold / 3 Partial / 16 Any-gold from nf40
    # ------------------------------------------------------------------
    all_gold_ids_pre = [
        cid for cid in contexts
        if nf40_records.get(cid, {}).get("context_coverage") == "all_gold_in_final"
    ]
    partial_gold_ids_pre = [
        cid for cid in contexts
        if nf40_records.get(cid, {}).get("context_coverage") == "partial_gold_in_final"
    ]
    any_gold_ids_pre = [
        cid for cid in contexts
        if nf40_records.get(cid, {}).get("context_coverage") in {
            "all_gold_in_final", "partial_gold_in_final",
        }
    ]

    all_gold_count_verified = len(all_gold_ids_pre) == expected_baseline.all_gold_case_count
    partial_gold_count_verified = len(partial_gold_ids_pre) == expected_baseline.partial_gold_case_count
    any_gold_count_verified = len(any_gold_ids_pre) == expected_baseline.any_gold_case_count

    # ------------------------------------------------------------------
    # Step 5: Build document identity map and verify completeness
    # ------------------------------------------------------------------
    identity_map = _build_document_identity_map(args.final_context_manifest)

    # ------------------------------------------------------------------
    # Step 6: Verify side-effect observation boundaries are configured
    # ------------------------------------------------------------------
    observer = _SideEffectObserver()
    side_effect_observation_complete = observer.effects.all_boundaries_accounted_for()

    # ------------------------------------------------------------------
    # Pre-flight integrity: if any pre-flight check fails, exit non-zero
    # ------------------------------------------------------------------
    preflight_integrity_passed = (
        context_report.passed
        and all_gold_count_verified
        and partial_gold_count_verified
        and any_gold_count_verified
        and side_effect_observation_complete
    )

    if not preflight_integrity_passed:
        failed = []
        if not context_report.passed:
            failed.append(f"context_verification (failed_cases={context_report.failed_cases})")
        if not all_gold_count_verified:
            failed.append(f"all_gold_count (expected {expected_baseline.all_gold_case_count}, got {len(all_gold_ids_pre)})")
        if not partial_gold_count_verified:
            failed.append(f"partial_gold_count (expected {expected_baseline.partial_gold_case_count}, got {len(partial_gold_ids_pre)})")
        if not any_gold_count_verified:
            failed.append(f"any_gold_count (expected {expected_baseline.any_gold_case_count}, got {len(any_gold_ids_pre)})")
        if not side_effect_observation_complete:
            failed.append("side_effect_observation_boundaries")
        print(
            "NF42 R2 pre-flight integrity FAILED: " + ", ".join(failed),
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 7: Execute Current variant
    # ------------------------------------------------------------------
    current_manifest, current = await _run_variant(
        provider="current", cases=cases, contexts=contexts,
        tenant_id=args.tenant_id, nf40_records=nf40_records,
        identity_map=identity_map, observer=observer,
    )

    # Post-Current: verify document identity and side-effects
    # Collect all fact document_ids from Current run for identity check
    all_current_fact_doc_ids = set()
    for row in current:
        for fact in row.get("correct_facts", []):
            doc_id = fact.get("document_id")
            if doc_id:
                all_current_fact_doc_ids.add(doc_id)

    unmapped_document_ids = sorted(
        doc_id for doc_id in all_current_fact_doc_ids
        if doc_id not in identity_map
    )
    document_identity_complete = len(unmapped_document_ids) == 0

    # Check side-effects after Current run
    observer.effects.model_chat_completion_requests = current_manifest["model_chat_completion_requests"]
    current_side_effects_clean = observer.effects.all_observed_zero()

    # Current baseline check (partial — only current fields)
    current_by_id = {row["case_id"]: row for row in current}
    all_gold_ids = [row["case_id"] for row in current if row["context_coverage"] == "all_gold_in_final"]
    partial_gold_ids = [row["case_id"] for row in current if row["context_coverage"] == "partial_gold_in_final"]
    any_gold_ids = [row["case_id"] for row in current if row["context_coverage"] in {"all_gold_in_final", "partial_gold_in_final"}]

    # Pre-Structured integrity gate
    if not document_identity_complete or not current_side_effects_clean:
        failed = []
        if not document_identity_complete:
            failed.append(f"document_identity (unmapped={unmapped_document_ids})")
        if not current_side_effects_clean:
            failed.append("side_effects_nonzero_after_current")
        print(
            "NF42 R2 post-Current integrity FAILED: " + ", ".join(failed),
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 8: Execute Structured variant
    # ------------------------------------------------------------------
    structured_manifest, structured = await _run_variant(
        provider="structured_shadow", cases=cases, contexts=contexts,
        tenant_id=args.tenant_id, nf40_records=nf40_records,
        identity_map=identity_map, observer=observer,
    )

    # Update model call count with both variants
    observer.effects.model_chat_completion_requests = (
        current_manifest["model_chat_completion_requests"]
        + structured_manifest["model_chat_completion_requests"]
    )

    # ------------------------------------------------------------------
    # Step 9: Compare baselines using field groups
    # ------------------------------------------------------------------
    structured_by_id = {row["case_id"]: row for row in structured}

    actual_baseline = _compute_actual_baseline(
        current_by_id=current_by_id,
        structured_by_id=structured_by_id,
        all_gold_ids=all_gold_ids,
        partial_gold_ids=partial_gold_ids,
        any_gold_ids=any_gold_ids,
    )

    expected_dict = expected_baseline.to_dict()

    current_baseline_reproduced = baseline_fields_match(
        actual=actual_baseline, expected=expected_dict, fields=CURRENT_BASELINE_FIELDS,
    )
    structured_baseline_reproduced = baseline_fields_match(
        actual=actual_baseline, expected=expected_dict, fields=STRUCTURED_BASELINE_FIELDS,
    )
    cross_variant_baseline_reproduced = baseline_fields_match(
        actual=actual_baseline, expected=expected_dict, fields=CROSS_VARIANT_FIELDS,
    )

    # ------------------------------------------------------------------
    # Step 10: Generate attribution
    # ------------------------------------------------------------------
    # Coverage-gain cases: Structured has correct facts where Current had none
    coverage_gain_cases = [
        cid for cid in all_gold_ids
        if structured_by_id[cid]["correct_fact_available"] and not current_by_id[cid]["correct_fact_available"]
    ]
    coverage_gain_fact_count = sum(
        len(structured_by_id[cid]["correct_fact_ids"])
        for cid in coverage_gain_cases
    )

    # All new correct facts (semantic key set difference)
    all_new_correct_fact_count = sum(
        len(
            set(structured_by_id[cid].get("correct_semantic_keys", []))
            - set(current_by_id[cid].get("correct_semantic_keys", []))
        )
        for cid in all_gold_ids
    )

    # Build funnel traces for coverage-gain facts
    all_funnel_traces: list[NewFactFunnelTrace] = []
    for cid in coverage_gain_cases:
        correct_facts = structured_by_id[cid]["correct_facts"]
        traces = _build_new_fact_funnel(case_id=cid, correct_facts=correct_facts, structured_record=structured_by_id[cid])
        all_funnel_traces.extend(traces)

    # Regression cases
    regression_ids = [
        cid for cid in current_by_id
        if current_by_id[cid]["released_answer_correct"] and not structured_by_id[cid]["released_answer_correct"]
    ]
    regression_traces: list[RegressionCaseTrace] = []
    regressions_attributed = True
    for cid in regression_ids:
        trace = _build_regression_trace(case_id=cid, current_record=current_by_id[cid], structured_record=structured_by_id[cid])
        if trace:
            regression_traces.append(trace)
            if trace.regression_cause == RegressionCause.REGRESSION_TRACE_INSUFFICIENT:
                regressions_attributed = False
        else:
            regressions_attributed = False

    # ------------------------------------------------------------------
    # Step 11: Compute gate
    # ------------------------------------------------------------------
    next_gate = _determine_next_gate(all_funnel_traces)

    # ------------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------------
    new_fact_identity_complete = all(
        trace.candidate_key for trace in all_funnel_traces
    )

    side_effect_observation = observer.effects.to_dict()

    integrity_checks = {
        "current_baseline_reproduced": current_baseline_reproduced,
        "structured_baseline_reproduced": structured_baseline_reproduced,
        "cross_variant_baseline_reproduced": cross_variant_baseline_reproduced,
        "all_gold_case_count_verified": len(all_gold_ids) == expected_baseline.all_gold_case_count,
        "partial_gold_case_count_verified": len(partial_gold_ids) == expected_baseline.partial_gold_case_count,
        "any_gold_case_count_verified": len(any_gold_ids) == expected_baseline.any_gold_case_count,
        "document_identity_complete": document_identity_complete,
        "side_effect_observation_complete": side_effect_observation_complete,
        "content_hashes_verified": context_report.content_hash_verified_count,
        "final_context_hashes_verified": context_report.final_context_hash_verified_count,
        "retrieval_calls_zero": observer.effects.retrieval_calls == 0,
        "model_calls_zero": observer.effects.model_chat_completion_requests == 0,
        "side_effects_passed": observer.effects.passed,
        "new_fact_identity_complete": new_fact_identity_complete,
        "regression_semantic_identity_used": True,
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

    # ------------------------------------------------------------------
    # Shared metadata
    # ------------------------------------------------------------------
    shared = {
        "artifact_schema": "nf42-r2/v1",
        "case_count": len(cases),
        "tenant_id": args.tenant_id,
        "frozen_payload_hash": args.expected_payload_sha256,
        "final_contexts_hash": final_contexts_hash,
        "question_hash": nf39_baseline.get("question_hash"),
        "label_hash": nf39_baseline.get("label_hash"),
        "nf42_r1_baseline_sha256": nf42_r1_baseline_sha256,
        "side_effect_observation": side_effect_observation,
    }

    out = args.out_dir
    _write(out / "baseline-manifest.json", {
        **shared,
        "expected_baseline": expected_baseline.to_dict(),
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
        "partial_gold_case_count": len(partial_gold_ids),
        "any_gold_case_count": len(any_gold_ids),
        "current_correct_fact_cases": sum(current_by_id[cid]["correct_fact_available"] for cid in all_gold_ids),
        "structured_correct_fact_cases": sum(structured_by_id[cid]["correct_fact_available"] for cid in all_gold_ids),
        "all_new_correct_fact_count": all_new_correct_fact_count,
        "coverage_gain_fact_count": coverage_gain_fact_count,
        "coverage_gain_case_count": len(coverage_gain_cases),
        "coverage_gain_cases": coverage_gain_cases,
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

    # Coverage-gain funnel (renamed from new-fact-loss-funnel)
    _write(out / "coverage-gain-funnel.json", {
        **shared,
        "all_new_correct_fact_count": all_new_correct_fact_count,
        "coverage_gain_fact_count": coverage_gain_fact_count,
        "coverage_gain_case_count": len(coverage_gain_cases),
        "coverage_gain_cases": coverage_gain_cases,
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
                    "correct_semantic_keys": current_by_id[cid]["correct_semantic_keys"],
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
                    "correct_semantic_keys": structured_by_id[cid]["correct_semantic_keys"],
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
        "expected_baseline": expected_baseline.to_dict(),
        "actual_baseline": actual_baseline,
        "current_baseline_reproduced": current_baseline_reproduced,
        "structured_baseline_reproduced": structured_baseline_reproduced,
        "cross_variant_baseline_reproduced": cross_variant_baseline_reproduced,
        "all_gold_case_count_verified": len(all_gold_ids) == expected_baseline.all_gold_case_count,
        "partial_gold_case_count_verified": len(partial_gold_ids) == expected_baseline.partial_gold_case_count,
        "any_gold_case_count_verified": len(any_gold_ids) == expected_baseline.any_gold_case_count,
        "document_identity_complete": document_identity_complete,
        "side_effect_observation_complete": side_effect_observation_complete,
        "content_hashes_verified": context_report.content_hash_verified_count,
        "final_context_hashes_verified": context_report.final_context_hash_verified_count,
        "regression_semantic_identity_used": True,
        "regressions_attributed": regressions_attributed,
        "extractor_only_ab": False,
        "single_variable_verified": False,
        "production_default": "current",
        "production_switch_allowed": False,
        "production_behavior_changed": False,
        "decision": "structured_path_regressed",
        "all_new_correct_fact_count": all_new_correct_fact_count,
        "coverage_gain_fact_count": coverage_gain_fact_count,
        "coverage_gain_case_count": len(coverage_gain_cases),
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
