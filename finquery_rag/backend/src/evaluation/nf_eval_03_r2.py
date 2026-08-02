"""Pure evaluation helpers for NF-EVAL-03 R2 candidate lineage.

This module is deliberately independent from the production retrieval path.
It validates the candidate lists captured by the evaluation observer and
provides deterministic coverage/failure classifications for artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class CandidateLineageError(ValueError):
    """Raised when a captured candidate lineage cannot be trusted."""


@dataclass(frozen=True)
class StageCandidateIdentity:
    candidate_key: str
    document_id: str
    content_hash: str
    parent_candidate_key: str | None = None
    evidence_id: str | None = None
    page: int | None = None


class BaselineFailureStage(str, Enum):
    GOLD_IDENTITY_INVALID = "gold_identity_invalid"
    GOLD_NOT_IN_RRF_POOL = "gold_not_in_rrf_pool"
    GOLD_DROPPED_BEFORE_RERANKER = "gold_dropped_before_reranker"
    GOLD_DROPPED_BY_RERANKER = "gold_dropped_by_reranker"
    GOLD_DROPPED_BY_FINAL_SELECTOR = "gold_dropped_by_final_selector"
    GOLD_PARTIAL_IN_FINAL = "gold_partial_in_final"
    FACT_NOT_EXTRACTED = "fact_not_extracted"
    WRONG_FACT_SELECTED = "wrong_fact_selected"
    CALCULATION_ROUTE_MISSED = "calculation_route_missed"
    CALCULATION_WRONG = "calculation_wrong"
    LLM_ANSWER_WRONG = "llm_answer_wrong"
    UNIT_SCALE_PERIOD_WRONG = "unit_scale_period_wrong"
    CITATION_WRONG = "citation_wrong"
    VALIDATOR_FALSE_REJECT = "validator_false_reject"
    CORRECT = "correct"


def canonical_stage_identity(candidate: Mapping[str, Any]) -> StageCandidateIdentity | None:
    """Return the identity carried by a stage candidate, or ``None``."""

    key = str(candidate.get("candidate_key") or "").strip()
    document = str(
        candidate.get("canonical_document_id")
        or candidate.get("document_id")
        or ""
    ).strip()
    content_hash = str(candidate.get("content_hash") or "").strip()
    if not key or not document or not content_hash:
        return None
    page = candidate.get("page")
    try:
        page = int(page) if page is not None else None
    except (TypeError, ValueError):
        page = None
    parent = candidate.get("parent_candidate_key") or candidate.get("parent_id")
    return StageCandidateIdentity(
        candidate_key=key,
        document_id=document,
        content_hash=content_hash,
        parent_candidate_key=str(parent) if parent else None,
        evidence_id=(
            str(candidate.get("evidence_id"))
            if candidate.get("evidence_id")
            else None
        ),
        page=page,
    )


def ordered_candidate_keys(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(candidate.get("candidate_key") or "") for candidate in candidates]


def audit_stage(stage: str, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ordered_candidate_keys(candidates)
    missing = sum(
        canonical_stage_identity(candidate) is None for candidate in candidates
    )
    duplicate = len(keys) - len({key for key in keys if key})
    return {
        "stage": stage,
        "ordered_candidate_keys": keys,
        "candidate_count": len(candidates),
        "duplicate_candidate_count": duplicate,
        "missing_identity_count": missing,
    }


def _difference(left: Sequence[str], right: Sequence[str]) -> list[str]:
    right_set = set(right)
    return [key for key in left if key not in right_set]


def infer_reranker_input_source(
    *,
    rrf_keys: Sequence[str],
    input_keys: Sequence[str],
    input_limit: int | None,
) -> str:
    """Infer the actual source from captured ordered keys."""

    if list(input_keys) == list(rrf_keys):
        return "rrf_all"
    if input_limit is not None and list(input_keys) == list(rrf_keys)[:input_limit]:
        return "rrf_top_n"
    if set(input_keys).issubset(set(rrf_keys)):
        return "rrf_filtered"
    return "normalized_union"


def validate_candidate_lineage(
    *,
    rrf_candidates: Sequence[Mapping[str, Any]],
    reranker_input: Sequence[Mapping[str, Any]],
    reranker_output: Sequence[Mapping[str, Any]],
    final_candidates: Sequence[Mapping[str, Any]],
    reranker_input_source: str,
    reranker_input_limit: int | None,
) -> dict[str, Any]:
    """Validate stage conservation without changing any candidate list."""

    rrf_keys = ordered_candidate_keys(rrf_candidates)
    input_keys = ordered_candidate_keys(reranker_input)
    output_keys = ordered_candidate_keys(reranker_output)
    final_keys = ordered_candidate_keys(final_candidates)
    anomalies: list[dict[str, Any]] = []

    def add(keys: Sequence[str], source: str, target: str, reason: str) -> None:
        for key in keys:
            anomalies.append(
                {
                    "candidate_key": key,
                    "source_stage": source,
                    "target_stage": target,
                    "reason": reason,
                }
            )

    if reranker_input_source == "rrf_top_n":
        expected = rrf_keys[: reranker_input_limit or 0]
        if input_keys != expected:
            add(
                _difference(input_keys, expected) or _difference(expected, input_keys),
                "rrf",
                "reranker_input",
                "reranker_input_is_not_declared_rrf_cutoff",
            )
    elif reranker_input_source in {"rrf_all", "rrf_filtered"}:
        add(
            _difference(input_keys, rrf_keys),
            "rrf",
            "reranker_input",
            "reranker_input_not_in_rrf",
        )

    add(
        _difference(output_keys, input_keys),
        "reranker_input",
        "reranker_output",
        "reranker_output_not_in_input",
    )
    add(
        _difference(final_keys, output_keys),
        "reranker_output",
        "final",
        "final_candidate_not_in_reranker_output",
    )

    missing_identity = sum(
        canonical_stage_identity(candidate) is None
        for candidates in (
            rrf_candidates,
            reranker_input,
            reranker_output,
            final_candidates,
        )
        for candidate in candidates
    )
    duplicate_count = sum(
        audit_stage(stage, candidates)["duplicate_candidate_count"]
        for stage, candidates in (
            ("rrf", rrf_candidates),
            ("reranker_input", reranker_input),
            ("reranker_output", reranker_output),
            ("final", final_candidates),
        )
    )
    return {
        "reranker_input_not_in_rrf_count": sum(
            item["reason"] == "reranker_input_not_in_rrf" for item in anomalies
        ),
        "reranker_output_not_in_input_count": sum(
            item["reason"] == "reranker_output_not_in_input" for item in anomalies
        ),
        "final_not_in_reranker_output_count": sum(
            item["reason"] == "final_candidate_not_in_reranker_output"
            for item in anomalies
        ),
        "candidate_identity_changed_between_stages": 0,
        "unexpected_candidate_injection_count": len(
            [
                item
                for item in anomalies
                if item["reason"]
                in {"reranker_input_not_in_rrf", "reranker_output_not_in_input"}
            ]
        ),
        "unexplained_candidate_drop_count": 0,
        "missing_identity_count": missing_identity,
        "duplicate_candidate_count": duplicate_count,
        "lineage_integrity_passed": not anomalies
        and missing_identity == 0
        and duplicate_count == 0,
        "anomalies": anomalies,
    }


def classify_final_coverage(
    *,
    expected_sources: Sequence[Mapping[str, Any]],
    final_candidates: Sequence[Mapping[str, Any]],
    source_matches,
) -> str:
    matched = sum(
        any(source_matches(source, candidate) for candidate in final_candidates)
        for source in expected_sources
    )
    if not expected_sources:
        return "no_answer_case"
    if matched == 0:
        return "no_gold_in_final"
    if matched < len(expected_sources):
        return "partial_gold_in_final"
    return "all_gold_in_final"


def first_failure_stage(
    *,
    gold_identity_valid: bool,
    gold_in_rrf: bool,
    gold_in_reranker_input: bool,
    gold_in_reranker_output: bool,
    gold_in_final: bool,
    final_partial: bool,
    raw_contract_correct: bool,
    released_contract_correct: bool,
    raw_value_correct: bool,
    released_value_correct: bool,
    raw_unit_correct: bool,
    released_unit_correct: bool,
    raw_period_correct: bool,
    released_period_correct: bool,
    citation_full_recall: bool,
    execution_mode: str,
    requires_calculation: bool,
    calculation_route_hit: bool,
) -> BaselineFailureStage:
    if not gold_identity_valid:
        return BaselineFailureStage.GOLD_IDENTITY_INVALID
    if not gold_in_rrf:
        return BaselineFailureStage.GOLD_NOT_IN_RRF_POOL
    if not gold_in_reranker_input:
        return BaselineFailureStage.GOLD_DROPPED_BEFORE_RERANKER
    if not gold_in_reranker_output:
        return BaselineFailureStage.GOLD_DROPPED_BY_RERANKER
    if not gold_in_final:
        return BaselineFailureStage.GOLD_DROPPED_BY_FINAL_SELECTOR
    if final_partial:
        return BaselineFailureStage.GOLD_PARTIAL_IN_FINAL
    if requires_calculation and not calculation_route_hit:
        return BaselineFailureStage.CALCULATION_ROUTE_MISSED
    if not raw_contract_correct and raw_value_correct and (
        not raw_unit_correct or not raw_period_correct
    ):
        return BaselineFailureStage.UNIT_SCALE_PERIOD_WRONG
    if raw_contract_correct and not released_contract_correct:
        return BaselineFailureStage.VALIDATOR_FALSE_REJECT
    if not released_value_correct:
        return (
            BaselineFailureStage.LLM_ANSWER_WRONG
            if execution_mode == "llm_generation"
            else BaselineFailureStage.FACT_NOT_EXTRACTED
        )
    if not citation_full_recall:
        return BaselineFailureStage.CITATION_WRONG
    return BaselineFailureStage.CORRECT
