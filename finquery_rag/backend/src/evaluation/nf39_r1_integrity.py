"""Evaluation-only NF39 R1 same-K attribution helpers."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Iterable

from src.evaluation.evaluation import EvaluationCase, ExpectedSource
from src.evaluation.nf39_attribution import (
    EvaluationIntegrityError, canonical_candidate_key, to_stage_candidate,
)


class RankTransition(str, Enum):
    NOT_IN_RRF_40 = "not_in_rrf_40"
    TRUNCATED_BEFORE_RERANKER = "truncated_before_reranker"
    PROMOTED_INTO_TOP5 = "promoted_into_top5"
    DEMOTED_OUT_OF_TOP5 = "demoted_out_of_top5"
    STAYED_IN_TOP5 = "stayed_in_top5"
    PROMOTED_BUT_OUTSIDE_TOP5 = "promoted_but_outside_top5"
    DEMOTED_BUT_ALREADY_OUTSIDE_TOP5 = "demoted_but_already_outside_top5"
    UNCHANGED_OUTSIDE_TOP5 = "unchanged_outside_top5"
    DROPPED_BY_FINAL_SELECTOR = "dropped_by_final_selector"
    PRESENT_IN_FINAL = "present_in_final"


@dataclass(frozen=True)
class SourceStageRanks:
    case_id: str
    expected_source_id: str
    rrf_rank: int | None
    reranker_input_rank: int | None
    reranker_rank: int | None
    final_rank: int | None
    match_granularity: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def retrieval_eligible_cases(cases: Iterable[EvaluationCase]) -> list[EvaluationCase]:
    return [c for c in cases if not c.expected_no_answer and c.expected_sources]


def denominator_report(cases: Iterable[EvaluationCase]) -> dict[str, int]:
    all_cases = list(cases)
    eligible = retrieval_eligible_cases(all_cases)
    return {
        "retrieval_case_count": len(eligible),
        "expected_source_count": sum(len(c.expected_sources) for c in eligible),
        "multi_source_case_count": sum(len(c.expected_sources) > 1 for c in eligible),
        "no_answer_case_count": sum(c.expected_no_answer for c in all_cases),
    }


def _source_id(source: ExpectedSource) -> str:
    if source.chunk_id:
        return "evidence:" + source.chunk_id
    if source.filename and source.page is not None:
        return f"document_page:{source.filename}:p{source.page}"
    return "document:" + (source.filename or "unlabeled")


def _granularity(source: ExpectedSource) -> str:
    if source.chunk_id:
        return "evidence_id"
    return "document_page" if source.page is not None else "document_only"


def deduplicate_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result, seen = [], set()
    for row in rows:
        key = canonical_candidate_key(to_stage_candidate(row))
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _match_rank(source: ExpectedSource, rows: Iterable[dict[str, Any]], k: int | None = None) -> int | None:
    for rank, row in enumerate(deduplicate_candidates(rows)[:k], 1):
        candidate = {
            "chunk_id": row.get("evidence_id") or row.get("candidate_id"),
            "filename": row.get("document_id"),
            "page": row.get("page"),
        }
        if source.matches(candidate):
            return rank
    return None


def _matched_count(case: EvaluationCase, rows: Iterable[dict[str, Any]], k: int) -> int:
    return sum(_match_rank(source, rows, k) is not None for source in case.expected_sources)


def stage_metrics_same_k(*, cases: Iterable[EvaluationCase], rankings: dict[str, list[dict[str, Any]]], ks: tuple[int, ...]) -> dict[str, Any]:
    all_cases = list(cases)
    eligible = retrieval_eligible_cases(all_cases)
    denoms = denominator_report(all_cases)
    multis = [case for case in eligible if len(case.expected_sources) > 1]
    output: dict[str, Any] = {"denominators": denoms}
    for k in ks:
        case_hits = source_hits = coverage_hits = 0
        reciprocals = []
        for case in eligible:
            rows = rankings.get(case.case_id, [])
            matched = _matched_count(case, rows, k)
            case_hits += int(matched > 0)
            source_hits += matched
            first = min(
                (_match_rank(source, rows, k) for source in case.expected_sources if _match_rank(source, rows, k) is not None),
                default=None,
            )
            reciprocals.append(1 / first if first else 0.0)
        for case in multis:
            coverage_hits += int(_matched_count(case, rankings.get(case.case_id, []), k) == len(case.expected_sources))
        output.update({
            f"case_hit_rate_at_{k}": case_hits / len(eligible) if eligible else 1.0,
            f"source_recall_at_{k}": source_hits / denoms["expected_source_count"] if denoms["expected_source_count"] else 1.0,
            f"all_source_coverage_at_{k}": coverage_hits / len(multis) if multis else 1.0,
            f"mrr_at_{k}": sum(reciprocals) / len(reciprocals) if reciprocals else 1.0,
            f"case_hit_count_at_{k}": case_hits,
            f"source_hit_count_at_{k}": source_hits,
            f"all_source_covered_case_count_at_{k}": coverage_hits,
        })
    return output


def classify_rank_transition(*, rrf_rank: int | None, reranker_input_top_n: int, reranker_rank: int | None, final_rank: int | None, rrf_top_n: int = 40) -> RankTransition:
    if rrf_rank is None or rrf_rank > rrf_top_n:
        return RankTransition.NOT_IN_RRF_40
    if rrf_rank > reranker_input_top_n:
        return RankTransition.TRUNCATED_BEFORE_RERANKER
    if reranker_rank is None:
        raise EvaluationIntegrityError("Candidate entered reranker but has no reranker rank")
    if reranker_rank <= 5:
        if final_rank is None:
            return RankTransition.DROPPED_BY_FINAL_SELECTOR
        return RankTransition.PROMOTED_INTO_TOP5 if rrf_rank > 5 else RankTransition.STAYED_IN_TOP5
    if rrf_rank <= 5:
        return RankTransition.DEMOTED_OUT_OF_TOP5
    if reranker_rank < rrf_rank:
        return RankTransition.PROMOTED_BUT_OUTSIDE_TOP5
    if reranker_rank > rrf_rank:
        return RankTransition.DEMOTED_BUT_ALREADY_OUTSIDE_TOP5
    return RankTransition.UNCHANGED_OUTSIDE_TOP5


def source_stage_transitions(*, cases: Iterable[EvaluationCase], rrf_rankings: dict[str, list[dict[str, Any]]], reranker_input_rankings: dict[str, list[dict[str, Any]]], reranker_rankings: dict[str, list[dict[str, Any]]], final_rankings: dict[str, list[dict[str, Any]]], reranker_input_top_n: int, rrf_top_n: int = 40) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records, counts = [], {item.value: 0 for item in RankTransition}
    for case in retrieval_eligible_cases(cases):
        for source in case.expected_sources:
            ranks = SourceStageRanks(
                case.case_id, _source_id(source),
                _match_rank(source, rrf_rankings.get(case.case_id, []), rrf_top_n),
                _match_rank(source, reranker_input_rankings.get(case.case_id, []), reranker_input_top_n),
                _match_rank(source, reranker_rankings.get(case.case_id, []), reranker_input_top_n),
                _match_rank(source, final_rankings.get(case.case_id, []), 5),
                _granularity(source),
            )
            transition = classify_rank_transition(
                rrf_rank=ranks.rrf_rank, reranker_input_top_n=reranker_input_top_n,
                reranker_rank=ranks.reranker_rank, final_rank=ranks.final_rank,
                rrf_top_n=rrf_top_n,
            )
            row = ranks.to_dict()
            row.update({"transition": transition.value, "generation_status": "not_evaluated", "validation_status": "not_evaluated"})
            records.append(row)
            counts[transition.value] += 1
    return records, counts


def case_stage_summary(*, cases: Iterable[EvaluationCase], rrf_rankings: dict[str, list[dict[str, Any]]], reranker_rankings: dict[str, list[dict[str, Any]]], final_rankings: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    for case in retrieval_eligible_cases(cases):
        expected = len(case.expected_sources)
        rrf = _matched_count(case, rrf_rankings.get(case.case_id, []), 5)
        reranked = _matched_count(case, reranker_rankings.get(case.case_id, []), 5)
        final = _matched_count(case, final_rankings.get(case.case_id, []), 5)
        output.append({"case_id": case.case_id, "expected_source_count": expected, "rrf_top5_matched_sources": rrf, "reranker_top5_matched_sources": reranked, "final_top5_matched_sources": final, "rrf_all_sources_covered": rrf == expected, "reranker_all_sources_covered": reranked == expected, "final_all_sources_covered": final == expected, "generation_status": "not_evaluated", "validation_status": "not_evaluated"})
    return output


def final_context_manifest(final_rankings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    cases, absent = {}, 0
    for case_id, rows in sorted(final_rankings.items()):
        payload, available = [], True
        for rank, row in enumerate(deduplicate_candidates(rows), 1):
            content_hash = row.get("content_hash")
            if not content_hash:
                content_hash, available = "unavailable", False
            payload.append({"candidate_key": canonical_candidate_key(to_stage_candidate(row)), "content_hash": content_hash, "rank": rank})
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        cases[case_id] = {"final_context_hash": digest, "candidate_count": len(payload), "content_hash_available": available, "generation_status": "not_evaluated", "validation_status": "not_evaluated"}
        absent += int(not available)
    return {"cases": cases, "cases_without_exported_content_hash": absent, "generation_evaluation": "not_evaluated"}


def attach_generation_outcome(*, ranking_manifest: dict[str, Any], generation_artifact: dict[str, Any] | None) -> dict[str, Any]:
    if generation_artifact is None:
        return {"generation_status": "not_evaluated", "validation_status": "not_evaluated", "reason": "no_generation_artifact"}
    fields = {"question_hash", "label_hash", "final_context_hash", "generator_model_identity", "prompt_hash", "generation_config_hash", "validator_config_hash", "calculator_config_hash"}
    mismatches = sorted(field for field in fields if ranking_manifest.get(field) != generation_artifact.get(field))
    if mismatches:
        return {"generation_status": "not_evaluated", "validation_status": "not_evaluated", "reason": "artifact_fingerprint_mismatch", "mismatches": mismatches}
    return {"generation_status": generation_artifact.get("generation_status", "evaluated"), "validation_status": generation_artifact.get("validation_status", "evaluated"), "outcome": generation_artifact.get("outcome")}


def fusion_execution_report(*, cases: Iterable[EvaluationCase], rrf_rankings: dict[str, list[dict[str, Any]]], reranker_rankings: dict[str, list[dict[str, Any]]], fusion_rankings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if fusion_rankings is reranker_rankings:
        raise EvaluationIntegrityError("Fusion rankings must be a distinct ranking object")
    details, set_changed, order_changed, entered, exited = [], 0, 0, 0, 0
    for case in cases:
        def keys(mapping: dict[str, list[dict[str, Any]]]) -> list[str]:
            return [canonical_candidate_key(to_stage_candidate(row)) for row in deduplicate_candidates(mapping.get(case.case_id, [])[:5])]
        rrf, reranked, fused = keys(rrf_rankings), keys(reranker_rankings), keys(fusion_rankings)
        changed_set, changed_order = set(fused) != set(reranked), fused != reranked
        set_changed += int(changed_set)
        order_changed += int(changed_order)
        if not case.expected_no_answer and case.expected_sources:
            base = _matched_count(case, reranker_rankings.get(case.case_id, []), 5)
            variant = _matched_count(case, fusion_rankings.get(case.case_id, []), 5)
            entered, exited = entered + int(variant > base), exited + int(variant < base)
        details.append({"case_id": case.case_id, "rrf_top5_keys": rrf, "reranker_top5_keys": reranked, "fusion_top5_keys": fused, "fusion_changed_top5_set": changed_set, "fusion_changed_top5_order": changed_order})
    return {"fusion_executed": True, "top5_set_changed_case_count": set_changed, "top5_order_changed_case_count": order_changed, "gold_entered_top5_case_count": entered, "gold_exited_top5_case_count": exited, "candidate_order_changed_but_gold_metrics_unchanged": order_changed > 0 and entered == 0 and exited == 0, "cases": details}

