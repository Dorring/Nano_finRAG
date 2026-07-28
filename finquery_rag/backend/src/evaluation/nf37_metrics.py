"""Ranking-only metrics for frozen NF37 candidate pools."""
from __future__ import annotations
from typing import Iterable
from .evaluation import EvaluationCase

def candidate_to_source(candidate: dict) -> dict:
    return {
        "chunk_id": candidate.get("candidate_id") or candidate.get("evidence_id") or candidate.get("doc_id"),
        "filename": candidate.get("filename") or candidate.get("document_id") or candidate.get("doc_name"),
        "page": candidate.get("page"),
    }

def case_hit_rate_at_k(cases: Iterable[EvaluationCase], rankings: dict[str, list[dict]], k: int) -> float:
    eligible = [case for case in cases if case.expected_sources]
    if not eligible:
        return 1.0
    hits = sum(any(expected.matches(candidate_to_source(candidate)) for expected in case.expected_sources for candidate in rankings.get(case.case_id, [])[:k]) for case in eligible)
    return hits / len(eligible)

def source_recall_at_k(cases: Iterable[EvaluationCase], rankings: dict[str, list[dict]], k: int) -> float:
    matched = total = 0
    for case in cases:
        for expected in case.expected_sources:
            total += 1
            matched += any(expected.matches(candidate_to_source(candidate)) for candidate in rankings.get(case.case_id, [])[:k])
    return matched / total if total else 1.0

def all_source_coverage_at_k(cases: Iterable[EvaluationCase], rankings: dict[str, list[dict]], k: int) -> float:
    eligible = [case for case in cases if len(case.expected_sources) > 1]
    if not eligible:
        return 1.0
    covered = sum(all(any(expected.matches(candidate_to_source(candidate)) for candidate in rankings.get(case.case_id, [])[:k]) for expected in case.expected_sources) for case in eligible)
    return covered / len(eligible)

def ranking_metrics(cases: Iterable[EvaluationCase], rankings: dict[str, list[dict]], ks: tuple[int, ...] = (5, 8, 20)) -> dict[str, float]:
    case_list = list(cases)
    report = {}
    for k in ks:
        report[f"case_hit_rate_at_{k}"] = case_hit_rate_at_k(case_list, rankings, k)
        report[f"source_recall_at_{k}"] = source_recall_at_k(case_list, rankings, k)
        report[f"all_source_coverage_at_{k}"] = all_source_coverage_at_k(case_list, rankings, k)
    reciprocal = []
    for case in case_list:
        if not case.expected_sources:
            continue
        rank = next((index for index, candidate in enumerate(rankings.get(case.case_id, []), 1) if any(expected.matches(candidate_to_source(candidate)) for expected in case.expected_sources)), None)
        reciprocal.append(1 / rank if rank else 0)
    report["mrr"] = sum(reciprocal) / len(reciprocal) if reciprocal else 1.0
    return report
