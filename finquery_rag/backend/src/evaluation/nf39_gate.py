"""NF39 Ranking Gate and end-to-end Gate logic.

Implements the three decision points from the NF39 spec:

1. **Trigger threshold** – decide whether to run Rank Fusion at all.
2. **Ranking-only Gate** – decide whether Fusion beats Current on ranking
   metrics without any regression.
3. **End-to-end Gate** – (optional) decide whether Fusion beats Current on
   full generation metrics.
"""
from __future__ import annotations

from typing import Any

from src.evaluation.evaluation import EvaluationCase
from src.evaluation.nf39_attribution import (
    FinalLossStage,
    contains_gold,
    gold_ranks,
)

# ---------------------------------------------------------------------------
# Trigger threshold (Section 十一)
# ---------------------------------------------------------------------------


def should_trigger_fusion(
    *,
    rrf_metrics: dict[str, Any],
    reranker_metrics: dict[str, Any],
    attribution_summary: dict[str, int],
) -> tuple[bool, list[str]]:
    """Return ``(trigger, reasons)`` based on Section 十一 rules.

    Fusion is triggered when **any** of these holds:

    - Reranker Source Recall@5 < RRF Source Recall@5
    - Reranker All-source Coverage@5 < RRF All-source Coverage@5
    - ``demoted_by_reranker`` case count ≥ 2
    """
    reasons: list[str] = []

    rrf_sr = rrf_metrics.get("source_recall_at_5", 0.0)
    rer_sr = reranker_metrics.get("source_recall_at_5", 0.0)
    if rer_sr < rrf_sr:
        reasons.append(
            f"reranker_source_recall_at_5 ({rer_sr:.4f}) < rrf ({rrf_sr:.4f})"
        )

    rrf_cov = rrf_metrics.get("all_source_coverage_at_5", 0.0)
    rer_cov = reranker_metrics.get("all_source_coverage_at_5", 0.0)
    if rer_cov < rrf_cov:
        reasons.append(
            f"reranker_all_source_coverage_at_5 ({rer_cov:.4f}) < rrf ({rrf_cov:.4f})"
        )

    demoted = attribution_summary.get(
        FinalLossStage.DEMOTED_BY_RERANKER.value, 0
    )
    if demoted >= 2:
        reasons.append(
            f"demoted_by_reranker case count ({demoted}) >= 2"
        )

    return (len(reasons) > 0, reasons)


# ---------------------------------------------------------------------------
# Case diff (Current vs Fusion)
# ---------------------------------------------------------------------------


def compute_fusion_case_diff(
    cases: list[EvaluationCase],
    baseline_rankings: dict[str, list[dict[str, Any]]],
    fusion_rankings: dict[str, list[dict[str, Any]]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compare per-case Top-K hit status between Current and Fusion.

    Returns a dict with ``improved``, ``regressed``, ``unchanged``,
    ``gold_source_promoted``, ``gold_source_demoted``, and a ``cases`` list.

    A case is *improved* when Fusion has a gold hit in Top-K but Current
    does not, or when the best gold rank improves.  *Regressed* is the
    reverse.  No-answer cases and cases without expected sources are
    skipped.
    """
    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []
    promoted = 0
    demoted = 0
    case_details: list[dict[str, Any]] = []

    for case in cases:
        if case.expected_no_answer or not case.expected_sources:
            continue

        base_top = baseline_rankings.get(case.case_id, [])[:top_k]
        fusion_top = fusion_rankings.get(case.case_id, [])[:top_k]

        base_hit = contains_gold(base_top, case.expected_sources)
        fusion_hit = contains_gold(fusion_top, case.expected_sources)

        base_ranks = gold_ranks(
            baseline_rankings.get(case.case_id, []), case.expected_sources
        )
        fusion_ranks = gold_ranks(
            fusion_rankings.get(case.case_id, []), case.expected_sources
        )

        base_best = min(base_ranks) if base_ranks else None
        fusion_best = min(fusion_ranks) if fusion_ranks else None

        if fusion_hit and not base_hit:
            improved.append(case.case_id)
        elif base_hit and not fusion_hit:
            regressed.append(case.case_id)
        elif base_hit and fusion_hit:
            if fusion_best is not None and base_best is not None:
                if fusion_best < base_best:
                    improved.append(case.case_id)
                elif fusion_best > base_best:
                    regressed.append(case.case_id)
                else:
                    unchanged.append(case.case_id)
            else:
                unchanged.append(case.case_id)
        else:
            unchanged.append(case.case_id)

        if base_best is not None and fusion_best is not None:
            if fusion_best < base_best:
                promoted += 1
            elif fusion_best > base_best:
                demoted += 1

        case_details.append(
            {
                "case_id": case.case_id,
                "baseline_top5_hit": base_hit,
                "fusion_top5_hit": fusion_hit,
                "baseline_best_gold_rank": base_best,
                "fusion_best_gold_rank": fusion_best,
                "improved": (
                    fusion_hit and not base_hit
                    or (
                        base_hit
                        and fusion_hit
                        and fusion_best is not None
                        and base_best is not None
                        and fusion_best < base_best
                    )
                ),
                "regressed": (
                    base_hit and not fusion_hit
                    or (
                        base_hit
                        and fusion_hit
                        and fusion_best is not None
                        and base_best is not None
                        and fusion_best > base_best
                    )
                ),
            }
        )

    return {
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "unchanged_count": len(unchanged),
        "improved_case_ids": improved,
        "regressed_case_ids": regressed,
        "gold_source_promoted": promoted,
        "gold_source_demoted": demoted,
        "cases": case_details,
    }


# ---------------------------------------------------------------------------
# Ranking-only Gate (Section 十四)
# ---------------------------------------------------------------------------


def evaluate_ranking_gate(
    *,
    baseline_metrics: dict[str, Any],
    fusion_metrics: dict[str, Any],
    case_diff: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate the Ranking-only Gate per Section 十四.

    Requirements (all must hold):

    - Final Case Hit@5 not decreased.
    - Final Source Recall@5 not decreased.
    - Final All-source Coverage@5 not decreased.
    - Final MRR not decreased.
    - Regressed case count == 0.

    And at least one of:

    - Final Case Hit@5 increased by ≥ 1 case.
    - Final Source Recall@5 increased by ≥ 1 source.
    - Final All-source Coverage@5 increased by ≥ 1 case.
    - MRR improved by ≥ 0.02.
    """
    k = top_k

    case_hit_ok = (
        fusion_metrics[f"case_hit_rate_at_{k}"]
        >= baseline_metrics[f"case_hit_rate_at_{k}"]
    )
    source_recall_ok = (
        fusion_metrics[f"source_recall_at_{k}"]
        >= baseline_metrics[f"source_recall_at_{k}"]
    )
    coverage_ok = (
        fusion_metrics[f"all_source_coverage_at_{k}"]
        >= baseline_metrics[f"all_source_coverage_at_{k}"]
    )
    mrr_ok = fusion_metrics["mrr"] >= baseline_metrics["mrr"]
    no_regression = case_diff["regressed_count"] == 0

    no_decline = case_hit_ok and source_recall_ok and coverage_ok and mrr_ok and no_regression

    case_count = baseline_metrics.get("case_count", 1) or 1
    source_count = max(
        baseline_metrics.get("source_recall_at_5", 0) * case_count, 1
    )

    case_hit_gain = (
        fusion_metrics[f"case_hit_rate_at_{k}"]
        - baseline_metrics[f"case_hit_rate_at_{k}"]
    ) * case_count
    source_recall_gain = (
        fusion_metrics[f"source_recall_at_{k}"]
        - baseline_metrics[f"source_recall_at_{k}"]
    ) * source_count
    coverage_gain = (
        fusion_metrics[f"all_source_coverage_at_{k}"]
        - baseline_metrics[f"all_source_coverage_at_{k}"]
    ) * case_count
    mrr_gain = fusion_metrics["mrr"] - baseline_metrics["mrr"]

    at_least_one_gain = (
        case_hit_gain >= 1
        or source_recall_gain >= 1
        or coverage_gain >= 1
        or mrr_gain >= 0.02
    )

    passed = no_decline and at_least_one_gain

    return {
        "passed": passed,
        "no_decline": no_decline,
        "at_least_one_gain": at_least_one_gain,
        "case_hit_ok": case_hit_ok,
        "source_recall_ok": source_recall_ok,
        "coverage_ok": coverage_ok,
        "mrr_ok": mrr_ok,
        "no_regression": no_regression,
        "case_hit_gain": case_hit_gain,
        "source_recall_gain": source_recall_gain,
        "coverage_gain": coverage_gain,
        "mrr_gain": mrr_gain,
        "improved_count": case_diff["improved_count"],
        "regressed_count": case_diff["regressed_count"],
        "gold_source_promoted": case_diff["gold_source_promoted"],
        "gold_source_demoted": case_diff["gold_source_demoted"],
        "baseline": {
            f"case_hit_rate_at_{k}": baseline_metrics[f"case_hit_rate_at_{k}"],
            f"source_recall_at_{k}": baseline_metrics[f"source_recall_at_{k}"],
            f"all_source_coverage_at_{k}": baseline_metrics[
                f"all_source_coverage_at_{k}"
            ],
            "mrr": baseline_metrics["mrr"],
        },
        "fusion": {
            f"case_hit_rate_at_{k}": fusion_metrics[f"case_hit_rate_at_{k}"],
            f"source_recall_at_{k}": fusion_metrics[f"source_recall_at_{k}"],
            f"all_source_coverage_at_{k}": fusion_metrics[
                f"all_source_coverage_at_{k}"
            ],
            "mrr": fusion_metrics["mrr"],
        },
    }


# ---------------------------------------------------------------------------
# End-to-end Gate (Section 十五)
# ---------------------------------------------------------------------------


def evaluate_e2e_gate(
    *,
    baseline_e2e: dict[str, Any],
    fusion_e2e: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the end-to-end Gate per Section 十五.

    Requirements:

    - Golden Pass not decreased.
    - Numeric Accuracy not decreased.
    - Citation Recall not decreased.
    - No-answer Accuracy not decreased.
    - P95 latency increase ≤ 0.5 seconds.

    And at least one of:

    - Golden Pass increased by ≥ 1 case.
    - Citation Recall increased by ≥ 1 source.
    """
    golden_ok = fusion_e2e["golden_pass"] >= baseline_e2e["golden_pass"]
    numeric_ok = fusion_e2e["numeric_accuracy"] >= baseline_e2e["numeric_accuracy"]
    citation_ok = fusion_e2e["citation_recall"] >= baseline_e2e["citation_recall"]
    no_answer_ok = (
        fusion_e2e["no_answer_accuracy"]
        >= baseline_e2e["no_answer_accuracy"]
    )

    p95_increase = fusion_e2e.get("p95_latency_ms", 0) - baseline_e2e.get(
        "p95_latency_ms", 0
    )
    latency_ok = p95_increase <= 500  # 0.5 seconds in ms

    no_decline = (
        golden_ok and numeric_ok and citation_ok and no_answer_ok and latency_ok
    )

    golden_gain = fusion_e2e["golden_pass"] - baseline_e2e["golden_pass"]
    citation_gain = fusion_e2e["citation_recall"] - baseline_e2e["citation_recall"]

    at_least_one_gain = golden_gain >= 1 or citation_gain >= 1

    passed = no_decline and at_least_one_gain

    return {
        "passed": passed,
        "no_decline": no_decline,
        "at_least_one_gain": at_least_one_gain,
        "golden_ok": golden_ok,
        "numeric_ok": numeric_ok,
        "citation_ok": citation_ok,
        "no_answer_ok": no_answer_ok,
        "latency_ok": latency_ok,
        "golden_gain": golden_gain,
        "citation_gain": citation_gain,
        "p95_increase_ms": p95_increase,
        "baseline": baseline_e2e,
        "fusion": fusion_e2e,
    }
