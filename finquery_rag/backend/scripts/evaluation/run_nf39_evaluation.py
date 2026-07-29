"""Run the NF39 RRF-to-Final attribution and rank-preserving fusion evaluation.

This script reads a frozen RRF candidate pool (exported by
``export_nf39_rrf_pool.py``) and:

1. Runs the current production reranker on the frozen pool to obtain
   Reranker Top-20 and Final Top-5 (Variant A: Current).
2. Computes stage metrics at K=5/8/20/40 for RRF, Reranker, and Final.
3. Classifies per-case gold evidence loss into five stages.
4. Computes Final Top-5 redundancy statistics.
5. Checks the Rank Fusion trigger threshold.
6. If triggered, runs equal-weight rank-preserving fusion (Variant B)
   and evaluates the Ranking-only Gate.

The production retrieval pipeline is **never** modified.  Fusion is an
offline experiment only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from src.evaluation.evaluation import load_jsonl_cases  # noqa: E402
from src.evaluation.nf39_attribution import (  # noqa: E402
    CaseAttribution,
    FinalLossStage,
    build_case_attribution,
    compute_redundancy,
    compute_stage_metrics,
)
from src.evaluation.nf39_gate import (  # noqa: E402
    compute_fusion_case_diff,
    evaluate_ranking_gate,
    should_trigger_fusion,
)
from src.retrieval.rank_preserving_fusion import rank_preserving_fusion  # noqa: E402
from src.services.reranker import build_reranker  # noqa: E402

# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def _pool_to_rrf_format(pool_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert RRF pool entries to the reranker input format."""
    return [
        {
            "doc_id": c.get("candidate_id") or c.get("evidence_id") or "",
            "score": float(c.get("rrf_score", c.get("score", 0))),
            "fused_score": float(c.get("rrf_score", c.get("score", 0))),
            "metadata": {
                "doc_name": c.get("document_id", ""),
                "page": c.get("page"),
                "type": c.get("block_type", "text"),
                "parent_id": c.get("parent_id"),
                "table_id": c.get("table_id"),
            },
        }
        for c in pool_candidates
    ]


def _rrf_to_summary(rrf_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert RRF pool entries to the summarize_candidates format."""
    return [
        {
            "rank": rank,
            "evidence_id": c.get("candidate_id") or c.get("evidence_id") or "",
            "document_id": c.get("document_id", ""),
            "page": c.get("page"),
            "block_type": c.get("block_type", "text"),
            "parent_id": c.get("parent_id"),
            "table_id": c.get("table_id"),
            "score": float(c.get("rrf_score", c.get("score", 0))),
            "rrf_score": float(c.get("rrf_score", c.get("score", 0))),
            "reranker_score": None,
        }
        for rank, c in enumerate(rrf_candidates, start=1)
    ]


def _reranker_output_to_summary(
    reranked: list[dict[str, Any]],
    rrf_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert reranker output to summarize_candidates format.

    ``rrf_lookup`` maps evidence_id → RRF pool entry (for rrf_score).
    """
    result = []
    for rank, chunk in enumerate(reranked, start=1):
        doc_id = chunk.get("doc_id", "")
        meta = chunk.get("metadata") or {}
        rrf_entry = rrf_lookup.get(doc_id, {})
        result.append(
            {
                "rank": rank,
                "evidence_id": doc_id,
                "document_id": meta.get("doc_name", ""),
                "page": meta.get("page"),
                "block_type": meta.get("type", "text"),
                "parent_id": meta.get("parent_id"),
                "table_id": meta.get("table_id"),
                "score": float(chunk.get("rerank_score", chunk.get("score", 0))),
                "rrf_score": float(
                    rrf_entry.get("rrf_score", rrf_entry.get("score", 0))
                ),
                "reranker_score": float(
                    chunk.get("rerank_score", chunk.get("score", 0))
                ),
            }
        )
    return result


def _fusion_to_summary(
    fused_candidates: list[dict[str, Any]],
    rrf_lookup: dict[str, dict[str, Any]],
    reranker_rank_map: dict[str, int],
) -> list[dict[str, Any]]:
    """Convert fusion output to summarize_candidates format."""
    result = []
    for rank, chunk in enumerate(fused_candidates, start=1):
        doc_id = (
            chunk.get("candidate_id")
            or chunk.get("evidence_id")
            or chunk.get("doc_id")
            or ""
        )
        meta = chunk.get("metadata") or {}
        rrf_entry = rrf_lookup.get(doc_id, {})
        rer_rank = reranker_rank_map.get(doc_id)
        result.append(
            {
                "rank": rank,
                "evidence_id": doc_id,
                "document_id": chunk.get("document_id", meta.get("doc_name", "")),
                "page": chunk.get("page", meta.get("page")),
                "block_type": chunk.get("block_type", meta.get("type", "text")),
                "parent_id": chunk.get("parent_id", meta.get("parent_id")),
                "table_id": chunk.get("table_id", meta.get("table_id")),
                "score": 0.0,  # fusion uses ranks, not scores
                "rrf_score": float(
                    rrf_entry.get("rrf_score", rrf_entry.get("score", 0))
                ),
                "reranker_score": None if rer_rank is None else float(rer_rank),
            }
        )
    return result


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run NF39 RRF-to-Final attribution evaluation"
    )
    parser.add_argument(
        "--pool-dir",
        required=True,
        help="Directory containing rrf-candidate-pool.json and manifest",
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reranker", default="heuristic")
    parser.add_argument("--reranker-output-top-n", type=int, default=20)
    parser.add_argument("--final-top-k", type=int, default=5)
    args = parser.parse_args()

    pool_dir = Path(args.pool_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation cases...")
    cases = load_jsonl_cases(args.cases)
    print(f"Loaded {len(cases)} cases")

    print("Loading frozen RRF candidate pool...")
    pool_data = json.loads(
        (pool_dir / "rrf-candidate-pool.json").read_text(encoding="utf-8")
    )
    pool_manifest = json.loads(
        (pool_dir / "rrf-candidate-pool-manifest.json").read_text(encoding="utf-8")
    )

    reranker = build_reranker(args.reranker)
    reranker_name = reranker.name if reranker else "noop"

    rrf_rankings: dict[str, list[dict[str, Any]]] = {}
    reranker_rankings: dict[str, list[dict[str, Any]]] = {}
    final_rankings: dict[str, list[dict[str, Any]]] = {}
    fusion_rankings: dict[str, list[dict[str, Any]]] = {}

    print("Running reranker on frozen RRF pool...")
    for case in cases:
        pool_entries = pool_data.get(case.case_id, [])
        if not pool_entries:
            rrf_rankings[case.case_id] = []
            reranker_rankings[case.case_id] = []
            final_rankings[case.case_id] = []
            continue

        # S0 / S1: RRF Top-40 (also the reranker input)
        rrf_summary = _rrf_to_summary(pool_entries)
        rrf_rankings[case.case_id] = rrf_summary

        # Convert to reranker input format
        rrf_for_reranker = _pool_to_rrf_format(pool_entries)
        rrf_lookup = {
            c.get("candidate_id") or c.get("evidence_id"): c
            for c in pool_entries
        }

        # S2: Reranker Top-N (output_top_n = 20)
        reranker_output = reranker.rerank(
            case.question,
            rrf_for_reranker,
            top_k=min(args.reranker_output_top_n, len(rrf_for_reranker)),
        ) if reranker else rrf_for_reranker[:args.reranker_output_top_n]

        reranker_summary = _reranker_output_to_summary(reranker_output, rrf_lookup)
        reranker_rankings[case.case_id] = reranker_summary

        # S4: Final Top-K (baseline / Variant A)
        final_rankings[case.case_id] = reranker_summary[: args.final_top_k]

        # Variant B: Rank-preserving fusion
        reranker_rank_map: dict[str, int] = {}
        for rank, chunk in enumerate(reranker_output, start=1):
            doc_id = chunk.get("doc_id", "")
            if doc_id not in reranker_rank_map:
                reranker_rank_map[doc_id] = rank

        fused = rank_preserving_fusion(
            rrf_candidates=pool_entries,
            reranked_candidates=reranker_output,
            fusion_k=60,
        )
        fusion_summary = _fusion_to_summary(
            fused, rrf_lookup, reranker_rank_map
        )
        fusion_rankings[case.case_id] = fusion_summary[: args.final_top_k]

    # ------------------------------------------------------------------
    # Stage metrics
    # ------------------------------------------------------------------
    print("Computing stage metrics...")
    rrf_ks = (5, 8, 20, 40)
    reranker_ks = (5, 8, 20)
    final_ks = (5,)

    rrf_metrics = compute_stage_metrics(
        cases=cases, rankings=rrf_rankings, ks=rrf_ks
    )
    reranker_metrics = compute_stage_metrics(
        cases=cases, rankings=reranker_rankings, ks=reranker_ks
    )
    final_metrics = compute_stage_metrics(
        cases=cases, rankings=final_rankings, ks=final_ks
    )

    stage_metrics = {
        "rrf": rrf_metrics,
        "reranker": reranker_metrics,
        "final": final_metrics,
    }
    _write_json(out_dir / "stage-metrics.json", stage_metrics)

    # ------------------------------------------------------------------
    # Stage attribution
    # ------------------------------------------------------------------
    print("Classifying final loss stages...")
    attribution_summary = {stage.value: 0 for stage in FinalLossStage}
    case_attributions: list[dict[str, Any]] = []

    for case in cases:
        if case.expected_no_answer or not case.expected_sources:
            attribution = CaseAttribution(
                case_id=case.case_id,
                bucket="no_answer" if case.expected_no_answer else "no_sources",
                loss_stage=FinalLossStage.PASSED,
            )
            attribution_summary[FinalLossStage.PASSED.value] += 1
            case_attributions.append(attribution.to_dict())
            continue

        rrf_top40 = rrf_rankings.get(case.case_id, [])
        reranker_input = rrf_top40  # S1 = S0 (reranker input = all RRF)
        reranker_ranked = reranker_rankings.get(case.case_id, [])
        final_top5 = final_rankings.get(case.case_id, [])

        attribution = build_case_attribution(
            case=case,
            rrf_top40=rrf_top40,
            reranker_input=reranker_input,
            reranker_ranked=reranker_ranked,
            final_top5=final_top5,
            golden_pass=False,  # No end-to-end evaluation in ranking-only mode
        )
        attribution_summary[attribution.loss_stage.value] += 1
        case_attributions.append(attribution.to_dict())

    stage_attribution = {
        "summary": attribution_summary,
        "stage_metrics": stage_metrics,
        "cases": case_attributions,
    }
    _write_json(out_dir / "stage-attribution.json", stage_attribution)

    # ------------------------------------------------------------------
    # Redundancy report
    # ------------------------------------------------------------------
    print("Computing redundancy statistics...")
    redundancy = compute_redundancy(final_rankings, top_k=args.final_top_k)
    _write_json(out_dir / "redundancy-report.json", redundancy.to_dict())

    # ------------------------------------------------------------------
    # Rank Fusion trigger check
    # ------------------------------------------------------------------
    print("Checking Rank Fusion trigger threshold...")
    trigger, trigger_reasons = should_trigger_fusion(
        rrf_metrics=rrf_metrics,
        reranker_metrics=reranker_metrics,
        attribution_summary=attribution_summary,
    )

    print(f"Trigger: {trigger}")
    for reason in trigger_reasons:
        print(f"  - {reason}")

    # ------------------------------------------------------------------
    # Rank Fusion evaluation (if triggered)
    # ------------------------------------------------------------------
    fusion_comparison: dict[str, Any] = {
        "triggered": trigger,
        "trigger_reasons": trigger_reasons,
    }

    if trigger:
        print("Running Rank Fusion evaluation...")
        fusion_metrics = compute_stage_metrics(
            cases=cases,
            rankings=fusion_rankings,
            ks=final_ks,
        )

        case_diff = compute_fusion_case_diff(
            cases=cases,
            baseline_rankings=final_rankings,
            fusion_rankings=fusion_rankings,
            top_k=args.final_top_k,
        )

        gate = evaluate_ranking_gate(
            baseline_metrics=final_metrics,
            fusion_metrics=fusion_metrics,
            case_diff=case_diff,
            top_k=args.final_top_k,
        )

        fusion_comparison.update(
            {
                "baseline_metrics": final_metrics,
                "fusion_metrics": fusion_metrics,
                "case_diff": case_diff,
                "gate": gate,
                "production_switch_allowed": gate["passed"],
                "decision": (
                    "apply_fusion" if gate["passed"] else "retain_current"
                ),
            }
        )

        _write_json(out_dir / "rank-fusion-comparison.json", fusion_comparison)
        _write_json(out_dir / "case-diff-report.json", case_diff)

        print(f"Ranking Gate: {'PASSED' if gate['passed'] else 'NOT PASSED'}")
    else:
        fusion_comparison.update(
            {
                "production_switch_allowed": False,
                "decision": "retain_current",
                "reason": "Trigger threshold not met; fusion not evaluated",
            }
        )
        _write_json(out_dir / "rank-fusion-comparison.json", fusion_comparison)

    # ------------------------------------------------------------------
    # Acceptance summary
    # ------------------------------------------------------------------
    acceptance = {
        "phase": "NF39",
        "case_count": len(cases),
        "rrf_top_n": pool_manifest.get("rrf_top_n", 40),
        "reranker_output_top_n": args.reranker_output_top_n,
        "final_top_k": args.final_top_k,
        "reranker": reranker_name,
        "rrf_metrics_at_5": {
            "case_hit_rate": rrf_metrics.get("case_hit_rate_at_5"),
            "source_recall": rrf_metrics.get("source_recall_at_5"),
            "all_source_coverage": rrf_metrics.get("all_source_coverage_at_5"),
            "mrr": rrf_metrics.get("mrr"),
        },
        "reranker_metrics_at_5": {
            "case_hit_rate": reranker_metrics.get("case_hit_rate_at_5"),
            "source_recall": reranker_metrics.get("source_recall_at_5"),
            "all_source_coverage": reranker_metrics.get(
                "all_source_coverage_at_5"
            ),
            "mrr": reranker_metrics.get("mrr"),
        },
        "final_metrics_at_5": {
            "case_hit_rate": final_metrics.get("case_hit_rate_at_5"),
            "source_recall": final_metrics.get("source_recall_at_5"),
            "all_source_coverage": final_metrics.get(
                "all_source_coverage_at_5"
            ),
            "mrr": final_metrics.get("mrr"),
        },
        "loss_stage_counts": attribution_summary,
        "redundancy": redundancy.to_dict(),
        "fusion_triggered": trigger,
        "fusion_gate_passed": (
            fusion_comparison.get("gate", {}).get("passed", False)
            if trigger
            else None
        ),
        "production_switch_allowed": fusion_comparison.get(
            "production_switch_allowed", False
        ),
        "decision": fusion_comparison.get("decision", "retain_current"),
    }
    _write_json(out_dir / "nf39-acceptance.json", acceptance)

    print("Done. Artifacts written to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
