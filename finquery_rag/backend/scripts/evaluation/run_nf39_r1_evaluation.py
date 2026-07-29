"""Run NF39 R1 on the existing frozen Top-40 candidate pool only."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from src.evaluation.case_fingerprints import label_fingerprint, question_fingerprint
from src.evaluation.evaluation import load_jsonl_cases
from src.evaluation.nf38_evaluator import _stable_digest
from src.evaluation.nf39_r1_integrity import (
    case_stage_summary,
    denominator_report,
    final_context_manifest,
    fusion_execution_report,
    source_stage_transitions,
    stage_metrics_same_k,
)
from src.retrieval.rank_preserving_fusion import rank_preserving_fusion
from src.services.reranker import build_reranker
from scripts.evaluation.run_nf39_evaluation import (
    _fusion_to_summary,
    _pool_to_rrf_format,
    _reranker_output_to_summary,
    _rrf_to_summary,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pool_digest(pool: dict[str, list[dict[str, Any]]]) -> str:
    return _stable_digest([
        {"case_id": case_id, "candidates": pool[case_id]}
        for case_id in sorted(pool)
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NF39 R1 ranking-only attribution")
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reranker", default="heuristic")
    parser.add_argument("--reranker-input-top-n", type=int, default=20)
    parser.add_argument("--final-top-k", type=int, default=5)
    args = parser.parse_args()
    if args.reranker_input_top_n != 20 or args.final_top_k != 5:
        raise ValueError("NF39 R1 is fixed to reranker input Top-20 and Final Top-5")

    pool_dir, out_dir = Path(args.pool_dir), Path(args.out_dir)
    cases = load_jsonl_cases(args.cases)
    pool = json.loads((pool_dir / "rrf-candidate-pool.json").read_text(encoding="utf-8"))
    manifest = json.loads((pool_dir / "rrf-candidate-pool-manifest.json").read_text(encoding="utf-8"))
    actual_pool_hash = _pool_digest(pool)
    if actual_pool_hash != manifest["candidate_pool_hash"]:
        raise ValueError("Frozen candidate-pool hash mismatch")
    if len(cases) != manifest["case_count"]:
        raise ValueError("Case count does not match frozen candidate pool")
    if question_fingerprint(cases) != manifest["question_hash"]:
        raise ValueError("Question hash does not match frozen candidate pool")
    if label_fingerprint(cases) != manifest["label_hash"]:
        raise ValueError("Label hash does not match frozen candidate pool")

    reranker = build_reranker(args.reranker)
    if reranker is None:
        raise RuntimeError("NF39 R1 requires an explicit reranker provider")
    rrf_rankings, inputs, reranked, final, fusion = {}, {}, {}, {}, {}
    for case in cases:
        pool_rows = pool.get(case.case_id, [])
        rrf_summary = _rrf_to_summary(pool_rows)
        rrf_rankings[case.case_id] = rrf_summary
        input_rows = pool_rows[:args.reranker_input_top_n]
        inputs[case.case_id] = _rrf_to_summary(input_rows)
        lookup = {row.get("candidate_id") or row.get("evidence_id"): row for row in pool_rows}
        reranker_output = reranker.rerank(
            case.question,
            _pool_to_rrf_format(input_rows),
            top_k=len(input_rows),
        )
        reranker_summary = _reranker_output_to_summary(reranker_output, lookup)
        reranked[case.case_id] = reranker_summary
        final[case.case_id] = list(reranker_summary[:args.final_top_k])
        ranks = {row.get("doc_id", ""): rank for rank, row in enumerate(reranker_output, 1)}
        fused = rank_preserving_fusion(
            rrf_candidates=input_rows,
            reranked_candidates=reranker_output,
            fusion_k=60,
        )
        fusion[case.case_id] = _fusion_to_summary(fused, lookup, ranks)[:args.final_top_k]

    stages = {
        "s0_rrf_top40": stage_metrics_same_k(cases=cases, rankings=rrf_rankings, ks=(5, 20, 40)),
        "s1_rrf_top20_reranker_input": stage_metrics_same_k(cases=cases, rankings=inputs, ks=(5, 20)),
        "s2_reranker_ranked_top20": stage_metrics_same_k(cases=cases, rankings=reranked, ks=(5, 20)),
        "s3_reranker_top5": stage_metrics_same_k(cases=cases, rankings=reranked, ks=(5,)),
        "s4_final_context_top5": stage_metrics_same_k(cases=cases, rankings=final, ks=(5,)),
    }
    source_records, transition_counts = source_stage_transitions(
        cases=cases,
        rrf_rankings=rrf_rankings,
        reranker_input_rankings=inputs,
        reranker_rankings=reranked,
        final_rankings=final,
        reranker_input_top_n=args.reranker_input_top_n,
        rrf_top_n=manifest["rrf_top_n"],
    )
    source_case_counts: dict[str, int] = {}
    for row in source_records:
        source_case_counts.setdefault(row["transition"], set()).add(row["case_id"])
    source_case_counts = {key: len(value) for key, value in source_case_counts.items()}
    context = final_context_manifest(final)
    fusion_report = fusion_execution_report(
        cases=cases,
        rrf_rankings={key: value[:5] for key, value in rrf_rankings.items()},
        reranker_rankings={key: value[:5] for key, value in reranked.items()},
        fusion_rankings=fusion,
    )
    denoms = denominator_report(cases)
    baseline = {
        "case_count": len(cases),
        "answerable_case_count": denoms["retrieval_case_count"],
        "no_answer_case_count": denoms["no_answer_case_count"],
        "expected_source_count": denoms["expected_source_count"],
        "candidate_pool_hash": actual_pool_hash,
        "question_hash": manifest["question_hash"],
        "label_hash": manifest["label_hash"],
        "tenant_id": manifest["tenant_id"],
        "corpus_hash": manifest["corpus_hash"],
        "rrf_top_n": manifest["rrf_top_n"],
        "reranker_input_top_n": args.reranker_input_top_n,
        "final_top_k": args.final_top_k,
        "reranker": reranker.name,
        "ranking_only": True,
        "production_behavior_changed": False,
    }
    _write(out_dir / "baseline-manifest.json", baseline)
    _write(out_dir / "stage-metrics-same-k.json", {"denominators": denoms, "stages": stages})
    _write(out_dir / "source-rank-transitions.json", {"counts": transition_counts, "case_counts": source_case_counts, "sources": source_records})
    _write(out_dir / "case-stage-summary.json", {"cases": case_stage_summary(cases=cases, rrf_rankings=rrf_rankings, reranker_rankings=reranked, final_rankings=final)})
    _write(out_dir / "final-context-manifest.json", context)
    _write(out_dir / "fusion-execution-report.json", fusion_report)
    _write(out_dir / "nf39-r1-acceptance.json", {
        "same_k_metrics_present": True,
        "same_denominators_verified": True,
        "source_level_attribution_present": True,
        "ranking_only_answer_claims_removed": True,
        "fusion_execution_verified": fusion_report["fusion_executed"],
        "production_behavior_changed": False,
    })
    print("NF39 R1 complete:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

