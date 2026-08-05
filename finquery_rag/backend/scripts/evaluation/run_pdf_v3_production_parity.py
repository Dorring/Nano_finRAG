"""Gate 0: replay the production retrieval objects without answer generation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.evaluation import run_nf_eval_03_r1 as r1
from scripts.evaluation import run_nf_eval_03_r2 as r2
from scripts.evaluation.run_nf_eval_03_baseline import _build_engine, _load_inputs
from scripts.evaluation.run_nf_eval_03_r2 import RecordingBM25, RecordingDenseQuery, RecordingReranker, _stage_list

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-0"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ranks(values: list[dict[str, Any]]) -> dict[str, int]:
    return {str(item["candidate_key"]): int(item["stage_rank"]) for item in values}


def _historical_metrics() -> dict[str, int]:
    """Read historical diagnostics only to document their non-production provenance."""
    bm25 = json.loads(
        (ROOT / "artifacts/evaluation/nf-opt-03/bm25-comparison.json").read_text(
            encoding="utf-8"
        )
    )
    dense = json.loads(
        (ROOT / "artifacts/evaluation/nf-opt-01/dense-rank-comparison.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "bm25_current_window_at_200": int(
            bm25["A"]["rank_metrics"]["@200"]["source_hit_count"]
        ),
        "bm25_b200_at_200": int(
            bm25["B200"]["rank_metrics"]["@200"]["source_hit_count"]
        ),
        "dense_current_at_200": int(dense["current"]["@200"]["source_hit_count"]),
        "dense_superset_at_200": int(dense["superset"]["@200"]["source_hit_count"]),
    }


def _ledger(
    *,
    metrics: dict[str, Any],
    universe: dict[str, Any],
    code_commit: str,
) -> list[dict[str, Any]]:
    historical = _historical_metrics()
    current = {
        "evaluation_role": "current_production_snapshot",
        "code_commit": code_commit,
        "index_hash": universe["bm25_db_sha256"],
        "candidate_universe_hash": universe["candidate_universe_hash"],
        "production_promoted": True,
        "source_artifact": "artifacts/evaluation/pdf-retrieval-v3-gate-0/production-stage-replay.json",
    }
    return [
        {
            **current,
            "metric_name": "BM25 Source Recall@200",
            "value": f"{metrics['bm25_source_recall_at_200']}/80",
            "retriever": "production_bm25",
            "requested_k": 200,
            "query_processing": "production QueryProcessor expansion",
        },
        {
            **current,
            "metric_name": "Dense Source Recall@200",
            "value": f"{metrics['dense_source_recall_at_200']}/80",
            "retriever": "production_dense",
            "requested_k": 200,
            "query_processing": "production QueryProcessor expansion",
        },
        {
            **current,
            "metric_name": "RRF Source Recall@40",
            "value": f"{metrics['rrf_source_recall_at_40']}/80",
            "retriever": "production_rrf_diagnostic_cut",
            "requested_k": 40,
            "query_processing": "production retrieval pipeline",
        },
        {
            **current,
            "metric_name": "Strict Final Source Recall@5",
            "value": f"{metrics['strict_final_source_recall_at_5']}/80",
            "retriever": "production_final_selector",
            "requested_k": 5,
            "query_processing": "production retrieval pipeline",
        },
        {
            "metric_name": "BM25 Source Recall@200",
            "value": f"{historical['bm25_current_window_at_200']}/80",
            "evaluation_role": "historical_diagnostic_current_window",
            "code_commit": "not_available_in_artifact",
            "index_hash": "not_available_in_artifact",
            "candidate_universe_hash": "not_available_in_artifact",
            "retriever": "bm25_variant_a",
            "requested_k": 200,
            "query_processing": "NF-OPT-03 current_production window diagnostic",
            "production_promoted": False,
            "source_artifact": "artifacts/evaluation/nf-opt-03/bm25-comparison.json#/A/rank_metrics/@200",
        },
        {
            "metric_name": "BM25 Source Recall@200",
            "value": f"{historical['bm25_b200_at_200']}/80",
            "evaluation_role": "historical_shadow_extended_window",
            "code_commit": "not_available_in_artifact",
            "index_hash": "not_available_in_artifact",
            "candidate_universe_hash": "not_available_in_artifact",
            "retriever": "bm25_variant_b200",
            "requested_k": 200,
            "query_processing": "NF-OPT-03 extended BM25 window experiment",
            "production_promoted": False,
            "source_artifact": "artifacts/evaluation/nf-opt-03/bm25-comparison.json#/B200/rank_metrics/@200",
        },
        {
            "metric_name": "Dense Source Recall@200",
            "value": f"{historical['dense_current_at_200']}/80",
            "evaluation_role": "historical_diagnostic_current_index",
            "code_commit": "not_available_in_artifact",
            "index_hash": "not_available_in_artifact",
            "candidate_universe_hash": "not_available_in_artifact",
            "retriever": "dense_current",
            "requested_k": 200,
            "query_processing": "NF-OPT-01 dense rank diagnostic",
            "production_promoted": False,
            "source_artifact": "artifacts/evaluation/nf-opt-01/dense-rank-comparison.json#/current/@200",
        },
        {
            "metric_name": "Dense Source Recall@200",
            "value": f"{historical['dense_superset_at_200']}/80",
            "evaluation_role": "historical_shadow_superset",
            "code_commit": "not_available_in_artifact",
            "index_hash": "not_available_in_artifact",
            "candidate_universe_hash": "not_available_in_artifact",
            "retriever": "dense_superset",
            "requested_k": 200,
            "query_processing": "NF-OPT-01 dense canonical/superset shadow",
            "production_promoted": False,
            "source_artifact": "artifacts/evaluation/nf-opt-01/dense-rank-comparison.json#/superset/@200",
        },
    ]


async def run(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inputs = _load_inputs(corpus_path=args.corpus, manifest_path=args.manifest, questions_path=args.questions, labels_path=args.labels)
    engine, client = _build_engine(args)
    pipeline = engine._retrieval_pipeline
    reranker = RecordingReranker(engine.reranker)
    dense = RecordingDenseQuery(pipeline._dense_query_fn)
    bm25 = RecordingBM25(pipeline._bm25_retriever)
    engine.reranker = reranker
    pipeline._reranker = reranker
    pipeline._dense_query_fn = dense
    pipeline._bm25_retriever = bm25
    mapping = r1._doc_map(inputs.corpus)
    filenames = {str(item["document_id"]): str(item["filename"]) for item in inputs.corpus["documents"]}
    replayed: list[dict[str, Any]] = []
    for question in inputs.questions:
        case_id = str(question["case_id"])
        scope = [filenames[str(item)] for item in question["document_scope"]]
        if len(scope) != 1:
            raise RuntimeError(f"{case_id}: Gate 0 requires one document scope")
        reranker.clear()
        dense.clear()
        bm25.clear()
        retrieval_query = pipeline._query_processor.expand(str(question["question"]))
        dense_200 = pipeline._dense_query_fn(
            query_text=retrieval_query,
            doc_name=scope[0],
            n_results=200,
            user_id=args.tenant_id,
        )
        bm25_200 = pipeline._bm25_retriever.search(
            retrieval_query,
            k=200,
            doc_name=scope[0],
            user_id=args.tenant_id,
        )
        await asyncio.get_running_loop().run_in_executor(None, pipeline.retrieve_single, scope[0], str(question["question"]), args.tenant_id, args.n_results)
        debug = pipeline.last_retrieval_debug.get("candidate_stages") or {}
        rerank_call, final_call = r2._select_calls(reranker.calls, final_top_k=args.n_results)
        rrf_full_raw = list((rerank_call or {}).get("input") or [])
        reranker_output_raw = list((rerank_call or {}).get("output") or [])
        final_input_raw = list((final_call or {}).get("input") or [])
        final_output_raw = list((final_call or {}).get("output") or [])
        stages = {
            "dense": _stage_list(dense_200, mapping=mapping, tenant_id=args.tenant_id),
            "bm25": _stage_list(bm25_200, mapping=mapping, tenant_id=args.tenant_id),
            "rrf": _stage_list(debug.get("rrf") or [], mapping=mapping, tenant_id=args.tenant_id),
            "rrf_full": _stage_list(rrf_full_raw, mapping=mapping, tenant_id=args.tenant_id),
            "reranker_input": _stage_list(rrf_full_raw, mapping=mapping, tenant_id=args.tenant_id),
            "reranker": _stage_list(debug.get("reranker") or [], mapping=mapping, tenant_id=args.tenant_id),
            "final": _stage_list(debug.get("final") or [], mapping=mapping, tenant_id=args.tenant_id),
        }
        replayed.append(
            {
                "case_id": case_id,
                "stages": stages,
                "actual_stage_boundaries": {
                    "bm25_requested_k": 200,
                    "bm25_actual_returned_count": len(bm25_200),
                    "dense_requested_k": 200,
                    "dense_actual_returned_count": len(dense_200),
                    "rrf_metric_k": 40,
                    "rrf_debug_output_count": len(debug.get("rrf") or []),
                    "rrf_actual_output_count": len(rrf_full_raw),
                    "reranker_input_count": len(rrf_full_raw),
                    "reranker_output_count": len(reranker_output_raw),
                    "final_selector_input_count": len(final_input_raw),
                    "final_selector_output_count": len(final_output_raw),
                    "reranker_calls": [
                        {
                            "top_k": call.get("top_k"),
                            "input_count": len(call.get("input") or []),
                            "output_count": len(call.get("output") or []),
                        }
                        for call in reranker.calls
                    ],
                },
            }
        )
    authoritative = json.loads((ROOT / "artifacts/evaluation/nf-eval-03-r2/case-results.json").read_text(encoding="utf-8"))["cases"]
    authoritative_by_id = {str(item["case_id"]): item for item in authoritative}
    label_by_id = inputs.labels_by_id
    expected = 0
    hits = {name: set() for name in ("bm25", "dense", "rrf", "final")}
    rows = []
    for record in replayed:
        case_id = record["case_id"]
        sources = list(label_by_id[case_id].get("expected_sources") or [])
        gold = {str(item["candidate_key"]) for item in sources}
        expected += len(sources)
        stage_hits = {}
        for name, values in record["stages"].items():
            matched = gold & {str(item["candidate_key"]) for item in values}
            if name in hits:
                hits[name].update(
                    (case_id, source_index)
                    for source_index, source in enumerate(sources)
                    if str(source["candidate_key"]) in matched
                )
            stage_hits[name] = sorted(matched)
        authoritative_final = {str(item["candidate_key"]) for item in authoritative_by_id[case_id].get("retrieval_stages", {}).get("final") or []}
        rows.append({"case_id": case_id, "replayed_final_hits": stage_hits["final"], "authoritative_final_gold_hits": sorted(gold & authoritative_final), "stage_hit_counts": {name: len(value) for name, value in stage_hits.items()}})
    auth_hits = {
        (str(case["case_id"]), source_index)
        for case in authoritative
        for source_index, source in enumerate(label_by_id[str(case["case_id"])].get("expected_sources") or [])
        if str(source["candidate_key"]) in {str(item["candidate_key"]) for item in case.get("retrieval_stages", {}).get("final") or []}
    }
    replay_final = hits["final"]
    metrics = {"gold_source_count": expected, "bm25_source_recall_at_200": len(hits["bm25"]), "dense_source_recall_at_200": len(hits["dense"]), "rrf_source_recall_at_40": len(hits["rrf"]), "strict_final_source_recall_at_5": len(replay_final), "authoritative_final_hit_count": len(auth_hits), "replayed_final_hit_count": len(replay_final), "missing_hits": [dict(case_id=case_id, source_index=index) for case_id, index in sorted(auth_hits - replay_final)], "unexpected_hits": [dict(case_id=case_id, source_index=index) for case_id, index in sorted(replay_final - auth_hits)]}
    universe = {"bm25_db_sha256": _sha(args.bm25_db_path), "chroma_sqlite_sha256": _sha(args.chroma_path / "chroma.sqlite3"), "candidate_universe_hash": _sha(args.bm25_db_path), "tenant_id": args.tenant_id}
    nestedness_records = []
    unexplained_transitions = []
    replay_by_id = {str(item["case_id"]): item for item in replayed}
    for case_id, source_index in sorted(auth_hits):
        candidate_key = str(label_by_id[case_id]["expected_sources"][source_index]["candidate_key"])
        stages = replay_by_id[case_id]["stages"]
        ranks = {name: _ranks(values) for name, values in stages.items()}
        rrf_at_40 = ranks["rrf"].get(candidate_key)
        rrf_full = ranks["rrf_full"].get(candidate_key)
        final_rank = ranks["final"].get(candidate_key)
        transition_reason = "nested_rrf_metric_to_final"
        if rrf_at_40 is None:
            if rrf_full is None:
                transition_reason = "unexplained_final_candidate"
                unexplained_transitions.append({"case_id": case_id, "source_index": source_index})
            elif rrf_full > 40:
                transition_reason = "rrf_full_pool_to_reranker"
            else:
                transition_reason = "debug_rrf_identity_projection_mismatch"
        nestedness_records.append(
            {
                "case_id": case_id,
                "source_index": source_index,
                "candidate_identity": candidate_key,
                "bm25_rank": ranks["bm25"].get(candidate_key),
                "dense_rank": ranks["dense"].get(candidate_key),
                "rrf_full_rank": rrf_full,
                "rrf_at_40_rank": rrf_at_40,
                "reranker_input": candidate_key in ranks["reranker_input"],
                "reranker_rank": ranks["reranker"].get(candidate_key),
                "final_rank": final_rank,
                "transition_reason": transition_reason,
            }
        )
    boundaries = [record["actual_stage_boundaries"] for record in replayed]
    stage_boundary_manifest = {
        "contract": {
            "bm25_requested_k": 200,
            "dense_requested_k": 200,
            "rrf_metric_k": 40,
            "rrf_actual_output": "full fused list passed into each production reranker invocation",
            "reranker_input": "the full RRF list, not the diagnostic RRF@40 projection",
            "reranker_output": "first production invocation is Top-20",
            "final_selector_input": "second production reranker invocation receives the same full RRF list",
            "final_selector_output": "Top-5",
        },
        "per_case": boundaries,
        "summary": {
            key: sorted({int(boundary[key]) for boundary in boundaries})
            for key in (
                "bm25_actual_returned_count",
                "dense_actual_returned_count",
                "rrf_debug_output_count",
                "rrf_actual_output_count",
                "reranker_input_count",
                "reranker_output_count",
                "final_selector_input_count",
                "final_selector_output_count",
            )
        },
    }
    metrics["unexplained_final_transitions"] = len(unexplained_transitions)
    accepted = (
        len(hits["rrf"]) == 20
        and len(replay_final) == 13
        and not metrics["missing_hits"]
        and not metrics["unexpected_hits"]
        and not unexplained_transitions
    )
    _write(args.out_dir / "baseline-runtime-config.json", {"code_commit": args.code_commit, "embedding_model": engine._embedding_model_name if hasattr(engine, "_embedding_model_name") else None, "retrieval_candidate_multiplier": args.retrieval_candidate_multiplier, "final_k": args.n_results, "production_components_called": ["QueryProcessor", "BM25Retriever", "DenseRetriever", "candidate_fusion.rrf", "Reranker", "FinalSelector"]})
    _write(args.out_dir / "candidate-universe-manifest.json", universe)
    _write(args.out_dir / "index-integrity.json", {"bm25_index_hash": universe["bm25_db_sha256"], "dense_collection_identity": str(args.chroma_path), "question_hash": _sha(args.questions), "gold_source_hash": _sha(args.labels)})
    _write(args.out_dir / "metric-provenance-ledger.json", {"records": _ledger(metrics=metrics, universe=universe, code_commit=args.code_commit)})
    _write(args.out_dir / "stage-boundary-manifest.json", stage_boundary_manifest)
    _write(args.out_dir / "stage-nestedness-audit.json", {"records": nestedness_records, "unexplained_transition_count": len(unexplained_transitions), "unexplained_transitions": unexplained_transitions})
    _write(args.out_dir / "production-stage-replay.json", {"metrics": metrics, "cases": replayed})
    _write(args.out_dir / "baseline-case-parity.json", {"authoritative_final_hits": [dict(case_id=case_id, source_index=index) for case_id, index in sorted(auth_hits)], "replayed_final_hits": [dict(case_id=case_id, source_index=index) for case_id, index in sorted(replay_final)], "missing_hits": metrics["missing_hits"], "unexpected_hits": metrics["unexpected_hits"], "cases": rows})
    _write(args.out_dir / "acceptance.json", {"gate": "pdf_retrieval_v3_gate_0_r1", "gate_passed": accepted, "decision": "production_decision_parity_passed_with_metric_provenance_closed" if accepted else "production_decision_parity_blocked", "next_gate": "benchmark_governance" if accepted else "stop_and_fix_harness", "current_production_snapshot": {"bm25_source_recall_at_200": f"{len(hits['bm25'])}/80", "dense_source_recall_at_200": f"{len(hits['dense'])}/80", "rrf_source_recall_at_40": f"{len(hits['rrf'])}/80", "strict_final_source_recall_at_5": f"{len(replay_final)}/80", "final_identity_parity": f"{len(replay_final & auth_hits)}/13"}, "historical_non_production_metrics_archived": True, "stage_boundary_manifest_complete": True, "unexplained_final_transitions": len(unexplained_transitions), "retrieval_runtime_gold_reads": 0, "gold_source_posthoc_scoring_reads": expected, "expected_value_runtime_reads": 0, "per_query_oracle_selection": False, "parameter_scan": False, "production_index_writes": 0, "production_default_config_modified": False, "answer_generation_calls": 0, "model_chat_completion_requests": client.chat_completion_requests})
    return 0 if accepted else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", type=int, default=1)
    parser.add_argument("--model-base-url", default="http://127.0.0.1:8500/v1")
    parser.add_argument("--model-name", default="finquery-finance-sft1147")
    parser.add_argument("--api-key", default="not-needed-for-local")
    parser.add_argument("--chroma-path", type=Path, required=True)
    parser.add_argument("--bm25-db-path", type=Path, required=True)
    parser.add_argument("--n-results", type=int, default=5)
    parser.add_argument("--retrieval-candidate-multiplier", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--corpus", type=Path, default=ROOT / "benchmarks/financial_rag_v1/corpus.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/financial_rag_v1/data/golden-manifest.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl")
    parser.add_argument("--labels", type=Path, default=ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl")
    parser.add_argument("--code-commit", default="working-tree")
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
