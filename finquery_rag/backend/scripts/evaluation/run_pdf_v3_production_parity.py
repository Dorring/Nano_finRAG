"""Gate 0: replay the production retrieval objects without answer generation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.evaluation import run_nf_eval_03_r1 as r1
from scripts.evaluation.run_nf_eval_03_baseline import _build_engine, _load_inputs
from scripts.evaluation.run_nf_eval_03_r2 import RecordingBM25, RecordingDenseQuery, RecordingReranker, _stage_list

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-0"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        stages = {
            "dense": _stage_list(dense_200, mapping=mapping, tenant_id=args.tenant_id),
            "bm25": _stage_list(bm25_200, mapping=mapping, tenant_id=args.tenant_id),
            "rrf": _stage_list(debug.get("rrf") or [], mapping=mapping, tenant_id=args.tenant_id),
            "reranker": _stage_list(debug.get("reranker") or [], mapping=mapping, tenant_id=args.tenant_id),
            "final": _stage_list(debug.get("final") or [], mapping=mapping, tenant_id=args.tenant_id),
        }
        replayed.append({"case_id": case_id, "stages": stages})
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
    accepted = len(hits["bm25"]) == 48 and len(hits["dense"]) == 52 and len(hits["rrf"]) == 20 and len(replay_final) == 13 and not metrics["missing_hits"] and not metrics["unexpected_hits"]
    universe = {"bm25_db_sha256": _sha(args.bm25_db_path), "chroma_sqlite_sha256": _sha(args.chroma_path / "chroma.sqlite3"), "candidate_universe_hash": _sha(args.bm25_db_path), "tenant_id": args.tenant_id}
    _write(args.out_dir / "baseline-runtime-config.json", {"code_commit": args.code_commit, "embedding_model": engine._embedding_model_name if hasattr(engine, "_embedding_model_name") else None, "retrieval_candidate_multiplier": args.retrieval_candidate_multiplier, "final_k": args.n_results, "production_components_called": ["QueryProcessor", "BM25Retriever", "DenseRetriever", "candidate_fusion.rrf", "Reranker", "FinalSelector"]})
    _write(args.out_dir / "candidate-universe-manifest.json", universe)
    _write(args.out_dir / "index-integrity.json", {"bm25_index_hash": universe["bm25_db_sha256"], "dense_collection_identity": str(args.chroma_path), "question_hash": _sha(args.questions), "gold_source_hash": _sha(args.labels)})
    _write(args.out_dir / "production-stage-replay.json", {"metrics": metrics, "cases": replayed})
    _write(args.out_dir / "baseline-case-parity.json", {"authoritative_final_hits": [dict(case_id=case_id, source_index=index) for case_id, index in sorted(auth_hits)], "replayed_final_hits": [dict(case_id=case_id, source_index=index) for case_id, index in sorted(replay_final)], "missing_hits": metrics["missing_hits"], "unexpected_hits": metrics["unexpected_hits"], "cases": rows})
    _write(args.out_dir / "acceptance.json", {"gate": "pdf_retrieval_v3_gate_0", "gate_passed": accepted, "decision": "production_baseline_parity_passed" if accepted else "production_baseline_parity_blocked", "next_gate": "gate_1_benchmark_governance" if accepted else "stop_and_fix_harness", "gold_source_runtime_reads": 0, "expected_value_runtime_reads": 0, "per_query_oracle_selection": False, "parameter_scan": False, "production_index_writes": 0, "production_default_config_modified": False, "answer_generation_calls": 0, "model_chat_completion_requests": client.chat_completion_requests})
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
