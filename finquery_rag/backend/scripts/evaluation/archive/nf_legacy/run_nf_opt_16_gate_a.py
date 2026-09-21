"""NF-OPT-16 Gate A: verify offline BGE-M3 sparse/multi-vector Shadow capability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.evaluation.nf_opt_16 import (
    assert_query_has_no_expected_fields,
    stable_smoke_sample,
    validate_model_output,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "benchmarks/financial_rag_v1/corpus.json"
DEFAULT_QUESTIONS = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"
DEFAULT_OUT = ROOT / "artifacts/evaluation/nf-opt-16"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _questions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(args: argparse.Namespace) -> int:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("NF-OPT-16 requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    if not args.model_path.exists():
        raise FileNotFoundError(f"missing local BGE-M3 snapshot: {args.model_path}")

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    allowed_documents = {str(item["filename"]) for item in corpus["documents"]}
    questions = _questions(args.questions)
    answerable = [item for item in questions if item.get("answerable")]
    no_answer = [item for item in questions if not item.get("answerable")]
    if len(answerable) != 64 or len(no_answer) != 8:
        raise ValueError("expected frozen 64 answerable and 8 no-answer questions")
    query = dict(sorted(answerable, key=lambda item: str(item["case_id"]))[0])
    assert_query_has_no_expected_fields(query)

    connection = sqlite3.connect(f"file:{args.candidate_db}?mode=ro", uri=True)
    try:
        source_rows = [
            {"doc_id": str(doc_id), "content": str(content or ""), "doc_name": str(doc_name)}
            for doc_id, content, doc_name in connection.execute("SELECT doc_id, content, doc_name FROM chunk_store")
            if str(doc_name) in allowed_documents
        ]
    finally:
        connection.close()
    sample = stable_smoke_sample(source_rows, limit=args.sample_size)
    sample_texts = [str(item["content"]) for item in sample]
    query_text = str(query["question"])

    # Import only after all fail-closed environment checks have passed.
    from FlagEmbedding import BGEM3FlagModel

    started = time.perf_counter()
    model = BGEM3FlagModel(
        str(args.model_path),
        use_fp16=True,
        devices=args.device,
        batch_size=args.batch_size,
    )
    candidate_output = model.encode(
        sample_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    query_output = model.encode(
        [query_text],
        batch_size=1,
        max_length=args.max_length,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    candidate_stats = validate_model_output(candidate_output, expected_rows=len(sample))
    query_stats = validate_model_output(query_output, expected_rows=1)

    model_manifest = {
        "model_name": "BAAI/bge-m3",
        "model_snapshot_path": str(args.model_path),
        "model_snapshot_sha256": _sha(args.model_path / "config.json"),
        "offline_mode": True,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "use_fp16": True,
        "dense_mode_requested": False,
        "sparse_mode_requested": True,
        "multi_vector_mode_requested": True,
        "model_inputs_exclude_expected_fields": True,
    }
    sample_manifest = {
        "candidate_store_access_mode": "sqlite_read_only",
        "corpus_candidate_count": len(source_rows),
        "sample_size": len(sample),
        "sample_identity_sha256": hashlib.sha256(
            "\n".join(str(item["doc_id"]) for item in sample).encode("utf-8")
        ).hexdigest(),
        "sample_content_sha256": hashlib.sha256("\n".join(sample_texts).encode("utf-8")).hexdigest(),
        "query_case_id": str(query["case_id"]),
        "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
        "gold_fields_read_for_model_inputs": False,
        "raw_candidate_content_committed": False,
    }
    capability = {
        "candidate_output": candidate_stats,
        "query_output": query_stats,
        "elapsed_ms": elapsed_ms,
        "sparse_shadow_retrieval_eligible": True,
        "multi_vector_full_corpus_index_built": False,
        "multi_vector_next_step": "measure_only_after_frozen_sparse_shadow_result",
    }
    acceptance = {
        "artifact_schema": "nf-opt-16/gate-a/acceptance/v1",
        "baseline_master_merge_commit": "4d5bcaf8bdc459bbc0329b0666949d6297592958",
        "baseline_tree_sha": "6c7410e6984b3234f9bbaa02ea7f7edbf98ffa57",
        "question_count": 72,
        "answerable_case_count": 64,
        "no_answer_case_count": 8,
        "input_hashes": {"corpus_sha256": _sha(args.corpus), "questions_sha256": _sha(args.questions)},
        "candidate_store_access_mode": "sqlite_read_only",
        "gold_fields_read_for_model_inputs": False,
        "model_network_calls": 0,
        "embedding_model_calls": 2,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "decision": "bge_m3_sparse_shadow_capability_validated",
        "next_gate": "nf-opt-16-gate-b-frozen-sparse-shadow",
    }
    _write(args.out_dir / "bge-m3-runtime-manifest.json", model_manifest)
    _write(args.out_dir / "shadow-input-manifest.json", sample_manifest)
    _write(args.out_dir / "bge-m3-capability-report.json", capability)
    _write(args.out_dir / "multi-vector-feasibility-report.json", {
        "status": "capability_confirmed_not_indexed",
        "reason": "Gate A validates output semantics only; full-corpus late-interaction indexing is a separate frozen decision.",
        "candidate_colbert": candidate_stats,
        "query_colbert": query_stats,
    })
    _write(args.out_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": acceptance["next_gate"], "production_switch_allowed": False})
    _write(args.out_dir / "nf-opt-16-gate-a-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--sample-size", type=int, default=8)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
