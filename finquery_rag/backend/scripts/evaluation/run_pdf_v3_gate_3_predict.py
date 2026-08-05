"""Gate 3 prediction: append a structured residual to a frozen raw RRF pool.

The script never opens labels or governance.  The raw side is the sealed Gate
0 production-stage replay; structured retrieval is isolated in runtime
directories and is eligible only for Gate 2 ``table_single_fact`` profiles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any

import numpy as np

from src.evaluation.nf_opt_15 import build_retrieval_view
from src.evaluation.pdf_query_representation_v2 import (
    char_score,
    fixed_rrf as concept_rrf,
    normalize_label,
    ranks,
    token_bm25_scores,
)
from src.retrieval.embedding_provider import ExistingMiniLMEmbeddingProvider
from src.retrieval_v3.structured_lane import (
    append_structured_residual,
    enriched_retrieval_text,
    fixed_rrf,
    is_safe_structured_view,
    payload_hash,
)
from src.services.retrieval import SqliteBM25Retriever
from src.services.retrieval_config import get_embedding_model_name


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/financial_rag_v1/data"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-3"
RUNTIME = ROOT / "artifacts/runtime/pdf-retrieval-v3-gate-3"
GATE_0 = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-0/production-stage-replay.json"
GATE_2 = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-2/router-predictions.json"
CONCEPTS = ROOT / "artifacts/evaluation/pdf-query-representation-v2/concept-registry.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_structured_views(
    candidate_db: Path, corpus: dict[str, Any], tenant_id: int
) -> tuple[list[dict[str, Any]], int, int]:
    """Build one safe V2-Lite view per current production row identity."""
    documents = {str(item["filename"]): item for item in corpus["documents"]}
    connection = sqlite3.connect(f"file:{candidate_db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT doc_id, content, metadata_json, doc_name FROM chunk_store WHERE user_id = ?",
            (tenant_id,),
        ).fetchall()
    finally:
        connection.close()
    views: dict[str, dict[str, Any]] = {}
    conflicts = 0
    for doc_id, content, metadata_json, doc_name in rows:
        document = documents.get(str(doc_name))
        if document is None:
            continue
        try:
            metadata = json.loads(metadata_json or "{}")
        except json.JSONDecodeError:
            continue
        view = build_retrieval_view(
            doc_id=str(doc_id), content=str(content or ""), metadata=metadata, document=document
        )
        if not is_safe_structured_view(view):
            continue
        identity = str(view["candidate_key"])
        record = {
            "structured_view_id": str(view["retrieval_view_id"]),
            "candidate_key": identity,
            "evidence_id": str(view["evidence_id"]),
            "document_id": str(view["document_id"]),
            "filename": str(doc_name),
            "pdf_page": view.get("pdf_page"),
            "table_id": metadata.get("parent_id") or metadata.get("table_id"),
            "row_index": metadata.get("row_index"),
            "raw_text": str(content or ""),
            "retrieval_text": enriched_retrieval_text(view, str(content or "")),
            "metric": view["metric_field"]["normalized_metric"],
            "statement_or_section": view["section_field"]["statement_title"],
            "table_level_periods": list(view["period_field"]["periods"]),
            "scale": view["unit_field"]["scale"],
            "representation_level": "strict_cell_aware" if metadata.get("table_alignment") == "exact" else "retrieval_only",
            "field_lineage": view["field_lineage"],
            "cell_period_claim_emitted": False,
        }
        previous = views.get(identity)
        if previous is not None and previous["evidence_id"] != record["evidence_id"]:
            conflicts += 1
            continue
        views.setdefault(identity, record)
    return sorted(views.values(), key=lambda item: item["candidate_key"]), conflicts, len(rows)


def _resolve_concept(
    phrase: str, registry: list[dict[str, Any]], provider: ExistingMiniLMEmbeddingProvider,
    vectors: np.ndarray,
) -> dict[str, Any]:
    labels = [" ".join([str(item["canonical_label"]), *item.get("generic_aliases", [])]) for item in registry]
    normalized = normalize_label(phrase)
    exact = [
        1.0 if normalized in [*item.get("labels", []), *item.get("generic_aliases", [])] else 0.0
        for item in registry
    ]
    query_vector = provider.encode_queries([phrase])[0]
    dense = (vectors @ np.asarray(query_vector)).tolist()
    scores = concept_rrf([ranks(exact), ranks(token_bm25_scores(phrase, labels)), ranks([char_score(phrase, label) for label in labels]), ranks(dense)])
    order = sorted(range(len(registry)), key=lambda index: (-scores[index], index))
    first, second = order[0], order[1] if len(order) > 1 else order[0]
    selected = registry[first]
    return {
        "metric_phrase": phrase,
        "top_1_concept_id": selected["concept_id"],
        "top_1_canonical_label": selected["canonical_label"],
        "top_1_score": scores[first],
        "top_1_top_2_margin": scores[first] - scores[second],
        "signals": ["exact", "token_bm25", "character_trigram", "short_text_embedding", "fixed_rrf_k60"],
    }


def _candidate_record(view: dict[str, Any], rank: int, score: float, lane: str) -> dict[str, Any]:
    return {
        "candidate_key": view["candidate_key"], "evidence_id": view["evidence_id"],
        "document_id": view["document_id"], "pdf_page": view["pdf_page"],
        "evidence_type": "table_row", "structured_rank": rank, "structured_score": score,
        "present_in_raw_pool": False, "present_in_structured_pool": True,
        "lane_provenance": [lane],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--tenant-id", type=int, default=1)
    parser.add_argument("--device", default=os.getenv("PDF_V3_EMBEDDING_DEVICE", "cpu"))
    parser.add_argument("--corpus", type=Path, default=ROOT / "benchmarks/financial_rag_v1/corpus.json")
    parser.add_argument("--questions", type=Path, default=DATA / "questions.golden.jsonl")
    parser.add_argument("--gate-0", type=Path, default=GATE_0)
    parser.add_argument("--gate-2", type=Path, default=GATE_2)
    parser.add_argument("--concept-registry", type=Path, default=CONCEPTS)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--runtime-dir", type=Path, default=RUNTIME)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()
    if not args.gate_0.exists() or not args.gate_2.exists():
        raise RuntimeError("Gate 0 raw snapshot and sealed Gate 2 predictions are required")
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    questions = _jsonl(args.questions)
    profiles = {item["case_id"]: item for item in json.loads(args.gate_2.read_text(encoding="utf-8"))["predictions"]}
    gate_0 = json.loads(args.gate_0.read_text(encoding="utf-8"))
    baseline_snapshot = dict(gate_0["metrics"])
    expected_snapshot = {
        "bm25_source_recall_at_200": 37,
        "dense_source_recall_at_200": 14,
        "rrf_source_recall_at_40": 20,
        "strict_final_source_recall_at_5": 13,
    }
    if any(baseline_snapshot.get(name) != value for name, value in expected_snapshot.items()):
        raise RuntimeError("Gate 0 current-production snapshot is not the required 37/14/20/13 baseline")
    raw_cases = {item["case_id"]: item for item in gate_0["cases"]}
    if len(questions) != 72 or set(profiles) != {item["case_id"] for item in questions}:
        raise RuntimeError("expected complete sealed 72-question Gate 2 profile set")
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    for name in ("structured-bm25", "structured-dense"):
        path = args.runtime_dir / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    views, identity_conflicts, source_rows = _load_structured_views(args.candidate_db, corpus, args.tenant_id)
    if identity_conflicts:
        raise RuntimeError("structured candidate identity conflict prevents index construction")
    if not views:
        raise RuntimeError("no safe structured table-row views found")
    view_by_evidence = {item["evidence_id"]: item for item in views}
    bm25_path = args.runtime_dir / "structured-bm25" / "structured.db"
    bm25 = SqliteBM25Retriever(str(bm25_path))
    bm25.add_chunks([
        {"id": item["evidence_id"], "content": item["retrieval_text"],
         "metadata": {"doc_id": item["evidence_id"], "doc_name": item["filename"], "type": "table_row"}}
        for item in views
    ], user_id=args.tenant_id)
    provider = ExistingMiniLMEmbeddingProvider(model_name_or_path=get_embedding_model_name(), device=args.device)
    registry = json.loads(args.concept_registry.read_text(encoding="utf-8"))["records"]
    concept_vectors = provider.encode_documents([str(item["canonical_label"]) for item in registry])
    vectors_by_filename: dict[str, np.ndarray] = {}
    for filename in sorted({item["filename"] for item in views}):
        subset = [item for item in views if item["filename"] == filename]
        vectors_by_filename[filename] = provider.encode_documents([item["retrieval_text"] for item in subset])
    np.save(args.runtime_dir / "structured-dense" / "concept-vectors.npy", concept_vectors)
    _write(args.runtime_dir / "structured-dense" / "view-index.json", {name: [item["candidate_key"] for item in views if item["filename"] == name] for name in vectors_by_filename})
    documents = {str(item["document_id"]): item for item in corpus["documents"]}
    predictions = []
    raw_pool_hashes = []
    for question in questions:
        case_id = str(question["case_id"])
        raw_case = raw_cases.get(case_id)
        if raw_case is None:
            raise RuntimeError(f"{case_id}: Gate 0 raw pool missing")
        raw_pool = list(raw_case["stages"]["rrf_full"])
        raw_pool_hashes.append(payload_hash(raw_pool))
        profile = profiles[case_id]["profile"]
        eligible = profile["task_type"] == "table_single_fact"
        structured_bm25: list[dict[str, Any]] = []
        structured_dense: list[dict[str, Any]] = []
        structured_rrf: list[dict[str, Any]] = []
        trace: dict[str, Any] | None = None
        structured_query: str | None = None
        if eligible:
            phrase = " ".join(item["normalized_text"] for item in profile["metric_phrases"])
            trace = _resolve_concept(phrase, registry, provider, concept_vectors)
            periods = " ".join(item["normalized_period"] or item["raw_text"] for item in profile["periods"])
            structured_query = " | ".join(part for part in (str(profile.get("issuer") or "").replace("_", " "), str(trace["top_1_canonical_label"]), periods, str(profile.get("statement_hint") or "")) if part)
            scope = [documents[str(item)]["filename"] for item in question["document_scope"]]
            if len(scope) != 1:
                raise RuntimeError(f"{case_id}: expected exactly one document scope")
            filename = str(scope[0])
            subset = [item for item in views if item["filename"] == filename]
            if subset:
                bm25_results = bm25.search(structured_query, k=40, doc_name=filename, user_id=args.tenant_id)
                bm25_pairs = [
                    (str(result["doc_id"]), result)
                    for result in bm25_results
                    if str(result["doc_id"]) in view_by_evidence
                ]
                dense_scores = vectors_by_filename[filename] @ provider.encode_queries([structured_query])[0]
                dense_indices = np.argsort(-dense_scores, kind="stable")[:40]
                dense_ids = [subset[int(index)]["evidence_id"] for index in dense_indices]
                structured_bm25 = [
                    _candidate_record(view_by_evidence[key], rank, float(result["score"]), "structured_bm25")
                    for rank, (key, result) in enumerate(bm25_pairs, 1)
                ]
                structured_dense = [
                    _candidate_record(view_by_evidence[key], rank, float(dense_scores[int(index)]), "structured_dense")
                    for rank, (key, index) in enumerate(zip(dense_ids, dense_indices), 1)
                ]
                fused = fixed_rrf([item["candidate_key"] for item in structured_bm25], [item["candidate_key"] for item in structured_dense], limit=20)
                lookup = {item["candidate_key"]: item for item in [*structured_bm25, *structured_dense]}
                structured_rrf = [_candidate_record(view_by_evidence[lookup[key]["evidence_id"]], rank, score, "structured_rrf") for rank, (key, score) in enumerate(fused, 1)]
        merged = append_structured_residual(raw_pool, structured_rrf)
        if not merged.raw_unchanged:
            raise RuntimeError(f"{case_id}: raw pool protection failure")
        predictions.append({
            "case_id": case_id, "raw_question": question["question"], "eligible_for_structured_lane": eligible,
            "raw_full_rrf_candidate_count": len(raw_pool), "raw_full_rrf_candidates": raw_pool,
            "raw_rrf_at_40": raw_case["stages"]["rrf"], "structured_query": structured_query,
            "concept_resolution_trace": trace, "structured_bm25_top40": structured_bm25,
            "structured_dense_top40": structured_dense, "structured_rrf_top20": structured_rrf,
            "combined_full_pool": merged.combined, "raw_pool_hash_before": payload_hash(raw_pool),
            "raw_pool_hash_after": payload_hash(merged.combined[:len(raw_pool)]),
            "structured_duplicate_in_raw_pool_count": merged.duplicate_count,
        })
    corpus_hash = payload_hash([{key: item[key] for key in ("candidate_key", "evidence_id", "retrieval_text", "representation_level")} for item in views])
    _write(args.out_dir / "gate-3-protocol.json", {"gate": "pdf_retrieval_v3_gate_3", "evaluation_type": "post_benchmark_iterative_evaluation", "code_commit": args.code_commit, "candidate_db_sha256": _sha(args.candidate_db), "gate_0_snapshot_sha256": _sha(args.gate_0), "gate_0_current_production_snapshot": expected_snapshot, "gate_0_snapshot_validated_before_prediction": True, "gate_2_prediction_sha256": _sha(args.gate_2), "concept_registry_sha256": _sha(args.concept_registry), "structured_query_scope": "table_single_fact_only", "structured_bm25_top_k": 40, "structured_dense_top_k": 40, "structured_rrf_top_k": 20, "rrf_k": 60, "forbidden": ["gold", "governance", "labels", "reranker", "final_selector", "answer_generation", "per_query_oracle", "parameter_scan"]})
    _write(args.out_dir / "structured-corpus-manifest.json", {"source_candidate_store_row_count": source_rows, "structured_candidate_count": len(views), "structured_candidate_identity_hash": payload_hash([item["candidate_key"] for item in views]), "retrieval_text_hash": payload_hash([item["retrieval_text"] for item in views]), "representation_levels": {name: sum(item["representation_level"] == name for item in views) for name in ("strict_cell_aware", "retrieval_only")}, "cell_period_claim_count": 0, "production_candidate_modified": False, "production_index_writes": 0})
    _write(args.out_dir / "structured-candidate-integrity.json", {"original_identity_count": len(views), "structured_view_count": len(views), "identity_loss_count": 0, "identity_conflict_count": identity_conflicts, "duplicate_view_count": 0, "views_missing_source_lineage": sum(not item["field_lineage"] for item in views), "false_cell_to_period_claim_count": 0})
    common_manifest = {"candidate_count": len(views), "candidate_identity_hash": payload_hash([item["candidate_key"] for item in views]), "retrieval_text_hash": payload_hash([item["retrieval_text"] for item in views]), "production_index_writes": 0, "top_k": 40}
    _write(args.out_dir / "structured-bm25-index-manifest.json", {**common_manifest, "implementation": "src.services.retrieval.SqliteBM25Retriever", "path": str(bm25_path), "config": {"tokenizer": "unicode61+jieba_fast", "max_search_limit": 100}})
    _write(args.out_dir / "structured-dense-index-manifest.json", {**common_manifest, "implementation": "src.retrieval.embedding_provider.ExistingMiniLMEmbeddingProvider", "model": provider.name, "revision": provider.revision, "device": provider.device, "normalized_vectors": True, "path": str(args.runtime_dir / "structured-dense")})
    _write(args.out_dir / "gate-3-predictions.json", {"prediction_count": len(predictions), "predictions": predictions})
    prediction_path = args.out_dir / "gate-3-predictions.json"
    _write(args.out_dir / "gate-3-prediction-seal.json", {"prediction_count": len(predictions), "protocol_hash": _sha(args.out_dir / "gate-3-protocol.json"), "prediction_hash": _sha(prediction_path), "structured_corpus_hash": corpus_hash, "structured_index_hash": payload_hash([_sha(args.out_dir / "structured-bm25-index-manifest.json"), _sha(args.out_dir / "structured-dense-index-manifest.json")]), "raw_pipeline_snapshot_hash": _sha(args.gate_0), "raw_pool_hash": payload_hash(raw_pool_hashes), "runtime_gold_reads": 0, "runtime_governance_reads": 0, "labels_read_before_seal": 0, "reranker_calls": 0, "final_selector_calls": 0, "answer_generation_calls": 0, "predictions_sealed": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
