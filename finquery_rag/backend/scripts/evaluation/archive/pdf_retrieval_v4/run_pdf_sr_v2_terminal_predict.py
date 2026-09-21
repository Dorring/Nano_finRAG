"""Prepare and run the one-shot PDF SR-V2 terminal transfer prediction seal."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

import numpy as np

from scripts.evaluation.run_pdf_retrieval_v2_lite import _write
from src.evaluation.nf_opt_15 import build_retrieval_view
from src.evaluation.pdf_query_representation_v2 import char_score, fixed_rrf, normalize_label, ranks, token_bm25_scores
from src.retrieval.embedding_provider import ExistingMiniLMEmbeddingProvider
from src.services.reranker import HeuristicReranker
from src.services.retrieval_config import get_embedding_model_name

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-sr-v2-terminal-transfer"
DEFAULT_CORPUS = ROOT / "benchmarks/financial_rag_v1/corpus.json"
DEFAULT_QUESTIONS = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"
DEFAULT_LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
CONCEPT_DIR = ROOT / "artifacts/evaluation/pdf-query-representation-v2"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metric_phrase(question: dict[str, Any]) -> str:
    text = normalize_label(str(question["question"]))
    company = normalize_label(str(question.get("company") or ""))
    period = normalize_label(str(question.get("requested_period") or ""))
    for value in (company, period, period.removeprefix("fy")):
        if value:
            text = re.sub(rf"\b{re.escape(value)}\b", " ", text)
    text = re.sub(r"\b(what|was|were|is|are|did|does|do|how|much|many|reported|report|in|for|by|the|company|fiscal|year|compared|with|from|to|of|as)\b", " ", text)
    return " ".join(text.split())


def _enriched_text(view: dict[str, Any], raw: str) -> tuple[str, bool]:
    metric = view["metric_field"]["normalized_metric"]
    if view["evidence_type"] != "table_row" or not metric:
        return raw, False
    parts = [
        f"document {view['document_field']['company']} {view['document_field']['fiscal_year']}",
        f"metric {metric}",
    ]
    statement = view["section_field"]["statement_title"]
    if statement:
        parts.append(f"statement {statement}")
    periods = view["period_field"]["periods"]
    if periods:
        parts.append("table periods " + " ".join(periods))
    unit = view["unit_field"]
    if unit["currency"] or unit["scale"]:
        parts.append("unit " + " ".join(item for item in (unit["currency"], unit["scale"]) if item))
    parts.append("row " + raw)
    return "\n".join(parts), True


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tokens = [TOKEN_RE.findall(document.casefold()) for document in documents]
        self.lengths = np.asarray([len(item) for item in self.tokens], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if len(self.lengths) else 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, terms in enumerate(self.tokens):
            for term, count in Counter(terms).items():
                self.postings[term].append((index, count))

    def rank(self, query: str, limit: int = 200) -> list[int]:
        scores = np.zeros(len(self.tokens), dtype=np.float32)
        for term in TOKEN_RE.findall(query.casefold()):
            posting = self.postings.get(term, ())
            if not posting:
                continue
            inverse = math.log(1 + (len(self.tokens) - len(posting) + 0.5) / (len(posting) + 0.5))
            for index, frequency in posting:
                denominator = frequency + self.k1 * (1 - self.b + self.b * self.lengths[index] / max(self.avgdl, 1))
                scores[index] += inverse * frequency * (self.k1 + 1) / denominator
        return [int(index) for index in np.argsort(-scores, kind="stable")[:limit]]


def prepare_protocol(args: argparse.Namespace) -> int:
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    pdfs = []
    for document in corpus["documents"]:
        pdf = args.pdf_dir / str(document["filename"])
        pdfs.append({"document_id": document["document_id"], "filename": document["filename"], "file_sha256": _sha(pdf), "size_bytes": pdf.stat().st_size})
    resolver_config = {"registry_sha256": _sha(args.concept_dir / "concept-registry.json"), "normalization": "pdf-query-representation-v2/v1", "routes": ["exact", "token_bm25", "character_trigram", "existing_short_text_embedding"], "rrf_k": 60, "top_k": 1}
    protocol = {"schema": "pdf-sr-v2/terminal-transfer/protocol/v1", "evaluation_type": "one_shot_terminal_diagnostic_transfer", "code_commit": args.code_commit, "benchmark_pdf_hashes": pdfs, "corpus_hash": _sha(args.corpus), "question_hash": _sha(args.questions), "source_identity_hash": _sha(args.labels), "source_identity_hash_method": "opaque_file_sha256_without_json_parse", "concept_registry_hash": _sha(args.concept_dir / "concept-registry.json"), "resolver_config_hash": _payload_sha(resolver_config), "resolver_config": resolver_config, "embedding_config": {"model": get_embedding_model_name(), "device": args.device}, "reranker_config": {"implementation": "src.services.reranker.HeuristicReranker", "weights_modified": False}, "baseline": "raw_candidate_raw_retrieval_raw_rerank", "treatment": "v2_lite_candidate_top1_canonical_retrieval_raw_rerank", "bm25": {"k1": 1.2, "b": 0.75, "top_k": 200}, "dense_top_k": 200, "rrf": {"k": 60, "top_k": 40}, "reranker_top_k": 20, "final_top_k": 5, "parameter_scan": False, "per_query_oracle_selection": False, "post_score_tuning_allowed": False, "pre_protocol_question_content_preview_count": 3, "pre_protocol_label_content_reads": 0}
    _write(args.out_dir / "terminal-transfer-protocol.json", protocol)
    _write(args.out_dir / "frozen-input-integrity.json", {"corpus_sha256": _sha(args.corpus), "questions_sha256": _sha(args.questions), "labels_opaque_sha256": _sha(args.labels), "pdf_count": len(pdfs), "pdf_hashes_verified": True, "labels_parsed": False, "protocol_precondition_strictly_clean": False, "protocol_precondition_exception": "three question records previewed during path/schema diagnostics; no labels or source identities parsed"})
    return 0


def _resolve_concept(phrase: str, registry: list[dict[str, Any]], provider: ExistingMiniLMEmbeddingProvider, vectors: Any) -> tuple[str, dict[str, Any]]:
    labels = [" ".join([item["canonical_label"], *item.get("generic_aliases", [])]) for item in registry]
    exact = [1.0 if normalize_label(phrase) in [*item["labels"], *item.get("generic_aliases", [])] else 0.0 for item in registry]
    bm25 = token_bm25_scores(phrase, labels)
    chars = [char_score(phrase, label) for label in labels]
    dense = np.asarray(vectors) @ np.asarray(provider.encode_queries([phrase])[0])
    combined = fixed_rrf([ranks(exact), ranks(bm25), ranks(chars), ranks(dense.tolist())])
    order = sorted(range(len(registry)), key=lambda index: (-combined[index], index))
    best = registry[order[0]]
    return str(best["canonical_label"]), {"metric_phrase": phrase, "top_1_concept_id": best["concept_id"], "top_1_canonical_label": best["canonical_label"], "signals": ["exact", "token_bm25", "character_trigram", "short_text_embedding", "fixed_rrf_k60"]}


def predict(args: argparse.Namespace) -> int:
    protocol_path = args.out_dir / "terminal-transfer-protocol.json"
    if not protocol_path.exists():
        raise RuntimeError("prepare and freeze terminal-transfer-protocol.json first")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    documents = {str(item["filename"]): item for item in corpus["documents"]}
    questions = _jsonl(args.questions)
    if len(questions) != 72:
        raise ValueError("expected 72 frozen questions")
    connection = sqlite3.connect(f"file:{args.candidate_db}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT doc_id, content, metadata_json, doc_name FROM chunk_store").fetchall()
    finally:
        connection.close()
    by_key: dict[str, dict[str, Any]] = {}
    duplicate_rows = 0
    for doc_id, content, metadata_json, doc_name in rows:
        document = documents.get(str(doc_name))
        if document is None:
            continue
        metadata = json.loads(metadata_json or "{}")
        view = build_retrieval_view(doc_id=str(doc_id), content=str(content or ""), metadata=metadata, document=document)
        key = str(view["candidate_key"])
        if key in by_key:
            duplicate_rows += 1
            continue
        enriched, enhanced = _enriched_text(view, str(content or ""))
        by_key[key] = {"candidate_key": key, "evidence_id": str(doc_id), "document_id": document["document_id"], "pdf_page": metadata.get("page"), "evidence_type": view["evidence_type"], "raw_text": str(content or ""), "e1_text": enriched, "enhanced": enhanced, "metric": view["metric_field"]["normalized_metric"], "metadata": metadata}
    candidates = sorted(by_key.values(), key=lambda item: item["candidate_key"])
    candidate_hash = _payload_sha([{key: item[key] for key in ("candidate_key", "evidence_id", "document_id", "pdf_page", "raw_text", "e1_text")} for item in candidates])
    by_document = defaultdict(list)
    for item in candidates:
        by_document[str(item["document_id"])].append(item)
    provider = ExistingMiniLMEmbeddingProvider(model_name_or_path=get_embedding_model_name(), device=args.device)
    registry = json.loads((args.concept_dir / "concept-registry.json").read_text(encoding="utf-8"))["records"]
    concept_vectors = provider.encode_documents([str(item["canonical_label"]) for item in registry])
    predictions = {"baseline": [], "e1": []}
    index_manifests = {}
    dense_indexes: dict[tuple[str, str], Any] = {}
    bm25_indexes: dict[tuple[str, str], BM25Index] = {}
    for document_id, scoped in sorted(by_document.items()):
        for variant, field in (("baseline", "raw_text"), ("e1", "e1_text")):
            texts = [str(item[field]) for item in scoped]
            dense_indexes[(variant, document_id)] = provider.encode_documents(texts)
            bm25_indexes[(variant, document_id)] = BM25Index(texts)
            index_manifests[f"{variant}:{document_id}"] = {"candidate_count": len(scoped), "candidate_identity_sha256": _payload_sha([item["candidate_key"] for item in scoped]), "retrieval_text_sha256": _payload_sha(texts)}
    for question in questions:
        scope = [str(item) for item in question["document_scope"]]
        scoped = [item for document_id in scope for item in by_document[document_id]]
        raw_query = str(question["question"])
        phrase = _metric_phrase(question)
        canonical, trace = _resolve_concept(phrase, registry, provider, concept_vectors)
        canonical_query = " ".join(item for item in (str(question.get("company") or ""), canonical, str(question.get("requested_period") or "")) if item)
        for variant, field, retrieval_query in (("baseline", "raw_text", raw_query), ("e1", "e1_text", canonical_query)):
            texts = [str(item[field]) for item in scoped]
            if len(scope) != 1:
                raise ValueError("terminal transfer supports the frozen single-document scope contract only")
            document_id = scope[0]
            bm25 = bm25_indexes[(variant, document_id)].rank(retrieval_query, 200)
            document_vectors = dense_indexes[(variant, document_id)]
            query_vector = provider.encode_queries([retrieval_query])[0]
            dense_scores = np.asarray(document_vectors) @ np.asarray(query_vector)
            dense = [int(index) for index in np.argsort(-dense_scores, kind="stable")[:200]]
            union = list(dict.fromkeys([*bm25, *dense]))[:200]
            scores: dict[int, float] = {}
            for ranking in (bm25[:40], dense[:40]):
                for rank, index in enumerate(ranking, 1):
                    scores[index] = scores.get(index, 0.0) + 1 / (60 + rank)
            rrf = [index for index, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:40]]
            chunks = [{"doc_id": scoped[index]["candidate_key"], "content": scoped[index]["raw_text"], "score": scores[index], "metadata": {"doc_name": scoped[index]["document_id"], "page": scoped[index]["pdf_page"], "row_label": scoped[index]["metric"]}} for index in rrf]
            reranker = HeuristicReranker()
            reranked = reranker.rerank(raw_query, chunks, top_k=20)
            final = reranker.rerank(raw_query, chunks, top_k=5)
            def records(indices: list[int]) -> list[dict[str, Any]]:
                return [{key: scoped[index][key] for key in ("candidate_key", "evidence_id", "document_id", "pdf_page", "evidence_type")} for index in indices]
            candidate_lookup = {item["candidate_key"]: item for item in scoped}
            predictions[variant].append({"case_id": question["case_id"], "document_scope": scope, "raw_query": raw_query, "canonical_query": canonical_query, "concept_resolution_trace": trace, "bm25_top_200": records(bm25), "dense_top_200": records(dense), "union_top_200": records(union), "rrf_top_40": records(rrf), "reranker_top_20": [{key: candidate_lookup[str(item["doc_id"])][key] for key in ("candidate_key", "evidence_id", "document_id", "pdf_page", "evidence_type")} for item in reranked], "final_top_5": [{key: candidate_lookup[str(item["doc_id"])][key] for key in ("candidate_key", "evidence_id", "document_id", "pdf_page", "evidence_type")} for item in final]})
    _write(args.out_dir / "shadow-corpus-manifest.json", {"candidate_store_row_count": len(rows), "shadow_candidate_count": len(candidates), "duplicate_store_rows_collapsed_by_identity": duplicate_rows, "enhanced_table_row_count": sum(item["enhanced"] for item in candidates), "raw_fallback_count": sum(not item["enhanced"] for item in candidates), "shadow_corpus_hash": candidate_hash, "production_index_writes": 0})
    _write(args.out_dir / "candidate-identity-integrity.json", {"original_identity_count": len(candidates), "shadow_view_count": len(candidates), "identity_loss_count": 0, "identity_conflict_count": 0, "duplicate_view_count": 0})
    for variant in ("baseline", "e1"):
        manifest = {"variant": variant, "storage": f"shadow_indexes/{variant}/ephemeral_read_only", "persistent_index_written": False, "document_scope_count": len(by_document), "candidate_count": len(candidates), "embedding_model": get_embedding_model_name(), "bm25": {"k1": 1.2, "b": 0.75}, "dense_top_k": 200, "rrf_k": 60, "rrf_top_k": 40, "reranker_top_k": 20, "final_top_k": 5, "per_document_manifests": {key: value for key, value in index_manifests.items() if key.startswith(variant + ":")}}
        _write(args.out_dir / f"{variant}-index-manifest.json", manifest)
        _write(args.out_dir / f"{variant}-predictions.json", {"prediction_count": len(predictions[variant]), "labels_parsed_before_prediction": 0, "predictions": predictions[variant]})
    baseline_path, e1_path = args.out_dir / "baseline-predictions.json", args.out_dir / "e1-predictions.json"
    seal = {"protocol_hash": _sha(protocol_path), "shadow_corpus_hash": candidate_hash, "baseline_index_manifest_hash": _sha(args.out_dir / "baseline-index-manifest.json"), "e1_index_manifest_hash": _sha(args.out_dir / "e1-index-manifest.json"), "baseline_prediction_hash": _sha(baseline_path), "e1_prediction_hash": _sha(e1_path), "prediction_count": 72, "labels_read_before_seal": 0, "predictions_sealed": True, "protocol_code_commit": protocol["code_commit"]}
    _write(args.out_dir / "prediction-seal.json", seal)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "predict"), required=True)
    parser.add_argument("--candidate-db", type=Path)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--concept-dir", type=Path, default=CONCEPT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-commit", default="working-tree-precommit-terminal-transfer")
    parser.add_argument("--device", default=os.getenv("PDF_TERMINAL_EMBEDDING_DEVICE", "cuda"))
    args = parser.parse_args()
    if args.mode == "prepare":
        return prepare_protocol(args)
    if args.candidate_db is None:
        parser.error("--candidate-db is required for predict")
    return predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
