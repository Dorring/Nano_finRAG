#!/usr/bin/env python3
"""T2-01 standard whole-context BM25/Dense/Hybrid retrieval prediction."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


EXPECTED_ROWS = 23_088
SUBSETS = ("FinQA", "ConvFinQA", "TAT-DQA")
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_SNAPSHOT = Path(
    "/home/mxf/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
) / "snapshots" / MODEL_REVISION
# The host's installed CUDA driver (12080) is older than the bundled torch
# CUDA runtime, so the reproducible runtime for this public baseline is CPU.
# This changes only execution placement, not the encoder/model contract.
DEVICE = "cpu"
EMBED_BATCH_SIZE = 64
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
TOP_K = 100
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def metadata_paths(root: Path) -> list[tuple[str, str, Path]]:
    result = [
        ("FinQA", split, root / "data" / "FinQA" / split / "metadata.jsonl")
        for split in ("train", "dev", "test")
    ]
    result.append(("ConvFinQA", "turn_0", root / "data" / "ConvFinQA" / "turn_0.jsonl"))
    result.extend(
        ("TAT-DQA", split, root / "data" / "TAT-DQA" / split / "metadata.jsonl")
        for split in ("train", "dev", "test")
    )
    return result


def query_text(row: dict[str, Any]) -> str:
    # This deliberately mirrors the published implementation, including the
    # behavior for empty strings and None; no fallback or normalization is
    # permitted in the standard track.
    return f"{row.get('company_name')} : {row.get('question')}"


def load_subset(root: Path, subset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = [(name, split, path) for name, split, path in metadata_paths(root) if name == subset]
    rows: list[dict[str, Any]] = []
    for _, default_split, path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                rows.append(
                    {
                        "query_id": str(source["id"]),
                        "subset": subset,
                        "split": source.get("split", default_split),
                        "query": query_text(source),
                        "company_name": source.get("company_name"),
                        "context_id": str(source["context_id"]),
                        "context": str(source.get("context") or ""),
                        "file_name": source.get("file_name"),
                    }
                )
    corpus_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        context_id = row["context_id"]
        existing = corpus_by_id.get(context_id)
        if existing is not None and existing["context"] != row["context"]:
            raise RuntimeError(f"context_id_content_conflict:{subset}:{context_id}")
        if existing is None:
            corpus_by_id[context_id] = {
                "context_id": context_id,
                "context": row["context"],
                "file_name": row["file_name"],
            }
    return rows, [corpus_by_id[key] for key in sorted(corpus_by_id)]


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


class FixedBM25:
    def __init__(self, documents: list[str]) -> None:
        self.k1 = BM25_K1
        self.b = BM25_B
        self.documents = documents
        self.doc_lens = np.asarray([len(tokenize(doc)) for doc in documents], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, doc in enumerate(documents):
            counts = Counter(tokenize(doc))
            for token, term_frequency in counts.items():
                self.postings[token].append((doc_idx, term_frequency))
        self.idf = {
            token: math.log((len(documents) - len(posting) + 0.5) / (len(posting) + 0.5) + 1.0)
            for token, posting in self.postings.items()
        }

    def rank(self, query: str, context_ids: list[str]) -> list[dict[str, Any]]:
        scores: dict[int, float] = defaultdict(float)
        query_tokens = set(tokenize(query))
        for token in query_tokens:
            posting = self.postings.get(token)
            if not posting:
                continue
            idf = self.idf[token]
            for doc_idx, term_frequency in posting:
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * float(self.doc_lens[doc_idx]) / max(self.avgdl, 1e-12)
                )
                scores[doc_idx] += idf * term_frequency * (self.k1 + 1.0) / denominator
        nonzero = sorted(scores, key=lambda idx: (-scores[idx], context_ids[idx]))
        zero = [idx for idx in range(len(context_ids)) if idx not in scores]
        zero.sort(key=lambda idx: context_ids[idx])
        ordered = (nonzero + zero)[: min(TOP_K, len(context_ids))]
        return [
            {"context_id": context_ids[idx], "rank": rank, "score": float(scores.get(idx, 0.0))}
            for rank, idx in enumerate(ordered, start=1)
        ]


def dense_rank(
    model: SentenceTransformer,
    queries: list[str],
    documents: list[str],
    context_ids: list[str],
) -> list[list[dict[str, Any]]]:
    document_embeddings = model.encode(
        documents,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    query_embeddings = model.encode(
        queries,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    result: list[list[dict[str, Any]]] = []
    cutoff = min(TOP_K, len(context_ids))
    for start in range(0, len(query_embeddings), 256):
        batch = query_embeddings[start : start + 256] @ document_embeddings.T
        for row in batch:
            candidate_indices = np.argpartition(-row, cutoff - 1)[:cutoff]
            candidate_indices = sorted(
                (int(idx) for idx in candidate_indices),
                key=lambda idx: (-float(row[idx]), context_ids[idx]),
            )
            result.append(
                [
                    {"context_id": context_ids[idx], "rank": rank, "score": float(row[idx])}
                    for rank, idx in enumerate(candidate_indices, start=1)
                ]
            )
    return result


def hybrid_rank(
    bm25: list[dict[str, Any]], dense: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for item in bm25:
        entry = candidates.setdefault(item["context_id"], {"score": 0.0, "best_rank": item["rank"]})
        entry["score"] += 1.0 / (RRF_K + item["rank"])
        entry["best_rank"] = min(entry["best_rank"], item["rank"])
    for item in dense:
        entry = candidates.setdefault(item["context_id"], {"score": 0.0, "best_rank": item["rank"]})
        entry["score"] += 1.0 / (RRF_K + item["rank"])
        entry["best_rank"] = min(entry["best_rank"], item["rank"])
    ordered = sorted(candidates, key=lambda key: (-candidates[key]["score"], candidates[key]["best_rank"], key))[:TOP_K]
    return [
        {"context_id": key, "rank": rank, "score": float(candidates[key]["score"])}
        for rank, key in enumerate(ordered, start=1)
    ]


def model_file_manifest() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    if MODEL_SNAPSHOT.is_dir():
        for path in sorted(p for p in MODEL_SNAPSHOT.rglob("*") if p.is_file()):
            files[str(path.relative_to(MODEL_SNAPSHOT))] = {
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "snapshot": str(MODEL_SNAPSHOT), "files": files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--closure-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    closure = args.closure_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    closure_acceptance_path = closure / "acceptance.json"
    closure_acceptance = json.loads(closure_acceptance_path.read_text(encoding="utf-8"))
    if not closure_acceptance.get("published_raw_track_ready"):
        raise RuntimeError("t2_00r1_not_closed")
    if closure_acceptance.get("headline_denominator") != EXPECTED_ROWS:
        raise RuntimeError("t2_denominator_contract")

    all_query_records: list[dict[str, Any]] = []
    corpus_summary: dict[str, Any] = {}
    query_manifest_path = output / "query-manifest.jsonl.gz"
    with gzip.open(query_manifest_path, "wt", encoding="utf-8", compresslevel=6) as query_handle:
        for subset in SUBSETS:
            rows, corpus = load_subset(dataset, subset)
            if not rows:
                raise RuntimeError(f"empty_subset:{subset}")
            all_query_records.extend(rows)
            docs = [str(item["context"]) for item in corpus]
            context_ids = [str(item["context_id"]) for item in corpus]
            corpus_summary[subset] = {
                "query_count": len(rows),
                "context_count": len(corpus),
                "context_ids": context_ids,
                "documents": [
                    {
                        "context_id": item["context_id"],
                        "file_name": item["file_name"],
                        "context_sha256": hashlib.sha256(item["context"].encode("utf-8")).hexdigest(),
                        "char_count": len(item["context"]),
                    }
                    for item in corpus
                ],
            }
            for row in rows:
                query_handle.write(
                    json.dumps(
                        {
                            "query_id": row["query_id"],
                            "subset": subset,
                            "split": row["split"],
                            "query": row["query"],
                            "query_sha256": hashlib.sha256(row["query"].encode("utf-8")).hexdigest(),
                            "company_name": row["company_name"],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    if len(all_query_records) != EXPECTED_ROWS:
        raise RuntimeError(f"query_count_contract:{len(all_query_records)}")

    model = SentenceTransformer(str(MODEL_SNAPSHOT), device=DEVICE, local_files_only=True)
    predictions_paths = {
        name: output / f"{name}-predictions.jsonl.gz"
        for name in ("bm25", "dense", "hybrid")
    }
    handles = {name: gzip.open(path, "wt", encoding="utf-8", compresslevel=6) for name, path in predictions_paths.items()}
    try:
        for subset in SUBSETS:
            rows, corpus = load_subset(dataset, subset)
            documents = [str(item["context"]) for item in corpus]
            context_ids = [str(item["context_id"]) for item in corpus]
            bm25_engine = FixedBM25(documents)
            bm25_lists = [bm25_engine.rank(row["query"], context_ids) for row in rows]
            dense_lists = dense_rank(model, [row["query"] for row in rows], documents, context_ids)
            for row, bm25_list, dense_list in zip(rows, bm25_lists, dense_lists):
                common = {
                    "query_id": row["query_id"],
                    "subset": subset,
                    "precomputed_candidate_count": len(context_ids),
                    "candidate_count": len(bm25_list),
                }
                handles["bm25"].write(json.dumps({**common, "ranked_contexts": bm25_list}, separators=(",", ":")) + "\n")
                handles["dense"].write(json.dumps({**common, "ranked_contexts": dense_list}, separators=(",", ":")) + "\n")
                handles["hybrid"].write(
                    json.dumps({**common, "ranked_contexts": hybrid_rank(bm25_list, dense_list)}, separators=(",", ":")) + "\n"
                )
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "dataset_commit": closure_acceptance["dataset_commit"],
        "closure_acceptance_sha256": sha256(closure_acceptance_path),
        "query_count": len(all_query_records),
        "corpus_count": sum(item["context_count"] for item in corpus_summary.values()),
        "subset_query_counts": {key: value["query_count"] for key, value in corpus_summary.items()},
        "subset_context_counts": {key: value["context_count"] for key, value in corpus_summary.items()},
        "query_manifest_sha256": sha256(query_manifest_path),
        "prediction_files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in predictions_paths.items()},
        "candidate_budget": TOP_K,
    }
    write_json(output / "corpus-manifest.json", {"corpora": corpus_summary})
    write_json(output / "prediction-manifest.json", manifest)
    write_json(
        output / "protocol.json",
        {
            "gate": "t2_ragbench_01_standard_retrieval",
            "dataset_repo": "G4KMU/t2-ragbench",
            "dataset_commit": closure_acceptance["dataset_commit"],
            "published_rows": EXPECTED_ROWS,
            "gold_unit": "context_id",
            "retrieval_unit": "whole_published_context",
            "query_template": "f'{company_name} : {question}'",
            "subsets": list(SUBSETS),
            "bm25": {"k1": BM25_K1, "b": BM25_B, "tokenizer": "casefold ASCII alphanumeric regex"},
            "dense": {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "device": DEVICE,
                "batch_size": EMBED_BATCH_SIZE,
                "normalize_embeddings": True,
                "instruction": None,
                "model_file_manifest": model_file_manifest(),
            },
            "hybrid": {"method": "RRF", "k": RRF_K, "component_top_k": TOP_K},
            "candidate_top_k": TOP_K,
            "pdf_parsing": 0,
            "chunking": 0,
            "query_plan": 0,
            "query_rewrite": 0,
            "hyde": 0,
            "cross_encoder": 0,
            "llm": 0,
            "parameter_scan": False,
            "gold_driven_scan": False,
            "gold_scoring_reads_before_seal": 0,
        },
    )
    write_json(
        output / "input-integrity.json",
        {
            "dataset_commit": closure_acceptance["dataset_commit"],
            "closure_acceptance_sha256": sha256(closure_acceptance_path),
            "published_rows": EXPECTED_ROWS,
            "query_count": len(all_query_records),
            "context_mutation": 0,
            "query_mutation": 0,
            "gold_scoring_reads_before_seal": 0,
        },
    )
    write_json(
        output / "prediction-seal.json",
        {
            "sealed": True,
            "gate": "t2_ragbench_01_standard_retrieval",
            "prediction_count": len(all_query_records),
            "output_sha256": {name: sha256(path) for name, path in predictions_paths.items()},
            "query_manifest_sha256": sha256(query_manifest_path),
            "candidate_budget": TOP_K,
            "gold_scoring_reads_before_seal": 0,
            "pdf_parsing": 0,
            "chunking": 0,
            "cross_encoder": 0,
            "llm": 0,
            "parameter_scan": False,
            "query_rewrite": 0,
            "query_plan": 0,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

