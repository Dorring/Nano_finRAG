"""Build two isolated dense indexes for NF38 Embedding A/B.

Reads the canonical corpus from artifacts/evaluation/nf38/canonical-records.jsonl,
builds a MiniLM index and a BGE-M3 index, and saves both to disk with manifests.

Usage:
    python -m scripts.evaluation.build_nf38_embedding_indexes \
        --corpus-dir artifacts/evaluation/nf38 \
        --out-dir artifacts/evaluation/nf38/indexes \
        --bge-device cuda:1

The script writes:
    indexes/minilm/vectors.npz + index-manifest.json
    indexes/bge-m3/vectors.npz + index-manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from src.evaluation.nf38_corpus import (  # noqa: E402
    CanonicalEvidenceRecord,
    hash_embedding_text,
)
from src.retrieval.embedding_provider import (  # noqa: E402
    BgeM3DenseEmbeddingProvider,
    ExistingMiniLMEmbeddingProvider,
)
from src.evaluation.nf38_dense_index import (  # noqa: E402
    assert_indexes_share_corpus,
    build_dense_index,
)


def load_canonical_records(corpus_dir: Path) -> tuple[list[CanonicalEvidenceRecord], dict]:
    """Load canonical records and manifest from the corpus directory."""
    records_path = corpus_dir / "canonical-records.jsonl"
    manifest_path = corpus_dir / "corpus-manifest.json"

    if not records_path.exists():
        raise FileNotFoundError(f"Canonical records not found: {records_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[CanonicalEvidenceRecord] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        # embedding_text is not stored in the JSONL (only the hash).
        # We need to reload from ChromaDB to get the actual text.
        # For now, use the hash as a placeholder.
        records.append(
            CanonicalEvidenceRecord(
                evidence_id=data["evidence_id"],
                document_id=data["document_id"],
                page=data.get("page"),
                block_type=data["block_type"],
                parent_id=data.get("parent_id"),
                table_id=data.get("table_id"),
                section_path=tuple(data.get("section_path") or []),
                embedding_text="",  # Will be loaded separately
                embedding_text_hash=data.get("embedding_text_hash", ""),
            )
        )
    return records, manifest


def load_embedding_texts(chroma_path: str, collection_name: str) -> dict[str, str]:
    """Load embedding texts from ChromaDB, keyed by evidence_id."""
    import chromadb

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name=collection_name)
    data = collection.get(include=["documents"])
    return {doc_id: doc or "" for doc_id, doc in zip(data["ids"], data["documents"])}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NF38 dense indexes")
    parser.add_argument("--corpus-dir", required=True, help="Directory with canonical-records.jsonl")
    parser.add_argument("--out-dir", required=True, help="Output directory for indexes")
    parser.add_argument("--chroma-path", default=None, help="ChromaDB path for loading embedding texts")
    parser.add_argument("--collection-name", default="rag_global_knowledge_base")
    parser.add_argument("--bge-device", default="cuda:1")
    parser.add_argument("--bge-max-length", type=int, default=None, help="Override BGE max_length (default: auto)")
    parser.add_argument("--skip-bge", action="store_true", help="Skip BGE-M3 index (MiniLM only)")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out_dir)

    records, corpus_manifest = load_canonical_records(corpus_dir)

    # Load embedding texts from ChromaDB
    chroma_path = args.chroma_path or str(BACKEND / "chroma_db")
    print(f"Loading embedding texts from {chroma_path}...")
    text_map = load_embedding_texts(chroma_path, args.collection_name)

    # Fill in embedding texts
    filled_records: list[CanonicalEvidenceRecord] = []
    missing = 0
    for record in records:
        text = text_map.get(record.evidence_id, "")
        if not text:
            missing += 1
        filled_records.append(
            CanonicalEvidenceRecord(
                evidence_id=record.evidence_id,
                document_id=record.document_id,
                page=record.page,
                block_type=record.block_type,
                parent_id=record.parent_id,
                table_id=record.table_id,
                section_path=record.section_path,
                embedding_text=text,
                embedding_text_hash=hash_embedding_text(text),
            )
        )
    if missing:
        print(f"WARNING: {missing} records have no embedding text")

    corpus_hash = corpus_manifest["corpus_hash"]
    evidence_ids_hash = corpus_manifest["evidence_ids_hash"]

    # Build MiniLM index
    print("Building MiniLM index...")
    minilm_provider = ExistingMiniLMEmbeddingProvider()
    minilm_index = build_dense_index(filled_records, minilm_provider, corpus_hash, evidence_ids_hash)
    minilm_dir = out_dir / "minilm"
    minilm_index.save(minilm_dir)
    print(f"MiniLM index saved to {minilm_dir} ({minilm_index.manifest().record_count} records)")

    if args.skip_bge:
        print("Skipping BGE-M3 index (--skip-bge)")
        return 0

    # Build BGE-M3 index
    max_length = args.bge_max_length or 512
    print(f"Building BGE-M3 index (max_length={max_length}, device={args.bge_device})...")
    bge_provider = BgeM3DenseEmbeddingProvider(
        device=args.bge_device,
        max_length=max_length,
    )
    bge_index = build_dense_index(filled_records, bge_provider, corpus_hash, evidence_ids_hash)
    bge_dir = out_dir / "bge-m3"
    bge_index.save(bge_dir)
    print(f"BGE-M3 index saved to {bge_dir} ({bge_index.manifest().record_count} records)")

    # Verify corpus isolation
    print("Verifying corpus isolation...")
    assert_indexes_share_corpus(minilm_index, bge_index)
    print("OK: both indexes share the same canonical corpus")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
