"""Export a frozen canonical evidence corpus for NF38 Embedding A/B.

The corpus is read from the current production ChromaDB collection (the same
dense records MiniLM uses). Both MiniLM and BGE-M3 indexes must be built from
this canonical corpus so the only experimental variable is the embedding model.

Usage:
    python -m scripts.evaluation.export_nf38_canonical_corpus \
        --out-dir artifacts/evaluation/nf38

The script writes:
    - canonical-records.jsonl: one record per line (metadata only, no full text)
    - corpus-manifest.json: corpus_hash, record_count, block_type_counts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

import chromadb  # noqa: E402

from src.evaluation.nf38_corpus import (  # noqa: E402
    CanonicalEvidenceRecord,
    build_corpus_manifest,
    hash_embedding_text,
)

CHROMA_PATH = os.getenv("CHROMA_PATH", str(BACKEND / "chroma_db"))
COLLECTION_NAME = "rag_global_knowledge_base"


def _build_record(doc_id: str, content: str, metadata: dict) -> CanonicalEvidenceRecord:
    section_raw = metadata.get("section_path") or metadata.get("section_title") or ""
    if isinstance(section_raw, str):
        section_parts = tuple(part.strip() for part in section_raw.split(">") if part.strip())
    else:
        section_parts = tuple(str(part).strip() for part in section_raw if str(part).strip())

    return CanonicalEvidenceRecord(
        evidence_id=doc_id,
        document_id=metadata.get("doc_name") or "",
        page=metadata.get("page"),
        block_type=metadata.get("type") or "text",
        parent_id=metadata.get("parent_id"),
        table_id=metadata.get("table_id"),
        section_path=section_parts,
        embedding_text=content,
        embedding_text_hash=hash_embedding_text(content or ""),
    )


def export_canonical_corpus(out_dir: Path, chroma_path: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_path = chroma_path or CHROMA_PATH
    client = chromadb.PersistentClient(path=resolved_path)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    data = collection.get(include=["metadatas", "documents"])

    records: list[CanonicalEvidenceRecord] = []
    for doc_id, content, metadata in zip(data["ids"], data["documents"], data["metadatas"]):
        records.append(_build_record(doc_id, content or "", metadata or {}))

    records.sort(key=lambda r: r.evidence_id)

    records_path = out_dir / "canonical-records.jsonl"
    with records_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    manifest = build_corpus_manifest(records)
    manifest_path = out_dir / "corpus-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Exported {len(records)} canonical records to {records_path}")
    print(f"Corpus hash: {manifest['corpus_hash']}")
    print(f"Block types: {manifest['block_type_counts']}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chroma-path", default=CHROMA_PATH)
    args = parser.parse_args()

    export_canonical_corpus(Path(args.out_dir), chroma_path=args.chroma_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
