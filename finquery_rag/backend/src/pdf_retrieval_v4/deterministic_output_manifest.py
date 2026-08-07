"""Deterministic output manifest builder for Gate 02 R2.

Generates a stable, sorted manifest of all MinerU output files across
the corpus.  Files are sorted by (document_id, relative_path) and each
entry includes size_bytes and sha256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.pdf_retrieval_v4.frozen_corpus_manifest import sha256_file


def build_output_manifest(
    output_root: Path,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic manifest of all output files.

    Returns a dict with:
      - document_count
      - page_count
      - files: sorted list of {document_id, relative_path, size_bytes, sha256}
      - manifest_hash: SHA256 of the manifest itself
    """
    files: list[dict[str, Any]] = []
    total_pages = 0

    sorted_docs = sorted(documents, key=lambda d: str(d.get("document_id") or ""))

    for doc in sorted_docs:
        doc_id = str(doc["document_id"])
        total_pages += int(doc.get("page_count", 0))
        doc_dir = output_root / doc_id
        if not doc_dir.is_dir():
            continue
        for f in sorted(doc_dir.rglob("*"), key=lambda p: str(p.relative_to(output_root))):
            if not f.is_file():
                continue
            rel_path = str(f.relative_to(output_root))
            files.append({
                "document_id": doc_id,
                "relative_path": rel_path,
                "size_bytes": f.stat().st_size,
                "sha256": sha256_file(f),
            })

    # Sort by (document_id, relative_path) for determinism
    files.sort(key=lambda f: (f["document_id"], f["relative_path"]))

    manifest = {
        "document_count": len(sorted_docs),
        "page_count": total_pages,
        "files": files,
    }

    # Compute manifest hash
    manifest_str = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest["manifest_hash"] = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()

    return manifest


def compute_manifest_hash(manifest: dict[str, Any]) -> str:
    """Compute the hash of a manifest (excluding the hash field itself)."""
    copy = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    manifest_str = json.dumps(copy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
