"""Standalone dense index for NF38 Embedding A/B.

This index is intentionally separate from production ChromaDB to guarantee
physical isolation. It stores L2-normalized vectors in a numpy matrix and
performs cosine similarity search via dot product. The index can be saved to
disk as .npz + .json metadata so builds are reproducible and auditable.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.nf38_corpus import CanonicalEvidenceRecord
from src.retrieval.embedding_provider import EmbeddingProvider


@dataclass(frozen=True)
class IndexManifest:
    """Manifest describing a built dense index."""

    provider: str
    model: str
    revision: str
    dimension: int
    max_length: int
    distance_metric: str = "cosine"
    normalized: bool = True
    corpus_hash: str = ""
    record_count: int = 0
    evidence_ids_hash: str = ""
    index_fingerprint: str = ""
    build_time_seconds: float = 0.0
    storage_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "revision": self.revision,
            "dimension": self.dimension,
            "max_length": self.max_length,
            "distance_metric": self.distance_metric,
            "normalized": self.normalized,
            "corpus_hash": self.corpus_hash,
            "record_count": self.record_count,
            "evidence_ids_hash": self.evidence_ids_hash,
            "index_fingerprint": self.index_fingerprint,
            "build_time_seconds": self.build_time_seconds,
            "storage_bytes": self.storage_bytes,
        }


@dataclass
class DenseIndex:
    """In-memory cosine-similarity dense index built from canonical records."""

    provider_name: str
    model: str
    revision: str
    dimension: int
    max_length: int
    corpus_hash: str
    evidence_ids_hash: str
    vectors: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    records: list[CanonicalEvidenceRecord] = field(default_factory=list)
    build_time_seconds: float = 0.0
    storage_bytes: int = 0

    def search(self, query_vector: np.ndarray, k: int = 50) -> list[dict[str, Any]]:
        """Return top-k candidates by cosine similarity.

        Each result dict contains evidence_id, document_id, page, block_type,
        score, and rank — matching the shape expected by nf37_metrics.
        """
        if self.vectors.shape[0] == 0 or k <= 0:
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)
        scores = (self.vectors @ query.T).ravel()

        k = min(k, len(scores))
        # argpartition for top-k, then sort by score descending
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_indices):
            record = self.records[idx]
            results.append(
                {
                    "candidate_id": record.evidence_id,
                    "evidence_id": record.evidence_id,
                    "document_id": record.document_id,
                    "page": record.page,
                    "block_type": record.block_type,
                    "score": float(scores[idx]),
                    "rank": rank,
                }
            )
        return results

    def manifest(self) -> IndexManifest:
        """Build a manifest summarizing this index."""
        fingerprint = _compute_index_fingerprint(
            self.provider_name,
            self.model,
            self.revision,
            self.corpus_hash,
            self.evidence_ids_hash,
            self.vectors,
        )
        return IndexManifest(
            provider=self.provider_name,
            model=self.model,
            revision=self.revision,
            dimension=self.dimension,
            max_length=self.max_length,
            corpus_hash=self.corpus_hash,
            record_count=len(self.records),
            evidence_ids_hash=self.evidence_ids_hash,
            index_fingerprint=fingerprint,
            build_time_seconds=self.build_time_seconds,
            storage_bytes=self.storage_bytes,
        )

    def save(self, path: Path) -> None:
        """Save vectors and metadata to disk."""
        path.mkdir(parents=True, exist_ok=True)
        vectors_path = path / "vectors.npz"
        meta_path = path / "index-manifest.json"

        np.savez(vectors_path, vectors=self.vectors)
        manifest = self.manifest()
        meta_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.storage_bytes = vectors_path.stat().st_size + meta_path.stat().st_size


def build_dense_index(
    records: list[CanonicalEvidenceRecord],
    provider: EmbeddingProvider,
    corpus_hash: str,
    evidence_ids_hash: str,
) -> DenseIndex:
    """Build a DenseIndex from canonical records using the given provider."""
    start = time.monotonic()
    texts = [record.embedding_text for record in records]
    vectors = provider.encode_documents(texts) if texts else np.empty((0, provider.dimension), dtype=np.float32)

    index = DenseIndex(
        provider_name=provider.name,
        model=provider.name,
        revision=provider.revision,
        dimension=provider.dimension,
        max_length=provider.max_length,
        corpus_hash=corpus_hash,
        evidence_ids_hash=evidence_ids_hash,
        vectors=vectors,
        records=records,
        build_time_seconds=time.monotonic() - start,
    )
    return index


def _compute_index_fingerprint(
    provider_name: str,
    model: str,
    revision: str,
    corpus_hash: str,
    evidence_ids_hash: str,
    vectors: np.ndarray,
) -> str:
    """Compute a stable fingerprint for the index."""
    hasher = hashlib.sha256()
    hasher.update(provider_name.encode("utf-8"))
    hasher.update(b"\n")
    hasher.update(model.encode("utf-8"))
    hasher.update(b"\n")
    hasher.update(revision.encode("utf-8"))
    hasher.update(b"\n")
    hasher.update(corpus_hash.encode("utf-8"))
    hasher.update(b"\n")
    hasher.update(evidence_ids_hash.encode("utf-8"))
    hasher.update(b"\n")
    if vectors.size > 0:
        hasher.update(vectors.tobytes())
    return hasher.hexdigest()


def assert_indexes_share_corpus(index_a: DenseIndex, index_b: DenseIndex) -> None:
    """Assert two indexes were built from the same canonical corpus."""
    assert index_a.corpus_hash == index_b.corpus_hash, (
        f"Corpus hash mismatch: {index_a.corpus_hash} vs {index_b.corpus_hash}"
    )
    assert index_a.evidence_ids_hash == index_b.evidence_ids_hash, (
        f"Evidence IDs hash mismatch: {index_a.evidence_ids_hash} vs {index_b.evidence_ids_hash}"
    )
    assert len(index_a.records) == len(index_b.records), (
        f"Record count mismatch: {len(index_a.records)} vs {len(index_b.records)}"
    )
    ids_a = {r.evidence_id for r in index_a.records}
    ids_b = {r.evidence_id for r in index_b.records}
    assert ids_a == ids_b, "Evidence ID sets differ between indexes"
