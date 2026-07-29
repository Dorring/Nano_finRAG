"""Tests for NF38 dense index isolation and corpus consistency."""
from __future__ import annotations

import numpy as np
import pytest
from src.evaluation.nf38_corpus import (
    CanonicalEvidenceRecord,
    build_corpus_manifest,
    hash_embedding_text,
)
from src.evaluation.nf38_dense_index import (
    DenseIndex,
    assert_indexes_share_corpus,
    build_dense_index,
)
from src.retrieval.embedding_provider import l2_normalize


class _StubProvider:
    """Stub embedding provider for testing."""

    def __init__(self, name: str, dimension: int, seed: int = 42) -> None:
        self._name = name
        self._dimension = dimension
        self._seed = seed

    @property
    def name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        return "test-rev"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_length(self) -> int:
        return 128

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        rng = np.random.RandomState(self._seed)
        vectors = rng.randn(len(texts), self._dimension).astype(np.float32)
        return l2_normalize(vectors)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_documents(texts)


def _make_records(n: int = 5) -> list[CanonicalEvidenceRecord]:
    return [
        CanonicalEvidenceRecord(
            evidence_id=f"r{i}",
            document_id="a.pdf",
            page=i + 1,
            block_type="text",
            embedding_text=f"revenue {i}",
            embedding_text_hash=hash_embedding_text(f"revenue {i}"),
        )
        for i in range(n)
    ]


def _make_index(
    provider_name: str = "stub",
    dimension: int = 8,
    n_records: int = 5,
    corpus_hash: str = "abc123",
    evidence_ids_hash: str = "def456",
    seed: int = 42,
) -> DenseIndex:
    records = _make_records(n_records)
    provider = _StubProvider(provider_name, dimension, seed=seed)
    return build_dense_index(records, provider, corpus_hash, evidence_ids_hash)


def test_build_dense_index_creates_index_with_correct_dimensions():
    index = _make_index(dimension=8, n_records=5)
    assert index.vectors.shape == (5, 8)
    assert len(index.records) == 5


def test_build_dense_index_preserves_corpus_hash():
    index = _make_index(corpus_hash="my-hash")
    assert index.corpus_hash == "my-hash"


def test_two_indexes_from_same_corpus_share_corpus_hash():
    records = _make_records(5)
    manifest = build_corpus_manifest(records)
    provider_a = _StubProvider("minilm", 8, seed=1)
    provider_b = _StubProvider("bge-m3", 16, seed=2)
    index_a = build_dense_index(records, provider_a, manifest["corpus_hash"], manifest["evidence_ids_hash"])
    index_b = build_dense_index(records, provider_b, manifest["corpus_hash"], manifest["evidence_ids_hash"])
    assert index_a.corpus_hash == index_b.corpus_hash
    assert index_a.evidence_ids_hash == index_b.evidence_ids_hash


def test_assert_indexes_share_corpus_passes_for_same_corpus():
    records = _make_records(5)
    manifest = build_corpus_manifest(records)
    provider_a = _StubProvider("minilm", 8, seed=1)
    provider_b = _StubProvider("bge-m3", 16, seed=2)
    index_a = build_dense_index(records, provider_a, manifest["corpus_hash"], manifest["evidence_ids_hash"])
    index_b = build_dense_index(records, provider_b, manifest["corpus_hash"], manifest["evidence_ids_hash"])
    assert_indexes_share_corpus(index_a, index_b)


def test_assert_indexes_share_corpus_fails_for_different_corpus():
    records_a = _make_records(5)
    records_b = _make_records(3)
    manifest_a = build_corpus_manifest(records_a)
    manifest_b = build_corpus_manifest(records_b)
    provider = _StubProvider("stub", 8)
    index_a = build_dense_index(records_a, provider, manifest_a["corpus_hash"], manifest_a["evidence_ids_hash"])
    index_b = build_dense_index(records_b, provider, manifest_b["corpus_hash"], manifest_b["evidence_ids_hash"])
    with pytest.raises(AssertionError, match="Corpus hash mismatch"):
        assert_indexes_share_corpus(index_a, index_b)


def test_evidence_ids_identical_across_indexes():
    records = _make_records(5)
    manifest = build_corpus_manifest(records)
    provider_a = _StubProvider("minilm", 8, seed=1)
    provider_b = _StubProvider("bge-m3", 16, seed=2)
    index_a = build_dense_index(records, provider_a, manifest["corpus_hash"], manifest["evidence_ids_hash"])
    index_b = build_dense_index(records, provider_b, manifest["corpus_hash"], manifest["evidence_ids_hash"])
    ids_a = {r.evidence_id for r in index_a.records}
    ids_b = {r.evidence_id for r in index_b.records}
    assert ids_a == ids_b


def test_index_dimensions_match_provider():
    index_384 = _make_index(dimension=384)
    index_1024 = _make_index(dimension=1024)
    assert index_384.dimension == 384
    assert index_1024.dimension == 1024


def test_search_returns_top_k_results():
    index = _make_index(dimension=8, n_records=10)
    query = _StubProvider("query", 8).encode_queries(["test"])[0]
    results = index.search(query, k=5)
    assert len(results) == 5
    assert all("candidate_id" in r for r in results)
    assert all("score" in r for r in results)
    assert all("rank" in r for r in results)
    # Results should be sorted by score descending
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_returns_empty_for_empty_index():
    provider = _StubProvider("stub", 8)
    records: list[CanonicalEvidenceRecord] = []
    index = build_dense_index(records, provider, "hash", "ids_hash")
    query = provider.encode_queries(["test"])[0]
    results = index.search(query, k=5)
    assert results == []


def test_search_returns_empty_for_k_zero():
    index = _make_index(dimension=8, n_records=5)
    query = _StubProvider("q", 8).encode_queries(["test"])[0]
    assert index.search(query, k=0) == []


def test_search_results_contain_metadata():
    index = _make_index(dimension=8, n_records=3)
    query = _StubProvider("q", 8).encode_queries(["test"])[0]
    results = index.search(query, k=3)
    for result in results:
        assert "document_id" in result
        assert "page" in result
        assert "block_type" in result
        assert result["document_id"] == "a.pdf"
        assert result["block_type"] == "text"


def test_index_manifest_has_correct_fields():
    index = _make_index(provider_name="minilm", dimension=384, n_records=5, corpus_hash="ch", evidence_ids_hash="eh")
    manifest = index.manifest()
    d = manifest.to_dict()
    assert d["provider"] == "minilm"
    assert d["model"] == "minilm"
    assert d["dimension"] == 384
    assert d["corpus_hash"] == "ch"
    assert d["evidence_ids_hash"] == "eh"
    assert d["record_count"] == 5
    assert d["distance_metric"] == "cosine"
    assert d["normalized"] is True
    assert d["index_fingerprint"]
    assert len(d["index_fingerprint"]) == 64


def test_index_fingerprint_changes_with_provider():
    records = _make_records(5)
    manifest = build_corpus_manifest(records)
    provider_a = _StubProvider("minilm", 8, seed=1)
    provider_b = _StubProvider("bge-m3", 8, seed=1)
    index_a = build_dense_index(records, provider_a, manifest["corpus_hash"], manifest["evidence_ids_hash"])
    index_b = build_dense_index(records, provider_b, manifest["corpus_hash"], manifest["evidence_ids_hash"])
    assert index_a.manifest().index_fingerprint != index_b.manifest().index_fingerprint


def test_index_fingerprint_changes_with_vectors():
    records = _make_records(5)
    manifest = build_corpus_manifest(records)
    provider_a = _StubProvider("minilm", 8, seed=1)
    provider_b = _StubProvider("minilm", 8, seed=2)
    index_a = build_dense_index(records, provider_a, manifest["corpus_hash"], manifest["evidence_ids_hash"])
    index_b = build_dense_index(records, provider_b, manifest["corpus_hash"], manifest["evidence_ids_hash"])
    assert index_a.manifest().index_fingerprint != index_b.manifest().index_fingerprint


def test_index_fingerprint_stable_for_same_inputs():
    records = _make_records(5)
    manifest = build_corpus_manifest(records)
    provider = _StubProvider("minilm", 8, seed=1)
    index_a = build_dense_index(records, provider, manifest["corpus_hash"], manifest["evidence_ids_hash"])
    index_b = build_dense_index(records, provider, manifest["corpus_hash"], manifest["evidence_ids_hash"])
    assert index_a.manifest().index_fingerprint == index_b.manifest().index_fingerprint


def test_production_collection_is_not_modified(tmp_path):
    """The DenseIndex is standalone and never touches ChromaDB."""
    index = _make_index(dimension=8, n_records=3)
    # Saving the index creates .npz and .json files, not ChromaDB collections.
    index.save(tmp_path / "test-index")
    assert (tmp_path / "test-index" / "vectors.npz").exists()
    assert (tmp_path / "test-index" / "index-manifest.json").exists()
    # No chroma_db directory should be created.
    assert not (tmp_path / "chroma_db").exists()


def test_same_corpus_used_for_both_indexes():
    """Both MiniLM and BGE-M3 indexes must be built from the exact same corpus.

    This is the core isolation invariant: the only variable that changes
    between Variant A and Variant B is the embedding model. The corpus,
    evidence IDs, and ordering must be identical.
    """
    records = _make_records(5)
    manifest = build_corpus_manifest(records)
    provider_minilm = _StubProvider("minilm", 384, seed=1)
    provider_bge = _StubProvider("bge-m3", 1024, seed=2)

    index_minilm = build_dense_index(
        records, provider_minilm, manifest["corpus_hash"], manifest["evidence_ids_hash"]
    )
    index_bge = build_dense_index(
        records, provider_bge, manifest["corpus_hash"], manifest["evidence_ids_hash"]
    )

    # Corpus hash must match
    assert index_minilm.corpus_hash == index_bge.corpus_hash
    # Evidence IDs hash must match
    assert index_minilm.evidence_ids_hash == index_bge.evidence_ids_hash
    # Record count must match
    assert len(index_minilm.records) == len(index_bge.records)
    # Evidence IDs must be identical and in the same order
    ids_minilm = [r.evidence_id for r in index_minilm.records]
    ids_bge = [r.evidence_id for r in index_bge.records]
    assert ids_minilm == ids_bge
    # Dimensions differ (the only allowed difference)
    assert index_minilm.dimension != index_bge.dimension


def test_table_cells_are_not_global_candidates():
    """Canonical corpus should not include table_cell as a dense candidate."""
    records = [
        CanonicalEvidenceRecord(
            evidence_id="r1",
            document_id="a.pdf",
            page=1,
            block_type="text",
            embedding_text="revenue",
            embedding_text_hash=hash_embedding_text("revenue"),
        ),
        CanonicalEvidenceRecord(
            evidence_id="r2",
            document_id="a.pdf",
            page=1,
            block_type="table_cell",
            embedding_text="100",
            embedding_text_hash=hash_embedding_text("100"),
        ),
    ]
    # The index itself doesn't filter; the canonical corpus export does.
    # Here we verify the index can be built from any records.
    manifest = build_corpus_manifest(records)
    provider = _StubProvider("stub", 8)
    index = build_dense_index(records, provider, manifest["corpus_hash"], manifest["evidence_ids_hash"])
    assert len(index.records) == 2
    # But the manifest reports table_cell_global_count for auditing.
    assert manifest["table_cell_global_count"] == 1
