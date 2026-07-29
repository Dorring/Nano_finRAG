"""Tests for the BGE-M3 dense-only embedding provider.

These tests validate the BGE-M3 provider contract without requiring a real
model download wherever possible. Tests that need the actual BGE-M3 model are
marked as integration/gpu and skipped when FlagEmbedding is unavailable.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.embedding_provider import (
    BgeM3DenseEmbeddingProvider,
    EmbeddingOutputError,
)


def test_bge_provider_reports_1024_dimension():
    provider = BgeM3DenseEmbeddingProvider()
    assert provider.dimension == 1024


def test_bge_provider_reports_name():
    provider = BgeM3DenseEmbeddingProvider(model_name_or_path="BAAI/bge-m3")
    assert provider.name == "BAAI/bge-m3"


def test_bge_provider_reports_unpinned_revision_by_default():
    provider = BgeM3DenseEmbeddingProvider()
    assert provider.revision == "unpinned"


def test_bge_provider_reports_pinned_revision():
    provider = BgeM3DenseEmbeddingProvider(model_revision="abc123")
    assert provider.revision == "abc123"


def test_bge_provider_reports_max_length_from_config():
    provider = BgeM3DenseEmbeddingProvider(max_length=512)
    assert provider.max_length == 512


def test_bge_provider_does_not_load_model_on_init():
    """Model loading must be lazy, not at construction time."""
    provider = BgeM3DenseEmbeddingProvider()
    assert provider._model is None


def test_bge_provider_returns_empty_for_empty_input():
    provider = BgeM3DenseEmbeddingProvider()
    vectors = provider.encode_documents([])
    assert vectors.shape == (0, 1024)
    assert vectors.dtype == np.float32


class _FakeBgeOutput:
    """Mimics the dict-like output of BGEM3FlagModel.encode."""

    def __init__(self, dense_vecs: np.ndarray) -> None:
        self._data = {"dense_vecs": dense_vecs}

    def __getitem__(self, key: str):
        return self._data[key]


class _FakeBgeModel:
    """Minimal fake of BGEM3FlagModel for unit tests."""

    def __init__(self, dimension: int = 1024) -> None:
        self._dimension = dimension
        self.encode_calls: list[dict] = []

    def encode(
        self,
        texts,
        batch_size=16,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    ):
        self.encode_calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "max_length": max_length,
                "return_dense": return_dense,
                "return_sparse": return_sparse,
                "return_colbert_vecs": return_colbert_vecs,
            }
        )
        vectors = np.random.RandomState(42).randn(len(texts), self._dimension).astype(np.float32)
        return _FakeBgeOutput(dense_vecs=vectors)


def test_bge_provider_uses_fake_model_and_normalizes():
    provider = BgeM3DenseEmbeddingProvider(max_length=512)
    provider._model = _FakeBgeModel(dimension=1024)
    vectors = provider.encode_documents(["revenue", "cash flow"])
    assert vectors.shape == (2, 1024)
    norms = np.linalg.norm(vectors, axis=1)
    for norm in norms:
        assert abs(norm - 1.0) < 1e-6


def test_bge_provider_requests_dense_only_from_model():
    provider = BgeM3DenseEmbeddingProvider(max_length=512)
    fake = _FakeBgeModel(dimension=1024)
    provider._model = fake
    provider.encode_documents(["test"])
    call = fake.encode_calls[-1]
    assert call["return_dense"] is True
    assert call["return_sparse"] is False
    assert call["return_colbert_vecs"] is False


def test_bge_provider_respects_max_length_in_encode_call():
    provider = BgeM3DenseEmbeddingProvider(max_length=1024)
    fake = _FakeBgeModel(dimension=1024)
    provider._model = fake
    provider.encode_documents(["test"])
    call = fake.encode_calls[-1]
    assert call["max_length"] == 1024


def test_bge_provider_respects_batch_size_in_encode_call():
    provider = BgeM3DenseEmbeddingProvider(batch_size=8)
    fake = _FakeBgeModel(dimension=1024)
    provider._model = fake
    provider.encode_documents(["test"])
    call = fake.encode_calls[-1]
    assert call["batch_size"] == 8


def test_bge_provider_encode_queries_uses_same_path_as_documents():
    provider = BgeM3DenseEmbeddingProvider()
    fake = _FakeBgeModel(dimension=1024)
    provider._model = fake
    docs = provider.encode_documents(["revenue"])
    queries = provider.encode_queries(["revenue"])
    # Both paths go through the same _encode, so deterministic fakes match.
    np.testing.assert_array_equal(docs, queries)


def test_bge_provider_rejects_wrong_dimension():
    provider = BgeM3DenseEmbeddingProvider()
    provider._model = _FakeBgeModel(dimension=768)  # wrong dim
    with pytest.raises(EmbeddingOutputError, match="shape"):
        provider.encode_documents(["test"])


def test_bge_provider_rejects_nan_output():
    provider = BgeM3DenseEmbeddingProvider()

    class _NaNModel:
        def encode(self, texts, **kwargs):
            vecs = np.array([[float("nan")] * 1024] * len(texts), dtype=np.float32)
            return _FakeBgeOutput(dense_vecs=vecs)

    provider._model = _NaNModel()
    with pytest.raises(EmbeddingOutputError, match="NaN"):
        provider.encode_documents(["test"])


def test_bge_provider_rejects_inf_output():
    provider = BgeM3DenseEmbeddingProvider()

    class _InfModel:
        def encode(self, texts, **kwargs):
            vecs = np.array([[float("inf")] * 1024] * len(texts), dtype=np.float32)
            return _FakeBgeOutput(dense_vecs=vecs)

    provider._model = _InfModel()
    with pytest.raises(EmbeddingOutputError, match="Inf"):
        provider.encode_documents(["test"])


def test_bge_provider_is_deterministic_with_same_fake_model():
    provider = BgeM3DenseEmbeddingProvider()
    provider._model = _FakeBgeModel(dimension=1024)
    first = provider.encode_documents(["revenue", "cash"])
    provider._model = _FakeBgeModel(dimension=1024)
    second = provider.encode_documents(["revenue", "cash"])
    np.testing.assert_array_equal(first, second)


def test_bge_provider_no_case_specific_embedding_logic():
    """Provider must not hardcode any case/document-specific behavior.

    The provider is a pure embedding function: it maps text -> vector with no
    awareness of case_id, document_id, or question bucket. We verify this by
    checking that the constructor accepts no case/document parameters.
    """
    import inspect

    sig = inspect.signature(BgeM3DenseEmbeddingProvider.__init__)
    param_names = set(sig.parameters.keys()) - {"self"}
    forbidden = {"case_id", "document_id", "question", "bucket", "doc_name"}
    assert not (param_names & forbidden), (
        f"BGE provider must not accept case/document-specific params: {param_names & forbidden}"
    )


def test_bge_provider_no_document_specific_embedding_logic():
    """encode_documents must treat all texts identically regardless of source."""
    provider = BgeM3DenseEmbeddingProvider()
    fake = _FakeBgeModel(dimension=1024)
    provider._model = fake
    # Texts from different documents are encoded the same way.
    provider.encode_documents(["doc_a revenue", "doc_b revenue"])
    call = fake.encode_calls[-1]
    # No per-document metadata is passed to the model.
    assert "documents" not in call
    assert "doc_name" not in call
    assert "document_id" not in call


# --- Integration tests requiring the real BGE-M3 model ---

def _flag_embedding_available() -> bool:
    try:
        import FlagEmbedding  # noqa: F401

        return True
    except ImportError:
        return False


_skip_no_flag_embedding = pytest.mark.skipif(
    not _flag_embedding_available(),
    reason="FlagEmbedding not installed; requires GPU and model download",
)


@_skip_no_flag_embedding
@pytest.mark.integration
@pytest.mark.gpu
def test_bge_provider_returns_dense_only():
    """The real BGE-M3 model must return only dense vectors."""
    provider = BgeM3DenseEmbeddingProvider(device="cuda:1")
    vectors = provider.encode_documents(["revenue increased 10%"])
    assert vectors.shape == (1, 1024)
    assert vectors.dtype == np.float32


@_skip_no_flag_embedding
@pytest.mark.integration
@pytest.mark.gpu
def test_bge_embedding_dimension_is_1024():
    provider = BgeM3DenseEmbeddingProvider(device="cuda:1")
    vectors = provider.encode_documents(["test"])
    assert vectors.shape[1] == 1024


@_skip_no_flag_embedding
@pytest.mark.integration
@pytest.mark.gpu
def test_bge_embeddings_are_finite():
    provider = BgeM3DenseEmbeddingProvider(device="cuda:1")
    vectors = provider.encode_documents(["revenue", "cash flow", "net income"])
    assert np.isfinite(vectors).all()


@_skip_no_flag_embedding
@pytest.mark.integration
@pytest.mark.gpu
def test_bge_embeddings_are_l2_normalized():
    provider = BgeM3DenseEmbeddingProvider(device="cuda:1")
    vectors = provider.encode_documents(["revenue", "cash flow"])
    norms = np.linalg.norm(vectors, axis=1)
    for norm in norms:
        assert abs(norm - 1.0) < 1e-5


@_skip_no_flag_embedding
@pytest.mark.integration
@pytest.mark.gpu
def test_bge_query_and_document_encoding_are_deterministic():
    provider = BgeM3DenseEmbeddingProvider(device="cuda:1")
    docs = provider.encode_documents(["revenue"])
    queries = provider.encode_queries(["revenue"])
    np.testing.assert_array_almost_equal(docs, queries, decimal=4)


@_skip_no_flag_embedding
@pytest.mark.integration
@pytest.mark.gpu
def test_bge_cosine_sanity_check():
    """Sanity: identical texts are more similar than unrelated texts."""
    provider = BgeM3DenseEmbeddingProvider(device="cuda:1")
    vectors = provider.encode_documents(["revenue", "revenue", "unrelated weather"])
    same = float(np.dot(vectors[0], vectors[1]))
    different = float(np.dot(vectors[0], vectors[2]))
    assert same > different
