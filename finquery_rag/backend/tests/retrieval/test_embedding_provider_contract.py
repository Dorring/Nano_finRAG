"""Contract tests for the EmbeddingProvider interface.

Both ExistingMiniLMEmbeddingProvider and BgeM3DenseEmbeddingProvider must
satisfy this contract. The tests use a stub provider to validate the contract
invariants without requiring real model downloads.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.embedding_provider import (
    EmbeddingOutputError,
    ExistingMiniLMEmbeddingProvider,
    l2_normalize,
    validate_embeddings,
)


class _StubProvider:
    """Minimal in-memory provider for contract tests."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def name(self) -> str:
        return "stub"

    @property
    def revision(self) -> str:
        return "stub-rev"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_length(self) -> int:
        return 128

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        vectors = np.random.RandomState(42).randn(len(texts), self._dimension).astype(np.float32)
        return l2_normalize(vectors)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_documents(texts)


def test_l2_normalize_produces_unit_vectors():
    vectors = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)
    norms = np.linalg.norm(normalized, axis=1)
    assert abs(norms[0] - 1.0) < 1e-6
    assert norms[1] == 0.0


def test_l2_normalize_preserves_row_count():
    vectors = np.random.rand(5, 4).astype(np.float32)
    normalized = l2_normalize(vectors)
    assert normalized.shape == (5, 4)


def test_validate_embeddings_rejects_wrong_ndim():
    with pytest.raises(EmbeddingOutputError, match="2D"):
        validate_embeddings(np.zeros((2,), dtype=np.float32), 2, 4)


def test_validate_embeddings_rejects_wrong_shape():
    with pytest.raises(EmbeddingOutputError, match="shape"):
        validate_embeddings(np.zeros((2, 4), dtype=np.float32), 2, 8)


def test_validate_embeddings_rejects_nan():
    vectors = np.array([[1.0, float("nan")], [2.0, 3.0]], dtype=np.float32)
    with pytest.raises(EmbeddingOutputError, match="NaN"):
        validate_embeddings(vectors, 2, 2)


def test_validate_embeddings_rejects_inf():
    vectors = np.array([[1.0, float("inf")], [2.0, 3.0]], dtype=np.float32)
    with pytest.raises(EmbeddingOutputError, match="Inf"):
        validate_embeddings(vectors, 2, 2)


def test_validate_embeddings_accepts_valid_matrix():
    vectors = np.ones((3, 4), dtype=np.float32)
    validate_embeddings(vectors, 3, 4)


def test_stub_provider_returns_empty_for_empty_input():
    provider = _StubProvider()
    vectors = provider.encode_documents([])
    assert vectors.shape == (0, 8)


def test_stub_provider_returns_correct_count():
    provider = _StubProvider()
    vectors = provider.encode_documents(["a", "b", "c"])
    assert vectors.shape == (3, 8)


def test_stub_provider_output_is_l2_normalized():
    provider = _StubProvider()
    vectors = provider.encode_documents(["a", "b"])
    norms = np.linalg.norm(vectors, axis=1)
    for norm in norms:
        assert abs(norm - 1.0) < 1e-6


def test_stub_provider_is_deterministic():
    provider = _StubProvider()
    first = provider.encode_documents(["revenue", "cash"])
    second = provider.encode_documents(["revenue", "cash"])
    np.testing.assert_array_equal(first, second)


def test_minilm_provider_reports_384_dimension():
    provider = ExistingMiniLMEmbeddingProvider()
    assert provider.dimension == 384


def test_minilm_provider_reports_name():
    provider = ExistingMiniLMEmbeddingProvider(model_name_or_path="all-MiniLM-L6-v2")
    assert provider.name == "all-MiniLM-L6-v2"


def test_minilm_provider_reports_revision():
    provider = ExistingMiniLMEmbeddingProvider()
    assert provider.revision == "production-minilm"


def test_minilm_provider_reports_max_length():
    provider = ExistingMiniLMEmbeddingProvider()
    assert provider.max_length == 256


def test_minilm_provider_returns_empty_for_empty_input():
    provider = ExistingMiniLMEmbeddingProvider()
    vectors = provider.encode_documents([])
    assert vectors.shape == (0, 384)
    assert vectors.dtype == np.float32


def test_minilm_provider_does_not_load_model_on_init():
    """Model loading must be lazy, not at construction time."""
    provider = ExistingMiniLMEmbeddingProvider()
    assert provider._model is None


# --- Integration tests requiring the real MiniLM model ---

def _minilm_model_available() -> bool:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        result = model.encode(["probe"], show_progress_bar=False)
        import numpy as np

        arr = np.asarray(result, dtype=np.float32)
        return arr.ndim == 2 and arr.shape == (1, 384)
    except Exception:
        return False


_skip_no_st = pytest.mark.skipif(
    not _minilm_model_available(),
    reason="MiniLM model not available; requires sentence_transformers and model download",
)


@_skip_no_st
@pytest.mark.integration
def test_minilm_provider_encodes_and_normalizes():
    """Integration-level: the real MiniLM model must produce normalized vectors."""
    provider = ExistingMiniLMEmbeddingProvider()
    vectors = provider.encode_documents(["revenue increased", "net income"])
    assert vectors.shape == (2, 384)
    norms = np.linalg.norm(vectors, axis=1)
    for norm in norms:
        assert abs(norm - 1.0) < 1e-5


@_skip_no_st
@pytest.mark.integration
def test_minilm_query_and_document_encoding_are_deterministic():
    provider = ExistingMiniLMEmbeddingProvider()
    docs = provider.encode_documents(["revenue"])
    queries = provider.encode_queries(["revenue"])
    np.testing.assert_array_almost_equal(docs, queries, decimal=5)


@_skip_no_st
@pytest.mark.integration
def test_cosine_sanity_check():
    """Sanity check: identical texts are more similar than unrelated texts."""
    provider = ExistingMiniLMEmbeddingProvider()
    vectors = provider.encode_documents(["revenue", "revenue", "unrelated weather"])
    same = float(np.dot(vectors[0], vectors[1]))
    different = float(np.dot(vectors[0], vectors[2]))
    assert same > different
