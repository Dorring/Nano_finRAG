"""Embedding provider abstraction for NF38 Dense A/B.

Both the existing MiniLM provider and the BGE-M3 Dense provider implement the
same interface so the evaluation harness can swap embeddings without changing
the retrieval pipeline. The provider is intentionally lazy: model loading
happens on first encode, not at import time.
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    """Interface implemented by all dense embedding providers."""

    @property
    def name(self) -> str:
        ...

    @property
    def revision(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    @property
    def max_length(self) -> int:
        ...

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        ...

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        ...


class EmbeddingOutputError(RuntimeError):
    """Raised when an embedding provider returns malformed output."""


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows. Zero-norm rows are left as zeros to avoid NaN."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return vectors / safe


def validate_embeddings(vectors: np.ndarray, expected_count: int, dimension: int) -> None:
    """Check shape, dimension, and finiteness of an embedding matrix."""
    if vectors.ndim != 2:
        raise EmbeddingOutputError(f"Expected a 2D embedding matrix, got {vectors.ndim}D")
    if vectors.shape != (expected_count, dimension):
        raise EmbeddingOutputError(
            f"Unexpected embedding shape: {vectors.shape}, expected ({expected_count}, {dimension})"
        )
    if not np.isfinite(vectors).all():
        raise EmbeddingOutputError("Embedding matrix contains NaN or Inf values")


class ExistingMiniLMEmbeddingProvider:
    """Wraps the current sentence-transformers MiniLM model.

    This provider delegates to the same SentenceTransformerEmbeddingFunction
    used by production ChromaDB, so the MiniLM variant in NF38 matches the
    current production dense retrieval exactly.
    """

    def __init__(
        self,
        model_name_or_path: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name_or_path
        self._device = device
        self._model: Any | None = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str:
        return "production-minilm"

    @property
    def dimension(self) -> int:
        return 384

    @property
    def max_length(self) -> int:
        return 256

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        model = self._load()
        raw = model.encode(texts, show_progress_bar=False)
        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        validate_embeddings(vectors, len(texts), self.dimension)
        return l2_normalize(vectors)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


class BgeM3DenseEmbeddingProvider:
    """BGE-M3 dense-only embedding provider for NF38 A/B.

    Only dense vectors are returned. Sparse and ColBERT outputs are explicitly
    disabled. The model loads lazily on first encode so module import never
    triggers a download or GPU allocation.
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        model_revision: str | None = None,
        device: str = "cuda:1",
        batch_size: int = 16,
        max_length: int = 512,
        use_fp16: bool = True,
    ) -> None:
        self._model_name = model_name_or_path
        self._model_revision = model_revision
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._use_fp16 = use_fp16
        self._model: Any | None = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str:
        return self._model_revision or "unpinned"

    @property
    def dimension(self) -> int:
        return 1024

    @property
    def max_length(self) -> int:
        return self._max_length

    def _load(self):
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise RuntimeError(
                    "FlagEmbedding is required for BGE-M3 dense evaluation"
                ) from exc
            self._model = BGEM3FlagModel(
                self._model_name,
                devices=[self._device],
                use_fp16=self._use_fp16,
            )
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        model = self._load()
        output = model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vectors = np.asarray(output["dense_vecs"], dtype=np.float32)
        validate_embeddings(vectors, len(texts), self.dimension)
        return l2_normalize(vectors)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)
