"""Reranking interfaces for FinQuery retrieval.

The default production path keeps reranking disabled. This module provides a
small dependency-free interface so cross-encoder reranking can be added later
without changing the RAG pipeline shape.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Protocol


class Reranker(Protocol):
    """Protocol implemented by all rerankers."""

    name: str

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
        """Return chunks sorted by reranker relevance."""


@dataclass
class NoopReranker:
    """Preserve retrieval order. Useful as the default / disabled reranker."""

    name: str = "noop"

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
        limit = top_k if top_k is not None else len(chunks)
        return list(chunks)[:limit]


@dataclass
class HeuristicReranker:
    """Dependency-free lexical reranker for deterministic tests and fallback.

    Score combines original retrieval score with query-token overlap. It is not
    a substitute for a cross-encoder, but gives the pipeline a stable reranker
    contract without model downloads or new dependencies.
    """

    original_score_weight: float = 0.7
    lexical_weight: float = 0.3
    name: str = "heuristic"

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
        if not chunks:
            return []

        query_terms = _tokenize(query)
        query_phrases = _query_phrases(query)
        scored = []
        for index, chunk in enumerate(chunks):
            original_score = _safe_float(chunk.get("score", 0.0))
            lexical_score = _lexical_overlap(query_terms, chunk.get("content", ""), query_phrases)
            rerank_score = (
                self.original_score_weight * original_score
                + self.lexical_weight * lexical_score
            )
            item = dict(chunk)
            item["rerank_score"] = rerank_score
            item["reranker"] = self.name
            scored.append((rerank_score, original_score, -index, item))

        scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        ordered = [item for _, _, _, item in scored]
        if top_k is not None:
            return ordered[:top_k]
        return ordered



@dataclass
class CrossEncoderReranker:
    """Optional cross-encoder reranker with lazy model loading.

    This reranker is only constructed when explicitly configured. A model name
    or local path is required so production does not accidentally download a
    model at startup.
    """

    model_name_or_path: str
    model: Any | None = None
    name: str = "cross-encoder"

    def __post_init__(self):
        if not self.model_name_or_path and self.model is None:
            raise ValueError("CrossEncoderReranker requires a model name or local path")

    def _get_model(self):
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for cross-encoder reranking"
                ) from exc
            self.model = CrossEncoder(self.model_name_or_path)
        return self.model

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
        if not chunks:
            return []

        pairs = [(query, chunk.get("content", "")) for chunk in chunks]
        raw_scores = self._get_model().predict(pairs)
        scored = []
        for index, (chunk, score) in enumerate(zip(chunks, raw_scores)):
            item = dict(chunk)
            item["rerank_score"] = _safe_float(score)
            item["reranker"] = self.name
            scored.append((item["rerank_score"], _safe_float(item.get("score", 0.0)), -index, item))

        scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        ordered = [item for _, _, _, item in scored]
        if top_k is not None:
            return ordered[:top_k]
        return ordered

def build_reranker(
    name: str | None,
    model_name_or_path: str | None = None,
) -> Reranker | None:
    """Build a reranker from config name.

    `None`, empty, "none", and "noop" all mean disabled/no-op.
    Cross-encoder reranking must be explicitly configured with a model path.
    """
    normalized = (name or "none").strip().lower()
    if normalized in {"", "none", "off", "disabled"}:
        return None
    if normalized == "noop":
        return NoopReranker()
    if normalized == "heuristic":
        return HeuristicReranker()
    if normalized in {"cross-encoder", "cross_encoder", "crossencoder"}:
        if not model_name_or_path:
            raise ValueError("RAG_RERANKER_MODEL is required for cross-encoder reranking")
        return CrossEncoderReranker(model_name_or_path=model_name_or_path)
    raise ValueError(f"Unknown reranker: {name}")


def _tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text or "")
        if token.strip()
    }


def _query_phrases(text: str) -> set[str]:
    """Return specific adjacent query phrases for dependency-free reranking.

    Token overlap alone treats a common word such as cash as equally useful
    in a note title and in the requested line item. Adjacent content-bearing
    terms preserve more of the user's retrieval intent while remaining
    language- and document-agnostic.
    """
    stopwords = {
        "what", "was", "were", "the", "and", "for", "with", "from", "this",
        "that", "how", "much", "many", "did", "does", "have", "has", "of",
        "in", "on", "at", "as", "to", "a", "an", "report", "document",
    }
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", text or "")
        if token.lower() not in stopwords and len(token) > 1
    ]
    return {
        f"{left} {right}"
        for left, right in zip(tokens, tokens[1:])
        if left != right
    }


def _lexical_overlap(
    query_terms: set[str],
    content: str,
    query_phrases: set[str] | None = None,
) -> float:
    if not query_terms:
        return 0.0
    content_terms = _tokenize(content)
    if not content_terms:
        return 0.0
    overlap = len(query_terms & content_terms)
    token_score = overlap / math.sqrt(len(query_terms) * len(content_terms))
    phrases = query_phrases or set()
    if not phrases:
        return token_score
    normalized_content = re.sub(r"\s+", " ", content or "").lower()
    phrase_hits = sum(1 for phrase in phrases if phrase in normalized_content)
    # The bounded bonus promotes exact metric phrases without suppressing
    # sparse lexical matches when PDFs split a phrase across a line boundary.
    phrase_score = min(0.45, 0.45 * phrase_hits / len(phrases))
    return token_score + phrase_score


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
