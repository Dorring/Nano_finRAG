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
        scored = []
        for index, chunk in enumerate(chunks):
            original_score = _safe_float(chunk.get("score", 0.0))
            lexical_score = _evidence_alignment(query, query_terms, chunk.get("content", ""))
            rerank_score = (
                self.original_score_weight * original_score
                + self.lexical_weight * lexical_score
            )
            item = dict(chunk)
            item["rerank_score"] = rerank_score
            item["evidence_alignment"] = lexical_score
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


def _lexical_overlap(query_terms: set[str], content: str) -> float:
    if not query_terms:
        return 0.0
    content_terms = _tokenize(content)
    if not content_terms:
        return 0.0
    overlap = len(query_terms & content_terms)
    return overlap / math.sqrt(len(query_terms) * len(content_terms))


_EVIDENCE_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "did", "do", "does", "for",
    "from", "have", "has", "how", "in", "is", "it", "its", "of", "on",
    "or", "the", "this", "to", "was", "were", "what", "which", "with",
    "company", "document", "report", "year", "ended", "december",
}


def _evidence_alignment(query: str, query_terms: set[str], content: str) -> float:
    """Prefer document-scoped metric phrases near answer-bearing values.

    Retrieval already filters by document, so proper-name tokens in a query
    are scope hints rather than metric evidence. This lightweight score
    rewards the remaining two-word phrases and, for value questions, phrases
    that occur close to a number. It is independent of any document/page or
    fixed financial metric.
    """
    base = _lexical_overlap(query_terms, content)
    query_tokens = _query_evidence_tokens(query)
    content_tokens = _content_evidence_tokens(content)
    if len(query_tokens) < 2 or len(content_tokens) < 2:
        return base

    query_phrases = {
        tuple(query_tokens[index:index + 2])
        for index in range(len(query_tokens) - 1)
    }
    content_phrases = {
        tuple(content_tokens[index:index + 2])
        for index in range(len(content_tokens) - 1)
    }
    matching_phrases = query_phrases.intersection(content_phrases)
    score = base + min(0.9, 0.45 * len(matching_phrases))
    if not matching_phrases or not _is_value_query(query):
        return score

    lowered = (content or "").lower()
    for phrase in matching_phrases:
        pattern = r"\b" + r"\W+".join(map(re.escape, phrase)) + r"\b"
        for match in re.finditer(pattern, lowered):
            window = lowered[max(0, match.start() - 60):match.end() + 140]
            if re.search(r"[$]?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|per\s+cent|million|billion|thousand))?", window):
                return score + 0.3
    return score


def _query_evidence_tokens(query: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", query or "")
    tokens = []
    for index, raw in enumerate(raw_tokens):
        normalized = raw.lower()
        if normalized in _EVIDENCE_STOPWORDS:
            continue
        # Proper names only establish the document/company scope. Preserve
        # the leading question word, but do not let a later title/acronym
        # displace a metric phrase inside an already scoped document.
        if index > 0 and (raw.isupper() or raw[:1].isupper()):
            continue
        if not tokens or normalized != tokens[-1]:
            tokens.append(normalized)
    return tokens


def _content_evidence_tokens(content: str) -> list[str]:
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", content or "")
        if token.lower() not in _EVIDENCE_STOPWORDS
    ]
    collapsed = []
    for token in tokens:
        if not collapsed or token != collapsed[-1]:
            collapsed.append(token)
    return collapsed


def _is_value_query(query: str) -> bool:
    normalized = (query or "").lower()
    return any(marker in normalized for marker in (
        "how much", "how many", "amount", "revenue", "cash", "margin",
        "income", "expense", "assets", "liabilities", "percentage",
        "percent", "rate", "growth", "profit", "loss",
    ))


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
