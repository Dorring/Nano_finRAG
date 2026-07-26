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
        evidence_texts = [_chunk_evidence_text(chunk) for chunk in chunks]
        page_alignment = _page_evidence_alignment(
            query, chunks, evidence_texts
        )
        scored = []
        for index, (chunk, evidence_text) in enumerate(zip(chunks, evidence_texts)):
            original_score = _safe_float(chunk.get("score", 0.0))
            lexical_score = _evidence_alignment(
                query, query_terms, evidence_text
            )
            coherence_score = page_alignment.get(_page_key(chunk), 0.0)
            rerank_score = (
                self.original_score_weight * original_score
                + self.lexical_weight * lexical_score
                + 0.10 * coherence_score
            )
            item = dict(chunk)
            item["rerank_score"] = rerank_score
            item["evidence_alignment"] = lexical_score
            item["page_evidence_alignment"] = coherence_score
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



def _chunk_evidence_text(chunk: dict) -> str:
    """Combine chunk text with compact structural metadata for reranking.

    Table rows often contain only a line item and numbers; their surrounding
    section or illustration name is stored in metadata.  Including that
    metadata lets a query's entity/qualifier terms disambiguate otherwise
    identical rows without changing the indexed document content.
    """
    metadata = chunk.get("metadata") or {}
    structural_values = [
        metadata.get(key)
        for key in ("section_path", "table_title", "row_label", "parent_title")
        if metadata.get(key)
    ]
    structural = " ".join(str(value) for value in structural_values)
    content = chunk.get("content", "") or ""
    return f"{structural}\n{content}".strip()



def _page_key(chunk: dict) -> tuple[str, int | None] | None:
    """Return a document/page key when the chunk has page-level provenance."""
    metadata = chunk.get("metadata") or {}
    doc_name = metadata.get("doc_name")
    page = metadata.get("page")
    if not doc_name or page is None:
        return None
    return str(doc_name), page


def _page_evidence_alignment(
    query: str,
    chunks: list[dict],
    evidence_texts: list[str],
) -> dict[tuple[str, int | None] | None, float]:
    """Reward pages where complementary chunks jointly cover rare query anchors.

    A page-level boost is useful only when it joins evidence that a normal
    chunk-level ranker cannot see together, such as an illustration heading
    and a separate table row. Terms common across the candidate set, such as
    cash, are intentionally insufficient: they would otherwise promote
    unrelated pages that happen to contain the same generic metric.
    """
    query_tokens = _query_page_tokens(query)
    if len(query_tokens) < 2:
        return {}

    chunk_hits = [
        query_tokens.intersection(_tokenize(text))
        for text in evidence_texts
    ]
    token_frequency = {
        token: sum(token in hits for hits in chunk_hits)
        for token in query_tokens
    }
    rare_tokens = {
        token
        for token, frequency in token_frequency.items()
        if frequency <= max(2, len(chunks) // 5)
    }
    if not rare_tokens:
        return {}

    page_hits: dict[tuple[str, int | None] | None, set[str]] = {}
    page_counts: dict[tuple[str, int | None] | None, int] = {}
    for chunk, hits in zip(chunks, chunk_hits):
        key = _page_key(chunk)
        if key is None or not hits:
            continue
        page_hits.setdefault(key, set()).update(hits)
        page_counts[key] = page_counts.get(key, 0) + 1

    aligned = {}
    for key, hits in page_hits.items():
        rare_hits = hits.intersection(rare_tokens)
        # Require both a discriminative anchor and a second query signal.
        if page_counts.get(key, 0) < 2 or not rare_hits or len(hits) < 2:
            continue
        coverage = len(hits) / len(query_tokens)
        rare_coverage = len(rare_hits) / len(rare_tokens)
        aligned[key] = 0.4 * coverage + 0.6 * rare_coverage
    return aligned


def _query_page_tokens(query: str) -> set[str]:
    """Keep query entity and metric tokens for complementary page matching."""
    generic = _EVIDENCE_STOPWORDS | {
        "amount", "given", "question", "questions", "shown", "show",
        "reported", "reporting", "financial", "statement", "statements",
        "practice", "basis", "company", "companies",
    }
    return {
        token
        for token in _tokenize(query)
        if len(token) >= 3 and token not in generic
    }


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
    score += _direct_value_alignment(matching_phrases, content)

    lowered = (content or "").lower()
    for phrase in matching_phrases:
        pattern = r"\b" + r"\W+".join(map(re.escape, phrase)) + r"\b"
        for match in re.finditer(pattern, lowered):
            window = lowered[max(0, match.start() - 60):match.end() + 140]
            if re.search(r"[$]?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|per\s+cent|million|billion|thousand))?", window):
                return score + 0.3
    return score


def _direct_value_alignment(matching_phrases: set[tuple[str, str]], content: str) -> float:
    """Reward a metric directly asserted to equal a nearby value.

    A term match alone is ambiguous in financial reports: it may occur in an
    interest-income explanation, a foreign-subsidiary disclosure, or a cash
    flow adjustment. This gives an extra signal only when the matching metric
    phrase is immediately followed by an assertion verb and a value. The
    phrase comes from the user's query, so the rule is document- and
    metric-agnostic.
    """
    lowered = (content or "").lower()
    value = r"[$€£¥]?\s*\(?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:%|per\s+cent|million|billion|thousand))?"
    verbs = r"(?:is|are|was|were|totaled|amounted(?:\s+to)?|stood(?:\s+at)?|equaled|reached|reported)"
    for phrase in matching_phrases:
        phrase_pattern = r"\b" + r"\W+".join(map(re.escape, phrase)) + r"\b"
        for match in re.finditer(phrase_pattern, lowered):
            # A metric can be mentioned as the object of another measure
            # ("interest income from cash ..."), which is not a direct
            # statement of the metric's own value.
            prefix = lowered[max(0, match.start() - 32):match.start()]
            relation = r"(?:from(?:\s+[a-z]+){0,3}|on|of|for|in|against|related\s+to)"
            if re.search(r"\b" + relation + r"\s*$", prefix):
                continue
            following = lowered[match.end():match.end() + 96]
            if re.match(r"\s+" + verbs + r"\b.{0,32}?" + value, following):
                return 0.8
    return 0.0


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
