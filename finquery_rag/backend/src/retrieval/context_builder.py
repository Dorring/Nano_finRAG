"""Context building from retrieved evidence chunks.

Extracted from RAGEngine to isolate context assembly logic.
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SufficiencyResult:
    """Result of evidence sufficiency evaluation."""
    is_sufficient: bool
    best_score: float
    average_score: float


class ContextBuilder:
    """Builds context string and source list from retrieved chunks."""

    def __init__(
        self,
        *,
        max_context_tokens: int = 1100,
        min_score_threshold: float = 0.0,
        tokenizer=None,
    ):
        self._max_context_tokens = max_context_tokens
        self._min_score_threshold = min_score_threshold
        self._tokenizer = tokenizer
        # The exact chunks whose text entered the built context.  Validation
        # must use these (including parent expansion/truncation), rather than
        # a stale pre-context child chunk set.
        self._last_context_evidence: list[dict] = []

    @property
    def last_context_evidence(self) -> list[dict]:
        """Return copies of the evidence chunks represented in the context."""
        return [dict(chunk) for chunk in self._last_context_evidence]

    def build(self, chunks: list) -> tuple[str, list[dict]]:
        """Build context from retrieved chunks with dedup, score threshold, and token budget."""
        self._last_context_evidence = []
        if not chunks:
            return "", []

        # Deduplicate chunks by content
        seen_content = set()
        deduped = []
        for chunk in chunks:
            content_key = chunk["content"][:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                deduped.append(chunk)
        chunks = deduped

        # Filter by minimum score threshold
        if self._min_score_threshold > 0:
            chunks = [c for c in chunks if c.get("score", 0) >= self._min_score_threshold]

        if not chunks:
            return "", []

        chunks = self._merge_parent_context_chunks(chunks)

        context_parts = []
        sources = []
        context_evidence = []
        current_tokens = 0
        safe_limit = self._max_context_tokens - 200

        for i, chunk in enumerate(chunks, 1):
            doc_id = chunk["doc_id"]
            content = chunk["content"]
            chunk_type = chunk["metadata"].get("type")
            page = chunk["metadata"].get("page")
            parent_id = chunk["metadata"].get("parent_id")
            section_path = chunk["metadata"].get("section_path")
            child_hit_count = chunk["metadata"].get("child_hit_count")

            if "::" in doc_id:
                parts = doc_id.split("::")[0]
                if parts.startswith("user_"):
                    parts = "_".join(parts.split("_")[2:])
                filename = parts
            else:
                filename = doc_id

            if chunk_type == "table":
                table_num = chunk["metadata"].get("table_num", "")
                source_ref = "%s, p%s(T%s)" % (filename, page, table_num)
            else:
                source_ref = "%s, p%s" % (filename, page)

            chunk_text = "[%s]\n%s" % (source_ref, content)

            if self._tokenizer:
                chunk_tokens = len(self._tokenizer.encode(chunk_text))
            else:
                chunk_tokens = len(chunk_text) / 3

            if current_tokens + chunk_tokens > safe_limit:
                remaining_tokens = safe_limit - current_tokens
                if remaining_tokens > 80:
                    if self._tokenizer:
                        truncated_tokens = self._tokenizer.encode(content)[:remaining_tokens-20]
                        truncated_content = self._tokenizer.decode(truncated_tokens) + "\n[...]"
                    else:
                        truncated_content = content[:int(remaining_tokens * 3)] + "\n[...]"
                    chunk_text = "[%s]\n%s" % (source_ref, truncated_content)
                    context_parts.append(chunk_text)
                    evidence_chunk = dict(chunk)
                    evidence_chunk["content"] = truncated_content
                    context_evidence.append(evidence_chunk)
                    sources.append({
                        "filename": filename, "page": page,
                        "type": chunk_type, "score": chunk.get("score", 0),
                        "chunk_id": doc_id,
                        "parent_id": parent_id,
                        "section_path": section_path,
                        "child_hit_count": child_hit_count,
                    })
                break

            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
            context_evidence.append(dict(chunk))
            sources.append({
                "filename": filename, "page": page,
                "type": chunk_type, "score": chunk.get("score", 0),
                "chunk_id": doc_id,
                "parent_id": parent_id,
                "section_path": section_path,
                "child_hit_count": child_hit_count,
            })

        self._last_context_evidence = context_evidence
        context_str = "\n\n---\n\n".join(context_parts)
        return context_str, sources

    def _merge_parent_context_chunks(self, chunks: list) -> list:
        """Expand child hits to their parent section/page excerpt and merge siblings."""
        merged = []
        by_parent: dict[str, dict] = {}

        for chunk in chunks:
            parent_key = self._parent_context_key(chunk)
            if not parent_key:
                merged.append(chunk)
                continue

            metadata = dict(chunk.get("metadata") or {})
            parent_excerpt = metadata.get("parent_excerpt")
            existing = by_parent.get(parent_key)
            if existing is None:
                expanded = dict(chunk)
                expanded_metadata = dict(metadata)
                expanded_metadata["context_expanded_from"] = "parent_excerpt"
                expanded_metadata["child_hit_count"] = 1
                expanded_metadata["child_chunk_ids"] = [chunk.get("doc_id")]
                expanded_metadata["matched_child_snippets"] = [
                    self._compact_child_snippet(chunk.get("content", ""))
                ]
                expanded["metadata"] = expanded_metadata
                expanded["content"] = self._compose_parent_context(
                    parent_excerpt,
                    expanded_metadata["matched_child_snippets"],
                )
                expanded["child_hit_count"] = 1
                by_parent[parent_key] = expanded
                merged.append(expanded)
                continue

            existing_score = float(existing.get("score", 0) or 0)
            current_score = float(chunk.get("score", 0) or 0)
            existing["score"] = max(existing_score, current_score)
            existing["child_hit_count"] = int(existing.get("child_hit_count", 1)) + 1
            existing_meta = existing.get("metadata") or {}
            child_ids = list(existing_meta.get("child_chunk_ids") or [])
            child_id = chunk.get("doc_id")
            if child_id and child_id not in child_ids:
                child_ids.append(child_id)
            existing_meta["child_chunk_ids"] = child_ids
            existing_meta["child_hit_count"] = existing["child_hit_count"]
            snippets = list(existing_meta.get("matched_child_snippets") or [])
            snippet = self._compact_child_snippet(chunk.get("content", ""))
            if snippet and snippet not in snippets:
                snippets.append(snippet)
            existing_meta["matched_child_snippets"] = snippets
            existing["content"] = self._compose_parent_context(
                existing_meta.get("parent_excerpt", existing.get("content", "")),
                snippets,
            )

        return merged

    @staticmethod
    def _parent_context_key(chunk: dict) -> str | None:
        metadata = chunk.get("metadata") or {}
        parent_id = metadata.get("parent_id")
        parent_excerpt = metadata.get("parent_excerpt")
        if not isinstance(parent_id, str) or not parent_id.strip():
            return None
        if not isinstance(parent_excerpt, str) or not parent_excerpt.strip():
            return None
        return parent_id.strip()

    @staticmethod
    def _compact_child_snippet(content: str, *, max_chars: int = 500) -> str:
        text = re.sub(r"\s+", " ", content or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + " [...]"

    @staticmethod
    def _compose_parent_context(parent_excerpt: str, child_snippets: list[str]) -> str:
        snippets = [item for item in child_snippets if item]
        if not snippets:
            return parent_excerpt
        evidence = "\n".join(f"- {item}" for item in snippets)
        return f"{parent_excerpt}\n\nMatched child evidence:\n{evidence}"


class EvidenceSufficiencyEvaluator:
    """Evaluates whether retrieved evidence is sufficient for reliable answering."""

    def __init__(
        self,
        *,
        rrf_sufficiency_threshold: float = 0.008,
        dense_sufficiency_threshold: float = 0.15,
    ):
        self._rrf_threshold = rrf_sufficiency_threshold
        self._dense_threshold = dense_sufficiency_threshold

    def evaluate(self, chunks: list) -> SufficiencyResult:
        """Check if retrieved context is sufficient for a reliable answer."""
        if not chunks:
            return SufficiencyResult(is_sufficient=False, best_score=0.0, average_score=0.0)

        scores = [c.get("score", 0) for c in chunks]
        best_score = max(scores)
        avg_score = sum(scores) / len(scores)

        score_kinds = {
            str(chunk.get("score_kind") or "").lower()
            for chunk in chunks
        }
        known_score_kinds = score_kinds - {"", "unknown"}
        if known_score_kinds == {"rrf"}:
            threshold = self._rrf_threshold
        elif known_score_kinds:
            threshold = self._dense_threshold
        else:
            # Compatibility for legacy stores that predate score provenance.
            threshold = self._rrf_threshold if best_score < 0.05 else self._dense_threshold

        is_sufficient = best_score >= threshold
        return SufficiencyResult(
            is_sufficient=is_sufficient,
            best_score=best_score,
            average_score=avg_score,
        )

    def confidence(self, chunks: list) -> float:
        """Compute answer confidence based on retrieval quality."""
        if not chunks:
            return 0.0

        scores = [c.get("score", 0) for c in chunks]
        best = max(scores)
        avg = sum(scores) / len(scores)

        confidence = 0.7 * best + 0.3 * avg
        return min(1.0, max(0.0, confidence))
