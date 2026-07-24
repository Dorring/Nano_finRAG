from collections import defaultdict

"""Candidate fusion utilities for hybrid retrieval.

Extracted from RAGEngine to isolate score normalization and deduplication.
"""


def normalize_scores(chunks: list) -> list:
    """Unify score fields while preserving their provenance.

    RRF scores are rank-derived and live on a much smaller numeric scale than
    dense similarity scores.  Callers must not infer the scale from the score
    value itself: a valid single-retriever RRF hit is commonly about 0.016.
    """
    for chunk in chunks:
        if "fused_score" in chunk:
            chunk["score"] = chunk["fused_score"]
            chunk["score_kind"] = "rrf"
        elif "score" not in chunk:
            chunk["score"] = 0
            chunk.setdefault("score_kind", "unknown")
        else:
            chunk.setdefault("score_kind", "dense")
    return chunks


def dedupe_chunks(chunks: list) -> list:
    """Remove duplicate chunks by doc_id or content fingerprint."""
    deduped = []
    seen = set()
    for chunk in chunks or []:
        doc_id = chunk.get("doc_id")
        key = doc_id or (
            (chunk.get("metadata") or {}).get("doc_name"),
            (chunk.get("metadata") or {}).get("page"),
            (chunk.get("content") or "")[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def chunk_doc_name(chunk: dict) -> str | None:
    """Extract document name from chunk metadata or doc_id."""
    metadata = chunk.get("metadata") or {}
    doc_name = metadata.get("doc_name")
    if doc_name:
        return doc_name
    doc_id = chunk.get("doc_id") or ""
    if "::" not in doc_id:
        return None
    prefix = doc_id.split("::", 1)[0]
    if prefix.startswith("user_"):
        return "_".join(prefix.split("_")[2:])
    return prefix or None


def ensure_multi_doc_coverage(
    candidates: list, selected: list, doc_names: list[str], top_k: int | None
) -> list:
    """Keep at least one candidate per requested document for multi-document questions."""
    if top_k is None or top_k <= 0 or len(doc_names or []) <= 1:
        return selected
    selected = list(selected or [])
    selected_ids = {chunk.get("doc_id") for chunk in selected}
    selected_docs = {chunk_doc_name(chunk) for chunk in selected}
    wanted_docs = [doc for doc in doc_names if doc not in selected_docs]
    if not wanted_docs:
        return selected[:top_k]

    for doc_name in wanted_docs:
        doc_candidates = [chunk for chunk in candidates if chunk_doc_name(chunk) == doc_name]
        best = max(
            doc_candidates,
            key=lambda chunk: float(chunk.get("score", 0) or 0),
            default=None,
        )
        if not best or best.get("doc_id") in selected_ids:
            continue
        if len(selected) < top_k:
            selected.append(best)
        else:
            replace_index = len(selected) - 1
            selected[replace_index] = best
        selected_ids.add(best.get("doc_id"))
        selected_docs.add(doc_name)
    return selected[:top_k]


def boost_front_matter_chunks(query: str, chunks: list, *, is_front_matter_query_fn) -> list:
    """Prefer page-1 evidence for title/author/abstract style questions."""
    if not chunks or not is_front_matter_query_fn(query):
        return chunks
    boosted = []
    for chunk in chunks:
        item = dict(chunk)
        metadata = dict(item.get("metadata") or {})
        item["metadata"] = metadata
        score = float(item.get("score", 0) or 0)
        if metadata.get("page") == 1:
            item["score"] = score + 0.02
            item["front_matter_boost"] = 0.02
        boosted.append(item)
    boosted.sort(key=lambda item: item.get("score", 0), reverse=True)
    return boosted


def summarize_retrieved_chunks(chunks: list) -> list:
    """Return eval-safe retrieval metadata without copying chunk content."""
    summary = []
    for chunk in chunks:
        meta = chunk.get("metadata", {}) or {}
        item = {
            "doc_id": chunk.get("doc_id", ""),
            "filename": meta.get("doc_name"),
            "page": meta.get("page"),
            "type": meta.get("type"),
            "score": chunk.get("score", 0),
            "rerank_score": chunk.get("rerank_score"),
            "reranker": chunk.get("reranker"),
        }
        for key in ("parent_id", "section_path", "context_expanded_from"):
            value = meta.get(key)
            if value is not None:
                item[key] = value
        summary.append(item)
    return summary


def source_from_chunk(chunk: dict) -> dict:
    """Extract source information from a chunk."""
    meta = chunk.get("metadata", {}) or {}
    doc_id = chunk.get("doc_id", "")
    filename = meta.get("doc_name")
    if not filename:
        filename = doc_id.split("::", 1)[0] if "::" in doc_id else doc_id
        if filename.startswith("user_"):
            filename = "_".join(filename.split("_")[2:])
    return {
        "filename": filename,
        "page": meta.get("page"),
        "type": meta.get("type"),
        "score": chunk.get("score", 0),
        "chunk_id": doc_id,
        "parent_id": meta.get("parent_id"),
        "section_path": meta.get("section_path"),
        "child_hit_count": meta.get("child_hit_count"),
    }


def rrf(ranked_lists, k: int = 60):
    """
    使用倒数排名融合算法合并多个排序列表。

    Args:
        ranked_lists: 包含多个排序列表的列表，每个排序列表包含字典元素，
                      字典需具有"doc_id"和"score"键。
        k: RRF算法的常数，用于控制排名的权重，默认为60。

    Returns:
        List: 按融合得分降序排列的文档信息列表，每个字典包含原始文档信息及新增的"fused_score"键。
    """
    fused_scores = defaultdict(float)
    doc_map = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            doc_id = item["doc_id"]
            fused_scores[doc_id] += 1 / (k + rank + 1)

            if doc_id not in doc_map:
                doc_map[doc_id] = item

    sorted_ids = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {**doc_map[doc_id], "fused_score": score, "score_kind": "rrf"}
        for doc_id, score in sorted_ids
    ]
