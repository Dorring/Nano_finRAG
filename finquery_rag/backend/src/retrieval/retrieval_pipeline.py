"""Retrieval pipeline: single and multi-document retrieval.

Extracted from RAGEngine to isolate retrieval orchestration.
Dependencies are injected via constructor, not read from environment.
"""
import asyncio
from typing import Callable

from src.retrieval.candidate_fusion import (
    normalize_scores,
    boost_front_matter_chunks,
    ensure_multi_doc_coverage,
)
from src.retrieval.query_processor import QueryProcessor


class RetrievalPipeline:
    """Orchestrates dense, BM25, and hybrid retrieval with optional reranking."""

    def __init__(
        self,
        *,
        dense_query_fn: Callable,
        bm25_retriever=None,
        reranker=None,
        query_processor: QueryProcessor | None = None,
        candidate_multiplier: int = 4,
        use_hybrid: bool = True,
    ):
        self._dense_query_fn = dense_query_fn
        self._bm25_retriever = bm25_retriever
        self._reranker = reranker
        self._query_processor = query_processor or QueryProcessor()
        self._candidate_multiplier = max(1, candidate_multiplier)
        self._use_hybrid = use_hybrid
        self._last_retrieval_debug = self._make_retrieval_debug(0, 0)

    @property
    def last_retrieval_debug(self) -> dict:
        """Public accessor for the last retrieval debug info."""
        return self._last_retrieval_debug

    def _make_retrieval_debug(self, candidate_count: int, returned_count: int) -> dict:
        return {
            "reranker": self._reranker.name if self._reranker else None,
            "reranker_enabled": self._reranker is not None,
            "candidate_count": candidate_count,
            "returned_count": returned_count,
            "candidate_multiplier": self._candidate_multiplier,
        }

    def _apply_reranker(self, query: str, chunks: list, top_k: int) -> list:
        candidate_count = len(chunks)
        if not self._reranker:
            selected = chunks[:top_k]
        else:
            selected = self._reranker.rerank(query, chunks, top_k=top_k)
        self._last_retrieval_debug = self._make_retrieval_debug(
            candidate_count,
            len(selected),
        )
        return selected

    def retrieve_single(
        self,
        document_name: str,
        query: str,
        user_id: int | None = None,
        top_k: int = 3,
    ) -> list:
        """Retrieve relevant chunks from a single document."""
        retrieval_query = self._query_processor.expand(query)

        if not self._use_hybrid:
            results = self._dense_query_fn(
                query_text=retrieval_query, doc_name=document_name,
                n_results=top_k, user_id=user_id,
            )
            results = normalize_scores(results)
            results = boost_front_matter_chunks(
                query, results,
                is_front_matter_query_fn=self._query_processor.is_front_matter_query,
            )
            selected = self._apply_reranker(query, results, top_k)
            return selected[:top_k] if top_k else selected

        candidate_k = top_k * self._candidate_multiplier
        dense_results = self._dense_query_fn(
            query_text=retrieval_query, doc_name=document_name,
            n_results=candidate_k, user_id=user_id,
        )

        bm25 = self._bm25_retriever
        if bm25:
            from src.retrieval.candidate_fusion import rrf
            sparse_results = bm25.search(
                retrieval_query, k=candidate_k,
                doc_name=document_name, user_id=user_id,
            )
            fused = rrf([dense_results, sparse_results])
            results = normalize_scores(fused)
            results = boost_front_matter_chunks(
                query, results,
                is_front_matter_query_fn=self._query_processor.is_front_matter_query,
            )
            selected = self._apply_reranker(query, results, top_k)
            return selected[:top_k] if top_k else selected

        results = normalize_scores(dense_results)
        results = boost_front_matter_chunks(
            query, results,
            is_front_matter_query_fn=self._query_processor.is_front_matter_query,
        )
        selected = self._apply_reranker(query, results, top_k)
        return selected[:top_k] if top_k else selected

    async def retrieve_multiple(
        self,
        document_names: list[str],
        query: str,
        user_id: int | None = None,
        top_k: int = 3,
    ) -> list:
        """Retrieve relevant chunks from multiple documents concurrently."""
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(
                None,
                self.retrieve_single,
                doc_name, query, user_id, top_k,
            )
            for doc_name in document_names
        ]

        results_list = await asyncio.gather(*tasks)

        all_results = []
        for results in results_list:
            all_results.extend(results)

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        selected = self._apply_reranker(query, all_results, top_k)
        return ensure_multi_doc_coverage(all_results, selected, document_names, top_k)
