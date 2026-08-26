"""I6 regression tests for explicit legacy query-rewrite bypass."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.application.rag_orchestrator import RAGOrchestrator
from src.domain.query import QueryRequest


def _orchestrator(rewritten: str = "rewritten query") -> tuple[RAGOrchestrator, AsyncMock]:
    rewrite_query = AsyncMock(return_value=rewritten)
    return (
        RAGOrchestrator(
            query_processor=None,
            retrieval_pipeline=None,
            context_builder=None,
            sufficiency_evaluator=None,
            llm_gateway=SimpleNamespace(rewrite_query=rewrite_query),
            deterministic_extractor=None,
            trace_logger=None,
            intent_classifier=lambda query: {
                "intent": "out_of_scope",
                "requires_retrieval": False,
                "confidence": 1.0,
            },
            list_all_documents_fn=lambda user_id: [],
            get_front_matter_chunks_fn=lambda **kwargs: [],
        ),
        rewrite_query,
    )


def test_unresolved_query_invokes_legacy_rewrite_once() -> None:
    orchestrator, rewrite_query = _orchestrator()
    request = QueryRequest(
        question="What about last year?",
        user_id=7,
        conversation_history=(
            {"role": "user", "content": "Apple FY2024 Revenue?"},
        ),
    )

    asyncio.run(orchestrator.answer(request))

    rewrite_query.assert_awaited_once_with(
        "What about last year?",
        [{"role": "user", "content": "Apple FY2024 Revenue?"}],
        None,
    )


def test_resolved_query_bypasses_legacy_rewrite() -> None:
    orchestrator, rewrite_query = _orchestrator()
    request = QueryRequest(
        question="What was Apple FY2023 Revenue?",
        user_id=7,
        query_as_resolved=True,
        conversation_history=(
            {"role": "user", "content": "Apple FY2024 Revenue?"},
        ),
    )

    result = asyncio.run(orchestrator.answer(request))

    rewrite_query.assert_not_awaited()
    assert result.answer
