"""Query request domain object."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QueryRequest:
    """Immutable representation of an incoming RAG query.

    This is an internal domain object; the HTTP API schema remains unchanged.
    """

    question: str
    document_names: tuple[str, ...] = ()
    user_id: int | None = None
    session_id: str | None = None
    conversation_history: tuple[dict[str, Any], ...] = ()
    memory_profile: dict[str, Any] | None = None
    query_as_resolved: bool = False
    stream: bool = False
    debug: bool = False
