"""Answer result and retrieval result domain objects.

These objects form the typed boundary between the RAG orchestrator and the
HTTP API layer. ``AnswerResult`` is intentionally a strict superset of the
legacy dict shape: each production code path constructs an ``AnswerResult``
and ``to_legacy_dict`` reproduces the exact field set that the pre-refactor
``RAGOrchestrator.query`` returned, so the public API remains bit-for-bit
compatible.

Phase 3 Commit 8 adds an optional ``calculations`` field carrying
structured calculation diagnostics when the deterministic calculation
pipeline is invoked. ``to_legacy_dict`` emits ``calculations`` only when
non-empty, so existing API consumers are unaffected.

Phase 4 Commit 9 adds optional ``answerability`` and ``validation`` fields
carrying the pre-generation answerability verdict and post-generation
validation verdict (as public-safe dicts). ``to_legacy_dict`` emits them
only when non-None, so existing API consumers are unaffected.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnswerPath(str, Enum):
    """Identifies which orchestrator branch produced this answer.

    The legacy orchestrator returned different dict shapes from different
    early-return branches (conversational / no-retrieval / no-documents /
    full). ``to_legacy_dict`` uses this to reproduce the exact field set
    per branch.
    """

    FULL = "full"
    CONVERSATIONAL = "conversational"
    NO_RETRIEVAL = "no_retrieval"
    NO_DOCUMENTS = "no_documents"


# Fields emitted by each legacy branch. ``rewritten_question`` is conditional:
# - ``FULL``: always present (None when no conversation history).
# - Other branches: present iff ``had_conversation_history`` is True.
_LEGACY_FIELDS_BY_PATH: dict[AnswerPath, tuple[str, ...]] = {
    AnswerPath.FULL: (
        "answer",
        "sources",
        "context",
        "searched_docs",
        "confidence",
        "context_sufficient",
        "intent",
        "intent_confidence",
        "rewritten_question",
        "retrieved_chunks",
        "retrieval_debug",
        "trace_id",
    ),
    AnswerPath.CONVERSATIONAL: (
        "answer",
        "sources",
        "context",
        "searched_docs",
        "context_sufficient",
        "intent",
        "intent_confidence",
        "rewritten_question",
        "trace_id",
    ),
    AnswerPath.NO_RETRIEVAL: (
        "answer",
        "sources",
        "context",
        "searched_docs",
        "context_sufficient",
        "intent",
        "intent_confidence",
        "rewritten_question",
        "trace_id",
    ),
    AnswerPath.NO_DOCUMENTS: (
        "answer",
        "sources",
        "context",
        "searched_docs",
        "context_sufficient",
        "rewritten_question",
        "trace_id",
    ),
}


@dataclass(frozen=True)
class RetrievalResult:
    """Immutable representation of a retrieval operation's output."""

    query: str
    retrieval_query: str
    evidence: tuple[Any, ...]
    debug: dict[str, Any]


@dataclass(frozen=True)
class AnswerResult:
    """Immutable representation of a final answer.

    All fields are optional or have safe defaults so that any orchestrator
    branch can construct an ``AnswerResult``. ``to_legacy_dict`` reproduces
    the exact dict shape that ``RAGOrchestrator.query`` returned before this
    refactor, including which fields are omitted per branch and whether
    ``rewritten_question`` is included when ``conversation_history`` is empty.

    ``calculations`` (added in Phase 3 Commit 8) carries structured
    calculation diagnostics when the deterministic calculation pipeline
    produces an answer. It defaults to an empty tuple and is omitted from
    ``to_legacy_dict`` when empty, so existing API consumers are unaffected.
    """

    answer: str
    sources: tuple[dict[str, Any], ...] = ()
    context: str | None = None
    searched_docs: tuple[str, ...] = ()
    confidence: float | None = None
    context_sufficient: bool = True
    intent: str | None = None
    intent_confidence: float | None = None
    rewritten_question: str | None = None
    retrieved_chunks: tuple[dict[str, Any], ...] = ()
    retrieval_debug: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    warnings: tuple[str, ...] = ()
    path: AnswerPath = AnswerPath.FULL
    had_conversation_history: bool = False
    calculations: tuple[dict[str, Any], ...] = ()
    answerability: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    repair: dict[str, Any] | None = None

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert to the legacy API response dict format.

        The output must match the dict that ``RAGOrchestrator.query``
        returned before the refactor. Field set is determined by ``path``;
        ``rewritten_question`` is included conditionally:
        - ``FULL``: always present (possibly None).
        - Other paths: present iff ``had_conversation_history`` is True.

        ``calculations`` and ``warnings`` are emitted only when non-empty
        (additive fields that do not affect the legacy field set).
        """
        result: dict[str, Any] = {}
        include_rewritten = (
            self.path is AnswerPath.FULL or self.had_conversation_history
        )

        for name in _LEGACY_FIELDS_BY_PATH[self.path]:
            if name == "rewritten_question" and not include_rewritten:
                continue
            result[name] = self._legacy_value(name)

        if self.warnings:
            result["warnings"] = list(self.warnings)
        if self.calculations:
            result["calculations"] = list(self.calculations)
        if self.answerability is not None:
            result["answerability"] = dict(self.answerability)
        if self.validation is not None:
            result["validation"] = dict(self.validation)
        if self.repair is not None:
            result["repair"] = dict(self.repair)
        return result

    def _legacy_value(self, name: str) -> Any:
        if name == "sources":
            return list(self.sources)
        if name == "searched_docs":
            return list(self.searched_docs)
        if name == "retrieved_chunks":
            return list(self.retrieved_chunks)
        if name == "retrieval_debug":
            return dict(self.retrieval_debug)
        if name == "confidence":
            return self.confidence
        if name == "intent":
            return self.intent
        if name == "intent_confidence":
            return self.intent_confidence
        if name == "rewritten_question":
            return self.rewritten_question
        if name == "trace_id":
            return self.trace_id
        if name == "context":
            return self.context
        if name == "context_sufficient":
            return self.context_sufficient
        if name == "answer":
            return self.answer
        raise KeyError(name)
