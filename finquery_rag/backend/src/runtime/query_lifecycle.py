"""Shared Conversation-aware execution for the query HTTP transports."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..conversation.contracts import ConversationTurnOutcome
from ..conversation.shadow_service import ConversationShadowService
from ..conversation.sqlite_store import ConversationStateConflictError
from .query_execution_service import QueryExecutionService
from .response_mapper import to_legacy_query_dict
from .runtime_adapters import LegacyFinancialRuntimeAdapter
from .runtime_contract import FinancialQARuntime, FinancialQueryRequest, FinancialQueryResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserTurnExecutionRequest:
    """Transport-neutral input for one user turn."""

    request_id: str
    user_id: int
    original_query: str
    session_id: str | None = None
    document_names: list[str] = field(default_factory=list)
    n_results: int = 3
    conversation_mode: str = "off"


@dataclass
class UserTurnExecutionResult:
    """Final business result shared by JSON and validated-final SSE."""

    status: str
    answer: str | None
    clarification: dict[str, Any] | None
    citations: list[dict[str, Any]]
    evidence_ids: list[str]
    citation_ids: list[str]
    calculation_ids: list[str]
    reason_codes: list[str]
    release_status: str
    runtime_version: str
    conversation_mode: str
    original_query: str
    standalone_query: str
    query_as_resolved: bool
    request_id: str
    session_id: str | None
    legacy_result: dict[str, Any]
    runtime_result: FinancialQueryResult | None = None
    idempotent_replay: bool = False


class QueryLifecycleService:
    """Single Conversation -> Runtime -> final commit business path."""

    def __init__(
        self,
        *,
        session_manager: Any,
        memory_store: Any,
        get_rag_engine: Callable[[], Any],
        get_conversation_service: Callable[[], ConversationShadowService],
        financial_runtime_adapter_enabled: Callable[[], bool],
        active_query_requires_context: Callable[[str], bool],
        active_query_is_out_of_scope: Callable[[str], bool],
        assistant_session_metadata: Callable[..., dict[str, Any]],
        execution_service_factory: Callable[[Any], QueryExecutionService] = QueryExecutionService,
        financial_runtime_factory: Callable[
            [Any, FinancialQueryRequest], FinancialQARuntime
        ] | None = None,
    ) -> None:
        self.session_manager = session_manager
        self.memory_store = memory_store
        self.get_rag_engine = get_rag_engine
        self.get_conversation_service = get_conversation_service
        self.financial_runtime_adapter_enabled = financial_runtime_adapter_enabled
        self.active_query_requires_context = active_query_requires_context
        self.active_query_is_out_of_scope = active_query_is_out_of_scope
        self.assistant_session_metadata = assistant_session_metadata
        self.execution_service_factory = execution_service_factory
        self.financial_runtime_factory = financial_runtime_factory

    @staticmethod
    def _value(value: Any) -> str:
        return str(getattr(value, "value", value))

    @staticmethod
    def _empty_legacy(answer: str, docs: Sequence[str]) -> dict[str, Any]:
        return {
            "answer": answer,
            "sources": [],
            "searched_docs": list(docs),
            "retrieved_chunks": [],
            "retrieval_debug": {},
            "calculations": [],
        }

    @staticmethod
    def _control_outcome(status: str) -> str:
        return {
            "CLARIFICATION_REQUIRED": ConversationTurnOutcome.CLARIFICATION,
            "OUT_OF_SCOPE": ConversationTurnOutcome.OUT_OF_SCOPE,
        }.get(status, ConversationTurnOutcome.ERROR)

    @classmethod
    def _runtime_outcome(cls, runtime: FinancialQueryResult | None) -> str:
        if runtime is None:
            return ConversationTurnOutcome.FINANCIAL_ANSWER
        return {
            "FAIL_CLOSED": ConversationTurnOutcome.FAIL_CLOSED,
            "OUT_OF_SCOPE": ConversationTurnOutcome.OUT_OF_SCOPE,
            "ERROR": ConversationTurnOutcome.ERROR,
        }.get(cls._value(runtime.status), ConversationTurnOutcome.FINANCIAL_ANSWER)

    def _history(
        self, request: UserTurnExecutionRequest
    ) -> list[dict[str, Any]] | None:
        if not request.session_id:
            return None
        return self.session_manager.get_recent_messages(
            request.session_id, request.user_id
        )

    def _profile(self, user_id: int) -> dict[str, Any] | None:
        profile = self.memory_store.get_profile(user_id)
        return profile if isinstance(profile, dict) else None

    def _commit_control(
        self,
        *,
        request: UserTurnExecutionRequest,
        answer: str,
        status: str,
        reason_codes: Sequence[str],
        service: ConversationShadowService | None,
        replay: bool,
    ) -> None:
        if not request.session_id or replay:
            return
        self.session_manager.add_message(
            request.session_id, request.user_id, "user", request.original_query
        )
        self.session_manager.add_message(
            request.session_id,
            request.user_id,
            "assistant",
            answer,
            metadata={
                "control_status": status,
                "reason_codes": list(reason_codes),
                "evidence_ids": [],
                "calculation_ids": [],
            },
        )
        if service is None:
            return
        try:
            service.record_control_turn(
                user_id=request.user_id,
                session_id=request.session_id,
                request_id=request.request_id,
                original_query=request.original_query,
                outcome=self._control_outcome(status),
                reason_codes=list(reason_codes),
            )
        except ConversationStateConflictError:
            raise
        except Exception:
            logger.exception("conversation control metadata failed")

    def _control_result(
        self,
        *,
        request: UserTurnExecutionRequest,
        status: str,
        answer: str,
        reason_codes: Sequence[str],
        service: ConversationShadowService | None,
        replay: bool,
        options: Sequence[str] | None = None,
    ) -> UserTurnExecutionResult:
        reason_codes = list(reason_codes)
        clarification = (
            {
                "question": answer,
                "reason_codes": reason_codes,
                "options": list(options or []),
            }
            if status == "CLARIFICATION_REQUIRED"
            else None
        )
        self._commit_control(
            request=request,
            answer=answer,
            status=status,
            reason_codes=reason_codes,
            service=service,
            replay=replay,
        )
        return UserTurnExecutionResult(
            status=status,
            answer=answer,
            clarification=clarification,
            citations=[],
            evidence_ids=[],
            citation_ids=[],
            calculation_ids=[],
            reason_codes=reason_codes,
            release_status="NOT_APPLICABLE",
            runtime_version="V1",
            conversation_mode=request.conversation_mode,
            original_query=request.original_query,
            standalone_query=request.original_query,
            query_as_resolved=False,
            request_id=request.request_id,
            session_id=request.session_id,
            legacy_result=self._empty_legacy(answer, request.document_names),
            idempotent_replay=replay,
        )

    async def _run_runtime(
        self,
        *,
        request: UserTurnExecutionRequest,
        query: str,
        history: list[dict[str, Any]] | None,
        profile: dict[str, Any] | None,
        query_as_resolved: bool,
        service: ConversationShadowService | None,
        replay: bool,
    ) -> UserTurnExecutionResult:
        runtime_request = FinancialQueryRequest(
            request_id=request.request_id,
            user_id=str(request.user_id),
            session_id=request.session_id or f"__stateless__:{request.request_id}",
            original_query=request.original_query,
            standalone_query=query,
            query_as_resolved=query_as_resolved,
            conversation_metadata={},
            request_metadata={
                "document_names": list(request.document_names),
                "n_results": request.n_results,
                "conversation_history": history,
                "memory_profile": profile,
            },
        )
        engine = self.get_rag_engine()
        runtime: FinancialQueryResult | None = None
        if self.financial_runtime_adapter_enabled():
            runtime_impl = (
                self.financial_runtime_factory(engine, runtime_request)
                if self.financial_runtime_factory is not None
                else LegacyFinancialRuntimeAdapter(engine)
            )
            runtime = await self.execution_service_factory(runtime_impl).execute(
                runtime_request
            )
            legacy = to_legacy_query_dict(runtime)
        else:
            kwargs = {
                "question": query,
                "doc_names": list(request.document_names),
                "n_results": request.n_results,
                "user_id": request.user_id,
                "conversation_history": history,
                "memory_profile": profile,
            }
            if query_as_resolved:
                kwargs["query_as_resolved"] = True
            legacy = dict(await engine.query(**kwargs))
        if query_as_resolved and query != request.original_query:
            legacy["rewritten_question"] = query

        if request.session_id and not replay:
            self.session_manager.add_message(
                request.session_id,
                request.user_id,
                "user",
                request.original_query,
            )
            self.session_manager.add_message(
                request.session_id,
                request.user_id,
                "assistant",
                legacy["answer"],
                metadata=self.assistant_session_metadata(result=legacy),
            )
        if service is not None and request.session_id and not replay:
            try:
                evidence_ids = list(runtime.evidence_ids) if runtime else []
                citation_ids = list(runtime.citation_ids) if runtime else []
                calculation_ids = list(runtime.calculation_ids) if runtime else []
                release_status = (
                    self._value(runtime.release_status) if runtime else "NOT_APPLICABLE"
                )
                service.record_assistant_turn(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    request_id=request.request_id,
                    original_query=request.original_query,
                    referenced_evidence_ids=evidence_ids,
                    citation_ids=citation_ids,
                    calculation_ids=calculation_ids,
                    release_status=release_status,
                    outcome=self._runtime_outcome(runtime),
                )
            except ConversationStateConflictError:
                raise
            except Exception:
                logger.exception("conversation assistant metadata failed")
        return UserTurnExecutionResult(
            status=self._value(runtime.status) if runtime else "ANSWER",
            answer=legacy.get("answer"),
            clarification=None,
            citations=list(legacy.get("sources", [])),
            evidence_ids=list(runtime.evidence_ids) if runtime else [],
            citation_ids=list(runtime.citation_ids) if runtime else [],
            calculation_ids=list(runtime.calculation_ids) if runtime else [],
            reason_codes=list(runtime.reason_codes) if runtime else [],
            release_status=(
                self._value(runtime.release_status) if runtime else "NOT_APPLICABLE"
            ),
            runtime_version=self._value(runtime.runtime_version) if runtime else "V1",
            conversation_mode=request.conversation_mode,
            original_query=request.original_query,
            standalone_query=query,
            query_as_resolved=query_as_resolved,
            request_id=request.request_id,
            session_id=request.session_id,
            legacy_result=legacy,
            runtime_result=runtime,
            idempotent_replay=replay,
        )

    async def execute_user_turn(
        self, request: UserTurnExecutionRequest
    ) -> UserTurnExecutionResult:
        """Resolve, gate, execute V1, commit, and return one final result."""
        history = self._history(request)
        profile = self._profile(request.user_id)
        service: ConversationShadowService | None = None
        replay = False
        if request.conversation_mode == "shadow":
            try:
                service = self.get_conversation_service()
                replay = bool(
                    service.is_request_processed(
                        user_id=request.user_id,
                        session_id=request.session_id,
                        request_id=request.request_id,
                        original_query=request.original_query,
                    )
                )
                service.observe(
                    request_id=request.request_id,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    original_query=request.original_query,
                    prior_history=history,
                )
            except ConversationStateConflictError:
                raise
            except Exception:
                logger.exception("conversation shadow invocation failed")

        query = request.original_query
        execution_history = history
        query_as_resolved = False
        if request.conversation_mode == "on":
            if request.session_id is None:
                if self.active_query_is_out_of_scope(request.original_query):
                    return self._control_result(
                        request=request,
                        status="OUT_OF_SCOPE",
                        answer="This question is outside the supported financial document scope.",
                        reason_codes=["OUT_OF_SCOPE"],
                        service=None,
                        replay=False,
                    )
                if self.active_query_requires_context(request.original_query):
                    answer = "Please restate the company, metric, and period so I can answer this safely."
                    return self._control_result(
                        request=request,
                        status="CLARIFICATION_REQUIRED",
                        answer=answer,
                        reason_codes=["CONTEXT_UNAVAILABLE"],
                        service=None,
                        replay=False,
                    )
                execution_history = None
            else:
                service = self.get_conversation_service()
                replay = bool(
                    service.is_request_processed(
                        user_id=request.user_id,
                        session_id=request.session_id,
                        request_id=request.request_id,
                        original_query=request.original_query,
                    )
                )
                if not history and self.active_query_requires_context(request.original_query):
                    answer = "I need the earlier company, metric, and period context before I can answer this safely."
                    return self._control_result(
                        request=request,
                        status="CLARIFICATION_REQUIRED",
                        answer=answer,
                        reason_codes=["CONTEXT_UNAVAILABLE"],
                        service=service,
                        replay=replay,
                    )
                try:
                    resolution = service.resolve_active(
                        request_id=request.request_id,
                        user_id=request.user_id,
                        session_id=request.session_id,
                        original_query=request.original_query,
                        prior_history=history,
                    )
                except ConversationStateConflictError:
                    raise
                except Exception as exc:
                    logger.exception("active conversation resolution failed")
                    if self.active_query_requires_context(request.original_query):
                        reason_codes = (
                            ["CONTEXT_STATE_UNAVAILABLE"]
                            if "STATE" in type(exc).__name__.upper()
                            or "SQLITE" in type(exc).__name__.upper()
                            else ["CONTEXT_RESOLUTION_FAILED"]
                        )
                        answer = "I need the company, metric, and period stated explicitly before I can answer this safely."
                        return self._control_result(
                            request=request,
                            status="CLARIFICATION_REQUIRED",
                            answer=answer,
                            reason_codes=reason_codes,
                            service=service,
                            replay=replay,
                        )
                    service = None
                    execution_history = None
                else:
                    codes = [
                        str(getattr(code, "value", code))
                        for code in (resolution.reason_codes or [])
                    ]
                    replay = replay or "IDEMPOTENT_REPLAY" in codes
                    if not resolution.supported:
                        return self._control_result(
                            request=request,
                            status="OUT_OF_SCOPE",
                            answer="This question is outside the supported financial document scope.",
                            reason_codes=codes or ["OUT_OF_SCOPE"],
                            service=service,
                            replay=replay,
                        )
                    if resolution.clarification_required:
                        answer = resolution.clarification_question or "Which company, metric, or period should I use?"
                        return self._control_result(
                            request=request,
                            status="CLARIFICATION_REQUIRED",
                            answer=answer,
                            reason_codes=codes or ["AMBIGUOUS_REFERENCE"],
                            options=resolution.clarification_options,
                            service=service,
                            replay=replay,
                        )
                    query = (resolution.standalone_query or "").strip()
                    if not query:
                        answer = "Please restate the company, metric, and period so I can answer this safely."
                        return self._control_result(
                            request=request,
                            status="CLARIFICATION_REQUIRED",
                            answer=answer,
                            reason_codes=["INVALID_RESOLUTION"],
                            service=service,
                            replay=replay,
                        )
                    query_as_resolved = bool(
                        resolution.requires_context or query != request.original_query
                    )
                    execution_history = None
        return await self._run_runtime(
            request=request,
            query=query,
            history=execution_history,
            profile=profile,
            query_as_resolved=query_as_resolved,
            service=service,
            replay=replay,
        )
