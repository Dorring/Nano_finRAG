"""Production Conversation Shadow integration.

The shadow service observes the real session lifecycle without becoming part
of financial answer execution. It reads prior raw turns from SessionManager,
projects them into the existing conversation contracts, persists only the
structured DialogueState through SQLiteConversationStateStore, and emits a
privacy-safe observation. It never returns a query to the financial runtime.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .bailian_client import BailianClient
from .context_budget import ContextBudgetManager
from .contracts import (
    AssistantProvenance,
    ConversationResolution,
    ConversationTurnOutcome,
    DialogueState,
    DialogueTurn,
    ReasonCode,
)
from .resolver import ContextualQueryResolver
from .service import ConversationContextManager
from .sqlite_store import (
    ConversationStateConflictError,
    SQLiteConversationStateStore,
)
from .store import ConversationStateStore

logger = logging.getLogger(__name__)
ObservationSink = Callable[[dict[str, Any]], None]


class UserScopedConversationStore(ConversationStateStore):
    """Adapt the historical one-key component API to production identity."""

    def __init__(self, backend: SQLiteConversationStateStore, user_id: int) -> None:
        self.backend = backend
        self.user_id = int(user_id)

    def get_state(self, conversation_id: str) -> DialogueState | None:
        return self.backend.get_state(conversation_id, user_id=self.user_id)

    def save_state(self, state: DialogueState) -> None:
        self.backend.save_state(state, user_id=self.user_id)

    def clear_state(self, conversation_id: str) -> None:
        self.backend.clear_state(conversation_id, user_id=self.user_id)


@dataclass
class ConversationShadowObservation:
    """Structured, non-answer telemetry for one shadow invocation."""

    request_id: str
    user_id: int
    session_id: str | None
    original_query: str
    shadow_standalone_query: str | None = None
    requires_context: bool = False
    supported: bool = True
    ambiguity_detected: bool = False
    clarification_required: bool = False
    topic_switch: bool = False
    inherited_fields: list[str] = field(default_factory=list)
    explicit_fields: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    relevant_turn_ids: list[str] = field(default_factory=list)
    resolver_invoked: bool = False
    resolver_latency_ms: float = 0.0
    selected_context_tokens: int = 0
    raw_history_turn_count: int = 0
    selected_turn_count: int = 0
    dropped_turn_count: int = 0
    compressed_history_tokens: int = 0
    resolver_input_tokens: int = 0
    total_session_turns: int = 0
    raw_history_tokens: int = 0
    current_turn_included_exactly_once: bool = True
    legacy_rewritten_query: str | None = None
    shadow_status: str = "BYPASSED"
    shadow_error_code: str | None = None
    resolver_model: str = "qwen3.6-flash"
    resolver_thinking: bool = False
    state_persisted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


class ConversationShadowService:
    """Best-effort observer for the real /query session lifecycle."""

    def __init__(
        self,
        state_store: SQLiteConversationStateStore,
        resolver_factory: Callable[[], ContextualQueryResolver] | None = None,
        manager_factory: Callable[
            [ConversationStateStore, ContextualQueryResolver],
            ConversationContextManager,
        ]
        | None = None,
        observation_sink: ObservationSink | None = None,
    ) -> None:
        self.state_store = state_store
        self.resolver_factory = resolver_factory or (
            lambda: ContextualQueryResolver(
                client=BailianClient(
                    model="qwen3.6-flash",
                    enable_thinking=False,
                ),
            )
        )
        self.manager_factory = manager_factory or (
            lambda store, resolver: ConversationContextManager(
                store=store, resolver=resolver
            )
        )
        self.observation_sink = observation_sink
        self._token_counter = ContextBudgetManager()

    @staticmethod
    def _history_to_turns(
        history: Sequence[Mapping[str, Any]] | None,
    ) -> list[DialogueTurn]:
        """Project prior SessionManager messages into transient turns.

        SessionManager history is loaded before the endpoint commits the
        current user message. The projection is read-only and is never
        persisted as raw history in the structured state store.
        """
        turns: list[DialogueTurn] = []
        pending: DialogueTurn | None = None
        for index, message in enumerate(history or []):
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role", "")).strip().lower()
            content = message.get("content")
            if not isinstance(content, str) or not content:
                continue
            if role == "user":
                if pending is not None:
                    turns.append(pending)
                pending = DialogueTurn(
                    turn_id=f"session_message_{index}",
                    user_query=content,
                    standalone_query=content,
                )
            elif role == "assistant" and pending is not None:
                pending.assistant_response = content
        if pending is not None:
            turns.append(pending)
        return turns

    def _new_manager(self, user_id: int) -> ConversationContextManager:
        scoped_store = UserScopedConversationStore(self.state_store, user_id)
        resolver = self.resolver_factory()
        return self.manager_factory(scoped_store, resolver)

    def get_state(self, *, user_id: int, session_id: str) -> DialogueState | None:
        """Return structured state at the production identity boundary."""
        return self.state_store.get(int(user_id), session_id)

    def is_request_processed(
        self,
        *,
        user_id: int,
        session_id: str | None,
        request_id: str,
        original_query: str,
    ) -> bool:
        """Check request idempotency without replaying financial execution."""
        if not session_id:
            return False
        state = self.get_state(user_id=user_id, session_id=session_id)
        if state is None or state.last_processed_request_id is None:
            return False
        if state.last_processed_request_id != request_id:
            return False
        if (
            state.last_processed_original_query is not None
            and state.last_processed_original_query != original_query
        ):
            raise ConversationStateConflictError(
                "request_id was reused with a different original query",
            )
        return True

    @staticmethod
    def _replay_resolution(
        state: DialogueState,
        original_query: str,
    ) -> ConversationResolution:
        """Reconstruct prior semantic outcome without another state write."""
        pending = state.pending_clarification
        if pending is not None:
            field_name = (
                pending.unresolved_fields[0]
                if pending.unresolved_fields
                else "metric"
            )
            options = list(pending.candidates)
            suffix = f" ({', '.join(options)})" if options else ""
            return ConversationResolution(
                supported=True,
                requires_context=True,
                standalone_query="",
                resolved_entity=pending.entity or state.active_entity,
                resolved_period=pending.period or state.active_period,
                ambiguity_detected=True,
                clarification_required=True,
                clarification_question=(
                    f"Which {field_name} should I use{suffix}?"
                ),
                clarification_options=options,
                reason_codes=list(pending.reason_codes)
                + [ReasonCode.IDEMPOTENT_REPLAY],
            )
        standalone = state.last_resolved_query or original_query
        return ConversationResolution(
            supported=True,
            requires_context=standalone != original_query,
            standalone_query=standalone,
            resolved_entity=state.active_entity,
            resolved_metric=state.active_metric,
            resolved_period=state.active_period,
            resolved_scope=state.active_scope,
            reason_codes=[ReasonCode.IDEMPOTENT_REPLAY],
        )

    def _emit(self, observation: ConversationShadowObservation) -> None:
        payload = observation.to_dict()
        if self.observation_sink is not None:
            try:
                self.observation_sink(payload)
            except Exception:
                logger.exception("conversation shadow observation sink failed")
        try:
            logger.info(
                "conversation_shadow_observation",
                extra={"conversation_shadow": payload},
            )
        except Exception:
            pass

    def _raw_history_tokens(self, history: Sequence[Mapping[str, Any]] | None) -> int:
        total = 0
        for message in history or []:
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    total += self._token_counter.count_tokens(content)
        return total

    def observe(
        self,
        *,
        request_id: str,
        user_id: int,
        session_id: str | None,
        original_query: str,
        prior_history: Sequence[Mapping[str, Any]] | None,
        resolution_sink: Callable[[ConversationResolution], None] | None = None,
        raise_errors: bool = False,
    ) -> ConversationShadowObservation:
        """Run one shadow resolution and never raise into the V1 request."""
        history = list(prior_history or [])
        history_turns = self._history_to_turns(history)
        observation = ConversationShadowObservation(
            request_id=request_id,
            user_id=int(user_id),
            session_id=session_id,
            original_query=original_query,
            raw_history_turn_count=len(history_turns),
            total_session_turns=len(history_turns),
            raw_history_tokens=self._raw_history_tokens(history),
        )
        if not session_id:
            self._emit(observation)
            return observation

        diagnostics: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            replay_state = self.get_state(
                user_id=int(user_id),
                session_id=session_id,
            )
            is_replay = bool(
                replay_state is not None
                and replay_state.last_processed_request_id == request_id
            )
            if is_replay:
                if (
                    replay_state.last_processed_original_query is not None
                    and replay_state.last_processed_original_query
                    != original_query
                ):
                    raise ConversationStateConflictError(
                        "request_id was reused with a different original query",
                    )
                resolution = self._replay_resolution(
                    replay_state,
                    original_query,
                )
                manager = None
            else:
                manager = self._new_manager(int(user_id))
                resolution = manager.process_user_turn(
                    session_id,
                    original_query,
                    history_turns=history_turns,
                    diagnostics=diagnostics,
                )
            if resolution_sink is not None:
                resolution_sink(resolution)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            reason_codes = [
                str(getattr(code, "value", code))
                for code in (resolution.reason_codes or [])
            ]
            resolver_invoked = not (
                ReasonCode.RESOLVER_BYPASS in reason_codes
                or (
                    ReasonCode.NO_CONTEXT_REQUIRED in reason_codes
                    and not resolution.requires_context
                )
            )
            client_error = getattr(
                getattr(manager, "resolver", None) if manager is not None else None,
                "client",
                None,
            )
            client_error_code = getattr(client_error, "last_error_code", None)
            state = self.state_store.get(int(user_id), session_id)
            compressed_tokens = (
                self._token_counter.count_tokens(state.compressed_history or "")
                if state is not None
                else 0
            )
            observation.shadow_standalone_query = resolution.standalone_query
            observation.requires_context = bool(resolution.requires_context)
            observation.supported = bool(resolution.supported)
            observation.ambiguity_detected = bool(resolution.ambiguity_detected)
            observation.clarification_required = bool(resolution.clarification_required)
            observation.topic_switch = bool(resolution.topic_switch)
            observation.inherited_fields = list(resolution.inherited_fields or [])
            observation.explicit_fields = list(resolution.explicit_fields or [])
            observation.reason_codes = reason_codes
            observation.relevant_turn_ids = list(resolution.relevant_turn_ids or [])
            observation.resolver_invoked = False if is_replay else resolver_invoked
            observation.resolver_latency_ms = round(elapsed_ms, 3)
            observation.selected_context_tokens = int(
                diagnostics.get("estimated_context_tokens", 0)
            )
            observation.resolver_input_tokens = observation.selected_context_tokens
            observation.raw_history_turn_count = int(
                diagnostics.get("raw_history_turn_count", len(history_turns))
            )
            observation.selected_turn_count = int(
                diagnostics.get("selected_turn_count", 0)
            )
            observation.dropped_turn_count = int(
                diagnostics.get(
                    "dropped_turn_count",
                    max(
                        0,
                        observation.raw_history_turn_count
                        - observation.selected_turn_count,
                    ),
                )
            )
            observation.compressed_history_tokens = compressed_tokens
            observation.shadow_error_code = client_error_code
            observation.shadow_status = (
                "ERROR"
                if client_error_code
                else "CLARIFICATION"
                if resolution.clarification_required
                else "OUT_OF_SCOPE"
                if not resolution.supported
                else "OK"
            )
            observation.state_persisted = bool(
                resolution.supported
            )
        except Exception as exc:
            observation.resolver_latency_ms = round(
                (time.perf_counter() - started) * 1000.0, 3
            )
            observation.shadow_status = "ERROR"
            observation.shadow_error_code = type(exc).__name__.upper()
            observation.resolver_invoked = True
            self._emit(observation)
            if raise_errors:
                raise
            return observation
        self._emit(observation)
        return observation

    def resolve_active(
        self,
        *,
        request_id: str,
        user_id: int,
        session_id: str,
        original_query: str,
        prior_history: Sequence[Mapping[str, Any]] | None,
    ) -> ConversationResolution:
        """Resolve one active request and propagate state/resolver failures.

        Active callers need the structured resolution and must not silently
        convert a resolver or state-store failure into a guessed financial
        query. The existing observe() path remains best-effort by default.
        """
        resolutions: list[ConversationResolution] = []
        self.observe(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            original_query=original_query,
            prior_history=prior_history,
            resolution_sink=resolutions.append,
            raise_errors=True,
        )
        if not resolutions:
            raise RuntimeError("active conversation resolution produced no result")
        return resolutions[0]

    @staticmethod
    def _clean_ids(values: Sequence[str] | None) -> list[str]:
        result: list[str] = []
        for value in values or []:
            if isinstance(value, str) and value and value not in result:
                result.append(value)
        return result

    def _save_final_state(
        self,
        *,
        user_id: int,
        state: DialogueState,
        expected_state_version: int | None,
    ) -> None:
        self.state_store.save_state(
            state,
            user_id=int(user_id),
            expected_state_version=expected_state_version,
        )

    def record_control_turn(
        self,
        *,
        user_id: int,
        session_id: str | None,
        request_id: str,
        original_query: str,
        outcome: str,
        reason_codes: Sequence[str] | None = None,
    ) -> bool:
        """Commit a user-visible control outcome exactly once."""
        if not session_id:
            return False
        state = self.get_state(user_id=int(user_id), session_id=session_id)
        expected_version = (
            self.state_store.get_state_version(int(user_id), session_id)
            if state is not None
            else None
        )
        if state is None:
            state = DialogueState(conversation_id=session_id)
        if state.last_processed_request_id == request_id:
            if (
                state.last_processed_original_query is not None
                and state.last_processed_original_query != original_query
            ):
                raise ConversationStateConflictError(
                    "request_id was reused with a different original query",
                )
            return False
        state.last_processed_request_id = request_id
        state.last_processed_original_query = original_query
        state.last_turn_outcome = outcome
        state.last_assistant_provenance = AssistantProvenance(
            assistant_turn_id=f"{request_id}:assistant",
            release_status="NOT_APPLICABLE",
            outcome=outcome,
        )
        self._save_final_state(
            user_id=int(user_id),
            state=state,
            expected_state_version=expected_version,
        )
        return True

    def record_assistant_turn(
        self,
        *,
        user_id: int,
        session_id: str | None,
        referenced_evidence_ids: Sequence[str] | None = None,
        citation_ids: Sequence[str] | None = None,
        calculation_ids: Sequence[str] | None = None,
        request_id: str | None = None,
        original_query: str | None = None,
        release_status: str = "NOT_APPLICABLE",
        outcome: str = ConversationTurnOutcome.FINANCIAL_ANSWER,
    ) -> bool:
        """Persist structured provenance after V1 finalization.

        The raw assistant message is written exactly once by SessionManager.
        The SQLite store strips runtime-only raw turns from structured state.
        """
        if not session_id:
            return False
        state = self.get_state(user_id=int(user_id), session_id=session_id)
        if state is None:
            return False
        expected_version = self.state_store.get_state_version(
            int(user_id),
            session_id,
        )
        if request_id is not None and state.last_processed_request_id == request_id:
            if (
                state.last_processed_original_query is not None
                and original_query is not None
                and state.last_processed_original_query != original_query
            ):
                raise ConversationStateConflictError(
                    "request_id was reused with a different original query",
                )
            return False
        evidence_ids = self._clean_ids(referenced_evidence_ids)
        citation_ids_clean = self._clean_ids(citation_ids)
        calculation_ids_clean = self._clean_ids(calculation_ids)
        state.referenced_evidence_ids = list(state.referenced_evidence_ids) + [
            value
            for value in evidence_ids
            if value not in state.referenced_evidence_ids
        ]
        assistant_turn_id = request_id or uuid.uuid4().hex
        state.last_assistant_provenance = AssistantProvenance(
            assistant_turn_id=assistant_turn_id,
            evidence_ids=evidence_ids,
            citation_ids=citation_ids_clean,
            calculation_ids=calculation_ids_clean,
            release_status=release_status,
            outcome=outcome,
        )
        state.last_turn_outcome = outcome
        if request_id is not None:
            state.last_processed_request_id = request_id
            state.last_processed_original_query = original_query
        self._save_final_state(
            user_id=int(user_id),
            state=state,
            expected_state_version=expected_version,
        )
        return True

    def delete_state(self, *, user_id: int, session_id: str) -> bool:
        return self.state_store.delete(int(user_id), session_id)

    def delete_all_states(self, *, user_id: int) -> int:
        return self.state_store.delete_all_for_user(int(user_id))
