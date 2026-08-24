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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .bailian_client import BailianClient
from .context_budget import ContextBudgetManager
from .contracts import DialogueState, DialogueTurn, ReasonCode
from .resolver import ContextualQueryResolver
from .service import ConversationContextManager
from .sqlite_store import SQLiteConversationStateStore
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
            manager = self._new_manager(int(user_id))
            resolution = manager.process_user_turn(
                session_id,
                original_query,
                history_turns=history_turns,
                diagnostics=diagnostics,
            )
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
                getattr(manager, "resolver", None),
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
            observation.resolver_invoked = resolver_invoked
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
                resolution.supported and not resolution.clarification_required
            )
        except Exception as exc:
            observation.resolver_latency_ms = round(
                (time.perf_counter() - started) * 1000.0, 3
            )
            observation.shadow_status = "ERROR"
            observation.shadow_error_code = type(exc).__name__.upper()
            observation.resolver_invoked = True
        self._emit(observation)
        return observation

    def record_assistant_turn(
        self,
        *,
        user_id: int,
        session_id: str | None,
        referenced_evidence_ids: Sequence[str] | None = None,
    ) -> None:
        """Persist structured provenance after V1 finalization.

        The raw assistant message is written exactly once by SessionManager.
        The SQLite store strips runtime-only raw turns from structured state.
        """
        if not session_id:
            return
        manager = self._new_manager(int(user_id))
        manager.record_assistant_turn(
            session_id,
            assistant_response=None,
            referenced_evidence_ids=[
                value
                for value in (referenced_evidence_ids or [])
                if isinstance(value, str) and value
            ],
        )

    def delete_state(self, *, user_id: int, session_id: str) -> bool:
        return self.state_store.delete(int(user_id), session_id)

    def delete_all_states(self, *, user_id: int) -> int:
        return self.state_store.delete_all_for_user(int(user_id))
