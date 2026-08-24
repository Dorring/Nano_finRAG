from __future__ import annotations

import json

from src.conversation.config import resolve_multiturn_context_mode
from src.conversation.contracts import ConversationResolution
from src.conversation.resolver import ContextualQueryResolver
from src.conversation.shadow_service import ConversationShadowService
from src.conversation.sqlite_store import SQLiteConversationStateStore
from src.services.session_manager import SessionManager


class OfflineClient:
    api_key = ""

    def chat_completion(self, *args, **kwargs):
        raise AssertionError("offline shadow tests must not call the network")


class RaisingResolver:
    def resolve(self, *args, **kwargs):
        raise TimeoutError("simulated resolver timeout")


class CaptureResolver:
    def __init__(self):
        self.calls = []

    def resolve(self, current_query, dialogue_state=None, filtered_turns=None):
        self.calls.append(
            {
                "query": current_query,
                "turns": list(filtered_turns or []),
            }
        )
        return ContextualQueryResolver(client=OfflineClient()).resolve(
            current_query,
            dialogue_state,
            filtered_turns,
        )


def make_service(tmp_path, sink=None, resolver_factory=None):
    store = SQLiteConversationStateStore(str(tmp_path / "sessions.db"))
    return ConversationShadowService(
        store,
        resolver_factory=resolver_factory
        or (lambda: ContextualQueryResolver(client=OfflineClient())),
        observation_sink=sink,
    ), store


def add_turn(session_manager, session_id, user_id, question, answer="ok"):
    session_manager.add_message(session_id, user_id, "user", question)
    session_manager.add_message(session_id, user_id, "assistant", answer)


def test_first_turn_and_relative_period_use_prior_session_history(tmp_path):
    session_manager = SessionManager(str(tmp_path / "sessions.db"))
    observations = []
    service, store = make_service(tmp_path, observations.append)

    first = service.observe(
        request_id="r1",
        user_id=7,
        session_id="s1",
        original_query="Apple FY2024 Revenue?",
        prior_history=[],
    )
    assert first.shadow_status == "OK"
    assert first.resolver_invoked is False
    assert first.current_turn_included_exactly_once is True
    assert store.get(7, "s1").active_entity == "Apple"

    add_turn(session_manager, "s1", 7, "Apple FY2024 Revenue?")
    prior = session_manager.get_recent_messages("s1", 7)
    second = service.observe(
        request_id="r2",
        user_id=7,
        session_id="s1",
        original_query="What about the previous year?",
        prior_history=prior,
    )
    assert second.shadow_status == "OK"
    assert second.resolver_invoked is True
    assert "Apple" in (second.shadow_standalone_query or "")
    assert "FY2023" in (second.shadow_standalone_query or "")
    assert second.raw_history_turn_count == 1
    assert second.selected_turn_count == 1
    assert len(observations) == 2


def test_topic_switch_and_follow_up_use_new_active_state(tmp_path):
    session_manager = SessionManager(str(tmp_path / "sessions.db"))
    service, store = make_service(tmp_path)

    service.observe(
        request_id="r1",
        user_id=1,
        session_id="switch",
        original_query="Apple FY2024 Revenue?",
        prior_history=[],
    )
    add_turn(session_manager, "switch", 1, "Apple FY2024 Revenue?")
    prior = session_manager.get_recent_messages("switch", 1)
    switched = service.observe(
        request_id="r2",
        user_id=1,
        session_id="switch",
        original_query="Tesla FY2024 Operating Margin?",
        prior_history=prior,
    )
    assert switched.topic_switch is True
    assert store.get(1, "switch").active_entity == "Tesla"

    add_turn(session_manager, "switch", 1, "Tesla FY2024 Operating Margin?")
    follow_up = service.observe(
        request_id="r3",
        user_id=1,
        session_id="switch",
        original_query="What about the previous year?",
        prior_history=session_manager.get_recent_messages("switch", 1),
    )
    assert "Tesla" in (follow_up.shadow_standalone_query or "")
    assert "Operating Margin" in (follow_up.shadow_standalone_query or "")
    assert "FY2023" in (follow_up.shadow_standalone_query or "")


def test_ambiguity_is_observed_without_mutating_active_metric(tmp_path):
    session_manager = SessionManager(str(tmp_path / "sessions.db"))
    service, store = make_service(tmp_path)

    service.observe(
        request_id="r1",
        user_id=2,
        session_id="ambiguous",
        original_query="Apple Revenue and Operating Margin FY2024?",
        prior_history=[],
    )
    state_before = store.get(2, "ambiguous")
    assert state_before.active_metric is None
    add_turn(
        session_manager,
        "ambiguous",
        2,
        "Apple Revenue and Operating Margin FY2024?",
    )

    observation = service.observe(
        request_id="r2",
        user_id=2,
        session_id="ambiguous",
        original_query="What about the previous year?",
        prior_history=session_manager.get_recent_messages("ambiguous", 2),
    )
    assert observation.shadow_status == "CLARIFICATION"
    assert observation.clarification_required is True
    state_after = store.get(2, "ambiguous")
    assert state_after.turn_count == state_before.turn_count
    assert state_after.active_metric is None


def test_shadow_failure_is_best_effort_and_state_is_not_fabricated(tmp_path):
    service, store = make_service(
        tmp_path,
        resolver_factory=lambda: RaisingResolver(),
    )
    observation = service.observe(
        request_id="failure",
        user_id=3,
        session_id="failure-session",
        original_query="What about last year?",
        prior_history=[
            {"role": "user", "content": "Apple FY2024 Revenue?"},
            {"role": "assistant", "content": "Revenue was $999B."},
        ],
    )
    assert observation.shadow_status == "ERROR"
    assert observation.shadow_error_code == "TIMEOUTERROR"
    assert store.get(3, "failure-session") is None


def test_state_isolation_and_process_restart(tmp_path):
    session_manager = SessionManager(str(tmp_path / "sessions.db"))
    service, store = make_service(tmp_path)

    service.observe(
        request_id="a1",
        user_id=10,
        session_id="same",
        original_query="Apple FY2024 Revenue?",
        prior_history=[],
    )
    service.observe(
        request_id="b1",
        user_id=11,
        session_id="same",
        original_query="Tesla FY2024 Revenue?",
        prior_history=[],
    )
    assert store.get(10, "same").active_entity == "Apple"
    assert store.get(11, "same").active_entity == "Tesla"

    add_turn(session_manager, "same", 10, "Apple FY2024 Revenue?")
    store.close()
    restarted_service, restarted_store = make_service(tmp_path)
    observation = restarted_service.observe(
        request_id="a2",
        user_id=10,
        session_id="same",
        original_query="What about the previous year?",
        prior_history=session_manager.get_recent_messages("same", 10),
    )
    assert "Apple" in (observation.shadow_standalone_query or "")
    assert "FY2023" in (observation.shadow_standalone_query or "")
    assert restarted_store.get(11, "same").active_entity == "Tesla"


def test_current_turn_is_supplied_once_and_context_metrics_are_recorded(tmp_path):
    session_manager = SessionManager(str(tmp_path / "sessions.db"))
    capture = CaptureResolver()
    service, _ = make_service(tmp_path, resolver_factory=lambda: capture)

    service.observe(
        request_id="once-1",
        user_id=4,
        session_id="once",
        original_query="Apple FY2024 Revenue?",
        prior_history=[],
    )
    assert len(capture.calls[0]["turns"]) == 0
    add_turn(session_manager, "once", 4, "Apple FY2024 Revenue?")
    observation = service.observe(
        request_id="once-2",
        user_id=4,
        session_id="once",
        original_query="What about the previous year?",
        prior_history=session_manager.get_recent_messages("once", 4),
    )
    assert len(capture.calls[1]["turns"]) == 1
    assert capture.calls[1]["turns"][0].user_query == "Apple FY2024 Revenue?"
    assert observation.current_turn_included_exactly_once is True
    assert observation.raw_history_tokens > 0
    assert observation.selected_context_tokens > 0


def test_assistant_metadata_contains_only_structured_provenance(tmp_path):
    service, store = make_service(tmp_path)
    service.observe(
        request_id="meta-1",
        user_id=5,
        session_id="meta",
        original_query="Apple FY2024 Revenue?",
        prior_history=[],
    )
    service.record_assistant_turn(
        user_id=5,
        session_id="meta",
        referenced_evidence_ids=["chunk-1", "citation-1"],
    )
    state = store.get(5, "meta")
    assert state.referenced_evidence_ids == ["chunk-1", "citation-1"]
    raw = (
        store._get_conn()
        .execute(
            "SELECT structured_state_json FROM conversation_states "
            "WHERE user_id = 5 AND session_id = 'meta'",
        )
        .fetchone()[0]
    )
    payload = json.loads(raw)
    assert payload["recent_turns"] == []
    assert "answer_numeric_value" not in payload
    assert "Revenue was $999B" not in raw


def test_delete_state_is_explicit_and_session_scoped(tmp_path):
    service, store = make_service(tmp_path)
    service.observe(
        request_id="delete-1",
        user_id=6,
        session_id="delete",
        original_query="Apple FY2024 Revenue?",
        prior_history=[],
    )
    assert store.exists(6, "delete")
    assert service.delete_state(user_id=6, session_id="delete") is True
    assert store.get(6, "delete") is None


def test_stateless_shadow_is_bypassed_without_affecting_v1(tmp_path):
    observations = []
    service, _ = make_service(tmp_path, observations.append)
    observation = service.observe(
        request_id="stateless",
        user_id=8,
        session_id=None,
        original_query="Apple FY2024 Revenue?",
        prior_history=[],
    )
    assert observation.shadow_status == "BYPASSED"
    assert observation.resolver_invoked is False
    assert observations[0]["shadow_status"] == "BYPASSED"


def test_mode_contract_has_no_active_on_state():
    assert resolve_multiturn_context_mode(environ={}) == "off"
    assert (
        resolve_multiturn_context_mode(environ={"MULTITURN_CONTEXT_MODE": "shadow"})
        == "shadow"
    )
    assert (
        resolve_multiturn_context_mode(
            environ={
                "MULTITURN_CONTEXT_MODE": "off",
                "MULTITURN_CONTEXT_ENABLED": "true",
            },
        )
        == "off"
    )
    assert (
        resolve_multiturn_context_mode(
            environ={"MULTITURN_CONTEXT_ENABLED": "true"},
        )
        == "shadow"
    )
    try:
        resolve_multiturn_context_mode(environ={"MULTITURN_CONTEXT_MODE": "on"})
    except ValueError as exc:
        assert "not available in I5" in str(exc)
    else:
        raise AssertionError("active context mode must fail validation")


class ErrorMarkedResolver:
    class client:
        last_error_code = "HTTP_429"

    def resolve(self, current_query, dialogue_state=None, filtered_turns=None):
        return ConversationResolution(
            supported=True,
            standalone_query=current_query,
            reason_codes=["REFERENCE_RESOLVED"],
        )


def test_provider_error_is_visible_even_when_deterministic_shadow_fallback_succeeds(tmp_path):
    service, store = make_service(
        tmp_path,
        resolver_factory=lambda: ErrorMarkedResolver(),
    )
    observation = service.observe(
        request_id="provider-error",
        user_id=9,
        session_id="provider-error",
        original_query="Apple FY2024 Revenue?",
        prior_history=[],
    )
    assert observation.shadow_status == "ERROR"
    assert observation.shadow_error_code == "HTTP_429"
    assert store.get(9, "provider-error") is not None
