"""I5-R1 real endpoint verification for Conversation Shadow isolation."""

from __future__ import annotations

import os
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.conversation.contracts import ConversationResolution
from src.conversation.resolver import ContextualQueryResolver
from src.conversation.shadow_service import ConversationShadowService
from src.conversation.sqlite_store import SQLiteConversationStateStore
from src.services.session_manager import SessionManager


class OfflineClient:
    api_key = ""

    def chat_completion(self, *args, **kwargs):
        raise AssertionError("API tests must not call an external provider")


class RaisingResolver:
    def resolve(self, *args, **kwargs):
        raise TimeoutError("simulated Qwen timeout")


class ErrorMarkedResolver:
    class client:
        last_error_code = "INVALID_JSON"

    def resolve(self, current_query, dialogue_state=None, filtered_turns=None):
        return ConversationResolution(
            supported=True,
            standalone_query=current_query,
            reason_codes=["REFERENCE_RESOLVED"],
        )


class CaptureResolver:
    def __init__(self):
        self.calls: list[dict] = []

    def resolve(self, current_query, dialogue_state=None, filtered_turns=None):
        self.calls.append({"query": current_query, "turns": list(filtered_turns or [])})
        return ContextualQueryResolver(client=OfflineClient()).resolve(
            current_query, dialogue_state, filtered_turns
        )


class FailingStateStore(SQLiteConversationStateStore):
    def save_state(self, *args, **kwargs):
        raise sqlite3.OperationalError("simulated shadow SQLite write failure")


class FakeRAGEngine:
    def __init__(self):
        self.calls: list[dict] = []

    async def query(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "answer": "Revenue was $100B.",
            "sources": [{"filename": "annual_report.pdf", "page": 4, "chunk_id": "chunk-revenue"}],
            "searched_docs": ["annual_report.pdf"],
            "rewritten_question": None,
            "confidence": 0.91,
            "context_sufficient": True,
            "intent": "document_qa",
            "intent_confidence": 0.88,
            "trace_id": "trace-i5-r1",
            "retrieved_chunks": [{"chunk_id": "chunk-revenue"}],
            "retrieval_debug": {"candidate_count": 1},
            "calculations": [],
        }


def _make_service(root, *, sink=None, resolver_factory=None, store_cls=SQLiteConversationStateStore):
    root.mkdir(parents=True, exist_ok=True)
    db_path = str(root / "session.db")
    store = store_cls(db_path)
    service = ConversationShadowService(
        store,
        resolver_factory=resolver_factory
        or (lambda: ContextualQueryResolver(client=OfflineClient())),
        observation_sink=sink.append if sink is not None else None,
    )
    return service, store, SessionManager(db_path)


@pytest.fixture
def api_context(tmp_path):
    from fastapi.testclient import TestClient
    from src.main import app
    from src.services.auth import get_current_user

    observations: list[dict] = []
    service, store, session_manager = _make_service(tmp_path, sink=observations)
    user_ref = {"id": 42}
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=user_ref["id"], email="i5-r1@example.test"
    )
    with TestClient(app) as client:
        yield SimpleNamespace(
            app=app,
            client=client,
            service=service,
            store=store,
            session_manager=session_manager,
            observations=observations,
            user_ref=user_ref,
        )
    app.dependency_overrides.clear()
    store.close()
    session_manager.close()


def _post(ctx, *, mode, question="Apple FY2024 Revenue?", session_id="shadow-session", service=None):
    engine = FakeRAGEngine()
    with (
        patch("src.main.get_rag_engine", return_value=engine),
        patch("src.main._resolve_query_document_names_for_user", return_value=["annual_report.pdf"]),
        patch("src.main.memory_store", SimpleNamespace(get_profile=lambda user_id: {})),
        patch("src.main.session_manager", ctx.session_manager),
        patch("src.main.get_conversation_shadow_service", return_value=service or ctx.service),
        patch.dict(
            os.environ,
            {"FINANCIAL_RUNTIME_ADAPTER_ENABLED": "true", "MULTITURN_CONTEXT_MODE": mode},
            clear=False,
        ),
    ):
        response = ctx.client.post(
            "/query",
            json={
                "question": question,
                "document_names": ["annual_report.pdf"],
                "n_results": 5,
                "session_id": session_id,
            },
        )
    return response, engine


def _clear(ctx, session_id):
    with (
        patch("src.main.session_manager", ctx.session_manager),
        patch("src.main.get_conversation_shadow_service", return_value=ctx.service),
    ):
        return ctx.client.post(
            "/sessions/clear",
            json={
                "question": "clear session",
                "session_id": session_id,
            },
        )


def test_off_shadow_response_and_raw_history_parity(api_context):
    off_service = Mock()
    off, off_engine = _post(api_context, mode="off", service=off_service)
    assert off.status_code == 200
    off_service.observe.assert_not_called()
    off_history = api_context.session_manager.get_recent_messages("shadow-session", 42)

    assert _clear(api_context, "shadow-session").status_code == 200
    shadow, shadow_engine = _post(api_context, mode="shadow")
    assert shadow.status_code == 200
    assert shadow.json() == off.json()
    assert api_context.session_manager.get_recent_messages("shadow-session", 42) == off_history
    assert shadow_engine.calls == off_engine.calls
    assert api_context.observations[-1]["shadow_status"] == "OK"


def test_shadow_provider_and_sqlite_failures_do_not_change_v1(api_context, tmp_path):
    timeout_observations: list[dict] = []
    timeout_service, timeout_store, timeout_sessions = _make_service(
        tmp_path / "timeout",
        sink=timeout_observations,
        resolver_factory=lambda: RaisingResolver(),
    )
    response, engine = _post(api_context, mode="shadow", service=timeout_service)
    assert response.status_code == 200
    assert response.json()["answer"] == "Revenue was $100B."
    assert engine.calls
    assert timeout_observations[-1]["shadow_error_code"] == "TIMEOUTERROR"

    invalid_observations: list[dict] = []
    invalid_service, invalid_store, invalid_sessions = _make_service(
        tmp_path / "invalid",
        sink=invalid_observations,
        resolver_factory=lambda: ErrorMarkedResolver(),
    )
    response, _ = _post(
        api_context,
        mode="shadow",
        question="Microsoft FY2024 Revenue?",
        session_id="invalid",
        service=invalid_service,
    )
    assert response.status_code == 200
    assert invalid_observations[-1]["shadow_error_code"] == "INVALID_JSON"

    failing_root = tmp_path / "failing"
    failing_root.mkdir()
    failing_store = FailingStateStore(str(failing_root / "session.db"))
    failing_service = ConversationShadowService(
        failing_store,
        resolver_factory=lambda: ContextualQueryResolver(client=OfflineClient()),
    )
    response, _ = _post(
        api_context,
        mode="shadow",
        question="Tesla FY2024 Revenue?",
        session_id="sqlite-failure",
        service=failing_service,
    )
    assert response.status_code == 200
    timeout_store.close()
    timeout_sessions.close()
    invalid_store.close()
    invalid_sessions.close()
    failing_store.close()


def test_restart_current_turn_once_clear_and_user_isolation(api_context, tmp_path):
    root = tmp_path / "restart"
    capture = CaptureResolver()
    service, store, sessions = _make_service(
        root, sink=api_context.observations, resolver_factory=lambda: capture
    )
    api_context.store.close()
    api_context.session_manager.close()
    api_context.service = service
    api_context.store = store
    api_context.session_manager = sessions

    first, _ = _post(
        api_context, mode="shadow", question="Apple FY2024 Revenue?", session_id="restart-session"
    )
    assert first.status_code == 200
    assert len(capture.calls[0]["turns"]) == 0
    assert store.get(42, "restart-session").active_entity == "Apple"

    store.close()
    restarted_store = SQLiteConversationStateStore(store.db_path)
    api_context.store = restarted_store
    api_context.service = ConversationShadowService(
        restarted_store,
        resolver_factory=lambda: capture,
        observation_sink=api_context.observations.append,
    )
    second, _ = _post(
        api_context,
        mode="shadow",
        question="What about the previous year?",
        session_id="restart-session",
    )
    assert second.status_code == 200
    assert len(capture.calls[-1]["turns"]) == 1
    assert "Apple" in (api_context.observations[-1]["shadow_standalone_query"] or "")
    assert "FY2023" in (api_context.observations[-1]["shadow_standalone_query"] or "")

    assert _clear(api_context, "restart-session").status_code == 200
    assert restarted_store.get(42, "restart-session") is None

    api_context.user_ref["id"] = 42
    _post(api_context, mode="shadow", question="Apple FY2024 Revenue?", session_id="same-session")
    api_context.user_ref["id"] = 43
    _post(api_context, mode="shadow", question="Tesla FY2024 Revenue?", session_id="same-session")
    assert restarted_store.get(42, "same-session").active_entity == "Apple"
    assert restarted_store.get(43, "same-session").active_entity == "Tesla"


def test_active_stateless_context_dependent_query_clarifies_without_calling_v1(api_context):
    engine = FakeRAGEngine()
    with (
        patch("src.main.get_rag_engine", return_value=engine),
        patch("src.main._resolve_query_document_names_for_user", return_value=["annual_report.pdf"]),
        patch("src.main.memory_store", SimpleNamespace(get_profile=lambda user_id: {})),
        patch("src.main.session_manager", api_context.session_manager),
        patch.dict(
            os.environ,
            {"FINANCIAL_RUNTIME_ADAPTER_ENABLED": "true", "MULTITURN_CONTEXT_MODE": "on"},
            clear=False,
        ),
    ):
        response = api_context.client.post(
            "/query",
            json={
                "question": "What about the previous year?",
                "document_names": ["annual_report.pdf"],
                "n_results": 5,
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "CLARIFICATION_REQUIRED"
    assert payload["clarification"]["reason_codes"] == ["CONTEXT_UNAVAILABLE"]
    assert payload["answer"]
    assert engine.calls == []


def test_active_context_dependent_query_without_prior_turn_clarifies(api_context):
    response, engine = _post(
        api_context,
        mode="on",
        question="What about the previous year?",
        session_id="active-empty-context",
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "CLARIFICATION_REQUIRED"
    assert payload["clarification"]["reason_codes"] == ["CONTEXT_UNAVAILABLE"]
    assert engine.calls == []


def test_active_relative_period_reaches_v1_with_rewrite_bypass(api_context):
    first, _ = _post(
        api_context,
        mode="on",
        question="Apple FY2024 Revenue?",
        session_id="active-relative",
    )
    assert first.status_code == 200

    second, engine = _post(
        api_context,
        mode="on",
        question="What about the previous year?",
        session_id="active-relative",
    )

    assert second.status_code == 200
    payload = second.json()
    assert payload["rewritten_question"].startswith("What was Apple FY2023")
    assert engine.calls[0]["question"].startswith("What was Apple FY2023")
    assert engine.calls[0]["conversation_history"] is None
    assert engine.calls[0]["query_as_resolved"] is True


def test_active_ambiguity_returns_control_response_without_v1(api_context):
    first, _ = _post(
        api_context,
        mode="on",
        question="Apple Revenue and Operating Margin FY2024?",
        session_id="active-ambiguous",
    )
    assert first.status_code == 200
    state_before = api_context.store.get(42, "active-ambiguous")
    assert state_before.active_metric is None
    turn_count_before = state_before.turn_count

    second, engine = _post(
        api_context,
        mode="on",
        question="What about the previous year?",
        session_id="active-ambiguous",
    )

    assert second.status_code == 200
    payload = second.json()
    assert payload["status"] == "CLARIFICATION_REQUIRED"
    assert payload["clarification"]["reason_codes"] == ["AMBIGUOUS_METRIC"]
    assert engine.calls == []
    state_after = api_context.store.get(42, "active-ambiguous")
    assert state_after.turn_count == turn_count_before
    assert state_after.active_metric is None


def test_active_resolver_failure_clarifies_contextual_query(api_context):
    first, _ = _post(
        api_context,
        mode="on",
        question="Apple FY2024 Revenue?",
        session_id="active-timeout",
    )
    assert first.status_code == 200
    failing_service = ConversationShadowService(
        api_context.store,
        resolver_factory=lambda: RaisingResolver(),
    )

    second, engine = _post(
        api_context,
        mode="on",
        question="What about the previous year?",
        session_id="active-timeout",
        service=failing_service,
    )

    assert second.status_code == 200
    payload = second.json()
    assert payload["status"] == "CLARIFICATION_REQUIRED"
    assert payload["clarification"]["reason_codes"] == ["CONTEXT_RESOLUTION_FAILED"]
    assert engine.calls == []


def test_active_resolver_failure_allows_self_contained_v1_without_history(api_context):
    first, _ = _post(
        api_context,
        mode="on",
        question="Apple FY2024 Revenue?",
        session_id="active-self-contained-timeout",
    )
    assert first.status_code == 200
    failing_service = ConversationShadowService(
        api_context.store,
        resolver_factory=lambda: RaisingResolver(),
    )

    second, engine = _post(
        api_context,
        mode="on",
        question="Microsoft FY2024 Revenue?",
        session_id="active-self-contained-timeout",
        service=failing_service,
    )

    assert second.status_code == 200
    assert second.json()["answer"] == "Revenue was $100B."
    assert engine.calls[0]["question"] == "Microsoft FY2024 Revenue?"
    assert engine.calls[0]["conversation_history"] is None
    assert "query_as_resolved" not in engine.calls[0]


def test_active_explicit_topic_switch_does_not_forward_stale_history(api_context):
    first, _ = _post(
        api_context,
        mode="on",
        question="Apple FY2024 Revenue?",
        session_id="active-switch",
    )
    assert first.status_code == 200

    second, engine = _post(
        api_context,
        mode="on",
        question="Tesla FY2024 Operating Margin?",
        session_id="active-switch",
    )

    assert second.status_code == 200
    assert engine.calls[0]["question"] == "Tesla FY2024 Operating Margin?"
    assert engine.calls[0]["conversation_history"] is None
    assert "Apple" not in engine.calls[0]["question"]
    assert api_context.store.get(42, "active-switch").active_entity == "Tesla"


def test_active_cross_turn_calculation_does_not_use_assistant_numeric_text(api_context):
    first, _ = _post(
        api_context,
        mode="on",
        question="Apple FY2024 Revenue?",
        session_id="active-calc",
    )
    assert first.status_code == 200
    second, _ = _post(
        api_context,
        mode="on",
        question="What about the previous year?",
        session_id="active-calc",
    )
    assert second.status_code == 200
    api_context.session_manager.add_message(
        "active-calc",
        42,
        "assistant",
        "Apple FY2023 revenue was $999B.",
    )

    third, engine = _post(
        api_context,
        mode="on",
        question="How much did it grow?",
        session_id="active-calc",
    )

    assert third.status_code == 200
    assert "Calculate the change in Apple" in engine.calls[0]["question"]
    assert "$999B" not in engine.calls[0]["question"]
    assert engine.calls[0]["conversation_history"] is None
