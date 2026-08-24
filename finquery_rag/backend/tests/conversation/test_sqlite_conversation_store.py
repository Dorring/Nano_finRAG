"""Tests for the I4 SQLite ConversationStateStore capability."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.conversation import (
    ConversationStateConflictError,
    ConversationStateIdentityError,
    ConversationStateSerializationError,
    ConversationStateVersionError,
    DialogueState,
    DialogueTurn,
    InMemoryConversationStore,
    SQLiteConversationStateStore,
)
from src.conversation.service import ConversationContextManager


def _state(
    session_id: str,
    *,
    entity: str = "Apple",
    metric: str = "Revenue",
    turn_count: int = 1,
) -> DialogueState:
    return DialogueState(
        conversation_id=session_id,
        active_entity=entity,
        active_metric=metric,
        active_period="FY2024",
        active_scope="annual",
        comparison_entity="Microsoft",
        comparison_metric="Revenue",
        comparison_period="FY2023",
        active_topic=f"{entity}_{metric}",
        last_resolved_query=f"What was {entity} {metric}?",
        referenced_turn_ids=["turn_1"],
        referenced_evidence_ids=["evidence-1"],
        recent_turns=[
            DialogueTurn(
                turn_id="turn_1",
                user_query="What was Apple FY2024 revenue?",
                standalone_query="What was Apple FY2024 revenue?",
                assistant_response="Apple FY2024 revenue was $999B.",
                referenced_evidence_ids=["evidence-1"],
            ),
        ],
        compressed_history="Earlier discussion covered Apple revenue.",
        turn_count=turn_count,
    )


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "sessions.db")


def test_basic_round_trip_persists_structured_state_only(db_path: str) -> None:
    store = SQLiteConversationStateStore(db_path)
    state = _state("session-1")

    store.save(7, state)
    loaded = store.get(7, "session-1")

    assert loaded is not None
    assert loaded.active_entity == "Apple"
    assert loaded.active_metric == "Revenue"
    assert loaded.active_period == "FY2024"
    assert loaded.active_scope == "annual"
    assert loaded.comparison_entity == "Microsoft"
    assert loaded.comparison_period == "FY2023"
    assert loaded.active_topic == "Apple_Revenue"
    assert loaded.referenced_turn_ids == ["turn_1"]
    assert loaded.referenced_evidence_ids == ["evidence-1"]
    assert loaded.compressed_history == "Earlier discussion covered Apple revenue."
    assert loaded.turn_count == 1
    # Raw turns and assistant text remain owned by SessionManager.
    assert loaded.recent_turns == []

    with sqlite3.connect(db_path) as conn:
        raw = conn.execute(
            "SELECT structured_state_json FROM conversation_states "
            "WHERE user_id = ? AND session_id = ?",
            (7, "session-1"),
        ).fetchone()[0]
    assert "assistant_response" not in raw
    assert "$999B" not in raw
    assert "referenced_evidence_ids" in raw
    assert store.get_state_version(7, "session-1") == 1
    store.close()


def test_update_increments_state_version(db_path: str) -> None:
    store = SQLiteConversationStateStore(db_path)
    store.save(7, _state("session-1", metric="Revenue"))
    store.save(7, _state("session-1", metric="Operating Income", turn_count=2))

    loaded = store.get(7, "session-1")
    assert loaded is not None
    assert loaded.active_metric == "Operating Income"
    assert loaded.turn_count == 2
    assert store.get_state_version(7, "session-1") == 2
    store.close()


def test_delete_and_exists(db_path: str) -> None:
    store = SQLiteConversationStateStore(db_path)
    store.save(7, _state("session-1"))

    assert store.exists(7, "session-1")
    assert store.delete(7, "session-1")
    assert not store.exists(7, "session-1")
    assert store.delete(7, "session-1") is False
    store.close()


def test_user_and_session_isolation(db_path: str) -> None:
    store = SQLiteConversationStateStore(db_path)
    store.save(1, _state("shared", entity="Apple"))
    store.save(2, _state("shared", entity="Tesla"))
    store.save(1, _state("other", entity="Microsoft"))

    assert store.get(1, "shared").active_entity == "Apple"
    assert store.get(2, "shared").active_entity == "Tesla"
    assert store.get(1, "other").active_entity == "Microsoft"
    assert store.get(2, "other") is None
    with pytest.raises(ConversationStateIdentityError):
        store.get_state("shared")
    store.close()


def test_process_restart_restores_state(db_path: str) -> None:
    first = SQLiteConversationStateStore(db_path)
    first.save(42, _state("restart", entity="NVDA"))
    first.close()

    second = SQLiteConversationStateStore(db_path)
    loaded = second.get(42, "restart")
    assert loaded is not None
    assert loaded.active_entity == "NVDA"
    assert loaded.active_metric == "Revenue"
    assert second.get_state_version(42, "restart") == 1
    second.close()


def test_historical_store_contract_remains_available(db_path: str) -> None:
    store = SQLiteConversationStateStore(db_path)
    state = _state("legacy-session")

    # Existing ConversationStateStore callers still have their one-key API.
    store.save_state(state)
    assert store.get_state("legacy-session").active_entity == "Apple"
    assert store.clear_state("legacy-session")
    assert store.get_state("legacy-session") is None
    store.close()


def test_corrupt_json_fails_explicitly(db_path: str) -> None:
    store = SQLiteConversationStateStore(db_path)
    store.save(7, _state("corrupt"))
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE conversation_states SET structured_state_json = ? "
            "WHERE user_id = ? AND session_id = ?",
            ("{not-json", 7, "corrupt"),
        )
        conn.commit()

    with pytest.raises(ConversationStateSerializationError):
        store.get(7, "corrupt")
    store.close()


def test_unknown_schema_version_fails_explicitly(db_path: str) -> None:
    store = SQLiteConversationStateStore(db_path)
    store.save(7, _state("unknown-schema"))
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE conversation_states SET schema_version = ? "
            "WHERE user_id = ? AND session_id = ?",
            (999, 7, "unknown-schema"),
        )
        conn.commit()

    with pytest.raises(ConversationStateVersionError):
        store.get(7, "unknown-schema")
    store.close()


def test_forbidden_authoritative_fields_fail_closed(db_path: str) -> None:
    store = SQLiteConversationStateStore(db_path)
    store.save(7, _state("forbidden"))
    with sqlite3.connect(db_path) as conn:
        raw = conn.execute(
            "SELECT structured_state_json FROM conversation_states "
            "WHERE user_id = ? AND session_id = ?",
            (7, "forbidden"),
        ).fetchone()[0]
        payload = json.loads(raw)
        payload["trusted_answer"] = "Revenue is $999B."
        conn.execute(
            "UPDATE conversation_states SET structured_state_json = ? "
            "WHERE user_id = ? AND session_id = ?",
            (json.dumps(payload), 7, "forbidden"),
        )
        conn.commit()

    with pytest.raises(ConversationStateSerializationError):
        store.get(7, "forbidden")
    store.close()


def test_expected_state_version_detects_conflict(db_path: str) -> None:
    store = SQLiteConversationStateStore(db_path)
    store.save(7, _state("cas"))
    store.save(7, _state("cas", metric="Operating Income"), expected_state_version=1)

    with pytest.raises(ConversationStateConflictError):
        store.save(
            7,
            _state("cas", metric="Net Income"),
            expected_state_version=1,
        )
    assert store.get_state_version(7, "cas") == 2
    assert store.get(7, "cas").active_metric == "Operating Income"
    store.close()


def test_concurrent_writers_use_transaction_safe_upsert(db_path: str) -> None:
    store = SQLiteConversationStateStore(db_path)
    store.save(7, _state("concurrent", turn_count=0))

    def write(index: int) -> None:
        store.save(
            7,
            _state("concurrent", metric=f"Metric {index}", turn_count=index),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write, range(1, 21)))

    loaded = store.get(7, "concurrent")
    assert loaded is not None
    assert loaded.active_metric.startswith("Metric ")
    assert store.get_state_version(7, "concurrent") == 21
    store.close()


def test_context_manager_default_store_remains_in_memory() -> None:
    manager = ConversationContextManager()
    assert isinstance(manager.store, InMemoryConversationStore)
