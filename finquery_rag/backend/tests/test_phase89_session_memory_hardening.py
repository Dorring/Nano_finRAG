"""Phase 89 tests: session memory hardening."""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.session_manager import SessionManager


def _close(mgr):
    try:
        mgr.close()
    except Exception:
        pass


def test_add_message_auto_prunes_to_max_history_pairs(tmp_path):
    mgr = SessionManager(db_path=str(tmp_path / "sessions.db"), max_history=2)
    try:
        for i in range(6):
            mgr.add_message("s1", 1, "user" if i % 2 == 0 else "assistant", f"m{i}")

        messages = mgr.get_recent_messages("s1", 1, n_pairs=10)

        assert [m["content"] for m in messages] == ["m2", "m3", "m4", "m5"]
        assert mgr.get_session_count("s1", 1) == 4
    finally:
        _close(mgr)


def test_get_recent_messages_rejects_bad_pair_count(tmp_path):
    mgr = SessionManager(db_path=str(tmp_path / "sessions.db"))
    try:
        mgr.add_message("s1", 1, "user", "hello")

        assert mgr.get_recent_messages("s1", 1, n_pairs="bad") == []
        assert mgr.get_recent_messages("s1", 1, n_pairs=-1) == []
    finally:
        _close(mgr)


def test_list_sessions_is_paginated_ordered_and_tenant_scoped(tmp_path):
    mgr = SessionManager(db_path=str(tmp_path / "sessions.db"))
    try:
        mgr.add_message("old", 1, "user", "old")
        mgr.add_message("new", 1, "user", "new")
        mgr.add_message("other", 2, "user", "other")
        with sqlite3.connect(mgr.db_path) as conn:
            conn.execute("UPDATE conversations SET created_at = ? WHERE session_id = ?", (1, "old"))
            conn.execute("UPDATE conversations SET created_at = ? WHERE session_id = ?", (2, "new"))
            conn.execute("UPDATE conversations SET created_at = ? WHERE session_id = ?", (3, "other"))
            conn.commit()

        page = mgr.list_sessions(1, limit=1, offset=0)

        assert page == [{"session_id": "new", "message_count": 1, "created_at": 2.0, "updated_at": 2.0}]
        assert mgr.count_sessions(1) == 2
        assert mgr.count_sessions(2) == 1
        assert mgr.list_sessions(1, limit="bad") == []
    finally:
        _close(mgr)


def test_clear_all_for_user_and_storage_summary_are_tenant_scoped(tmp_path):
    mgr = SessionManager(db_path=str(tmp_path / "sessions.db"), ttl_seconds=0, max_history=3)
    try:
        mgr.add_message("s1", 1, "user", "a")
        mgr.add_message("s2", 1, "assistant", "b")
        mgr.add_message("s1", 2, "user", "c")

        summary = mgr.storage_summary(1)
        assert summary == {
            "user_id": 1,
            "sessions": 2,
            "messages": 2,
            "ttl_seconds": 0,
            "max_history_pairs": 3,
        }
        assert mgr.clear_all_for_user(1) == 2
        assert mgr.storage_summary(1)["messages"] == 0
        assert mgr.storage_summary(2)["messages"] == 1
    finally:
        _close(mgr)


def test_prune_session_history_rejects_bad_max_messages(tmp_path):
    mgr = SessionManager(db_path=str(tmp_path / "sessions.db"))
    try:
        mgr.add_message("s1", 1, "user", "a")
        assert mgr.prune_session_history("s1", 1, max_messages="bad") == 0
        assert mgr.get_session_count("s1", 1) == 1
    finally:
        _close(mgr)


def test_sessions_endpoint_and_query_paths_use_validated_session_id_static():
    root = os.path.join(os.path.dirname(__file__), "..")
    main = open(os.path.join(root, "src", "main.py"), encoding="utf-8").read()
    sessions_block = main[main.index('@app.get("/sessions")'):main.index('@app.post("/sessions/clear")')]
    query_block = main[main.index('@app.post("/query"'):main.index('@app.post("/query/stream"')]
    stream_block = main[main.index('@app.post("/query/stream"'):main.index('# <---------------------- Session endpoints')]

    assert "session_manager.list_sessions" in sessions_block
    assert '@app.delete("/sessions")' in main
    assert "session_manager.clear_all_for_user(current_user.id)" in main
    assert '"total_sessions": total_sessions' in sessions_block
    assert '"has_more": normalized_offset + len(sessions) < total_sessions' in sessions_block
    assert "def _build_user_turn_execution_request" in main
    assert "session_id = _validate_session_id(request.session_id) if request.session_id else None" in main
    lifecycle = open(os.path.join(root, "src", "runtime", "query_lifecycle.py"), encoding="utf-8").read()
    assert "get_recent_messages(" in lifecycle
    assert "execute_user_turn" in query_block
    assert "execute_user_turn" in stream_block
