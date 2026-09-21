"""Phase 47 tests: health snapshot runtime diagnostics."""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.document_registry import DocumentRegistry
from services.feedback import FeedbackStore
from services.health import collect_config_snapshot, collect_health_snapshot
from services.session_manager import SessionManager
from services.trace import TraceLogger


def _create_bm25_db(path):
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE chunk_store (
                doc_id TEXT PRIMARY KEY,
                content TEXT,
                metadata_json TEXT,
                user_id INTEGER,
                doc_name TEXT
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE fts_index USING fts5(
                content,
                doc_id UNINDEXED,
                tokenize='unicode61'
            )
        """)


def _close_session(manager):
    try:
        manager.close()
    except Exception:
        pass


def test_config_snapshot_reports_feedback_path_and_runtime_limits(monkeypatch):
    monkeypatch.setenv("FEEDBACK_DB_PATH", "custom_feedback.db")
    cfg = collect_config_snapshot()

    assert cfg["storage"]["feedback_db_path"] == "custom_feedback.db"
    assert cfg["limits"]["query_n_results_max"] == 20
    assert cfg["limits"]["session_id_max_chars"] == 128
    assert cfg["limits"]["feedback_comment_max_chars"] == 2000
    assert cfg["limits"]["trace_text_max_chars"] == 50000


def test_health_snapshot_includes_feedback_store_without_blocking_readiness(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    monkeypatch.setattr("services.health.CHROMA_PATH", str(chroma_dir))

    registry = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    feedback = FeedbackStore(db_path=str(tmp_path / "feedback.db"))
    TraceLogger(db_path=str(tmp_path / "trace.db"))
    bm25_path = str(tmp_path / "bm25.db")
    _create_bm25_db(bm25_path)

    try:
        snapshot = collect_health_snapshot(
            document_registry=registry,
            session_manager=sessions,
            feedback_store=feedback,
            bm25_db_path=bm25_path,
            trace_db_path=str(tmp_path / "trace.db"),
        )

        assert snapshot["ready"] is True
        assert snapshot["checks"]["feedback"]["ok"] is True
        assert snapshot["checks"]["feedback"]["required"] is False
        assert snapshot["checks"]["feedback"]["missing_tables"] == []
    finally:
        _close_session(sessions)


def test_health_snapshot_missing_feedback_is_reported_but_not_unready(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    monkeypatch.setattr("services.health.CHROMA_PATH", str(chroma_dir))

    registry = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    TraceLogger(db_path=str(tmp_path / "trace.db"))
    bm25_path = str(tmp_path / "bm25.db")
    _create_bm25_db(bm25_path)

    try:
        snapshot = collect_health_snapshot(
            document_registry=registry,
            session_manager=sessions,
            bm25_db_path=bm25_path,
            trace_db_path=str(tmp_path / "trace.db"),
            feedback_db_path=str(tmp_path / "missing_feedback.db"),
        )

        assert snapshot["ready"] is True
        assert snapshot["status"] == "ready"
        assert snapshot["checks"]["feedback"]["ok"] is False
        assert snapshot["checks"]["feedback"]["required"] is False
    finally:
        _close_session(sessions)


def test_health_snapshot_blocks_v2_when_production_assets_are_missing(monkeypatch):
    monkeypatch.setenv("FINANCIAL_RUNTIME_MODE", "v2")

    snapshot = collect_health_snapshot(trusted_v2_preflight=True)

    trusted_v2 = snapshot["checks"]["trusted_v2"]
    assert trusted_v2["required"] is True
    assert trusted_v2["ok"] is False
    assert trusted_v2["status"] == "blocked"
    assert "TRUSTED_V2_RUNTIME_BUILDER" in trusted_v2["error"]
    assert "TRUSTED_V2_R4_INDEX_DIR" in trusted_v2["error"]


def test_health_snapshot_skips_v2_preflight_for_explicit_v1(monkeypatch):
    monkeypatch.setenv("FINANCIAL_RUNTIME_MODE", "v1")

    snapshot = collect_health_snapshot(trusted_v2_preflight=True)

    trusted_v2 = snapshot["checks"]["trusted_v2"]
    assert trusted_v2 == {
        "ok": True,
        "required": False,
        "mode": "v1",
        "status": "skipped",
    }


def test_readyz_passes_feedback_store_static():
    root = os.path.join(os.path.dirname(__file__), "..")
    main = open(os.path.join(root, "src", "main.py"), encoding="utf-8").read()
    readyz_block = main[main.index('@app.get("/readyz")'):main.index('@app.get("/me"')]

    assert "feedback_store=feedback_store" in readyz_block
