"""Phase 9A tests: operational health and configuration diagnostics."""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.document_registry import DocumentRegistry
from services.health import collect_config_snapshot, collect_health_snapshot
from services.session_manager import SessionManager
from services.trace import TraceLogger


def _close_session(manager):
    try:
        manager.close()
    except Exception:
        pass



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


def test_config_snapshot_redacts_secret_values(monkeypatch):
    monkeypatch.setenv("LLM_API_BASE_URL", "http://internal.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "secret-value")
    monkeypatch.setenv("RAG_RERANKER", "heuristic")
    cfg = collect_config_snapshot()
    assert cfg["ok"] is True
    assert cfg["llm"]["base_url_configured"] is True
    assert cfg["llm"]["api_key_configured"] is True
    assert "secret-value" not in repr(cfg)
    assert cfg["retrieval"]["reranker"] == "heuristic"


def test_config_snapshot_rejects_bad_numeric_env(monkeypatch):
    monkeypatch.setenv("RAG_CANDIDATE_MULTIPLIER", "0")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "-1")
    cfg = collect_config_snapshot()
    assert cfg["ok"] is False
    assert "RAG_CANDIDATE_MULTIPLIER must be >= 1" in cfg["errors"]
    assert "SESSION_TTL_SECONDS must be >= 0" in cfg["errors"]


def test_config_snapshot_uses_wider_default_candidate_pool(monkeypatch):
    monkeypatch.delenv("RAG_CANDIDATE_MULTIPLIER", raising=False)
    cfg = collect_config_snapshot()
    assert cfg["retrieval"]["candidate_multiplier"] == 4


def test_health_snapshot_ready_when_required_stores_exist(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    monkeypatch.setattr("services.health.CHROMA_PATH", str(chroma_dir))

    registry = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    bm25_path = str(tmp_path / "bm25.db")
    _create_bm25_db(bm25_path)
    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    TraceLogger(db_path=str(tmp_path / "trace.db"))

    try:
        snapshot = collect_health_snapshot(
            document_registry=registry,
            session_manager=sessions,
            bm25_db_path=bm25_path,
            trace_db_path=str(tmp_path / "trace.db"),
        )
        assert snapshot["ready"] is True
        assert snapshot["status"] == "ready"
        assert snapshot["checks"]["document_registry"]["ok"] is True
        assert snapshot["checks"]["bm25"]["ok"] is True
        assert snapshot["checks"]["bm25"]["integrity"]["ok"] is True
        assert snapshot["checks"]["sessions"]["ok"] is True
        assert snapshot["checks"]["trace"]["ok"] is True
    finally:
        _close_session(sessions)


def test_health_snapshot_degraded_when_required_store_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("services.health.CHROMA_PATH", str(tmp_path / "chroma_db"))
    registry = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    try:
        snapshot = collect_health_snapshot(
            document_registry=registry,
            session_manager=sessions,
            bm25_db_path=str(tmp_path / "missing_bm25.db"),
            trace_db_path=str(tmp_path / "missing_trace.db"),
        )
        assert snapshot["ready"] is False
        assert snapshot["status"] == "degraded"
        assert snapshot["checks"]["bm25"]["ok"] is False
        # Trace is useful for diagnostics but should not block readiness.
        assert snapshot["checks"]["trace"]["required"] is False
    finally:
        _close_session(sessions)


def test_health_snapshot_detects_missing_required_table(tmp_path):
    bad_db = tmp_path / "bad_bm25.db"
    with sqlite3.connect(bad_db) as conn:
        conn.execute("CREATE TABLE unrelated (id INTEGER)")
    registry = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    try:
        snapshot = collect_health_snapshot(
            document_registry=registry,
            session_manager=sessions,
            bm25_db_path=str(bad_db),
            trace_db_path=str(tmp_path / "missing_trace.db"),
        )
        assert snapshot["ready"] is False
        assert snapshot["checks"]["bm25"]["missing_tables"] == [
            "chunk_store",
            "fts_index",
        ]
    finally:
        _close_session(sessions)



def test_health_snapshot_degraded_when_bm25_index_inconsistent(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    monkeypatch.setattr("services.health.CHROMA_PATH", str(chroma_dir))
    registry = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    bm25_path = tmp_path / "bm25.db"
    _create_bm25_db(bm25_path)
    with sqlite3.connect(bm25_path) as conn:
        conn.execute(
            "INSERT INTO chunk_store VALUES (?, ?, ?, ?, ?)",
            ("doc::1", "content", "{}", 1, "doc.pdf"),
        )
        conn.commit()

    try:
        snapshot = collect_health_snapshot(
            document_registry=registry,
            session_manager=sessions,
            bm25_db_path=str(bm25_path),
            trace_db_path=str(tmp_path / "missing_trace.db"),
        )

        assert snapshot["ready"] is False
        assert snapshot["checks"]["bm25"]["ok"] is False
        assert snapshot["checks"]["bm25"]["error"] == "bm25 index integrity check failed"
        assert snapshot["checks"]["bm25"]["integrity"]["missing_fts_count"] == 1
        assert "doc::1" not in repr(snapshot["checks"]["bm25"]["integrity"])
    finally:
        _close_session(sessions)


def test_health_snapshot_reports_bm25_duplicate_and_orphan_counts(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    monkeypatch.setattr("services.health.CHROMA_PATH", str(chroma_dir))
    registry = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    bm25_path = tmp_path / "bm25.db"
    _create_bm25_db(bm25_path)
    with sqlite3.connect(bm25_path) as conn:
        conn.execute(
            "INSERT INTO chunk_store VALUES (?, ?, ?, ?, ?)",
            ("doc::1", "content", "{}", 1, "doc.pdf"),
        )
        conn.execute("INSERT INTO fts_index(content, doc_id) VALUES (?, ?)", ("content", "doc::1"))
        conn.execute("INSERT INTO fts_index(content, doc_id) VALUES (?, ?)", ("content", "doc::1"))
        conn.execute("INSERT INTO fts_index(content, doc_id) VALUES (?, ?)", ("orphan", "orphan::1"))
        conn.commit()

    try:
        snapshot = collect_health_snapshot(
            document_registry=registry,
            session_manager=sessions,
            bm25_db_path=str(bm25_path),
            trace_db_path=str(tmp_path / "missing_trace.db"),
        )
        integrity = snapshot["checks"]["bm25"]["integrity"]

        assert snapshot["ready"] is False
        assert integrity["duplicate_doc_id_count"] == 1
        assert integrity["duplicate_fts_rows"] == 1
        assert integrity["orphan_fts_count"] == 1
        assert "orphan::1" not in repr(integrity)
    finally:
        _close_session(sessions)
