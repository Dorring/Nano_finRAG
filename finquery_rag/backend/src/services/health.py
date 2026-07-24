"""Operational health and configuration checks for FinQuery RAG.

The checks in this module are intentionally lightweight:
- no outbound LLM call
- no embedding request
- no Chroma collection creation
- no tenant data exposure
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from .retrieval_config import build_retrieval_model_config

BM25_DB_PATH = "rag_bm25.db"
CHROMA_PATH = "./chroma_db"
DOCUMENT_REGISTRY_DB_PATH = "document_registry.db"
TRACE_DB_PATH = "trace_log.db"
FEEDBACK_DB_PATH = "feedback.db"
GLOBAL_COLLECTION_NAME = "rag_global_knowledge_base"

RUNTIME_LIMITS = {
    "query_question_min_chars": 2,
    "query_n_results_min": 1,
    "query_n_results_max": 20,
    "query_question_max_chars": 4000,
    "query_document_names_max_items": 20,
    "query_document_name_max_chars": 180,
    "session_id_max_chars": 128,
    "trace_id_max_chars": 128,
    "upload_filename_max_chars": 180,
    "feedback_comment_max_chars": 2000,
    "trace_text_max_chars": 50000,
    "trace_json_max_chars": 50000,
}


def _runtime_path(env_name: str, default: str) -> str:
    return os.getenv(env_name, default)


def _safe_int(value: str | None, default: int) -> tuple[int, str | None]:
    if value is None:
        return default, None
    try:
        return int(value), None
    except (TypeError, ValueError):
        return default, f"invalid integer: {value!r}"


def _path_check(path: str, *, expect_dir: bool = False) -> dict[str, Any]:
    p = Path(path)
    exists = p.exists()
    parent = p if expect_dir else p.parent
    parent_exists = parent.exists()
    writable = os.access(parent, os.W_OK) if parent_exists else False
    ok = parent_exists and writable
    if expect_dir and exists and not p.is_dir():
        ok = False
    return {
        "ok": ok,
        "path": str(p),
        "exists": exists,
        "parent_exists": parent_exists,
        "parent_writable": writable,
    }


def _bm25_integrity_summary(path: str) -> dict[str, Any]:
    """Return non-content BM25/FTS consistency counts for health checks."""
    summary: dict[str, Any] = {
        "ok": False,
        "chunk_store_count": 0,
        "fts_count": 0,
        "missing_fts_count": 0,
        "duplicate_doc_id_count": 0,
        "duplicate_fts_rows": 0,
        "orphan_fts_count": 0,
    }
    try:
        with sqlite3.connect(path, timeout=2) as conn:
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunk_store").fetchone()[0]
            fts_count = conn.execute("SELECT COUNT(*) FROM fts_index").fetchone()[0]
            missing_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM chunk_store c
                LEFT JOIN fts_index f ON f.doc_id = c.doc_id
                WHERE f.doc_id IS NULL
                """
            ).fetchone()[0]
            duplicate_rows = conn.execute(
                """
                SELECT COUNT(*) AS duplicate_doc_ids, COALESCE(SUM(n - 1), 0) AS duplicate_rows
                FROM (
                    SELECT doc_id, COUNT(*) AS n
                    FROM fts_index
                    GROUP BY doc_id
                    HAVING n > 1
                )
                """
            ).fetchone()
            orphan_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM fts_index f
                LEFT JOIN chunk_store c ON c.doc_id = f.doc_id
                WHERE c.doc_id IS NULL
                """
            ).fetchone()[0]
    except Exception as exc:  # pragma: no cover - exact sqlite errors vary by platform
        summary["error"] = str(exc)
        return summary

    summary.update(
        {
            "chunk_store_count": int(chunk_count),
            "fts_count": int(fts_count),
            "missing_fts_count": int(missing_count),
            "duplicate_doc_id_count": int(duplicate_rows[0] or 0),
            "duplicate_fts_rows": int(duplicate_rows[1] or 0),
            "orphan_fts_count": int(orphan_count),
        }
    )
    summary["ok"] = not any(
        summary[key]
        for key in ("missing_fts_count", "duplicate_doc_id_count", "orphan_fts_count")
    )
    return summary


def _sqlite_check(path: str, *, required_tables: tuple[str, ...] = ()) -> dict[str, Any]:
    p = Path(path)
    base = _path_check(str(p))
    result: dict[str, Any] = {
        **base,
        "kind": "sqlite",
        "required_tables": list(required_tables),
        "missing_tables": [],
    }
    if not base["ok"]:
        return result

    if not p.exists():
        result["ok"] = False
        result["error"] = "database file does not exist"
        return result

    try:
        with sqlite3.connect(str(p), timeout=2) as conn:
            conn.execute("SELECT 1").fetchone()
            if required_tables:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                ).fetchall()
                tables = {row[0] for row in rows}
                missing = [table for table in required_tables if table not in tables]
                result["missing_tables"] = missing
                if missing:
                    result["ok"] = False
                    result["error"] = "missing required tables"
    except Exception as exc:  # pragma: no cover - exact sqlite errors vary by platform
        result["ok"] = False
        result["error"] = str(exc)
    return result


def collect_config_snapshot() -> dict[str, Any]:
    """Return non-secret runtime configuration relevant to RAG operations."""
    candidate_multiplier, candidate_error = _safe_int(
        os.getenv("RAG_CANDIDATE_MULTIPLIER"), 4
    )
    session_ttl, ttl_error = _safe_int(os.getenv("SESSION_TTL_SECONDS"), 0)

    errors = [err for err in (candidate_error, ttl_error) if err]
    if candidate_multiplier < 1:
        errors.append("RAG_CANDIDATE_MULTIPLIER must be >= 1")
    if session_ttl < 0:
        errors.append("SESSION_TTL_SECONDS must be >= 0")
    retrieval_models = build_retrieval_model_config()
    errors.extend(retrieval_models.get("errors", []))

    return {
        "ok": not errors,
        "errors": errors,
        "llm": {
            "base_url_configured": bool(os.getenv("LLM_API_BASE_URL")),
            "api_key_configured": bool(os.getenv("LLM_API_KEY")),
            "model_name": os.getenv("LLM_MODEL_NAME", "nanochat"),
        },
        "retrieval": {
            "hybrid_enabled": True,
            "candidate_multiplier": candidate_multiplier,
            **retrieval_models,
        },
        "storage": {
            "chroma_path": _runtime_path("CHROMA_PATH", CHROMA_PATH),
            "chroma_collection": GLOBAL_COLLECTION_NAME,
            "document_registry_db_path": _runtime_path(
                "DOCUMENT_REGISTRY_DB_PATH", DOCUMENT_REGISTRY_DB_PATH
            ),
            "bm25_db_path": _runtime_path("BM25_DB_PATH", BM25_DB_PATH),
            "sessions_db_path": os.getenv("SESSIONS_DB_PATH", "sessions.db"),
            "trace_db_path": _runtime_path("TRACE_DB_PATH", TRACE_DB_PATH),
            "feedback_db_path": _runtime_path("FEEDBACK_DB_PATH", FEEDBACK_DB_PATH),
        },
        "sessions": {
            "ttl_seconds": session_ttl,
        },
        "limits": RUNTIME_LIMITS,
    }


def collect_health_snapshot(
    *,
    document_registry: Any | None = None,
    session_manager: Any | None = None,
    feedback_store: Any | None = None,
    bm25_db_path: str | None = None,
    trace_db_path: str | None = None,
    feedback_db_path: str | None = None,
) -> dict[str, Any]:
    """Return a readiness snapshot without reading tenant content."""
    config = collect_config_snapshot()
    registry_path = getattr(
        document_registry,
        "db_path",
        _runtime_path("DOCUMENT_REGISTRY_DB_PATH", DOCUMENT_REGISTRY_DB_PATH),
    )
    session_path = getattr(
        session_manager, "db_path", os.getenv("SESSIONS_DB_PATH", "sessions.db")
    )
    feedback_path = getattr(
        feedback_store,
        "db_path",
        feedback_db_path or _runtime_path("FEEDBACK_DB_PATH", FEEDBACK_DB_PATH),
    )
    bm25_path = bm25_db_path or _runtime_path("BM25_DB_PATH", BM25_DB_PATH)
    trace_path = trace_db_path or _runtime_path("TRACE_DB_PATH", TRACE_DB_PATH)
    chroma_path = _runtime_path("CHROMA_PATH", CHROMA_PATH)

    bm25_check = _sqlite_check(bm25_path, required_tables=("chunk_store", "fts_index"))
    bm25_check["required"] = True
    if bm25_check.get("ok"):
        bm25_integrity = _bm25_integrity_summary(bm25_path)
        bm25_check["integrity"] = bm25_integrity
        if not bm25_integrity.get("ok", False):
            bm25_check["ok"] = False
            bm25_check["error"] = "bm25 index integrity check failed"

    checks = {
        "config": config,
        "chroma_path": {
            **_path_check(chroma_path, expect_dir=True),
            "kind": "directory",
            "required": False,
        },
        "document_registry": {
            **_sqlite_check(registry_path, required_tables=("document_registry",)),
            "required": True,
        },
        "bm25": bm25_check,
        "sessions": {
            **_sqlite_check(session_path, required_tables=("conversations",)),
            "required": True,
        },
        "trace": {
            **_sqlite_check(trace_path, required_tables=("trace_log",)),
            "required": False,
        },
        "feedback": {
            **_sqlite_check(feedback_path, required_tables=("answer_feedback",)),
            "required": False,
        },
    }

    required_ok = all(
        check.get("ok", False)
        for check in checks.values()
        if isinstance(check, dict) and check.get("required", True)
    )
    ready = bool(config["ok"] and required_ok)

    return {
        "status": "ready" if ready else "degraded",
        "ready": ready,
        "checked_at": time.time(),
        "checks": checks,
    }
