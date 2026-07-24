"""
Document Registry - lifecycle management for uploaded documents.
"""
import sqlite3
import hashlib
import time
import os
from typing import Optional, Dict, Any, List

from .sqlite_migrations import run_component_migrations

SCHEMA_VERSION = 1
MAX_ERROR_MESSAGE_CHARS = 2000

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS document_registry (
    document_id   TEXT PRIMARY KEY,
    tenant_id     INTEGER NOT NULL,
    filename      TEXT NOT NULL,
    file_hash     TEXT NOT NULL,
    content_hash  TEXT,
    chunk_count   INTEGER DEFAULT 0,
    page_count    INTEGER DEFAULT 0,
    version       INTEGER DEFAULT 1,
    status        TEXT DEFAULT 'pending',
    parser_version TEXT,
    splitter_version TEXT,
    embedding_version TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_registry_tenant ON document_registry(tenant_id);
CREATE INDEX IF NOT EXISTS idx_registry_file_hash ON document_registry(file_hash);
CREATE INDEX IF NOT EXISTS idx_registry_content_hash ON document_registry(content_hash);
CREATE INDEX IF NOT EXISTS idx_registry_status ON document_registry(status);
"""

VALID_TRANSITIONS = {
    "pending":   {"parsing", "failed"},
    "parsing":   {"indexing", "failed"},
    "indexing":  {"ready", "failed"},
    "ready":     set(),
    "failed":    {"pending"},
}
VALID_STATUSES = set(VALID_TRANSITIONS)

def validate_transition(current, target):
    allowed = VALID_TRANSITIONS.get(current)
    if allowed is None:
        raise ValueError("Unknown state: %r" % (current,))
    if target not in allowed:
        raise ValueError("Invalid transition %r -> %r; allowed: %s" % (current, target, allowed or "terminal"))

class DocumentRegistry:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.getenv("DOCUMENT_REGISTRY_DB_PATH", "document_registry.db")
        self.db_path = db_path
        self._init_schema()

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript(_SCHEMA_SQL)
            run_component_migrations(
                conn,
                "document_registry",
                SCHEMA_VERSION,
                {},
            )

    @staticmethod
    def file_hash(data):
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def content_hash(chunks):
        sorted_chunks = sorted(chunks, key=lambda c: c.get("metadata", {}).get("doc_id", ""))
        combined = chr(10).join(c.get("content", "") for c in sorted_chunks)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def find_by_file_hash(self, tenant_id, file_hash, *, parser_version=None,
                          splitter_version=None, embedding_version=None):
        """Find a ready duplicate, optionally constrained to ingest lineage.

        Version-constrained matching permits a deliberate re-upload after a
        parser, splitter, or embedding change while retaining normal duplicate
        protection for identical processing configurations.
        """
        conditions = ["tenant_id = ?", "file_hash = ?", "status = 'ready'"]
        params = [tenant_id, file_hash]
        for column, value in (
            ("parser_version", parser_version),
            ("splitter_version", splitter_version),
            ("embedding_version", embedding_version),
        ):
            if value is not None:
                conditions.append(f"{column} = ?")
                params.append(value)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM document_registry WHERE " + " AND ".join(conditions)
                + " ORDER BY version DESC LIMIT 1",
                tuple(params),
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def find_by_content_hash(self, tenant_id, content_hash, *, parser_version=None,
                             splitter_version=None, embedding_version=None):
        """Find a same-content duplicate scoped to an optional ingest lineage."""
        conditions = ["tenant_id = ?", "content_hash = ?", "status = 'ready'"]
        params = [tenant_id, content_hash]
        for column, value in (
            ("parser_version", parser_version),
            ("splitter_version", splitter_version),
            ("embedding_version", embedding_version),
        ):
            if value is not None:
                conditions.append(f"{column} = ?")
                params.append(value)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM document_registry WHERE " + " AND ".join(conditions)
                + " ORDER BY version DESC LIMIT 1",
                tuple(params),
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def get_latest_version(self, tenant_id, filename):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM document_registry WHERE tenant_id = ? AND filename = ? ORDER BY version DESC LIMIT 1",
                (tenant_id, filename)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def register(self, document_id, tenant_id, filename, file_hash,
                 content_hash=None, chunk_count=0, page_count=0,
                 status="pending", parser_version=None, splitter_version=None, embedding_version=None):
        if status not in VALID_STATUSES:
            raise ValueError("Invalid document status: %r" % (status,))
        if tenant_id is None:
            raise ValueError("tenant_id is required")
        if not document_id or not filename or not file_hash:
            raise ValueError("document_id, filename, and file_hash are required")
        now = time.time()
        existing = self.get_latest_version(tenant_id, filename)
        version = (existing["version"] + 1) if existing else 1
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO document_registry (document_id, tenant_id, filename, file_hash, content_hash, chunk_count, page_count, version, status, parser_version, splitter_version, embedding_version, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (document_id, tenant_id, filename, file_hash, content_hash, chunk_count, page_count, version, status, parser_version, splitter_version, embedding_version, now, now)
            )
            conn.commit()
        return version

    def transition(self, document_id, target_status, error_message=None):
        with self._conn() as conn:
            row = conn.execute("SELECT status FROM document_registry WHERE document_id = ?", (document_id,)).fetchone()
            if not row:
                raise ValueError("Document %r not found" % (document_id,))
            validate_transition(row[0], target_status)
            now = time.time()
            conn.execute("UPDATE document_registry SET status = ?, updated_at = ?, error_message = ? WHERE document_id = ?", (target_status, now, self._clean_error(error_message), document_id))
            conn.commit()

    def mark_indexing(self, document_id):
        self.transition(document_id, "indexing")

    def mark_ready(self, document_id, chunk_count, content_hash, page_count=None):
        with self._conn() as conn:
            row = conn.execute("SELECT status FROM document_registry WHERE document_id = ?", (document_id,)).fetchone()
            if not row:
                raise ValueError("Document %r not found" % (document_id,))
            validate_transition(row[0], "ready")
            now = time.time()
            if page_count is None:
                conn.execute("UPDATE document_registry SET status = ?, chunk_count = ?, content_hash = ?, updated_at = ?, error_message = NULL WHERE document_id = ?", ("ready", chunk_count, content_hash, now, document_id))
            else:
                conn.execute("UPDATE document_registry SET status = ?, chunk_count = ?, content_hash = ?, page_count = ?, updated_at = ?, error_message = NULL WHERE document_id = ?", ("ready", chunk_count, content_hash, page_count, now, document_id))
            conn.commit()

    def mark_failed(self, document_id, error_message):
        self.transition(document_id, "failed", error_message=error_message)

    def delete(self, tenant_id, filename):
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM document_registry WHERE tenant_id = ? AND filename = ?", (tenant_id, filename))
            conn.commit()
            return cur.rowcount

    def delete_all_for_tenant(self, tenant_id):
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM document_registry WHERE tenant_id = ?", (tenant_id,))
            conn.commit()
            return cur.rowcount

    def list_documents(self, tenant_id):
        if tenant_id is None:
            return []
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM document_registry WHERE tenant_id = ? AND status = 'ready' ORDER BY updated_at DESC, rowid DESC", (tenant_id,)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_all(self, tenant_id, status=None, limit=None, offset=0):
        """List registry rows for one tenant, optionally filtered by status and paginated."""
        if tenant_id is None:
            return []
        if status is not None and status not in VALID_TRANSITIONS:
            return []
        try:
            normalized_offset = max(0, int(offset or 0))
            normalized_limit = None if limit is None else max(0, int(limit))
        except (TypeError, ValueError):
            return []

        sql = "SELECT * FROM document_registry WHERE tenant_id = ?"
        params = [tenant_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY updated_at DESC, rowid DESC"
        if normalized_limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([normalized_limit, normalized_offset])
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_all(self, tenant_id, status=None):
        """Count registry rows for one tenant, optionally filtered by status."""
        if tenant_id is None:
            return 0
        if status is not None and status not in VALID_TRANSITIONS:
            return 0
        sql = "SELECT COUNT(*) FROM document_registry WHERE tenant_id = ?"
        params = [tenant_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        with self._conn() as conn:
            return int(conn.execute(sql, params).fetchone()[0])

    def status_summary(self, tenant_id):
        """Return zero-filled lifecycle counts plus aggregate totals."""
        counts = self.status_counts(tenant_id)
        if not counts:
            return {
                "counts": {},
                "total": 0,
                "active": 0,
                "ready": 0,
                "failed": 0,
            }
        return {
            "counts": counts,
            "total": sum(counts.values()),
            "active": counts.get("pending", 0) + counts.get("parsing", 0) + counts.get("indexing", 0),
            "ready": counts.get("ready", 0),
            "failed": counts.get("failed", 0),
        }

    def status_counts(self, tenant_id):
        """Return lifecycle status counts for one tenant."""
        if tenant_id is None:
            return {}
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM document_registry WHERE tenant_id = ? GROUP BY status",
                (tenant_id,),
            ).fetchall()
        counts = {status: 0 for status in VALID_TRANSITIONS}
        counts.update({row[0]: row[1] for row in rows})
        return counts

    def get_pending_for_retry(self, tenant_id=None, limit=100, offset=0):
        try:
            normalized_limit = max(1, min(int(limit or 100), 1000))
            normalized_offset = max(0, int(offset or 0))
        except (TypeError, ValueError):
            return []
        if tenant_id is not None:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM document_registry WHERE tenant_id = ? AND status = 'failed' ORDER BY updated_at DESC, rowid DESC LIMIT ? OFFSET ?",
                    (tenant_id, normalized_limit, normalized_offset),
                ).fetchall()
        else:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM document_registry WHERE status = 'failed' ORDER BY updated_at DESC, rowid DESC LIMIT ? OFFSET ?",
                    (normalized_limit, normalized_offset),
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _clean_error(error_message):
        if error_message is None:
            return None
        cleaned = " ".join(str(error_message).replace("\x00", " ").split())
        if not cleaned:
            return None
        return cleaned[:MAX_ERROR_MESSAGE_CHARS]

    @staticmethod
    def _row_to_dict(row):
        columns = ["document_id", "tenant_id", "filename", "file_hash", "content_hash", "chunk_count", "page_count", "version", "status", "parser_version", "splitter_version", "embedding_version", "created_at", "updated_at", "error_message"]
        return dict(zip(columns, row))
