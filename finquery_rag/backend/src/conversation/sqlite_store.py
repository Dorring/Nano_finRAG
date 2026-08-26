"""SQLite-backed persistence for structured conversation state.

Raw dialogue messages remain owned by SessionManager. This store persists only
the semantic DialogueState projection and provenance-reference metadata.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections.abc import Mapping
from typing import Any

from ..services.sqlite_migrations import run_component_migrations
from .contracts import DialogueState
from .store import ConversationStateStore


class ConversationStateStoreError(RuntimeError):
    """Base error for SQLite conversation-state failures."""


class ConversationStateSerializationError(ConversationStateStoreError):
    """Raised when structured state cannot be encoded or decoded safely."""


class ConversationStateVersionError(ConversationStateStoreError):
    """Raised when a persisted state uses an unsupported schema version."""


class ConversationStateConflictError(ConversationStateStoreError):
    """Raised when an expected state version no longer matches."""


class ConversationStateIdentityError(ConversationStateStoreError):
    """Raised when a user/session identity is missing or ambiguous."""


class SQLiteConversationStateStore(ConversationStateStore):
    """Thread-local SQLite store implementing the existing state-store contract.

    The explicit user-scoped methods use (user_id, session_id) as the storage
    identity. The one-argument methods required by the historical
    ConversationStateStore interface remain available for component tests and
    use a reserved legacy user id when saving an unscoped state. Production
    wiring should always use the explicit user-scoped methods.
    """

    SCHEMA_VERSION = 1
    COMPONENT_NAME = "conversation_state_store"
    DEFAULT_DB_ENV = "SESSIONS_DB_PATH"
    DEFAULT_DB_PATH = "sessions.db"
    LEGACY_USER_ID = 0
    MAX_SESSION_ID_LENGTH = 128
    FORBIDDEN_STATE_KEYS = frozenset(
        {
            "answer_numeric_value",
            "trusted_answer",
            "calculator_operand_from_history",
            "last_answer_numeric_value",
            "trusted_operand_json",
        },
    )

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = os.getenv(self.DEFAULT_DB_ENV, self.DEFAULT_DB_PATH)
        if not isinstance(db_path, str) or not db_path:
            raise ValueError("db_path must be a non-empty string")
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                component TEXT PRIMARY KEY,
                version INTEGER NOT NULL
            );
            """,
        )
        run_component_migrations(
            conn,
            self.COMPONENT_NAME,
            self.SCHEMA_VERSION,
            {1: self._migrate_to_v1},
        )

    @staticmethod
    def _migrate_to_v1(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversation_states (
                user_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                state_version INTEGER NOT NULL,
                structured_state_json TEXT NOT NULL,
                compressed_history TEXT,
                turn_count INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (user_id, session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_conversation_states_session
                ON conversation_states(session_id);
            """,
        )

    @classmethod
    def _session_id(cls, session_id: str) -> str:
        if (
            not isinstance(session_id, str)
            or not session_id
            or len(session_id) > cls.MAX_SESSION_ID_LENGTH
        ):
            raise ConversationStateIdentityError(
                "session_id must be 1-128 characters",
            )
        return session_id

    @classmethod
    def _user_id(cls, user_id: Any) -> int:
        if user_id is None or isinstance(user_id, bool):
            raise ConversationStateIdentityError("user_id is required")
        try:
            normalized = int(user_id)
        except (TypeError, ValueError) as exc:
            raise ConversationStateIdentityError(
                "user_id must be an integer",
            ) from exc
        if normalized < 0:
            raise ConversationStateIdentityError("user_id must be non-negative")
        return normalized

    @classmethod
    def _serialize_state(cls, state: DialogueState) -> tuple[str, str | None, int]:
        if not isinstance(state, DialogueState):
            raise ConversationStateSerializationError(
                "state must be a DialogueState",
            )
        cls._session_id(state.conversation_id)
        payload = state.to_dict()
        # Recent raw turns, including assistant text, remain SessionManager's
        # source of truth and are intentionally runtime-only here.
        payload["recent_turns"] = []
        payload["schema_version"] = cls.SCHEMA_VERSION
        forbidden = cls.FORBIDDEN_STATE_KEYS.intersection(payload)
        if forbidden:
            raise ConversationStateSerializationError(
                f"authoritative state fields are forbidden: {sorted(forbidden)}",
            )
        if payload.get("compressed_history") is not None and not isinstance(
            payload["compressed_history"],
            str,
        ):
            raise ConversationStateSerializationError(
                "compressed_history must be a string or None",
            )
        try:
            # Validate against the formal contract before writing JSON.
            DialogueState.from_dict(payload)
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConversationStateSerializationError(
                "DialogueState could not be serialized",
            ) from exc
        return encoded, payload["compressed_history"], int(payload["turn_count"])

    @classmethod
    def _deserialize_state(cls, row: sqlite3.Row) -> DialogueState:
        try:
            row_schema_version = int(row["schema_version"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ConversationStateSerializationError(
                "persisted schema_version is invalid",
            ) from exc
        if row_schema_version != cls.SCHEMA_VERSION:
            raise ConversationStateVersionError(
                f"unsupported persisted schema_version={row_schema_version}",
            )

        try:
            payload = json.loads(row["structured_state_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ConversationStateSerializationError(
                "structured_state_json is invalid JSON",
            ) from exc
        if not isinstance(payload, Mapping):
            raise ConversationStateSerializationError(
                "structured_state_json must contain an object",
            )
        payload = dict(payload)
        payload_schema_version = payload.get("schema_version")
        if payload_schema_version != cls.SCHEMA_VERSION:
            raise ConversationStateVersionError(
                f"unsupported payload schema_version={payload_schema_version}",
            )
        forbidden = cls.FORBIDDEN_STATE_KEYS.intersection(payload)
        if forbidden:
            raise ConversationStateSerializationError(
                f"authoritative state fields are forbidden: {sorted(forbidden)}",
            )
        session_id = cls._session_id(row["session_id"])
        if payload.get("conversation_id") != session_id:
            raise ConversationStateSerializationError(
                "persisted conversation_id does not match session_id",
            )
        if payload.get("recent_turns") not in (None, []):
            raise ConversationStateSerializationError(
                "recent_turns are runtime-only and cannot be persisted",
            )
        try:
            return DialogueState.from_dict(payload)
        except (TypeError, ValueError, KeyError) as exc:
            raise ConversationStateSerializationError(
                "structured DialogueState payload is invalid",
            ) from exc

    def _row_for(
        self,
        session_id: str,
        user_id: int | None,
    ) -> sqlite3.Row | None:
        session_id = self._session_id(session_id)
        conn = self._get_conn()
        try:
            if user_id is not None:
                return conn.execute(
                    """
                    SELECT * FROM conversation_states
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (self._user_id(user_id), session_id),
                ).fetchone()
            rows = conn.execute(
                """
                SELECT * FROM conversation_states
                WHERE session_id = ?
                ORDER BY user_id
                """,
                (session_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise ConversationStateStoreError(
                "conversation state read failed",
            ) from exc
        if len(rows) > 1:
            raise ConversationStateIdentityError(
                "user_id is required when session_id is shared by multiple users",
            )
        return rows[0] if rows else None

    def get_state(
        self,
        conversation_id: str,
        user_id: int | None = None,
    ) -> DialogueState | None:
        row = self._row_for(conversation_id, user_id)
        return None if row is None else self._deserialize_state(row)

    def save_state(
        self,
        state: DialogueState,
        user_id: int | None = None,
        expected_state_version: int | None = None,
    ) -> None:
        encoded, compressed_history, turn_count = self._serialize_state(state)
        session_id = self._session_id(state.conversation_id)
        normalized_user_id = (
            self.LEGACY_USER_ID if user_id is None else self._user_id(user_id)
        )
        if expected_state_version is not None and (
            isinstance(expected_state_version, bool)
            or not isinstance(expected_state_version, int)
            or expected_state_version < 0
        ):
            raise ConversationStateVersionError(
                "expected_state_version must be a non-negative integer",
            )

        conn = self._get_conn()
        now = time.time()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT state_version FROM conversation_states
                WHERE user_id = ? AND session_id = ?
                """,
                (normalized_user_id, session_id),
            ).fetchone()
            current_version = None if row is None else int(row["state_version"])
            if (
                expected_state_version is not None
                and current_version != expected_state_version
            ):
                raise ConversationStateConflictError(
                    "conversation state version conflict",
                )
            if row is None:
                conn.execute(
                    """
                    INSERT INTO conversation_states (
                        user_id, session_id, schema_version, state_version,
                        structured_state_json, compressed_history, turn_count,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_user_id,
                        session_id,
                        self.SCHEMA_VERSION,
                        encoded,
                        compressed_history,
                        turn_count,
                        now,
                        now,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE conversation_states
                    SET schema_version = ?, state_version = state_version + 1,
                        structured_state_json = ?, compressed_history = ?,
                        turn_count = ?, updated_at = ?
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (
                        self.SCHEMA_VERSION,
                        encoded,
                        compressed_history,
                        turn_count,
                        now,
                        normalized_user_id,
                        session_id,
                    ),
                )
            conn.commit()
        except ConversationStateConflictError:
            conn.rollback()
            raise
        except sqlite3.Error as exc:
            conn.rollback()
            raise ConversationStateStoreError(
                "conversation state write failed",
            ) from exc

    def clear_state(
        self,
        conversation_id: str,
        user_id: int | None = None,
    ) -> bool:
        session_id = self._session_id(conversation_id)
        row = self._row_for(session_id, user_id)
        if row is None:
            return False
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                DELETE FROM conversation_states
                WHERE user_id = ? AND session_id = ?
                """,
                (int(row["user_id"]), session_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as exc:
            conn.rollback()
            raise ConversationStateStoreError(
                "conversation state delete failed",
            ) from exc

    def get(self, user_id: int, session_id: str) -> DialogueState | None:
        """Get state using the explicit production identity boundary."""
        return self.get_state(session_id, user_id=user_id)

    def save(
        self,
        user_id: int,
        state: DialogueState,
        expected_state_version: int | None = None,
    ) -> None:
        """Save state using the explicit production identity boundary."""
        self.save_state(
            state,
            user_id=user_id,
            expected_state_version=expected_state_version,
        )

    put = save

    def delete(self, user_id: int, session_id: str) -> bool:
        """Delete state using the explicit production identity boundary."""
        return self.clear_state(session_id, user_id=user_id)

    def exists(self, user_id: int, session_id: str) -> bool:
        return self.get(user_id, session_id) is not None

    def delete_all_for_user(self, user_id: int) -> int:
        """Delete all structured states for one user during session cleanup."""
        normalized_user_id = self._user_id(user_id)
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM conversation_states WHERE user_id = ?",
                (normalized_user_id,),
            )
            conn.commit()
            return cursor.rowcount
        except sqlite3.Error as exc:
            conn.rollback()
            raise ConversationStateStoreError(
                "conversation state bulk delete failed",
            ) from exc

    def get_state_version(self, user_id: int, session_id: str) -> int | None:
        row = self._row_for(session_id, user_id)
        return None if row is None else int(row["state_version"])

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
