from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


DEFAULT_TITLE = "New chat"
MAX_TITLE_CHARACTERS = 200
MAX_MODEL_ID_CHARACTERS = 300
MAX_MESSAGE_CHARACTERS = 1_000_000
MAX_REASONING_CHARACTERS = 1_000_000
MAX_MESSAGES_PER_SESSION = 2_000
MAX_SESSIONS_PER_LIST = 500
MAX_MEDIA_PER_SESSION = 128
MAX_MEDIA_BYTES = 512 * 1024 * 1024
MAX_ORIGINAL_NAME_CHARACTERS = 255
MAX_MIME_TYPE_CHARACTERS = 127
MAX_STORED_PATH_CHARACTERS = 4_096
MAX_JSON_BYTES = 524_288
MAX_JSON_DEPTH = 12
MAX_JSON_ITEMS = 20_000
MAX_JSON_KEY_CHARACTERS = 200
MAX_JSON_STRING_CHARACTERS = 262_144
ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
ALLOWED_MEDIA_TYPES = frozenset({"image", "video"})
_MIME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SessionStore:
    def __init__(self, db_path: str | os.PathLike[str]):
        if not isinstance(db_path, (str, os.PathLike)):
            raise TypeError("db_path must be a path")
        value = os.fspath(db_path)
        if not value or value == ":memory:":
            raise ValueError("db_path must be a non-empty filesystem path")
        self.db_path = Path(value)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            connection.close()
            raise RuntimeError("SQLite foreign keys could not be enabled")
        return connection

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = NORMAL")
                schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
                if schema_version not in {0, 1}:
                    raise RuntimeError(
                        f"unsupported sessions database version: {schema_version}"
                    )
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY NOT NULL,
                        title TEXT NOT NULL,
                        model_id TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY NOT NULL,
                        session_id TEXT NOT NULL,
                        position INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        reasoning TEXT,
                        metrics_json TEXT,
                        created_at REAL NOT NULL,
                        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                        UNIQUE(session_id, position)
                    );
                    CREATE TABLE IF NOT EXISTS media (
                        id TEXT PRIMARY KEY NOT NULL,
                        session_id TEXT NOT NULL,
                        message_id TEXT,
                        position INTEGER NOT NULL,
                        media_type TEXT NOT NULL,
                        original_name TEXT NOT NULL,
                        mime_type TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        sha256 TEXT,
                        stored_path TEXT NOT NULL UNIQUE,
                        metadata_json TEXT,
                        created_at REAL NOT NULL,
                        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                        FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE SET NULL,
                        UNIQUE(session_id, position)
                    );
                    CREATE INDEX IF NOT EXISTS idx_messages_session_position
                        ON messages(session_id, position);
                    CREATE INDEX IF NOT EXISTS idx_media_session_position
                        ON media(session_id, position);
                    CREATE INDEX IF NOT EXISTS idx_media_message
                        ON media(message_id);
                    PRAGMA user_version = 1;
                    COMMIT;
                    """
                )
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.close()

    def create_session(
        self,
        model_id: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        validated_model_id = self._optional_text(
            model_id,
            "model_id",
            MAX_MODEL_ID_CHARACTERS,
        )
        validated_title = (
            DEFAULT_TITLE
            if title is None
            else self._required_text(title, "title", MAX_TITLE_CHARACTERS)
        )
        session_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO sessions "
                    "(id, title, model_id, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, validated_title, validated_model_id, now, now),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
        return {
            "id": session_id,
            "title": validated_title,
            "model_id": validated_model_id,
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "media": [],
        }

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        validated_limit = self._bounded_integer(
            limit,
            "limit",
            1,
            MAX_SESSIONS_PER_LIST,
        )
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    "SELECT id, title, model_id, created_at, updated_at "
                    "FROM sessions ORDER BY updated_at DESC, id ASC LIMIT ?",
                    (validated_limit,),
                ).fetchall()
            finally:
                connection.close()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        validated_session_id = self._uuid(session_id, "session_id")
        with self._lock:
            connection = self._connect()
            try:
                session = connection.execute(
                    "SELECT id, title, model_id, created_at, updated_at "
                    "FROM sessions WHERE id = ?",
                    (validated_session_id,),
                ).fetchone()
                if session is None:
                    return None
                messages = connection.execute(
                    "SELECT id, session_id, position, role, content, reasoning, "
                    "metrics_json, created_at FROM messages "
                    "WHERE session_id = ? ORDER BY position ASC",
                    (validated_session_id,),
                ).fetchall()
                media = connection.execute(
                    "SELECT id, session_id, message_id, position, media_type, "
                    "original_name, mime_type, size_bytes, sha256, metadata_json, "
                    "created_at FROM media WHERE session_id = ? ORDER BY position ASC",
                    (validated_session_id,),
                ).fetchall()
            finally:
                connection.close()
        result = dict(session)
        result["messages"] = [self._message_from_row(row) for row in messages]
        result["media"] = [self._public_media_from_row(row) for row in media]
        return result

    def rename_session(self, session_id: str, title: str) -> bool:
        validated_session_id = self._uuid(session_id, "session_id")
        validated_title = self._required_text(
            title,
            "title",
            MAX_TITLE_CHARACTERS,
        )
        now = time.time()
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (validated_title, now, validated_session_id),
                )
                connection.commit()
                return cursor.rowcount == 1
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def delete_session(self, session_id: str) -> list[Path] | None:
        validated_session_id = self._uuid(session_id, "session_id")
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                exists = connection.execute(
                    "SELECT 1 FROM sessions WHERE id = ?",
                    (validated_session_id,),
                ).fetchone()
                if exists is None:
                    connection.rollback()
                    return None
                rows = connection.execute(
                    "SELECT stored_path FROM media WHERE session_id = ? "
                    "ORDER BY position ASC",
                    (validated_session_id,),
                ).fetchall()
                connection.execute(
                    "DELETE FROM sessions WHERE id = ?",
                    (validated_session_id,),
                )
                connection.commit()
                return [Path(row["stored_path"]) for row in rows]
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def reset_conversation(self, session_id: str) -> list[Path] | None:
        validated_session_id = self._uuid(session_id, "session_id")
        now = time.time()
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                exists = connection.execute(
                    "SELECT 1 FROM sessions WHERE id = ?",
                    (validated_session_id,),
                ).fetchone()
                if exists is None:
                    connection.rollback()
                    return None
                rows = connection.execute(
                    "SELECT stored_path FROM media WHERE session_id = ? "
                    "ORDER BY position ASC",
                    (validated_session_id,),
                ).fetchall()
                connection.execute(
                    "DELETE FROM media WHERE session_id = ?",
                    (validated_session_id,),
                )
                connection.execute(
                    "DELETE FROM messages WHERE session_id = ?",
                    (validated_session_id,),
                )
                connection.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (DEFAULT_TITLE, now, validated_session_id),
                )
                connection.commit()
                return [Path(row["stored_path"]) for row in rows]
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def reset_session(self, session_id: str) -> list[Path] | None:
        return self.reset_conversation(session_id)

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        reasoning: str | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        validated_session_id = self._uuid(session_id, "session_id")
        if not isinstance(role, str) or role not in ALLOWED_ROLES:
            raise ValueError(f"role must be one of {sorted(ALLOWED_ROLES)}")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        if len(content) > MAX_MESSAGE_CHARACTERS:
            raise ValueError(
                f"content must contain at most {MAX_MESSAGE_CHARACTERS} characters"
            )
        validated_reasoning = self._optional_text(
            reasoning,
            "reasoning",
            MAX_REASONING_CHARACTERS,
            allow_blank=False,
        )
        if not content.strip() and validated_reasoning is None:
            raise ValueError("content or reasoning must be non-empty")
        metrics_json, validated_metrics = self._json_mapping(metrics, "metrics")
        message_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = connection.execute(
                    "SELECT title FROM sessions WHERE id = ?",
                    (validated_session_id,),
                ).fetchone()
                if session is None:
                    raise KeyError(f"unknown session_id: {validated_session_id}")
                count = connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                    (validated_session_id,),
                ).fetchone()[0]
                if count >= MAX_MESSAGES_PER_SESSION:
                    raise ValueError(
                        f"session cannot contain more than {MAX_MESSAGES_PER_SESSION} messages"
                    )
                position = count
                connection.execute(
                    "INSERT INTO messages "
                    "(id, session_id, position, role, content, reasoning, metrics_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        message_id,
                        validated_session_id,
                        position,
                        role,
                        content,
                        validated_reasoning,
                        metrics_json,
                        now,
                    ),
                )
                title = session["title"]
                if role == "user" and title == DEFAULT_TITLE:
                    first_user = connection.execute(
                        "SELECT COUNT(*) FROM messages "
                        "WHERE session_id = ? AND role = 'user'",
                        (validated_session_id,),
                    ).fetchone()[0]
                    if first_user == 1:
                        title = content.strip()[:MAX_TITLE_CHARACTERS] or DEFAULT_TITLE
                connection.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (title, now, validated_session_id),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
        return {
            "id": message_id,
            "session_id": validated_session_id,
            "position": position,
            "role": role,
            "content": content,
            "reasoning": validated_reasoning,
            "metrics": validated_metrics,
            "created_at": now,
        }

    def append_turn(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        *,
        reasoning: str | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        validated_session_id = self._uuid(session_id, "session_id")
        validated_user = self._required_text(
            user_content,
            "user_content",
            MAX_MESSAGE_CHARACTERS,
        )
        if not isinstance(assistant_content, str):
            raise TypeError("assistant_content must be a string")
        if len(assistant_content) > MAX_MESSAGE_CHARACTERS:
            raise ValueError(
                f"assistant_content must contain at most {MAX_MESSAGE_CHARACTERS} characters"
            )
        validated_reasoning = self._optional_text(
            reasoning,
            "reasoning",
            MAX_REASONING_CHARACTERS,
            allow_blank=False,
        )
        if not assistant_content.strip() and validated_reasoning is None:
            raise ValueError("assistant_content or reasoning must be non-empty")
        metrics_json, validated_metrics = self._json_mapping(metrics, "metrics")
        user_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = connection.execute(
                    "SELECT title FROM sessions WHERE id = ?",
                    (validated_session_id,),
                ).fetchone()
                if session is None:
                    raise KeyError(f"unknown session_id: {validated_session_id}")
                count = connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                    (validated_session_id,),
                ).fetchone()[0]
                if count > MAX_MESSAGES_PER_SESSION - 2:
                    raise ValueError(
                        f"session cannot contain more than {MAX_MESSAGES_PER_SESSION} messages"
                    )
                connection.execute(
                    "INSERT INTO messages "
                    "(id, session_id, position, role, content, reasoning, metrics_json, created_at) "
                    "VALUES (?, ?, ?, 'user', ?, NULL, NULL, ?)",
                    (user_id, validated_session_id, count, validated_user, now),
                )
                connection.execute(
                    "INSERT INTO messages "
                    "(id, session_id, position, role, content, reasoning, metrics_json, created_at) "
                    "VALUES (?, ?, ?, 'assistant', ?, ?, ?, ?)",
                    (
                        assistant_id,
                        validated_session_id,
                        count + 1,
                        assistant_content,
                        validated_reasoning,
                        metrics_json,
                        now,
                    ),
                )
                title = session["title"]
                if title == DEFAULT_TITLE:
                    title = validated_user[:MAX_TITLE_CHARACTERS]
                connection.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (title, now, validated_session_id),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
        return (
            {
                "id": user_id,
                "session_id": validated_session_id,
                "position": count,
                "role": "user",
                "content": validated_user,
                "reasoning": None,
                "metrics": None,
                "created_at": now,
            },
            {
                "id": assistant_id,
                "session_id": validated_session_id,
                "position": count + 1,
                "role": "assistant",
                "content": assistant_content,
                "reasoning": validated_reasoning,
                "metrics": validated_metrics,
                "created_at": now,
            },
        )

    def register_media(
        self,
        session_id: str,
        *,
        stored_path: str | os.PathLike[str],
        media_type: str,
        original_name: str,
        mime_type: str,
        size_bytes: int,
        sha256: str | None = None,
        message_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        validated_session_id = self._uuid(session_id, "session_id")
        validated_message_id = (
            None if message_id is None else self._uuid(message_id, "message_id")
        )
        if not isinstance(media_type, str) or media_type not in ALLOWED_MEDIA_TYPES:
            raise ValueError(f"media_type must be one of {sorted(ALLOWED_MEDIA_TYPES)}")
        validated_name = self._required_text(
            original_name,
            "original_name",
            MAX_ORIGINAL_NAME_CHARACTERS,
        )
        if validated_name in {".", ".."} or any(
            character in validated_name for character in ("/", "\\", "\x00")
        ):
            raise ValueError("original_name must be a plain file name")
        if not isinstance(mime_type, str):
            raise TypeError("mime_type must be a string")
        validated_mime_type = mime_type.strip().lower()
        if (
            not validated_mime_type
            or len(validated_mime_type) > MAX_MIME_TYPE_CHARACTERS
            or _MIME_PATTERN.fullmatch(validated_mime_type) is None
        ):
            raise ValueError("mime_type is invalid")
        if not validated_mime_type.startswith(f"{media_type}/"):
            raise ValueError("mime_type does not match media_type")
        validated_size = self._bounded_integer(
            size_bytes,
            "size_bytes",
            1,
            MAX_MEDIA_BYTES,
        )
        if not isinstance(stored_path, (str, os.PathLike)):
            raise TypeError("stored_path must be a path")
        stored_path_value = os.fspath(stored_path)
        if (
            not stored_path_value
            or len(stored_path_value) > MAX_STORED_PATH_CHARACTERS
            or "\x00" in stored_path_value
            or not Path(stored_path_value).is_absolute()
        ):
            raise ValueError("stored_path must be a bounded absolute path")
        if sha256 is not None:
            if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
                raise ValueError(
                    "sha256 must contain 64 lowercase hexadecimal characters"
                )
        metadata_json, validated_metadata = self._json_mapping(metadata, "metadata")
        media_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                exists = connection.execute(
                    "SELECT 1 FROM sessions WHERE id = ?",
                    (validated_session_id,),
                ).fetchone()
                if exists is None:
                    raise KeyError(f"unknown session_id: {validated_session_id}")
                if validated_message_id is not None:
                    message = connection.execute(
                        "SELECT session_id FROM messages WHERE id = ?",
                        (validated_message_id,),
                    ).fetchone()
                    if message is None:
                        raise KeyError(f"unknown message_id: {validated_message_id}")
                    if message["session_id"] != validated_session_id:
                        raise ValueError("message_id belongs to another session")
                count = connection.execute(
                    "SELECT COUNT(*) FROM media WHERE session_id = ?",
                    (validated_session_id,),
                ).fetchone()[0]
                if count >= MAX_MEDIA_PER_SESSION:
                    raise ValueError(
                        f"session cannot contain more than {MAX_MEDIA_PER_SESSION} media items"
                    )
                position = count
                connection.execute(
                    "INSERT INTO media "
                    "(id, session_id, message_id, position, media_type, original_name, "
                    "mime_type, size_bytes, sha256, stored_path, metadata_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        media_id,
                        validated_session_id,
                        validated_message_id,
                        position,
                        media_type,
                        validated_name,
                        validated_mime_type,
                        validated_size,
                        sha256,
                        stored_path_value,
                        metadata_json,
                        now,
                    ),
                )
                connection.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (now, validated_session_id),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
        return {
            "id": media_id,
            "session_id": validated_session_id,
            "message_id": validated_message_id,
            "position": position,
            "media_type": media_type,
            "original_name": validated_name,
            "mime_type": validated_mime_type,
            "size_bytes": validated_size,
            "sha256": sha256,
            "metadata": validated_metadata,
            "created_at": now,
        }

    def list_media(
        self,
        session_id: str,
        message_id: str | None = None,
    ) -> list[dict[str, Any]]:
        validated_session_id = self._uuid(session_id, "session_id")
        validated_message_id = (
            None if message_id is None else self._uuid(message_id, "message_id")
        )
        query = (
            "SELECT id, session_id, message_id, position, media_type, original_name, "
            "mime_type, size_bytes, sha256, metadata_json, created_at FROM media "
            "WHERE session_id = ?"
        )
        parameters: tuple[str, ...] = (validated_session_id,)
        if validated_message_id is not None:
            query += " AND message_id = ?"
            parameters = (validated_session_id, validated_message_id)
        query += " ORDER BY position ASC"
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(query, parameters).fetchall()
            finally:
                connection.close()
        return [self._public_media_from_row(row) for row in rows]

    def get_media(
        self,
        media_id: str,
        *,
        include_stored_path: bool = False,
    ) -> dict[str, Any] | None:
        validated_media_id = self._uuid(media_id, "media_id")
        if not isinstance(include_stored_path, bool):
            raise TypeError("include_stored_path must be a boolean")
        columns = (
            "id, session_id, message_id, position, media_type, original_name, "
            "mime_type, size_bytes, sha256, metadata_json, created_at"
        )
        if include_stored_path:
            columns += ", stored_path"
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    f"SELECT {columns} FROM media WHERE id = ?",
                    (validated_media_id,),
                ).fetchone()
            finally:
                connection.close()
        if row is None:
            return None
        result = self._public_media_from_row(row)
        if include_stored_path:
            result["stored_path"] = Path(row["stored_path"])
        return result

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "position": row["position"],
            "role": row["role"],
            "content": row["content"],
            "reasoning": row["reasoning"],
            "metrics": (
                None if row["metrics_json"] is None else json.loads(row["metrics_json"])
            ),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _public_media_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "message_id": row["message_id"],
            "position": row["position"],
            "media_type": row["media_type"],
            "original_name": row["original_name"],
            "mime_type": row["mime_type"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
            "metadata": (
                None
                if row["metadata_json"] is None
                else json.loads(row["metadata_json"])
            ),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _uuid(value: str, field: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a UUID string")
        try:
            parsed = uuid.UUID(value)
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"{field} must be a canonical UUID") from exc
        canonical = str(parsed)
        if value != canonical:
            raise ValueError(f"{field} must be a canonical UUID")
        return canonical

    @staticmethod
    def _required_text(value: str, field: str, maximum: int) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string")
        stripped = value.strip()
        if not stripped or len(stripped) > maximum or "\x00" in stripped:
            raise ValueError(f"{field} must contain 1 to {maximum} characters")
        return stripped

    @staticmethod
    def _optional_text(
        value: str | None,
        field: str,
        maximum: int,
        *,
        allow_blank: bool = False,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string or None")
        if "\x00" in value or len(value) > maximum:
            raise ValueError(f"{field} must contain at most {maximum} characters")
        if not allow_blank and not value.strip():
            raise ValueError(f"{field} cannot be blank")
        return value

    @staticmethod
    def _bounded_integer(value: int, field: str, minimum: int, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(f"{field} must be between {minimum} and {maximum}")
        return value

    @classmethod
    def _json_mapping(
        cls,
        value: Mapping[str, Any] | None,
        field: str,
    ) -> tuple[str | None, dict[str, Any] | None]:
        if value is None:
            return None, None
        if not isinstance(value, Mapping):
            raise TypeError(f"{field} must be a mapping or None")
        normalized = dict(value)
        item_count = [0]
        cls._validate_json_value(normalized, field, 0, item_count)
        try:
            serialized = json.dumps(
                normalized,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f"{field} must contain bounded JSON values") from exc
        if len(serialized.encode("utf-8")) > MAX_JSON_BYTES:
            raise ValueError(f"{field} must encode to at most {MAX_JSON_BYTES} bytes")
        return serialized, json.loads(serialized)

    @classmethod
    def _validate_json_value(
        cls,
        value: Any,
        field: str,
        depth: int,
        item_count: list[int],
    ) -> None:
        if depth > MAX_JSON_DEPTH:
            raise ValueError(f"{field} exceeds the maximum nesting depth")
        item_count[0] += 1
        if item_count[0] > MAX_JSON_ITEMS:
            raise ValueError(f"{field} contains too many values")
        if value is None or isinstance(value, (bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"{field} contains a non-finite number")
            return
        if isinstance(value, str):
            if len(value) > MAX_JSON_STRING_CHARACTERS or "\x00" in value:
                raise ValueError(f"{field} contains an invalid string")
            return
        if isinstance(value, list):
            for item in value:
                cls._validate_json_value(item, field, depth + 1, item_count)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or len(key) > MAX_JSON_KEY_CHARACTERS
                    or "\x00" in key
                ):
                    raise ValueError(f"{field} contains an invalid key")
                cls._validate_json_value(item, field, depth + 1, item_count)
            return
        raise ValueError(f"{field} contains a non-JSON value")
