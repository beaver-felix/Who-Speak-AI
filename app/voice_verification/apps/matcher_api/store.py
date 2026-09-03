"""Small SQLite repository for ciphertext and public metadata only."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class StoredContext:
    context_id: str
    public_context_b64: str
    public_context_sha256: str
    created_at: datetime


@dataclass(frozen=True)
class StoredIdentity:
    identity_id: str
    context_id: str
    display_name: str
    encrypted_template_b64: str
    created_at: datetime


class MatcherStore:
    def __init__(self, database_path: str) -> None:
        self.path = Path(database_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS voice_contexts (
                    context_id TEXT PRIMARY KEY,
                    public_context_b64 TEXT NOT NULL,
                    public_context_sha256 TEXT NOT NULL,
                    model_profile TEXT NOT NULL,
                    preprocessing_profile TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    he_profile TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS voice_identities (
                    identity_id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL REFERENCES voice_contexts(context_id) ON DELETE CASCADE,
                    display_name TEXT NOT NULL,
                    encrypted_template_b64 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_voice_identities_context ON voice_identities(context_id);
                """
            )

    def get_context(self, context_id: str) -> StoredContext | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT context_id, public_context_b64, public_context_sha256, created_at FROM voice_contexts WHERE context_id = ?",
                (context_id,),
            ).fetchone()
        return self._context(row) if row else None

    def create_context(self, *, context_id: str, public_context_b64: str, public_context_sha256: str, model_profile: str, preprocessing_profile: str, embedding_dimension: int, he_profile: str) -> StoredContext:
        created = self._now()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO voice_contexts
                (context_id, public_context_b64, public_context_sha256, model_profile, preprocessing_profile, embedding_dimension, he_profile, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (context_id, public_context_b64, public_context_sha256, model_profile, preprocessing_profile, embedding_dimension, he_profile, created),
            )
        result = self.get_context(context_id)
        assert result is not None
        return result

    def create_identity(self, *, identity_id: str, context_id: str, display_name: str, encrypted_template_b64: str) -> StoredIdentity:
        created = self._now()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO voice_identities (identity_id, context_id, display_name, encrypted_template_b64, created_at) VALUES (?, ?, ?, ?, ?)",
                (identity_id, context_id, display_name, encrypted_template_b64, created),
            )
        return StoredIdentity(identity_id, context_id, display_name, encrypted_template_b64, self._parse_time(created))

    def list_identities(self, context_id: str, target_identity_id: str | None = None) -> list[StoredIdentity]:
        statement = "SELECT identity_id, context_id, display_name, encrypted_template_b64, created_at FROM voice_identities WHERE context_id = ?"
        values: tuple[str, ...] = (context_id,)
        if target_identity_id is not None:
            statement += " AND identity_id = ?"
            values = (context_id, target_identity_id)
        with self._connection() as connection:
            rows = connection.execute(statement + " ORDER BY created_at ASC", values).fetchall()
        return [self._identity(row) for row in rows]

    def count_identities(self, context_id: str) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS identity_count FROM voice_identities WHERE context_id = ?",
                (context_id,),
            ).fetchone()
        assert row is not None
        return int(row["identity_count"])

    def delete_context(self, context_id: str) -> bool:
        with self._connection() as connection:
            result = connection.execute("DELETE FROM voice_contexts WHERE context_id = ?", (context_id,))
        return result.rowcount > 0

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _parse_time(value: str) -> datetime:
        return datetime.fromisoformat(value)

    def _context(self, row: sqlite3.Row) -> StoredContext:
        return StoredContext(row["context_id"], row["public_context_b64"], row["public_context_sha256"], self._parse_time(row["created_at"]))

    def _identity(self, row: sqlite3.Row) -> StoredIdentity:
        return StoredIdentity(row["identity_id"], row["context_id"], row["display_name"], row["encrypted_template_b64"], self._parse_time(row["created_at"]))
