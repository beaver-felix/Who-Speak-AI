"""SQLite persistence for local accounts, sessions, voice profiles, and demo events."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from app.assistant_gateway.security import digest_session_token


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class User:
    id: str
    email: str
    display_name: str


@dataclass(frozen=True)
class VoiceProfile:
    user_id: str
    context_id: str
    matcher_identity_id: str
    display_name: str


@dataclass(frozen=True)
class CalendarEvent:
    id: str
    user_id: str
    title: str
    start: datetime
    end: datetime


class GatewayStore:
    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_digest TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS voice_profiles (
                    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    context_id TEXT NOT NULL,
                    matcher_identity_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    enrolled_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mock_calendar_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    starts_at TEXT NOT NULL,
                    ends_at TEXT NOT NULL
                );
                """
            )

    def create_user(self, *, email: str, password_hash: str, display_name: str) -> User:
        user = User(str(uuid4()), email.strip().lower(), display_name.strip())
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO users (id, email, password_hash, display_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (user.id, user.email, password_hash, user.display_name, _now().isoformat()),
            )
        return user

    def user_by_email(self, email: str) -> tuple[User, str] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, email, display_name, password_hash FROM users WHERE email = ?", (email.strip().lower(),)
            ).fetchone()
        return (User(row["id"], row["email"], row["display_name"]), row["password_hash"]) if row else None

    def user_by_id(self, user_id: str) -> User | None:
        with self._connection() as connection:
            row = connection.execute("SELECT id, email, display_name FROM users WHERE id = ?", (user_id,)).fetchone()
        return User(row["id"], row["email"], row["display_name"]) if row else None

    def create_session(self, user_id: str, token: str, *, ttl: timedelta) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO sessions (token_digest, user_id, expires_at) VALUES (?, ?, ?)",
                (digest_session_token(token), user_id, (_now() + ttl).isoformat()),
            )

    def session_user(self, token: str) -> User | None:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT users.id, users.email, users.display_name, sessions.expires_at
                   FROM sessions JOIN users ON users.id = sessions.user_id
                   WHERE sessions.token_digest = ?""",
                (digest_session_token(token),),
            ).fetchone()
            if row is None:
                return None
            if datetime.fromisoformat(row["expires_at"]) <= _now():
                connection.execute("DELETE FROM sessions WHERE token_digest = ?", (digest_session_token(token),))
                return None
        return User(row["id"], row["email"], row["display_name"])

    def delete_session(self, token: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM sessions WHERE token_digest = ?", (digest_session_token(token),))

    def save_voice_profile(self, profile: VoiceProfile) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO voice_profiles (user_id, context_id, matcher_identity_id, display_name, enrolled_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                     context_id = excluded.context_id,
                     matcher_identity_id = excluded.matcher_identity_id,
                     display_name = excluded.display_name,
                     enrolled_at = excluded.enrolled_at""",
                (profile.user_id, profile.context_id, profile.matcher_identity_id, profile.display_name, _now().isoformat()),
            )

    def voice_profile(self, user_id: str) -> VoiceProfile | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT user_id, context_id, matcher_identity_id, display_name FROM voice_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return VoiceProfile(row["user_id"], row["context_id"], row["matcher_identity_id"], row["display_name"]) if row else None

    def list_mock_events(self, user_id: str, *, start: datetime, end: datetime) -> list[CalendarEvent]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT id, user_id, title, starts_at, ends_at FROM mock_calendar_events
                   WHERE user_id = ? AND starts_at < ? AND ends_at > ? ORDER BY starts_at""",
                (user_id, end.isoformat(), start.isoformat()),
            ).fetchall()
        return [CalendarEvent(row["id"], row["user_id"], row["title"], datetime.fromisoformat(row["starts_at"]), datetime.fromisoformat(row["ends_at"])) for row in rows]

    def create_mock_event(self, *, user_id: str, title: str, start: datetime, end: datetime) -> CalendarEvent:
        event = CalendarEvent(str(uuid4()), user_id, title.strip(), start, end)
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO mock_calendar_events (id, user_id, title, starts_at, ends_at) VALUES (?, ?, ?, ?, ?)",
                (event.id, event.user_id, event.title, event.start.isoformat(), event.end.isoformat()),
            )
        return event

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
