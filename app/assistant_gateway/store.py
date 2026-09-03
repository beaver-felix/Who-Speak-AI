"""SQLite persistence for accounts, sessions, voice profiles, and integrations."""

from __future__ import annotations

import sqlite3
import json
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
    location: str | None = None


@dataclass(frozen=True)
class GoogleCalendarConnection:
    user_id: str
    google_subject: str
    google_email: str
    encrypted_refresh_token: str
    granted_scopes: tuple[str, ...]
    status: str
    token_expires_at: datetime | None
    created_at: datetime
    updated_at: datetime
    revoked_at: datetime | None = None


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
                CREATE TABLE IF NOT EXISTS google_calendar_connections (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                    google_subject TEXT NOT NULL UNIQUE,
                    google_email TEXT NOT NULL,
                    encrypted_refresh_token TEXT NOT NULL,
                    granted_scopes TEXT NOT NULL,
                    status TEXT NOT NULL,
                    token_expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                CREATE TABLE IF NOT EXISTS google_oauth_states (
                    state_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    session_binding TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT
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

    def session_user_by_digest(self, token_digest: str) -> User | None:
        """Resolve a session using only its already-derived digest.

        Agent-to-gateway calls use this method so an opaque browser session
        token never needs to cross the process boundary.
        """
        with self._connection() as connection:
            row = connection.execute(
                """SELECT users.id, users.email, users.display_name, sessions.expires_at
                   FROM sessions JOIN users ON users.id = sessions.user_id
                   WHERE sessions.token_digest = ?""",
                (token_digest,),
            ).fetchone()
            if row is None:
                return None
            if datetime.fromisoformat(row["expires_at"]) <= _now():
                connection.execute("DELETE FROM sessions WHERE token_digest = ?", (token_digest,))
                return None
        return User(row["id"], row["email"], row["display_name"])

    def create_google_oauth_state(
        self,
        *,
        state_hash: str,
        user_id: str,
        session_binding: str,
        expires_at: datetime,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO google_oauth_states
                   (state_hash, user_id, session_binding, expires_at)
                   VALUES (?, ?, ?, ?)""",
                (state_hash, user_id, session_binding, expires_at.isoformat()),
            )

    def consume_google_oauth_state(
        self, *, state_hash: str, session_binding: str
    ) -> str | None:
        """Atomically consume a short-lived OAuth state for this session."""
        with self._connection() as connection:
            row = connection.execute(
                """SELECT user_id, session_binding, expires_at, consumed_at
                   FROM google_oauth_states WHERE state_hash = ?""",
                (state_hash,),
            ).fetchone()
            if row is None or row["consumed_at"] is not None:
                return None
            if row["session_binding"] != session_binding:
                return None
            if datetime.fromisoformat(row["expires_at"]) <= _now():
                connection.execute(
                    "UPDATE google_oauth_states SET consumed_at = ? WHERE state_hash = ?",
                    (_now().isoformat(), state_hash),
                )
                return None
            connection.execute(
                "UPDATE google_oauth_states SET consumed_at = ? WHERE state_hash = ? AND consumed_at IS NULL",
                (_now().isoformat(), state_hash),
            )
            return row["user_id"]

    def save_google_calendar_connection(self, connection_data: GoogleCalendarConnection) -> None:
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT user_id FROM google_calendar_connections WHERE google_subject = ?",
                (connection_data.google_subject,),
            ).fetchone()
            if existing is not None and existing["user_id"] != connection_data.user_id:
                raise ValueError("This Google account is already linked to another Who Speak account.")
            connection.execute(
                """INSERT INTO google_calendar_connections
                   (id, user_id, google_subject, google_email, encrypted_refresh_token,
                    granted_scopes, status, token_expires_at, created_at, updated_at, revoked_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                     google_subject = excluded.google_subject,
                     google_email = excluded.google_email,
                     encrypted_refresh_token = excluded.encrypted_refresh_token,
                     granted_scopes = excluded.granted_scopes,
                     status = excluded.status,
                     token_expires_at = excluded.token_expires_at,
                     updated_at = excluded.updated_at,
                     revoked_at = excluded.revoked_at""",
                (
                    str(uuid4()),
                    connection_data.user_id,
                    connection_data.google_subject,
                    connection_data.google_email,
                    connection_data.encrypted_refresh_token,
                    json.dumps(list(connection_data.granted_scopes), separators=(",", ":")),
                    connection_data.status,
                    connection_data.token_expires_at.isoformat() if connection_data.token_expires_at else None,
                    connection_data.created_at.isoformat(),
                    connection_data.updated_at.isoformat(),
                    connection_data.revoked_at.isoformat() if connection_data.revoked_at else None,
                ),
            )

    def google_calendar_connection(self, user_id: str) -> GoogleCalendarConnection | None:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT user_id, google_subject, google_email, encrypted_refresh_token,
                          granted_scopes, status, token_expires_at, created_at, updated_at, revoked_at
                   FROM google_calendar_connections WHERE user_id = ?""",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return GoogleCalendarConnection(
            user_id=row["user_id"],
            google_subject=row["google_subject"],
            google_email=row["google_email"],
            encrypted_refresh_token=row["encrypted_refresh_token"],
            granted_scopes=tuple(json.loads(row["granted_scopes"])),
            status=row["status"],
            token_expires_at=datetime.fromisoformat(row["token_expires_at"]) if row["token_expires_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            revoked_at=datetime.fromisoformat(row["revoked_at"]) if row["revoked_at"] else None,
        )

    def mark_google_calendar_status(self, user_id: str, status: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE google_calendar_connections SET status = ?, updated_at = ? WHERE user_id = ?",
                (status, _now().isoformat(), user_id),
            )

    def disconnect_google_calendar(self, user_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM google_calendar_connections WHERE user_id = ?", (user_id,)
            )

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
