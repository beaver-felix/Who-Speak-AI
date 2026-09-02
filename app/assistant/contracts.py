"""Framework-neutral contracts shared between voice auth and the future agent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class AuthState(StrEnum):
    GUEST = "guest"
    AUTH_PENDING = "auth_pending"
    AUTHENTICATED = "authenticated"
    SESSION_EXPIRED = "session_expired"


@dataclass(frozen=True)
class AuthDecision:
    state: AuthState
    identity_id: str | None = None
    display_name: str | None = None
    expires_at: datetime | None = None

    @property
    def may_use_private_tools(self) -> bool:
        if self.state is not AuthState.AUTHENTICATED or self.identity_id is None or self.expires_at is None:
            return False
        try:
            return self.expires_at > datetime.now(UTC)
        except TypeError:
            # A naive or malformed expiry is never authorization evidence.
            return False
