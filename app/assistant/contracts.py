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


class AuthChallengePhase(StrEnum):
    """Audio-ingress phase for one explicit voice-auth challenge.

    This is deliberately separate from :class:`AuthState`. A successful
    verification can grant authenticated capabilities while conversation
    audio remains blocked until the user explicitly resumes the Agent.
    """

    IDLE = "idle"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    WAITING_FOR_RESUME = "waiting_for_resume"
    CONVERSATION_READY = "conversation_ready"


@dataclass(frozen=True)
class AuthChallengeStatus:
    """Safe progress information exposed to the browser."""

    phase: AuthChallengePhase
    elapsed_ms: int
    target_ms: int
    can_resume: bool = False

    @classmethod
    def idle(cls, target_ms: int) -> "AuthChallengeStatus":
        return cls(
            phase=AuthChallengePhase.IDLE,
            elapsed_ms=0,
            target_ms=target_ms,
            can_resume=False,
        )


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
