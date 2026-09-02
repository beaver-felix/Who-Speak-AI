"""Trusted, framework-neutral voice-authentication session state."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from app.assistant.contracts import AuthDecision, AuthState
from voiceauth.matching import VerificationResult


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AuthSession:
    """Own the auth state for one LiveKit participant connection.

    The LLM and transcript processing do not receive a mutator for this
    object.  Reconnects create a fresh ``AuthSession`` and therefore start as
    guests.
    """

    def __init__(self, *, ttl: timedelta, clock: Callable[[], datetime] = _utc_now) -> None:
        if ttl <= timedelta():
            raise ValueError("Authentication TTL must be positive.")
        self._ttl = ttl
        self._clock = clock
        self._decision = AuthDecision(AuthState.GUEST)

    def current(self) -> AuthDecision:
        if (
            self._decision.state is AuthState.AUTHENTICATED
            and self._decision.expires_at is not None
            and self._clock() >= self._decision.expires_at
        ):
            self._decision = AuthDecision(AuthState.SESSION_EXPIRED)
        return self._decision

    def begin_private_auth(self) -> AuthDecision:
        self._decision = AuthDecision(AuthState.AUTH_PENDING)
        return self._decision

    def complete_verification(self, result: VerificationResult) -> AuthDecision:
        if self._decision.state is not AuthState.AUTH_PENDING:
            raise RuntimeError("Cannot complete voice verification unless authentication is pending.")
        if result.matched and result.identity_id == result.target_identity_id:
            self._decision = AuthDecision(
                AuthState.AUTHENTICATED,
                identity_id=result.identity_id,
                display_name=result.display_name,
                expires_at=self._clock() + self._ttl,
            )
        else:
            self._decision = AuthDecision(AuthState.GUEST)
        return self._decision

    def invalidate(self) -> AuthDecision:
        self._decision = AuthDecision(AuthState.GUEST)
        return self._decision
