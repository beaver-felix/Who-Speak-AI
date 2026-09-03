"""Future LiveKit task contract for gating private tools with speaker verification."""

from __future__ import annotations

from app.assistant.contracts import AuthDecision, AuthState


class VoiceAuthenticationTask:
    """Adapter point: call VoiceAuthGate, never inspect transcript text."""

    def __init__(self, voice_auth_gate) -> None:
        self.voice_auth_gate = voice_auth_gate

    def authenticate(self, audio, *, target_identity_id: str) -> AuthDecision:
        result = self.voice_auth_gate.verify(audio, target_identity_id=target_identity_id)
        if result.matched:
            # This adapter deliberately has no permission to invent expiry.
            # LiveKit owns it through AuthSession.
            return AuthDecision(AuthState.AUTHENTICATED, result.identity_id, result.display_name)
        return AuthDecision(AuthState.GUEST)
