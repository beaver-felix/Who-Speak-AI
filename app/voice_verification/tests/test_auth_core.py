from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.assistant.auth_session import AuthSession
from app.assistant.contracts import AuthState
from app.assistant.tasks.voice_authentication import VoiceAuthenticationTask
from voiceauth.gate import VoiceAuthGate
from voiceauth.matching import IdentificationResult, VerificationResult


class FakeVoiceService:
    def __init__(self) -> None:
        self.verify_target = None

    def identify(self, audio):
        return IdentificationResult(True, "identified", "An", 0.9, 1)

    def verify(self, audio, *, target_identity_id):
        self.verify_target = target_identity_id
        return VerificationResult(True, str(target_identity_id), "An", 0.9, 1, str(target_identity_id))


def test_gate_verification_uses_explicit_target_identity() -> None:
    service = FakeVoiceService()
    target = uuid4()

    result = VoiceAuthGate(service).verify(object(), target_identity_id=target)

    assert result.matched is True
    assert service.verify_target == target


def test_voice_authentication_task_requires_a_target_identity() -> None:
    with pytest.raises(TypeError):
        VoiceAuthenticationTask(FakeVoiceService()).authenticate(object())


def test_auth_session_only_grants_private_access_after_successful_verification() -> None:
    now = datetime.now(UTC)
    session = AuthSession(ttl=timedelta(minutes=5), clock=lambda: now)

    assert session.current().state is AuthState.GUEST
    assert session.begin_private_auth().state is AuthState.AUTH_PENDING

    result = VerificationResult(True, "owner", "An", 0.9, 1, "owner")
    decision = session.complete_verification(result)

    assert decision.state is AuthState.AUTHENTICATED
    assert decision.may_use_private_tools is True
    assert decision.expires_at == now + timedelta(minutes=5)


def test_auth_session_expires_and_revokes_private_access() -> None:
    current = [datetime.now(UTC)]
    session = AuthSession(ttl=timedelta(seconds=1), clock=lambda: current[0])
    session.begin_private_auth()
    session.complete_verification(VerificationResult(True, "owner", "An", 0.9, 1, "owner"))
    current[0] += timedelta(seconds=2)

    expired = session.current()

    assert expired.state is AuthState.SESSION_EXPIRED
    assert expired.may_use_private_tools is False
