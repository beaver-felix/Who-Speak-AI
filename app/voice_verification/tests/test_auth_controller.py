from __future__ import annotations

from datetime import timedelta

import numpy as np

from app.assistant.auth_controller import VoiceAuthChallengeController
from app.assistant.auth_session import AuthSession
from app.assistant.contracts import AuthState
from voiceauth.matching import VerificationResult


class Gate:
    def __init__(self, matched: bool) -> None:
        self.matched = matched
        self.calls = 0

    def verify(self, recording, *, target_identity_id):
        self.calls += 1
        return VerificationResult(self.matched, target_identity_id if self.matched else None, "An", 0.9, 1, target_identity_id)


def test_voice_verification_only_runs_after_explicit_private_mode_request() -> None:
    gate = Gate(matched=True)
    controller = VoiceAuthChallengeController(
        gate=gate, target_identity_id="owner", session=AuthSession(ttl=timedelta(minutes=5))
    )

    assert controller.append_pcm(np.ones(48_000 * 5, dtype=np.float32), sample_rate=48_000) is None
    assert gate.calls == 0
    assert controller.request_private_mode().state is AuthState.AUTH_PENDING

    decision = controller.append_pcm(np.ones(48_000 * 5, dtype=np.float32), sample_rate=48_000)

    assert gate.calls == 1
    assert decision is not None and decision.state is AuthState.AUTHENTICATED


def test_failed_voice_verification_returns_to_guest() -> None:
    controller = VoiceAuthChallengeController(
        gate=Gate(matched=False), target_identity_id="owner", session=AuthSession(ttl=timedelta(minutes=5))
    )
    controller.request_private_mode()

    decision = controller.append_pcm(np.ones(48_000 * 5, dtype=np.float32), sample_rate=48_000)

    assert decision is not None and decision.state is AuthState.GUEST
