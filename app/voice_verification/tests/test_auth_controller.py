from __future__ import annotations

from datetime import timedelta
from threading import Event, Thread

import numpy as np

from app.assistant.auth_controller import VoiceAuthChallengeController
from app.assistant.auth_session import AuthSession
from app.assistant.contracts import AuthState
from app.assistant.contracts import AuthChallengePhase
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


def test_failed_voice_verification_requires_an_explicit_guest_or_retry_action() -> None:
    controller = VoiceAuthChallengeController(
        gate=Gate(matched=False), target_identity_id="owner", session=AuthSession(ttl=timedelta(minutes=5))
    )
    controller.request_private_mode()

    decision = controller.append_pcm(np.ones(48_000 * 5, dtype=np.float32), sample_rate=48_000)

    assert decision is not None and decision.state is AuthState.GUEST
    assert controller.phase is AuthChallengePhase.WAITING_FOR_RESUME
    assert controller.challenge_status().can_resume is True

    # Audio after a failed challenge is still isolated until the user chooses
    # what should happen next; it must not silently enter the Agent.
    assert controller.append_pcm(np.ones(48_000, dtype=np.float32), sample_rate=48_000) is None

    resumed = controller.resume_conversation()
    assert resumed.state is AuthState.GUEST
    assert controller.phase is AuthChallengePhase.CONVERSATION_READY

    # Retrying starts a fresh challenge and is also an explicit transition.
    retry = controller.request_private_mode()
    assert retry.state is AuthState.AUTH_PENDING
    assert controller.phase is AuthChallengePhase.CAPTURING


def test_terminal_frame_is_clamped_and_result_waits_for_explicit_resume() -> None:
    class InspectingGate(Gate):
        def __init__(self) -> None:
            super().__init__(matched=True)
            self.recordings = []

        def verify(self, recording, *, target_identity_id):
            self.recordings.append(recording)
            return super().verify(recording, target_identity_id=target_identity_id)

    gate = InspectingGate()
    controller = VoiceAuthChallengeController(
        gate=gate, target_identity_id="owner", session=AuthSession(ttl=timedelta(minutes=5))
    )
    controller.request_private_mode()

    # A frame can contain substantially more than the remaining challenge
    # duration. Only the first five seconds may be retained.
    decision = controller.append_pcm(np.ones(48_000 * 6, dtype=np.float32), sample_rate=48_000)

    assert decision is not None and decision.state is AuthState.AUTHENTICATED
    assert len(gate.recordings) == 1
    assert gate.recordings[0].duration_seconds == 5.0
    status = controller.challenge_status()
    assert status.phase is AuthChallengePhase.WAITING_FOR_RESUME
    assert status.elapsed_ms == 5_000
    assert status.can_resume is True
    assert controller.append_pcm(np.ones(48_000, dtype=np.float32), sample_rate=48_000) is None
    assert gate.calls == 1

    resumed = controller.resume_conversation()
    assert resumed.state is AuthState.AUTHENTICATED
    assert controller.phase is AuthChallengePhase.CONVERSATION_READY


def test_cancel_during_processing_cannot_apply_an_old_verification_result() -> None:
    class BlockingGate(Gate):
        def __init__(self) -> None:
            super().__init__(matched=True)
            self.started = Event()
            self.release = Event()

        def verify(self, recording, *, target_identity_id):
            self.started.set()
            self.release.wait(timeout=1)
            return super().verify(recording, target_identity_id=target_identity_id)

    gate = BlockingGate()
    controller = VoiceAuthChallengeController(
        gate=gate, target_identity_id="owner", session=AuthSession(ttl=timedelta(minutes=5))
    )
    controller.request_private_mode()
    result: list[object] = []
    worker = Thread(
        target=lambda: result.append(
            controller.append_pcm(np.ones(48_000 * 5, dtype=np.float32), sample_rate=48_000)
        )
    )
    worker.start()
    assert gate.started.wait(timeout=1)

    controller.cancel()
    gate.release.set()
    worker.join(timeout=1)
    assert result == [None]
    assert controller.phase is AuthChallengePhase.IDLE
