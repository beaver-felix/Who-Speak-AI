"""Explicit private-mode challenge controller, independent of LiveKit APIs."""

from __future__ import annotations

from threading import RLock

from app.assistant.audio_buffer import AuthChallengeBuffer
from app.assistant.auth_session import AuthSession
from app.assistant.contracts import (
    AuthChallengePhase,
    AuthChallengeStatus,
    AuthDecision,
    AuthState,
)


class VoiceAuthChallengeController:
    """Buffer audio only after an explicit private-mode request.

    This object is the sole owner of the authorization transition.  It is not
    exposed to ASR, the LLM, or any tool implementation.
    """

    def __init__(self, *, gate, target_identity_id: str, session: AuthSession, capture_seconds: float = 5.0) -> None:
        if not 4.0 <= capture_seconds <= 6.0:
            raise ValueError("SV challenge duration must be between 4 and 6 seconds.")
        self._gate = gate
        self._target_identity_id = target_identity_id
        self._session = session
        self._capture_seconds = capture_seconds
        self._buffer = AuthChallengeBuffer(minimum_seconds=4.0, maximum_seconds=6.0)
        self._lock = RLock()
        self._phase = AuthChallengePhase.IDLE
        self._generation = 0
        self._captured_elapsed_ms = 0
        self._pending_recording = None

    def current(self) -> AuthDecision:
        with self._lock:
            return self._session.current()

    @property
    def phase(self) -> AuthChallengePhase:
        with self._lock:
            return self._phase

    def challenge_status(self) -> AuthChallengeStatus:
        with self._lock:
            self._session.current()
            elapsed_ms = (
                round(self._buffer.duration_seconds * 1000)
                if self._phase is AuthChallengePhase.CAPTURING
                else self._captured_elapsed_ms
            )
            return AuthChallengeStatus(
                phase=self._phase,
                elapsed_ms=elapsed_ms,
                target_ms=round(self._capture_seconds * 1000),
                can_resume=self._phase is AuthChallengePhase.WAITING_FOR_RESUME,
            )

    def request_private_mode(self) -> AuthDecision:
        with self._lock:
            self._generation += 1
            self._buffer.reset()
            self._captured_elapsed_ms = 0
            self._pending_recording = None
            decision = self._session.begin_private_auth()
            self._phase = AuthChallengePhase.CAPTURING
            return decision

    def capture_pcm(self, samples, *, sample_rate: int) -> bool:
        """Consume challenge audio and stop at the exact configured boundary."""

        with self._lock:
            if self._phase is not AuthChallengePhase.CAPTURING:
                return False
            if self._session.current().state is not AuthState.AUTH_PENDING:
                return False
            # Clamp the terminal frame to the configured target. Any speech
            # after the boundary is intentionally never retained or routed to
            # the conversation pipeline.
            self._buffer.append_pcm(
                samples,
                sample_rate=sample_rate,
                max_duration_seconds=self._capture_seconds,
            )
            if self._buffer.duration_seconds < self._capture_seconds:
                return False
            self._captured_elapsed_ms = round(self._buffer.duration_seconds * 1000)
            self._phase = AuthChallengePhase.PROCESSING
            self._pending_recording = self._buffer.finish()
            self._buffer.reset()
            return True

    def verify_pending(self) -> AuthDecision | None:
        """Verify the captured challenge without holding the session lock."""

        generation: int
        with self._lock:
            if self._phase is not AuthChallengePhase.PROCESSING or self._pending_recording is None:
                return None
            generation = self._generation
            recording = self._pending_recording
            self._pending_recording = None

        result = self._gate.verify(recording, target_identity_id=self._target_identity_id)

        with self._lock:
            # A cancel/retry can happen while local verification is running.
            # Never apply an old result to the new challenge.
            if generation != self._generation or self._phase is not AuthChallengePhase.PROCESSING:
                return None
            try:
                decision = self._session.complete_verification(result)
            finally:
                self._phase = AuthChallengePhase.WAITING_FOR_RESUME
            return decision

    def append_pcm(self, samples, *, sample_rate: int) -> AuthDecision | None:
        """Compatibility helper that captures and verifies in one call."""

        if not self.capture_pcm(samples, sample_rate=sample_rate):
            return None
        return self.verify_pending()

    def resume_conversation(self) -> AuthDecision:
        with self._lock:
            if self._phase is not AuthChallengePhase.WAITING_FOR_RESUME:
                raise RuntimeError("Conversation can resume only after the voice challenge has a result.")
            self._phase = AuthChallengePhase.CONVERSATION_READY
            return self._session.current()

    def cancel(self) -> AuthDecision:
        with self._lock:
            self._generation += 1
            self._buffer.reset()
            self._captured_elapsed_ms = 0
            self._pending_recording = None
            self._phase = AuthChallengePhase.IDLE
            return self._session.invalidate()
