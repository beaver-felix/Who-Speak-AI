"""Explicit private-mode challenge controller, independent of LiveKit APIs."""

from __future__ import annotations

from threading import RLock

from app.assistant.audio_buffer import AuthChallengeBuffer
from app.assistant.auth_session import AuthSession
from app.assistant.contracts import AuthDecision, AuthState


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

    def current(self) -> AuthDecision:
        with self._lock:
            return self._session.current()

    def request_private_mode(self) -> AuthDecision:
        with self._lock:
            self._buffer.reset()
            return self._session.begin_private_auth()

    def append_pcm(self, samples, *, sample_rate: int) -> AuthDecision | None:
        with self._lock:
            if self._session.current().state is not AuthState.AUTH_PENDING:
                return None
            self._buffer.append_pcm(samples, sample_rate=sample_rate)
            if self._buffer.duration_seconds < self._capture_seconds:
                return None
            try:
                result = self._gate.verify(self._buffer.finish(), target_identity_id=self._target_identity_id)
                return self._session.complete_verification(result)
            finally:
                self._buffer.reset()

    def cancel(self) -> AuthDecision:
        with self._lock:
            self._buffer.reset()
            return self._session.invalidate()
