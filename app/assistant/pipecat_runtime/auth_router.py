"""Application-specific audio gate before Pipecat VAD/STT."""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from collections.abc import Awaitable, Callable

import numpy as np
from pipecat.frames.frames import Frame, UserAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.assistant.auth_controller import VoiceAuthChallengeController
from app.assistant.contracts import (
    AuthChallengePhase,
    AuthChallengeStatus,
    AuthDecision,
    AuthState,
)

logger = logging.getLogger(__name__)


class AuthRouterProcessor(FrameProcessor):
    """Consume challenge audio and forward ordinary conversation audio.

    The challenge is consumed before Pipecat's VAD and STT. This is the
    security boundary that prevents the unlock phrase from becoming a chat
    transcript. Only the trusted descriptor-selected participant is accepted.
    """

    def __init__(
        self,
        *,
        controller: VoiceAuthChallengeController,
        participant_id: str,
        on_decision: Callable[[AuthDecision], Awaitable[None]],
        on_error: Callable[[str], Awaitable[None]],
        on_auth_progress: Callable[[AuthDecision, AuthChallengeStatus], Awaitable[None]] | None = None,
        resume_guard_seconds: float = 0.25,
    ) -> None:
        super().__init__(name="auth-router")
        if resume_guard_seconds < 0:
            raise ValueError("resume_guard_seconds cannot be negative.")
        self._controller = controller
        self._participant_id = participant_id
        self._on_decision = on_decision
        self._on_error = on_error
        self._on_auth_progress = on_auth_progress
        self._resume_guard_seconds = resume_guard_seconds
        self._last_notified_state = self.auth_state.state
        self._conversation_ready = False
        self._resume_guard_until = 0.0
        self._last_progress_elapsed_ms = -1
        self.challenge_frames_consumed = 0
        self.post_challenge_frames_dropped = 0
        self.resume_guard_frames_dropped = 0
        self.startup_frames_dropped = 0

    @property
    def auth_state(self) -> AuthDecision:
        return self._controller.current()

    def request_private_mode(self) -> AuthDecision:
        decision = self._controller.request_private_mode()
        self._last_notified_state = decision.state
        self._last_progress_elapsed_ms = -1
        return decision

    def cancel_private_mode(self) -> AuthDecision:
        decision = self._controller.cancel()
        self._last_notified_state = decision.state
        self._resume_guard_until = monotonic() + self._resume_guard_seconds
        self._last_progress_elapsed_ms = -1
        return decision

    def resume_conversation(self) -> AuthDecision:
        decision = self._controller.resume_conversation()
        self._last_notified_state = decision.state
        self._resume_guard_until = monotonic() + self._resume_guard_seconds
        return decision

    def set_conversation_ready(self) -> None:
        """Allow ordinary microphone audio after the local engine is ready."""

        self._conversation_ready = True

    @property
    def conversation_ready(self) -> bool:
        return self._conversation_ready

    @property
    def conversation_ingress_open(self) -> bool:
        phase = self._challenge_phase()
        return (
            self._conversation_ready
            and phase is AuthChallengePhase.CONVERSATION_READY
            and monotonic() >= self._resume_guard_until
        ) or (
            self._conversation_ready
            and phase is AuthChallengePhase.IDLE
            and monotonic() >= self._resume_guard_until
        )

    def challenge_status(self) -> AuthChallengeStatus:
        return self._controller.challenge_status()

    def _challenge_phase(self) -> AuthChallengePhase:
        # Keep test doubles and older callers compatible while all production
        # controllers expose the explicit phase contract.
        status_method = getattr(self._controller, "challenge_status", None)
        if status_method is None:
            return (
                AuthChallengePhase.CAPTURING
                if self.auth_state.state is AuthState.AUTH_PENDING
                else AuthChallengePhase.IDLE
            )
        return status_method().phase

    async def _maybe_publish_auth_progress(self, *, force: bool = False) -> None:
        if self._on_auth_progress is None:
            return
        status = self.challenge_status()
        if status.phase is not AuthChallengePhase.CAPTURING:
            return
        if not force and status.elapsed_ms - self._last_progress_elapsed_ms < 250:
            return
        self._last_progress_elapsed_ms = status.elapsed_ms
        await self._on_auth_progress(self.auth_state, status)

    async def _notify_state_change(self) -> AuthDecision:
        """Publish expiration transitions observed while normal audio flows."""

        decision = self.auth_state
        if decision.state is not self._last_notified_state:
            self._last_notified_state = decision.state
            await self._on_decision(decision)
        return decision

    def should_consume(self, frame: UserAudioRawFrame) -> bool:
        return frame.user_id == self._participant_id and self._challenge_phase() in {
            AuthChallengePhase.CAPTURING,
            AuthChallengePhase.PROCESSING,
            AuthChallengePhase.WAITING_FOR_RESUME,
        }

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserAudioRawFrame):
            # Never mix a second participant's audio into this account's auth
            # session, even if the room token was misconfigured.
            if frame.user_id != self._participant_id:
                logger.warning("pipecat_audio_dropped participant=%s reason=unexpected_participant", frame.user_id)
                return
            if not self._conversation_ready:
                self.startup_frames_dropped += 1
                logger.debug(
                    "pipecat_audio_dropped participant=%s reason=conversation_starting",
                    frame.user_id,
                )
                return
            decision = await self._notify_state_change()
            phase = self._challenge_phase()
            if phase is AuthChallengePhase.CAPTURING:
                self.challenge_frames_consumed += 1
                try:
                    values = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
                    capture_method = getattr(self._controller, "capture_pcm", None)
                    if capture_method is None:
                        # Compatibility path for older test doubles/callers.
                        decision = await asyncio.to_thread(
                            self._controller.append_pcm, values, sample_rate=frame.sample_rate
                        )
                    else:
                        captured = await asyncio.to_thread(
                            capture_method, values, sample_rate=frame.sample_rate
                        )
                        decision = None
                        if captured:
                            # Publish PROCESSING before the matcher runs. The
                            # controller still owns the pending recording.
                            await self._maybe_publish_auth_processing()
                            decision = await asyncio.to_thread(self._controller.verify_pending)
                    if decision is not None:
                        self._last_notified_state = decision.state
                        await self._on_decision(decision)
                    else:
                        await self._maybe_publish_auth_progress()
                except Exception:
                    # A malformed/failed verification must not leak audio to
                    # STT and must not accidentally grant private access.
                    logger.exception("voice_challenge_failed")
                    failure_decision = self.cancel_private_mode()
                    await self._on_decision(failure_decision)
                    await self._on_error("Voice verification could not be completed. Please try again.")
                return
            if phase in {
                AuthChallengePhase.PROCESSING,
                AuthChallengePhase.WAITING_FOR_RESUME,
            }:
                self.post_challenge_frames_dropped += 1
                logger.debug(
                    "pipecat_audio_dropped participant=%s reason=auth_phase phase=%s",
                    frame.user_id,
                    phase.value,
                )
                return
            if not self.conversation_ingress_open:
                self.resume_guard_frames_dropped += 1
                logger.debug(
                    "pipecat_audio_dropped participant=%s reason=resume_boundary",
                    frame.user_id,
                )
                return
        await self.push_frame(frame, direction)

    async def _maybe_publish_auth_processing(self) -> None:
        if self._on_auth_progress is None:
            return
        status = self.challenge_status()
        if status.phase is AuthChallengePhase.PROCESSING:
            await self._on_auth_progress(self.auth_state, status)
