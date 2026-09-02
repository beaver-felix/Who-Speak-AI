"""Application-specific audio gate before Pipecat VAD/STT."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import numpy as np
from pipecat.frames.frames import Frame, UserAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.assistant.auth_controller import VoiceAuthChallengeController
from app.assistant.contracts import AuthDecision, AuthState

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
    ) -> None:
        super().__init__(name="auth-router")
        self._controller = controller
        self._participant_id = participant_id
        self._on_decision = on_decision
        self._on_error = on_error
        self._last_notified_state = self.auth_state.state
        self.challenge_frames_consumed = 0

    @property
    def auth_state(self) -> AuthDecision:
        return self._controller.current()

    def request_private_mode(self) -> AuthDecision:
        decision = self._controller.request_private_mode()
        self._last_notified_state = decision.state
        return decision

    def cancel_private_mode(self) -> AuthDecision:
        decision = self._controller.cancel()
        self._last_notified_state = decision.state
        return decision

    async def _notify_state_change(self) -> AuthDecision:
        """Publish expiration transitions observed while normal audio flows."""

        decision = self.auth_state
        if decision.state is not self._last_notified_state:
            self._last_notified_state = decision.state
            await self._on_decision(decision)
        return decision

    def should_consume(self, frame: UserAudioRawFrame) -> bool:
        return frame.user_id == self._participant_id and self.auth_state.state is AuthState.AUTH_PENDING

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserAudioRawFrame):
            # Never mix a second participant's audio into this account's auth
            # session, even if the room token was misconfigured.
            if frame.user_id != self._participant_id:
                logger.warning("pipecat_audio_dropped participant=%s reason=unexpected_participant", frame.user_id)
                return
            decision = await self._notify_state_change()
            if decision.state is AuthState.AUTH_PENDING:
                self.challenge_frames_consumed += 1
                try:
                    values = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
                    decision = await asyncio.to_thread(
                        self._controller.append_pcm, values, sample_rate=frame.sample_rate
                    )
                    if decision is not None:
                        self._last_notified_state = decision.state
                        await self._on_decision(decision)
                except Exception:
                    # A malformed/failed verification must not leak audio to
                    # STT and must not accidentally grant private access.
                    logger.exception("voice_challenge_failed")
                    failure_decision = self.cancel_private_mode()
                    await self._on_decision(failure_decision)
                    await self._on_error("Voice verification could not be completed. Please try again.")
                return
        await self.push_frame(frame, direction)
