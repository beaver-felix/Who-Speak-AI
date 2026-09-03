"""Silero VAD + local Smart Turn bridge for the Pipecat runtime.

Pipecat's ``VADProcessor`` emits a raw VAD stop as soon as Silero observes
silence.  This processor keeps that stop from reaching segmented STT until
the local Smart Turn model confirms that the user's thought is complete.

The processor is intentionally placed after VAD and before STT:

    Silero VAD -> Smart Turn gate -> segmented local STT

The AuthRouterProcessor remains upstream, so authentication challenge audio
never reaches this component.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)


class SmartTurnAnalyzer(Protocol):
    """Small protocol so the gate can be tested without loading ONNX."""

    def set_sample_rate(self, sample_rate: int) -> None: ...

    def update_vad_start_secs(self, vad_start_secs: float) -> None: ...

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState: ...

    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, object | None]: ...

    def clear(self) -> None: ...

    async def cleanup(self) -> None: ...


class SmartTurnGateProcessor(FrameProcessor):
    """Delay the raw Silero stop until local Smart Turn confirms completion."""

    def __init__(
        self,
        *,
        analyzer: SmartTurnAnalyzer,
        max_wait_seconds: float = 2.0,
    ) -> None:
        super().__init__(name="smart-turn-gate")
        if max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be positive.")
        self._analyzer = analyzer
        self._max_wait_seconds = max_wait_seconds
        self._vad_speaking = False
        self._pending_stop: VADUserStoppedSpeakingFrame | None = None
        self._force_stop_task: asyncio.Task | None = None

    async def setup(self, setup):
        await super().setup(setup)
        self._analyzer.set_sample_rate(setup.audio_in_sample_rate)

    async def cleanup(self):
        await self._cancel_force_stop()
        await self._analyzer.cleanup()
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._reset()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InputAudioRawFrame):
            # Feed Smart Turn once per PCM frame.  Audio is still forwarded
            # immediately so SegmentedSTTService retains the whole utterance.
            self._analyzer.append_audio(frame.audio, self._vad_speaking)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._vad_speaking = True
            await self._cancel_force_stop()
            self._analyzer.update_vad_start_secs(frame.start_secs)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_speaking = False
            self._pending_stop = frame
            state, _metrics = await self._analyzer.analyze_end_of_turn()
            if state is EndOfTurnState.COMPLETE:
                await self._release_stop()
            else:
                # A short pause is not necessarily the end of the sentence.
                # Bound the wait so a soft/incomplete prediction cannot leave
                # SegmentedSTTService waiting forever.
                await self._start_force_stop()
            return

        if isinstance(frame, EndFrame):
            await self._reset()

        await self.push_frame(frame, direction)

    async def _start_force_stop(self) -> None:
        if self._force_stop_task is not None and not self._force_stop_task.done():
            return
        self._force_stop_task = self.create_task(
            self._force_stop_after_timeout(), name="smart-turn-force-stop"
        )

    async def _force_stop_after_timeout(self) -> None:
        try:
            await asyncio.sleep(self._max_wait_seconds)
            if self._pending_stop is not None and not self._vad_speaking:
                logger.debug("smart_turn_force_stop timeout_seconds=%s", self._max_wait_seconds)
                # This coroutine owns the task being cleared.  Clear the
                # reference before releasing the stop so cleanup does not try
                # to cancel the current task.
                self._force_stop_task = None
                self._analyzer.clear()
                await self._release_stop()
        except asyncio.CancelledError:
            raise

    async def _release_stop(self) -> None:
        stop = self._pending_stop
        self._pending_stop = None
        await self._cancel_force_stop()
        if stop is not None:
            await self.push_frame(stop, FrameDirection.DOWNSTREAM)

    async def _cancel_force_stop(self) -> None:
        task = self._force_stop_task
        self._force_stop_task = None
        if task is not None and not task.done():
            await self.cancel_task(task)

    async def _reset(self) -> None:
        await self._cancel_force_stop()
        self._pending_stop = None
        self._vad_speaking = False
        self._analyzer.clear()
