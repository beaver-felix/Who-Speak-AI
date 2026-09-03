"""Streaming TTS to Pipecat PCM adapter.

The adapter owns no network/model task outside Pipecat's task manager and
publishes only signed-16 mono PCM frames to the LiveKit output transport.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.assistant.providers.streaming_tts import StreamingTTSProvider, TTSChunk

logger = logging.getLogger(__name__)


@dataclass
class VoiceTTSSpeakFrame(TTSSpeakFrame):
    """A sentence plus the UI turn it belongs to."""

    turn_id: str = ""


@dataclass
class VoiceTTSTurnEndFrame(Frame):
    """Marker emitted after all sentences in one assistant turn."""

    turn_id: str = ""


class StreamingTTSPipecatService(FrameProcessor):
    """Stream queued sentences and convert them to transport PCM.

    The primary provider may yield audio progressively. A fallback is only
    attempted when the primary fails before yielding its first chunk; once
    audio has reached LiveKit, restarting the sentence would duplicate sound.
    """

    def __init__(
        self,
        provider: StreamingTTSProvider,
        *,
        fallback_provider: StreamingTTSProvider | None = None,
        timeout_seconds: float = 10.0,
        on_audio_started: Callable[[str], Awaitable[None]],
        on_audio_completed: Callable[[str], Awaitable[None]],
        on_error: Callable[[str, str], Awaitable[None]],
    ) -> None:
        super().__init__(name="streaming-tts")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self._provider = provider
        self._fallback_provider = fallback_provider
        self._timeout_seconds = timeout_seconds
        self._on_audio_started = on_audio_started
        self._on_audio_completed = on_audio_completed
        self._on_error = on_error
        self._active_task: asyncio.Task | None = None
        self._started_turns: set[str] = set()
        self._error_turns: set[str] = set()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            if self._active_task is not None and not self._active_task.done():
                await self.cancel_task(self._active_task)
            self._active_task = None
            self._started_turns.clear()
            self._error_turns.clear()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, VoiceTTSSpeakFrame):
            if frame.text.strip():
                logger.info(
                    "tts_speak_frame_received turn_id=%s text_chars=%s",
                    frame.turn_id,
                    len(frame.text),
                )
                self._active_task = self.create_task(
                    self._synthesize(frame), name=f"synthesize-{frame.turn_id}"
                )
                try:
                    await self._active_task
                except asyncio.CancelledError:
                    raise
                finally:
                    self._active_task = None
            return
        if isinstance(frame, VoiceTTSTurnEndFrame):
            self._started_turns.discard(frame.turn_id)
            if frame.turn_id not in self._error_turns:
                await self._on_audio_completed(frame.turn_id)
            self._error_turns.discard(frame.turn_id)
            return
        await self.push_frame(frame, direction)

    async def _synthesize(self, frame: VoiceTTSSpeakFrame) -> None:
        emitted_audio = False
        emitted_this_sentence = [False]
        providers = (self._provider, self._fallback_provider)
        for index, provider in enumerate(providers):
            if provider is None:
                continue
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    emitted_audio = await self._consume_provider(
                        provider,
                        frame,
                        emitted_audio=emitted_audio,
                        emitted_state=emitted_this_sentence,
                    )
                if emitted_audio:
                    await self.push_frame(TTSStoppedFrame(context_id=frame.turn_id))
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                # _consume_provider may have already pushed a chunk before
                # its generator failed; preserve that fact across the
                # exception so fallback cannot replay a partial sentence.
                emitted_audio = emitted_audio or emitted_this_sentence[0]
                if index == 0 and not emitted_audio and self._fallback_provider is not None:
                    logger.warning(
                        "tts_primary_failed_using_fallback turn_id=%s primary=%s fallback=%s",
                        frame.turn_id,
                        self._provider.name,
                        self._fallback_provider.name,
                        exc_info=True,
                    )
                    continue
                logger.exception("tts_failed turn_id=%s provider=%s", frame.turn_id, provider.name)
                self._error_turns.add(frame.turn_id)
                await self._on_error(
                    frame.turn_id,
                    "Audio unavailable. The text response is still available.",
                )
                return

    async def _consume_provider(
        self,
        provider: StreamingTTSProvider,
        frame: VoiceTTSSpeakFrame,
        *,
        emitted_audio: bool,
        emitted_state: list[bool],
    ) -> bool:
        chunk_count = 0
        audio_bytes = 0
        started_at = asyncio.get_running_loop().time()
        logger.info(
            "tts_provider_started turn_id=%s provider=%s text_chars=%s",
            frame.turn_id,
            provider.name,
            len(frame.text),
        )
        async for chunk in provider.stream(frame.text):
            if not isinstance(chunk, TTSChunk):
                raise TypeError(f"{provider.name} returned an invalid TTS chunk.")
            if not chunk.audio:
                continue
            if chunk.format != "s16":
                raise ValueError(f"{provider.name} returned unsupported audio format {chunk.format!r}.")
            if chunk.num_channels != 1:
                raise ValueError(f"{provider.name} must return mono audio.")
            if chunk.sample_rate <= 0:
                raise ValueError(f"{provider.name} returned an invalid sample rate.")
            if not emitted_audio:
                emitted_audio = True
                emitted_state[0] = True
                if frame.turn_id not in self._started_turns:
                    self._started_turns.add(frame.turn_id)
                    await self.push_frame(TTSStartedFrame(context_id=frame.turn_id))
                    await self._on_audio_started(frame.turn_id)
                    logger.info(
                        "tts_first_pcm_enqueued turn_id=%s provider=%s sample_rate=%s audio_bytes=%s",
                        frame.turn_id,
                        provider.name,
                        chunk.sample_rate,
                        len(chunk.audio),
                    )
            chunk_count += 1
            audio_bytes += len(chunk.audio)
            await self.push_frame(
                TTSAudioRawFrame(
                    audio=chunk.audio,
                    sample_rate=chunk.sample_rate,
                    num_channels=chunk.num_channels,
                    context_id=frame.turn_id,
                )
            )
        logger.info(
            "tts_provider_finished turn_id=%s provider=%s chunks=%s audio_bytes=%s elapsed_ms=%.1f",
            frame.turn_id,
            provider.name,
            chunk_count,
            audio_bytes,
            (asyncio.get_running_loop().time() - started_at) * 1000,
        )
        if not emitted_audio:
            raise RuntimeError(f"{provider.name} returned no audio.")
        return emitted_audio

    async def cleanup(self):
        if self._active_task is not None and not self._active_task.done():
            await self.cancel_task(self._active_task)
        self._active_task = None
        self._error_turns.clear()
        for provider in (self._provider, self._fallback_provider):
            if provider is not None:
                try:
                    await provider.close()
                except Exception:
                    logger.exception("tts_provider_cleanup_failed provider=%s", provider.name)
        await super().cleanup()


# Kept as an import-compatible name for existing callers while the service is
# now provider-neutral.
EdgeTTSPipecatService = StreamingTTSPipecatService
