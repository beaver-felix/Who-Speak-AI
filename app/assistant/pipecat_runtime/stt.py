"""Pipecat STT adapter backed by the project's local faster-whisper provider."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

import numpy as np
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.settings import STTSettings
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from app.assistant.providers.whisper_local import LocalWhisperProvider
from voiceauth.audio import AudioRecording

logger = logging.getLogger(__name__)


class LocalWhisperSTTService(SegmentedSTTService):
    """Run one local Whisper request per Pipecat VAD speech segment."""

    def __init__(
        self,
        provider: LocalWhisperProvider,
        *,
        sample_rate: int = 16_000,
        language: str = "vi",
    ) -> None:
        # Pipecat 1.8.x validates service settings at StartFrame.  A local
        # adapter still needs to provide concrete values even though the
        # actual model is owned by faster-whisper rather than Pipecat.
        super().__init__(
            sample_rate=sample_rate,
            audio_passthrough=False,
            settings=STTSettings(
                model="local-faster-whisper",
                language=language,
            ),
        )
        self._provider = provider
        self._language = language

    @property
    def supports_ttfs(self) -> bool:
        """Tell Pipecat that this adapter has no extra server-side TTFS wait.

        Turn completion is owned by the upstream Silero VAD + Smart Turn
        processors. Advertising the default STT P99 here would make Pipecat
        assume an additional one-second wait and would also produce a noisy
        startup warning.
        """

        return False

    @property
    def wants_wav_segments(self) -> bool:
        # faster-whisper consumes the canonical float waveform, not a WAV
        # header. SegmentedSTTService therefore gives us raw signed-16 PCM.
        return False

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        if not audio:
            return
        try:
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = self.sample_rate or self._init_sample_rate or 16_000
            recording = AudioRecording(
                waveform=np.ascontiguousarray(samples),
                sample_rate=sample_rate,
                original_sample_rate=sample_rate,
                original_channels=1,
            )
            text = await asyncio.to_thread(
                self._provider.transcribe, recording, language=self._language
            )
        except Exception as error:
            logger.exception("local_whisper_failed")
            yield ErrorFrame(
                "Local speech recognition is unavailable for this turn.",
                exception=error,
                processor=self,
            )
            return
        if text.strip():
            yield TranscriptionFrame(
                text=text.strip(),
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=Language.VI,
                finalized=True,
            )
