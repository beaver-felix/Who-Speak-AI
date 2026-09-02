"""Provider-neutral streaming TTS contracts and the buffered Edge fallback."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

import numpy as np

from app.assistant.providers.edge_tts_local import EdgeTTSProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTSChunk:
    """Transport-ready audio returned by a streaming TTS provider."""

    audio: bytes
    sample_rate: int
    num_channels: int
    format: str = "s16"


class StreamingTTSProvider(Protocol):
    """Minimal lifecycle contract used by the Pipecat TTS adapter."""

    name: str

    async def warmup(self) -> None:
        """Load and validate the provider before the room accepts turns."""

    async def stream(self, text: str) -> AsyncIterator[TTSChunk]:
        """Yield transport-ready audio without buffering the full utterance."""

    async def close(self) -> None:
        """Release provider resources during room/process shutdown."""


class BufferedEdgeTTSProvider:
    """Compatibility fallback for Edge-TTS's complete-MP3 API.

    Edge-TTS exposes websocket audio chunks, but this fallback intentionally
    keeps the existing complete-MP3 decode behavior. ZeroTTS is the streaming
    primary; an incremental MP3 demuxer is deliberately out of this change.
    """

    name = "edge"

    def __init__(self, provider: EdgeTTSProvider, *, output_sample_rate: int = 48_000) -> None:
        if output_sample_rate <= 0:
            raise ValueError("output_sample_rate must be positive.")
        self._provider = provider
        self._output_sample_rate = output_sample_rate

    async def warmup(self) -> None:
        """Validate imports without making a network synthesis request."""
        try:
            import av  # noqa: F401
        except ImportError as error:
            raise RuntimeError("PyAV is required for the Edge-TTS fallback.") from error

    async def stream(self, text: str) -> AsyncIterator[TTSChunk]:
        audio = await self._provider.synthesize(text)
        pcm_chunks = await asyncio.to_thread(
            decode_mp3_to_pcm_chunks,
            audio,
            self._output_sample_rate,
        )
        for pcm in pcm_chunks:
            if pcm:
                yield TTSChunk(
                    audio=pcm,
                    sample_rate=self._output_sample_rate,
                    num_channels=1,
                )

    async def close(self) -> None:
        return None


def decode_mp3_to_pcm_chunks(
    audio: bytes,
    output_sample_rate: int = 48_000,
    *,
    samples_per_chunk: int = 960,
) -> list[bytes]:
    """Decode one complete MP3 payload into signed-16 mono PCM chunks."""

    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required for the Edge-TTS fallback.") from error
    if not audio:
        return []
    if output_sample_rate <= 0 or samples_per_chunk <= 0:
        raise ValueError("Audio output settings must be positive.")

    chunks: list[bytes] = []
    with av.open(BytesIO(audio)) as container:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=output_sample_rate,
        )
        for packet in container.demux(stream):
            for decoded in packet.decode():
                for converted in resampler.resample(decoded):
                    pcm = converted.to_ndarray()
                    chunks.extend(_chunk_pcm(pcm.tobytes(), samples_per_chunk))
        for converted in resampler.resample(None):
            pcm = converted.to_ndarray()
            chunks.extend(_chunk_pcm(pcm.tobytes(), samples_per_chunk))
    return chunks


def _chunk_pcm(pcm: bytes, samples_per_chunk: int) -> list[bytes]:
    width = samples_per_chunk * 2
    return [
        pcm[start : start + width]
        for start in range(0, len(pcm), width)
        if pcm[start : start + width]
    ]


def float32_to_s16_mono(audio: object) -> bytes:
    """Convert a ZeroTTS float waveform to little-endian signed-16 PCM."""

    values = np.asarray(audio)
    if values.size == 0:
        return b""
    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError("ZeroTTS output must be mono or shaped (1, samples).")
        values = values[0]
    if values.ndim != 1:
        raise ValueError("ZeroTTS output must be a one-dimensional waveform.")

    if np.issubdtype(values.dtype, np.floating):
        values = np.clip(values.astype(np.float32, copy=False), -1.0, 1.0) * 32767.0
    elif not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Unsupported ZeroTTS audio dtype: {values.dtype}")
    values = np.clip(values, -32768, 32767).astype("<i2", copy=False)
    return np.ascontiguousarray(values).tobytes()
