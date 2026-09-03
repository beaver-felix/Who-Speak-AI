from __future__ import annotations

import asyncio

import numpy as np

from app.assistant.pipecat_runtime.edge_tts_service import (
    StreamingTTSPipecatService,
    VoiceTTSSpeakFrame,
)
from app.assistant.providers.streaming_tts import TTSChunk, float32_to_s16_mono
from app.assistant.providers.zerotts_local import ZeroTTSLocalProvider


def test_float32_waveform_is_converted_to_mono_signed_16_bit_pcm() -> None:
    pcm = float32_to_s16_mono(np.array([[-1.0, 0.0, 1.0]], dtype=np.float32))

    assert pcm == np.array([-32767, 0, 32767], dtype="<i2").tobytes()


def test_zerotts_provider_yields_stream_chunks_without_waiting_for_full_audio() -> None:
    class Model:
        def list_voices(self):
            return ["maichi"]

        def synthesize_stream(self, text, *, voice):
            assert text == "xin chao"
            assert voice == "maichi"
            yield np.array([[0.1, 0.2]], dtype=np.float32)
            yield np.array([[0.3, 0.4]], dtype=np.float32)

    provider = ZeroTTSLocalProvider(queue_max_chunks=2)
    provider._model = Model()

    async def collect() -> list[TTSChunk]:
        return [chunk async for chunk in provider.stream("xin chao")]

    chunks = asyncio.run(collect())

    assert len(chunks) == 2
    assert all(chunk.sample_rate == 48_000 for chunk in chunks)
    assert all(chunk.num_channels == 1 for chunk in chunks)
    assert all(chunk.format == "s16" for chunk in chunks)
    assert chunks[0].audio == np.array([3276, 6553], dtype="<i2").tobytes()


def test_zerotts_provider_flushes_short_audio_when_worker_finishes_before_sentinel() -> None:
    class Model:
        def synthesize_stream(self, text, *, voice):
            yield np.array([[0.25]], dtype=np.float32)

    provider = ZeroTTSLocalProvider(queue_max_chunks=2, startup_buffer_ms=250)
    provider._model = Model()

    async def collect() -> list[TTSChunk]:
        return [chunk async for chunk in provider.stream("ok")]

    chunks = asyncio.run(collect())

    assert len(chunks) == 1
    assert chunks[0].audio == np.array([8191], dtype="<i2").tobytes()


def test_tts_service_uses_fallback_only_before_primary_first_audio() -> None:
    class Provider:
        def __init__(self, name, chunks=None, error=None):
            self.name = name
            self.chunks = chunks or []
            self.error = error
            self.calls = 0

        async def warmup(self):
            return None

        async def stream(self, _text):
            self.calls += 1
            for chunk in self.chunks:
                yield chunk
            if self.error:
                raise self.error

        async def close(self):
            return None

    primary = Provider("zerotts", error=RuntimeError("primary unavailable"))
    fallback = Provider(
        "edge",
        chunks=[TTSChunk(b"fallback", 48_000, 1)],
    )
    pushed = []
    started = []
    errors = []

    async def run() -> None:
        service = StreamingTTSPipecatService(
            primary,
            fallback_provider=fallback,
            on_audio_started=lambda turn_id: _append_async(started, turn_id),
            on_audio_completed=lambda _turn_id: _noop_async(),
            on_error=lambda turn_id, message: _append_async(errors, (turn_id, message)),
        )

        async def push(frame, _direction=None):
            pushed.append(frame)

        service.push_frame = push
        await service._synthesize(VoiceTTSSpeakFrame("hello", turn_id="turn-1"))

    asyncio.run(run())

    assert primary.calls == 1
    assert fallback.calls == 1
    assert errors == []
    assert started == ["turn-1"]
    assert any(getattr(frame, "audio", None) == b"fallback" for frame in pushed)


def test_tts_service_does_not_replay_with_fallback_after_partial_primary_audio() -> None:
    class Provider:
        def __init__(self, name, chunks=None, error=None):
            self.name = name
            self.chunks = chunks or []
            self.error = error
            self.calls = 0

        async def warmup(self):
            return None

        async def stream(self, _text):
            self.calls += 1
            for chunk in self.chunks:
                yield chunk
            if self.error:
                raise self.error

        async def close(self):
            return None

    primary = Provider(
        "zerotts",
        chunks=[TTSChunk(b"partial", 48_000, 1)],
        error=RuntimeError("stream stopped"),
    )
    fallback = Provider("edge", chunks=[TTSChunk(b"duplicate", 48_000, 1)])
    pushed = []
    errors = []

    async def run() -> None:
        service = StreamingTTSPipecatService(
            primary,
            fallback_provider=fallback,
            on_audio_started=lambda _turn_id: _noop_async(),
            on_audio_completed=lambda _turn_id: _noop_async(),
            on_error=lambda turn_id, message: _append_async(errors, (turn_id, message)),
        )

        async def push(frame, _direction=None):
            pushed.append(frame)

        service.push_frame = push
        await service._synthesize(VoiceTTSSpeakFrame("hello", turn_id="turn-2"))

    asyncio.run(run())

    assert primary.calls == 1
    assert fallback.calls == 0
    assert any(getattr(frame, "audio", None) == b"partial" for frame in pushed)
    assert errors and errors[0][0] == "turn-2"


async def _append_async(target: list, value) -> None:
    target.append(value)


async def _noop_async() -> None:
    return None
