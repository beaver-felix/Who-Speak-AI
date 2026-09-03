"""Publish Edge-TTS output back into a local LiveKit room."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from io import BytesIO


async def publish_mp3(
    source,
    audio_bytes: bytes,
    *,
    on_first_frame: Callable[[], Awaitable[None] | None] | None = None,
) -> bool:
    """Decode MP3 in memory and publish mono 48 kHz PCM frames.

    Failure here never changes the trusted ``AuthDecision``; the text response
    remains available over the agent data topic.
    """
    import av
    from livekit import rtc

    container = av.open(BytesIO(audio_bytes), mode="r")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
    sent_first_frame = False

    async def capture(pcm) -> None:
        nonlocal sent_first_frame
        mono = pcm.reshape(-1)
        frame_size = 960  # 20 ms at 48 kHz, matching WebRTC's common packet cadence.
        for start in range(0, mono.size, frame_size):
            chunk = mono[start : start + frame_size]
            if not chunk.size:
                continue
            if not sent_first_frame:
                sent_first_frame = True
                if on_first_frame is not None:
                    callback_result = on_first_frame()
                    if inspect.isawaitable(callback_result):
                        await callback_result
            await source.capture_frame(
                rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=48000,
                    num_channels=1,
                    samples_per_channel=chunk.size,
                )
            )

    try:
        stream = container.streams.audio[0]
        for decoded in container.decode(stream):
            for frame in resampler.resample(decoded):
                pcm = frame.to_ndarray().astype("int16", copy=False)
                await capture(pcm)
        for frame in resampler.resample(None):
            pcm = frame.to_ndarray().astype("int16", copy=False)
            await capture(pcm)
    finally:
        container.close()
    return sent_first_frame
