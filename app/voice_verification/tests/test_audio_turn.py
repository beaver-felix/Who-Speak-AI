from __future__ import annotations

import numpy as np

from app.assistant.audio_turn import LocalSpeechTurnBuffer


def test_local_turn_buffer_requires_speech_then_trailing_silence() -> None:
    buffer = LocalSpeechTurnBuffer(min_speech_seconds=0.5, silence_seconds=0.3, maximum_seconds=3)

    assert buffer.append_pcm(np.zeros(1_000, dtype=np.float32), sample_rate=1_000) is None
    assert buffer.append_pcm(np.full(600, 0.1, dtype=np.float32), sample_rate=1_000) is None
    recording = buffer.append_pcm(np.zeros(300, dtype=np.float32), sample_rate=1_000)

    assert recording is not None
    assert recording.sample_rate == 16_000
    assert recording.duration_seconds == 0.9


def test_local_turn_buffer_does_not_keep_audio_after_a_completed_turn() -> None:
    buffer = LocalSpeechTurnBuffer(min_speech_seconds=0.1, silence_seconds=0.1, maximum_seconds=2)
    buffer.append_pcm(np.full(100, 0.1, dtype=np.float32), sample_rate=1_000)
    assert buffer.append_pcm(np.zeros(100, dtype=np.float32), sample_rate=1_000) is not None

    assert buffer.append_pcm(np.zeros(500, dtype=np.float32), sample_rate=1_000) is None
