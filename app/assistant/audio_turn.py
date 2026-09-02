"""In-memory local VAD/turn buffer used after the independent Auth Gate."""

from __future__ import annotations

import numpy as np

from voiceauth.audio import AudioRecording, canonicalize_audio


class LocalSpeechTurnBuffer:
    """Small RMS VAD with bounded, non-persistent PCM storage.

    This is deliberately separate from ``AuthChallengeBuffer``: guest/general
    conversation may use ASR, whereas speaker verification captures only after
    the user explicitly requests private mode.
    """

    def __init__(self, *, threshold: float = 0.012, min_speech_seconds: float = 0.8, silence_seconds: float = 0.8, maximum_seconds: float = 15.0) -> None:
        self._threshold = threshold
        self._min_speech_seconds = min_speech_seconds
        self._silence_seconds = silence_seconds
        self._maximum_seconds = maximum_seconds
        self.reset()

    def reset(self) -> None:
        self._frames: list[np.ndarray] = []
        self._sample_rate: int | None = None
        self._speech_seconds = 0.0
        self._trailing_silence = 0.0

    @property
    def has_audio(self) -> bool:
        """Whether this in-memory buffer is collecting a potential turn."""
        return bool(self._frames)

    def append_pcm(self, samples: np.ndarray, *, sample_rate: int) -> AudioRecording | None:
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 2:
            values = values.mean(axis=1)
        if values.ndim != 1 or not values.size or sample_rate <= 0:
            return None
        if self._sample_rate is not None and self._sample_rate != sample_rate:
            self.reset()
        self._sample_rate = sample_rate
        seconds = values.size / sample_rate
        rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
        is_speech = rms >= self._threshold
        if not self._frames and not is_speech:
            return None
        self._frames.append(np.ascontiguousarray(values))
        if is_speech:
            self._speech_seconds += seconds
            self._trailing_silence = 0.0
        else:
            self._trailing_silence += seconds
        total = sum(frame.size for frame in self._frames) / sample_rate
        complete = self._speech_seconds >= self._min_speech_seconds and self._trailing_silence >= self._silence_seconds
        if not complete and total < self._maximum_seconds:
            return None
        waveform = canonicalize_audio(np.concatenate(self._frames), sample_rate=sample_rate)
        self.reset()
        return AudioRecording(waveform, 16000, sample_rate, 1)
