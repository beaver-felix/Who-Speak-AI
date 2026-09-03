"""Bounded local audio collection for an explicit SV challenge."""

from __future__ import annotations

import numpy as np

from voiceauth.audio import AudioRecording, canonicalize_audio
from voiceauth.errors import AudioValidationError


class AuthChallengeBuffer:
    """Collect 4--6 seconds of PCM only while the auth state is pending."""

    def __init__(self, *, minimum_seconds: float = 4.0, maximum_seconds: float = 6.0) -> None:
        if not 0 < minimum_seconds <= maximum_seconds:
            raise ValueError("Challenge duration must satisfy 0 < minimum <= maximum.")
        self.minimum_seconds = minimum_seconds
        self.maximum_seconds = maximum_seconds
        self._waveform: list[np.ndarray] = []
        self._sample_rate: int | None = None

    def reset(self) -> None:
        self._waveform.clear()
        self._sample_rate = None

    def append_pcm(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int,
        max_duration_seconds: float | None = None,
    ) -> bool:
        """Append a frame and return true once enough audio has been collected."""

        if sample_rate <= 0:
            raise AudioValidationError("Audio frame sample rate must be positive.")
        if self._sample_rate is None:
            self._sample_rate = sample_rate
        elif self._sample_rate != sample_rate:
            raise AudioValidationError("Audio frame sample rate changed during the verification challenge.")
        mono = np.asarray(samples)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)
        if mono.ndim != 1 or not np.isfinite(mono).all():
            raise AudioValidationError("The verification audio frame is invalid.")
        current = sum(frame.size for frame in self._waveform)
        maximum_seconds = self.maximum_seconds if max_duration_seconds is None else max_duration_seconds
        if maximum_seconds <= 0:
            raise AudioValidationError("The maximum challenge duration must be positive.")
        capacity = round(maximum_seconds * sample_rate) - current
        if capacity > 0:
            self._waveform.append(np.ascontiguousarray(mono[:capacity], dtype=np.float32))
        return self.duration_seconds >= self.minimum_seconds

    @property
    def duration_seconds(self) -> float:
        return 0.0 if self._sample_rate is None else sum(frame.size for frame in self._waveform) / self._sample_rate

    def finish(self) -> AudioRecording:
        if self._sample_rate is None or self.duration_seconds < self.minimum_seconds:
            raise AudioValidationError(
                f"Record at least {self.minimum_seconds:.1f} seconds for the private-mode voice challenge."
            )
        raw = np.concatenate(self._waveform)
        canonical = canonicalize_audio(raw, sample_rate=self._sample_rate)
        return AudioRecording(
            waveform=canonical,
            sample_rate=16_000,
            original_sample_rate=self._sample_rate,
            original_channels=1,
        )
