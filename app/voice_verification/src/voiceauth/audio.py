"""In-memory audio decoding and canonicalization for the RawNet3 contract."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from voiceauth.config import TARGET_SAMPLE_RATE
from voiceauth.errors import AudioValidationError


@dataclass(frozen=True)
class AudioRecording:
    waveform: np.ndarray
    sample_rate: int
    original_sample_rate: int
    original_channels: int

    @property
    def duration_seconds(self) -> float:
        return self.waveform.size / self.sample_rate


def decode_recording(audio_bytes: bytes, *, target_sample_rate: int = TARGET_SAMPLE_RATE) -> AudioRecording:
    """Decode a browser WAV in memory; raw audio is never written to disk."""
    if not audio_bytes:
        raise AudioValidationError("Record a voice sample before continuing.")
    try:
        samples, original_rate = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as error:
        raise AudioValidationError("The recorded audio could not be decoded.") from error
    canonical = canonicalize_audio(samples, sample_rate=int(original_rate), target_sample_rate=target_sample_rate)
    return AudioRecording(
        waveform=canonical,
        sample_rate=target_sample_rate,
        original_sample_rate=int(original_rate),
        original_channels=int(samples.shape[1]),
    )


def canonicalize_audio(
    waveform: np.ndarray, *, sample_rate: int, target_sample_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """Apply the ViMD RawNet3 preprocessing without amplitude normalization."""
    if sample_rate <= 0 or target_sample_rate <= 0:
        raise AudioValidationError("Audio sample rates must be positive.")
    values = np.asarray(waveform)
    if values.ndim not in {1, 2} or values.shape[0] == 0:
        raise AudioValidationError("The recording contains no audio samples.")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise AudioValidationError("The recording contains invalid audio samples.")
    mono = values.astype(np.float64, copy=False).mean(axis=1) if values.ndim == 2 else values.astype(np.float64, copy=False)
    if sample_rate != target_sample_rate:
        divisor = gcd(sample_rate, target_sample_rate)
        resampled = resample_poly(mono, target_sample_rate // divisor, sample_rate // divisor)
        expected = (mono.size * target_sample_rate + sample_rate // 2) // sample_rate
        if resampled.size > expected:
            resampled = resampled[:expected]
        elif resampled.size < expected:
            resampled = np.pad(resampled, (0, expected - resampled.size))
        mono = resampled
    result = np.ascontiguousarray(mono, dtype=np.float32)
    if result.size == 0 or not np.isfinite(result).all():
        raise AudioValidationError("The recording is empty after preprocessing.")
    return result


def validate_speech_sample(recording: AudioRecording, *, minimum_seconds: float = 4.0) -> None:
    """Reject clips that would otherwise be repeated or contain near silence."""
    if recording.duration_seconds < minimum_seconds:
        raise AudioValidationError(
            f"Record at least {minimum_seconds:.1f} seconds of clear speech before continuing."
        )
    rms = float(np.sqrt(np.mean(np.square(recording.waveform, dtype=np.float64))))
    if not np.isfinite(rms) or rms < 0.002:
        raise AudioValidationError("No clear speech was detected. Check the microphone and record again.")
