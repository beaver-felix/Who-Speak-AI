"""Deterministic waveform segmentation for training and evaluation."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray


class SegmentError(ValueError):
    """Raised when a waveform cannot produce a valid fixed segment."""


def stable_segment_seed(
    *,
    global_seed: int,
    epoch: int,
    utterance_id: str,
) -> int:
    """Derive a process-independent crop seed for one utterance and epoch.

    Python's built-in string hash changes between interpreter processes. A
    SHA-256-derived seed makes crops reproducible across local and Kaggle runs
    while still changing between epochs.
    """
    if global_seed < 0:
        raise SegmentError("global_seed must be non-negative.")
    if epoch < 0:
        raise SegmentError("epoch must be non-negative.")
    if not utterance_id or not utterance_id.strip():
        raise SegmentError("utterance_id must be a non-empty string.")

    payload = f"{global_seed}\0{epoch}\0{utterance_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def random_fixed_segment(
    waveform: NDArray[np.generic],
    *,
    num_samples: int,
    seed: int,
) -> NDArray[np.float32]:
    """Select one deterministic pseudo-random fixed-length segment.

    Signals shorter than the target are repeated instead of silence-padded.
    This avoids teaching the speaker encoder that dataset-specific recording
    duration is an identity cue.
    """
    samples = _validated_waveform(waveform)
    _validate_num_samples(num_samples)
    if seed < 0:
        raise SegmentError("seed must be non-negative.")

    if samples.size <= num_samples:
        return _repeat_to_length(samples, num_samples)

    maximum_start = samples.size - num_samples
    generator = np.random.default_rng(seed)
    start = int(generator.integers(0, maximum_start + 1))
    return np.ascontiguousarray(
        samples[start : start + num_samples],
        dtype=np.float32,
    )


def evenly_spaced_segments(
    waveform: NDArray[np.generic],
    *,
    num_samples: int,
    segment_count: int,
) -> NDArray[np.float32]:
    """Return deterministic evaluation segments spanning an utterance.

    A short utterance produces one repeated segment because averaging multiple
    identical copies changes neither the embedding nor the evidence. Longer
    utterances include both endpoints and evenly spaced intermediate crops.
    """
    samples = _validated_waveform(waveform)
    _validate_num_samples(num_samples)
    if segment_count <= 0:
        raise SegmentError("segment_count must be positive.")

    if samples.size <= num_samples:
        return _repeat_to_length(samples, num_samples)[None, :]

    maximum_start = samples.size - num_samples
    starts = np.rint(
        np.linspace(0, maximum_start, num=segment_count)
    ).astype(np.int64)
    unique_starts = np.unique(starts)
    segments = np.stack(
        [
            samples[start : start + num_samples]
            for start in unique_starts
        ],
        axis=0,
    )
    return np.ascontiguousarray(segments, dtype=np.float32)


def _validated_waveform(
    waveform: NDArray[np.generic],
) -> NDArray[np.float32]:
    """Return one valid contiguous mono waveform."""
    samples = np.asarray(waveform)
    if samples.ndim != 1:
        raise SegmentError("waveform must be one-dimensional mono audio.")
    if samples.size == 0:
        raise SegmentError("waveform must contain at least one sample.")
    if not np.issubdtype(samples.dtype, np.number):
        raise SegmentError("waveform samples must be numeric.")
    if not np.isfinite(samples).all():
        raise SegmentError("waveform samples must all be finite.")
    return np.ascontiguousarray(samples, dtype=np.float32)


def _repeat_to_length(
    waveform: NDArray[np.float32],
    num_samples: int,
) -> NDArray[np.float32]:
    """Repeat a non-empty waveform until it reaches the target length."""
    repetitions = (num_samples + waveform.size - 1) // waveform.size
    repeated = np.tile(waveform, repetitions)[:num_samples]
    return np.ascontiguousarray(repeated, dtype=np.float32)


def _validate_num_samples(num_samples: int) -> None:
    """Require a positive integer segment length."""
    if (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, int)
        or num_samples <= 0
    ):
        raise SegmentError("num_samples must be a positive integer.")
