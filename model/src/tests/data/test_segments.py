"""Tests for deterministic training and evaluation segmentation."""

import numpy as np
import pytest

from speaker_recognition.data.segments import (
    SegmentError,
    evenly_spaced_segments,
    random_fixed_segment,
    stable_segment_seed,
)


def test_stable_seed_is_repeatable_and_epoch_specific() -> None:
    """Identical metadata should repeat while epochs change crop streams."""
    first = stable_segment_seed(
        global_seed=42,
        epoch=3,
        utterance_id="tidyvoice:train:id000001:en:a",
    )
    repeated = stable_segment_seed(
        global_seed=42,
        epoch=3,
        utterance_id="tidyvoice:train:id000001:en:a",
    )
    next_epoch = stable_segment_seed(
        global_seed=42,
        epoch=4,
        utterance_id="tidyvoice:train:id000001:en:a",
    )

    assert first == repeated
    assert first != next_epoch


def test_random_segment_is_deterministic_for_a_seed() -> None:
    """A seeded crop should be identical across repeated calls."""
    waveform = np.arange(100, dtype=np.float32)

    first = random_fixed_segment(waveform, num_samples=20, seed=17)
    repeated = random_fixed_segment(waveform, num_samples=20, seed=17)

    np.testing.assert_array_equal(first, repeated)
    assert first.shape == (20,)


def test_short_waveform_is_repeated_without_silence() -> None:
    """Short speech should wrap to the target instead of adding zeros."""
    waveform = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    segment = random_fixed_segment(
        waveform,
        num_samples=8,
        seed=42,
    )

    np.testing.assert_array_equal(
        segment,
        np.array([1, 2, 3, 1, 2, 3, 1, 2], dtype=np.float32),
    )


def test_evenly_spaced_segments_include_both_endpoints() -> None:
    """Evaluation crops should span the complete recording timeline."""
    waveform = np.arange(12, dtype=np.float32)

    segments = evenly_spaced_segments(
        waveform,
        num_samples=4,
        segment_count=3,
    )

    np.testing.assert_array_equal(
        segments,
        np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            dtype=np.float32,
        ),
    )


def test_short_evaluation_waveform_produces_one_segment() -> None:
    """Repeated duplicate evaluation crops should be avoided."""
    segments = evenly_spaced_segments(
        np.array([1.0, 2.0], dtype=np.float32),
        num_samples=5,
        segment_count=5,
    )

    assert segments.shape == (1, 5)
    np.testing.assert_array_equal(
        segments[0],
        np.array([1, 2, 1, 2, 1], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "invalid_waveform",
    [
        np.array([], dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.array([0.0, np.inf], dtype=np.float32),
    ],
)
def test_reject_invalid_segmentation_waveforms(
    invalid_waveform: np.ndarray,
) -> None:
    """Segmentation must reject malformed canonical audio."""
    with pytest.raises(SegmentError):
        evenly_spaced_segments(
            invalid_waveform,
            num_samples=10,
            segment_count=2,
        )
