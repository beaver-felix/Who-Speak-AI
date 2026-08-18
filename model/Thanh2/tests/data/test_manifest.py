"""Tests for the canonical speaker-recognition manifest."""

from __future__ import annotations

import pytest

from speaker_recognition.data.manifest import (
    AudioStorage,
    ManifestRecord,
    ManifestValidationError,
    Split,
    validate_manifest,
)


def make_record(
    utterance_id: str,
    speaker_id: str,
    recording_id: str,
    split: Split,
    *,
    audio_path: str | None = None,
) -> ManifestRecord:
    """Create a minimal valid standalone-audio record for tests."""
    return ManifestRecord(
        dataset="example",
        source_split=split.value,
        split=split,
        utterance_id=utterance_id,
        speaker_id=speaker_id,
        recording_id=recording_id,
        audio_storage=AudioStorage.FILE,
        audio_path=audio_path or f"audio\\{utterance_id}.wav",
    )


def test_file_record_normalizes_path_separators() -> None:
    """Manifest paths use portable POSIX separators."""
    record = make_record("utt-1", "speaker-1", "rec-1", Split.TRAIN)

    assert record.audio_path == "audio/utt-1.wav"
    assert record.audio_row_index is None


def test_parquet_record_requires_row_index() -> None:
    """Embedded Parquet audio cannot be addressed without a row index."""
    with pytest.raises(
        ManifestValidationError,
        match="non-negative audio_row_index",
    ):
        ManifestRecord(
            dataset="vimd",
            source_split="train",
            split=Split.TRAIN,
            utterance_id="utt-1",
            speaker_id="speaker-1",
            recording_id="rec-1",
            audio_storage=AudioStorage.PARQUET,
            audio_path="data/train-00000.parquet",
        )


def test_manifest_accepts_disjoint_groups() -> None:
    """Independent speakers and recordings pass all integrity checks."""
    records = [
        make_record("utt-1", "speaker-1", "rec-1", Split.TRAIN),
        make_record("utt-2", "speaker-2", "rec-2", Split.VALIDATION),
        make_record("utt-3", "speaker-3", "rec-3", Split.TEST),
    ]

    assert validate_manifest(records) == tuple(records)


def test_manifest_rejects_duplicate_utterance_id() -> None:
    """A logical utterance identifier must be globally unique."""
    records = [
        make_record("utt-1", "speaker-1", "rec-1", Split.TRAIN),
        make_record(
            "utt-1",
            "speaker-2",
            "rec-2",
            Split.TRAIN,
            audio_path="audio/second.wav",
        ),
    ]

    with pytest.raises(
        ManifestValidationError,
        match="Duplicate utterance IDs",
    ):
        validate_manifest(records)


def test_manifest_rejects_duplicate_physical_audio() -> None:
    """Two logical rows must not silently reference the same waveform."""
    records = [
        make_record(
            "utt-1",
            "speaker-1",
            "rec-1",
            Split.TRAIN,
            audio_path="audio/shared.wav",
        ),
        make_record(
            "utt-2",
            "speaker-1",
            "rec-1",
            Split.TRAIN,
            audio_path="audio/shared.wav",
        ),
    ]

    with pytest.raises(
        ManifestValidationError,
        match="Duplicate physical audio references",
    ):
        validate_manifest(records)


def test_manifest_rejects_speaker_leakage() -> None:
    """The same speaker cannot cross canonical experiment splits."""
    records = [
        make_record("utt-1", "speaker-1", "rec-1", Split.TRAIN),
        make_record("utt-2", "speaker-1", "rec-2", Split.TEST),
    ]

    with pytest.raises(
        ManifestValidationError,
        match="Cross-split speaker leakage",
    ):
        validate_manifest(records)


def test_manifest_rejects_recording_leakage() -> None:
    """Recording reuse across speakers or splits is also leakage."""
    records = [
        make_record("utt-1", "speaker-1", "rec-1", Split.TRAIN),
        make_record("utt-2", "speaker-2", "rec-1", Split.TEST),
    ]

    with pytest.raises(
        ManifestValidationError,
        match="Cross-split recording leakage",
    ):
        validate_manifest(records)