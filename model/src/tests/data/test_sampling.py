"""Tests for deterministic speaker-capped epoch membership."""

from collections import Counter
from pathlib import Path

import pytest

from speaker_recognition.data.dataset import TrainingSpeakerDataset
from speaker_recognition.data.manifest import (
    AudioStorage,
    ManifestRecord,
    Split,
)
from speaker_recognition.data.sampling import (
    EpochSamplingError,
    select_speaker_capped_epoch,
    utterance_id_sha256,
)


def _records(
    *,
    speaker_count: int = 3,
    utterances_per_speaker: int = 8,
) -> tuple[ManifestRecord, ...]:
    """Create deterministic metadata-only Train records."""
    records = []
    for speaker_index in range(speaker_count):
        speaker_id = f"speaker-{speaker_index}"
        for utterance_index in range(utterances_per_speaker):
            utterance_id = f"{speaker_id}:utt-{utterance_index}"
            records.append(
                ManifestRecord(
                    dataset="fixture",
                    source_split="train",
                    split=Split.TRAIN,
                    utterance_id=utterance_id,
                    speaker_id=speaker_id,
                    recording_id=utterance_id,
                    audio_storage=AudioStorage.FILE,
                    audio_path=f"audio/{utterance_id}.wav",
                )
            )
    return tuple(records)


def test_epoch_selection_caps_every_speaker() -> None:
    """High-resource identities must contribute no more than the declared cap."""
    selected = select_speaker_capped_epoch(
        _records(),
        max_utterances_per_speaker=2,
        seed=42,
        epoch_index=0,
    )

    counts = Counter(record.speaker_id for record in selected)
    assert counts == {
        "speaker-0": 2,
        "speaker-1": 2,
        "speaker-2": 2,
    }


def test_epoch_selection_is_repeatable_and_input_order_independent() -> None:
    """Canonical membership must not depend on caller order or global RNG."""
    records = _records()

    first = select_speaker_capped_epoch(
        records,
        max_utterances_per_speaker=3,
        seed=42,
        epoch_index=4,
    )
    repeated = select_speaker_capped_epoch(
        tuple(reversed(records)),
        max_utterances_per_speaker=3,
        seed=42,
        epoch_index=4,
    )

    assert repeated == first
    assert utterance_id_sha256(repeated) == utterance_id_sha256(first)


def test_epoch_selection_rotates_high_resource_exposure() -> None:
    """A later epoch should expose a different deterministic membership."""
    records = _records(utterances_per_speaker=12)

    first = select_speaker_capped_epoch(
        records,
        max_utterances_per_speaker=4,
        seed=42,
        epoch_index=0,
    )
    second = select_speaker_capped_epoch(
        records,
        max_utterances_per_speaker=4,
        seed=42,
        epoch_index=1,
    )

    assert utterance_id_sha256(second) != utterance_id_sha256(first)


def test_sparse_speakers_keep_every_available_utterance() -> None:
    """The cap must not duplicate data for identities below the maximum."""
    records = _records(speaker_count=2, utterances_per_speaker=2)

    selected = select_speaker_capped_epoch(
        records,
        max_utterances_per_speaker=4,
        seed=42,
        epoch_index=0,
    )

    assert selected == records


def test_pilot_speaker_cap_is_repeatable_and_rotates() -> None:
    """A bounded pilot must sample identities reproducibly across restarts."""
    records = _records(speaker_count=20, utterances_per_speaker=2)

    first = select_speaker_capped_epoch(
        records,
        max_utterances_per_speaker=1,
        max_speakers=5,
        seed=42,
        epoch_index=0,
    )
    repeated = select_speaker_capped_epoch(
        tuple(reversed(records)),
        max_utterances_per_speaker=1,
        max_speakers=5,
        seed=42,
        epoch_index=0,
    )
    second_epoch = select_speaker_capped_epoch(
        records,
        max_utterances_per_speaker=1,
        max_speakers=5,
        seed=42,
        epoch_index=1,
    )

    assert repeated == first
    assert len(first) == 5
    assert len({record.speaker_id for record in first}) == 5
    assert {
        record.speaker_id for record in second_epoch
    } != {record.speaker_id for record in first}


def test_training_dataset_updates_capped_membership_at_set_epoch(
    tmp_path: Path,
) -> None:
    """The DataLoader-facing length and records must follow the epoch cap."""
    records = _records(speaker_count=2, utterances_per_speaker=8)
    dataset = TrainingSpeakerDataset(
        records,
        dataset_roots={"fixture": tmp_path},
        segment_samples=48_000,
        seed=42,
        max_utterances_per_speaker=4,
        max_speakers_per_epoch=1,
    )
    first_hash = utterance_id_sha256(dataset.epoch_records)

    dataset.set_epoch(1)

    assert len(dataset.records) == 16
    assert len(dataset) == 4
    assert len(dataset.epoch_records) == 4
    assert utterance_id_sha256(dataset.epoch_records) != first_hash


def test_epoch_selection_rejects_non_train_records() -> None:
    """Evaluation partitions must never enter epoch membership selection."""
    record = _records()[0]
    invalid = ManifestRecord(
        dataset=record.dataset,
        source_split="valid",
        split=Split.VALIDATION,
        utterance_id=record.utterance_id,
        speaker_id=record.speaker_id,
        recording_id=record.recording_id,
        audio_storage=record.audio_storage,
        audio_path=record.audio_path,
    )

    with pytest.raises(EpochSamplingError, match="Train records only"):
        select_speaker_capped_epoch(
            (invalid,),
            max_utterances_per_speaker=4,
            seed=42,
            epoch_index=0,
        )
