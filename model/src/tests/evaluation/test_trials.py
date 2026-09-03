"""Tests for deterministic speaker-balanced verification trials."""

from speaker_recognition.data.manifest import (
    AudioStorage,
    ManifestRecord,
    Split,
)
from speaker_recognition.evaluation.trials import (
    TrialConstructionError,
    build_verification_trials,
    trial_list_sha256,
)


def test_trials_are_deterministic_and_input_order_independent() -> None:
    """Seeded protocols should ignore manifest insertion order."""
    records = _records(
        {
            "speaker-a": 3,
            "speaker-b": 3,
            "speaker-c": 2,
        }
    )

    first = build_verification_trials(
        records,
        split=Split.TEST,
        seed=42,
        max_genuine_per_speaker=2,
        impostor_trial_count=8,
    )
    reversed_result = build_verification_trials(
        list(reversed(records)),
        split=Split.TEST,
        seed=42,
        max_genuine_per_speaker=2,
        impostor_trial_count=8,
    )

    assert reversed_result == first
    assert trial_list_sha256(reversed_result) == trial_list_sha256(first)


def test_genuine_pairs_are_capped_per_speaker() -> None:
    """High-resource speakers should not dominate the genuine trials."""
    records = _records({"speaker-a": 5, "speaker-b": 2})

    trials = build_verification_trials(
        records,
        split=Split.TEST,
        seed=42,
        max_genuine_per_speaker=3,
        impostor_trial_count=5,
    )

    genuine_trials = [trial for trial in trials if trial.label == 1]
    per_speaker: dict[str, int] = {}
    for trial in genuine_trials:
        assert trial.left_speaker_id == trial.right_speaker_id
        speaker_id = trial.left_speaker_id
        per_speaker[speaker_id] = per_speaker.get(speaker_id, 0) + 1

    assert per_speaker == {"speaker-a": 3, "speaker-b": 1}


def test_impostor_trials_are_unique_and_cross_speaker() -> None:
    """Every impostor pair must contain two distinct identities."""
    records = _records(
        {
            "speaker-a": 3,
            "speaker-b": 3,
            "speaker-c": 3,
        }
    )

    trials = build_verification_trials(
        records,
        split=Split.TEST,
        seed=7,
        max_genuine_per_speaker=1,
        impostor_trial_count=12,
    )
    impostor_trials = [trial for trial in trials if trial.label == 0]

    assert len(impostor_trials) == 12
    assert len({trial.trial_id for trial in impostor_trials}) == 12
    assert all(
        trial.left_speaker_id != trial.right_speaker_id
        for trial in impostor_trials
    )


def test_trial_builder_filters_to_requested_split() -> None:
    """Records from another partition must not enter a trial protocol."""
    test_records = _records({"speaker-a": 2, "speaker-b": 2})
    validation_record = _record(
        speaker_id="validation-speaker",
        utterance_index=0,
        split=Split.VALIDATION,
    )

    trials = build_verification_trials(
        test_records + [validation_record],
        split=Split.TEST,
        seed=42,
        max_genuine_per_speaker=2,
        impostor_trial_count=3,
    )

    assert all(
        "validation-speaker" not in trial.trial_id
        for trial in trials
    )


def test_reject_impossible_impostor_count() -> None:
    """A request beyond unique cross-speaker capacity must fail."""
    records = _records({"speaker-a": 2, "speaker-b": 1})

    try:
        build_verification_trials(
            records,
            split=Split.TEST,
            seed=42,
            impostor_trial_count=3,
        )
    except TrialConstructionError as error:
        assert "only 2 unique pairs" in str(error)
    else:
        raise AssertionError("Expected TrialConstructionError.")


def _records(speaker_sizes: dict[str, int]) -> list[ManifestRecord]:
    """Create deterministic file-backed records for trial tests."""
    return [
        _record(
            speaker_id=speaker_id,
            utterance_index=utterance_index,
            split=Split.TEST,
        )
        for speaker_id, utterance_count in speaker_sizes.items()
        for utterance_index in range(utterance_count)
    ]


def _record(
    *,
    speaker_id: str,
    utterance_index: int,
    split: Split,
) -> ManifestRecord:
    """Create one canonical manifest fixture."""
    utterance_id = f"{speaker_id}-utterance-{utterance_index}"
    return ManifestRecord(
        dataset="fixture",
        source_split=split.value,
        split=split,
        utterance_id=utterance_id,
        speaker_id=speaker_id,
        recording_id=utterance_id,
        audio_storage=AudioStorage.FILE,
        audio_path=f"{utterance_id}.wav",
    )
