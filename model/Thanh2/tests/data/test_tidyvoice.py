"""Tests for converting TidyVoice paths into canonical manifest records."""

from pathlib import Path

import pytest

from speaker_recognition.data.manifest import AudioStorage, Split
from speaker_recognition.data.tidyvoice import (
    TidyVoicePathError,
    collect_tidyvoice_speaker_language_counts,
    iter_tidyvoice_audio_paths,
    parse_tidyvoice_audio_path,
)


def test_parse_train_path_into_canonical_record(tmp_path: Path) -> None:
    """A valid training path should preserve all identity metadata."""
    dataset_root = tmp_path / "TidyVoiceX_ASV"
    audio_path = (
        dataset_root
        / "TidyVoiceX_Train"
        / "TidyVoiceX_Train"
        / "id012711"
        / "fa"
        / "fa_29855668.wav"
    )

    record = parse_tidyvoice_audio_path(
        audio_path,
        dataset_root=dataset_root,
        split=Split.TRAIN,
    )

    assert record.dataset == "tidyvoice"
    assert record.source_split == "train"
    assert record.split is Split.TRAIN
    assert record.speaker_id == "tidyvoice:id012711"
    assert record.language == "fa"
    assert record.audio_storage is AudioStorage.FILE
    assert record.audio_row_index is None
    assert record.audio_path == (
        "TidyVoiceX_Train/TidyVoiceX_Train/"
        "id012711/fa/fa_29855668.wav"
    )
    assert record.utterance_id == (
        "tidyvoice:train:id012711:fa:fa_29855668"
    )
    assert record.recording_id == record.utterance_id


def test_parse_dev_path_into_validation_record(tmp_path: Path) -> None:
    """A source-dev utterance may be assigned to canonical validation."""
    dataset_root = tmp_path / "TidyVoiceX_ASV"
    audio_path = (
        dataset_root
        / "TidyVoiceX_Dev"
        / "TidyVoiceX_Dev"
        / "id014060"
        / "lg"
        / "lg_39353814.wav"
    )

    record = parse_tidyvoice_audio_path(
        audio_path,
        dataset_root=dataset_root,
        split=Split.VALIDATION,
    )

    assert record.source_split == "dev"
    assert record.split is Split.VALIDATION
    assert record.speaker_id == "tidyvoice:id014060"
    assert record.language == "lg"


def test_reject_path_outside_dataset_root(tmp_path: Path) -> None:
    """A manifest must not silently reference another dataset directory."""
    dataset_root = tmp_path / "TidyVoiceX_ASV"
    audio_path = tmp_path / "outside" / "id000001" / "en" / "sample.wav"

    with pytest.raises(TidyVoicePathError, match="outside dataset root"):
        parse_tidyvoice_audio_path(
            audio_path,
            dataset_root=dataset_root,
            split=Split.TRAIN,
        )


def test_reject_unsupported_tidyvoice_layout(tmp_path: Path) -> None:
    """Unexpected nesting should fail instead of guessing speaker labels."""
    dataset_root = tmp_path / "TidyVoiceX_ASV"
    audio_path = (
        dataset_root
        / "TidyVoiceX_Train"
        / "wrong_directory"
        / "id012711"
        / "fa"
        / "sample.wav"
    )

    with pytest.raises(TidyVoicePathError, match="Unsupported TidyVoice branch"):
        parse_tidyvoice_audio_path(
            audio_path,
            dataset_root=dataset_root,
            split=Split.TRAIN,
        )


def test_reject_invalid_source_to_canonical_split_mapping(
    tmp_path: Path,
) -> None:
    """Source-train speakers must remain in canonical training."""
    dataset_root = tmp_path / "TidyVoiceX_ASV"
    audio_path = (
        dataset_root
        / "TidyVoiceX_Train"
        / "TidyVoiceX_Train"
        / "id012711"
        / "fa"
        / "sample.wav"
    )

    with pytest.raises(TidyVoicePathError, match="must map to canonical train"):
        parse_tidyvoice_audio_path(
            audio_path,
            dataset_root=dataset_root,
            split=Split.TEST,
        )


def test_reject_non_wav_file(tmp_path: Path) -> None:
    """TidyVoice manifests should contain only documented WAV audio."""
    dataset_root = tmp_path / "TidyVoiceX_ASV"
    audio_path = (
        dataset_root
        / "TidyVoiceX_Dev"
        / "TidyVoiceX_Dev"
        / "id014060"
        / "en"
        / "sample.mp3"
    )

    with pytest.raises(TidyVoicePathError, match="WAV"):
        parse_tidyvoice_audio_path(
            audio_path,
            dataset_root=dataset_root,
            split=Split.TEST,
        )

def test_scan_source_split_in_deterministic_order(tmp_path: Path) -> None:
    """Scanning order must not depend on filesystem enumeration order."""
    dataset_root = tmp_path / "TidyVoiceX_ASV"
    branch_root = dataset_root / "TidyVoiceX_Train" / "TidyVoiceX_Train"

    relative_paths = (
        Path("id000002/en/z.wav"),
        Path("id000001/vi/b.wav"),
        Path("id000001/en/c.wav"),
        Path("id000001/en/a.wav"),
    )
    for relative_path in relative_paths:
        path = branch_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    discovered = tuple(
        path.relative_to(branch_root).as_posix()
        for path in iter_tidyvoice_audio_paths(
            dataset_root,
            source_split="train",
        )
    )

    assert discovered == (
        "id000001/en/a.wav",
        "id000001/en/c.wav",
        "id000001/vi/b.wav",
        "id000002/en/z.wav",
    )


def test_scan_rejects_unknown_source_split(tmp_path: Path) -> None:
    """Only source-provided train and dev branches are valid."""
    with pytest.raises(TidyVoicePathError, match="source split"):
        tuple(
            iter_tidyvoice_audio_paths(
                tmp_path,
                source_split="validation",
            )
        )


def test_scan_rejects_missing_branch_directory(tmp_path: Path) -> None:
    """A missing mount should fail before producing an empty manifest."""
    dataset_root = tmp_path / "TidyVoiceX_ASV"

    with pytest.raises(TidyVoicePathError, match="does not exist"):
        tuple(
            iter_tidyvoice_audio_paths(
                dataset_root,
                source_split="dev",
            )
        )


def test_scan_rejects_unexpected_file_format(tmp_path: Path) -> None:
    """Unexpected files should be reported instead of silently omitted."""
    dataset_root = tmp_path / "TidyVoiceX_ASV"
    language_root = (
        dataset_root
        / "TidyVoiceX_Dev"
        / "TidyVoiceX_Dev"
        / "id000001"
        / "en"
    )
    language_root.mkdir(parents=True)
    (language_root / "sample.wav").touch()
    (language_root / "unexpected.txt").touch()

    with pytest.raises(TidyVoicePathError, match="Unexpected file"):
        tuple(
            iter_tidyvoice_audio_paths(
                dataset_root,
                source_split="dev",
            )
        )

def test_collect_speaker_language_counts(tmp_path: Path) -> None:
    """Dev profiles should count languages without decoding waveforms."""
    dataset_root = tmp_path / "TidyVoiceX_ASV"
    branch_root = dataset_root / "TidyVoiceX_Dev" / "TidyVoiceX_Dev"

    relative_paths = (
        Path("id000001/en/a.wav"),
        Path("id000001/en/b.wav"),
        Path("id000001/vi/c.wav"),
        Path("id000002/vi/d.wav"),
    )
    for relative_path in relative_paths:
        path = branch_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    profiles = collect_tidyvoice_speaker_language_counts(
        dataset_root,
        source_split="dev",
    )

    assert profiles == {
        "tidyvoice:id000001": {"en": 2, "vi": 1},
        "tidyvoice:id000002": {"vi": 1},
    }