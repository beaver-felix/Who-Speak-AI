"""Tests for converting ViMD Parquet metadata into manifest records."""

from pathlib import Path

import pytest

from speaker_recognition.data.manifest import AudioStorage, Split
from speaker_recognition.data.vimd import (
    ViMDMetadataError,
    parse_vimd_metadata_row,
)


def make_metadata_row(
    *,
    speaker_id: str = "spk_11_0001",
    filename: str = "11_0001.wav",
) -> dict[str, object]:
    """Create one representative ViMD metadata row."""
    return {
        "region": "North",
        "province_code": 11,
        "province_name": "CaoBang",
        "filename": filename,
        "text": "Representative Vietnamese speech.",
        "speakerID": speaker_id,
        "gender": 1,
    }


def test_parse_train_row_into_canonical_record(tmp_path: Path) -> None:
    """A source Train row should remain in canonical Train."""
    dataset_root = tmp_path / "vimd-dataset"
    parquet_path = dataset_root / "data" / "train-00000.parquet"

    record = parse_vimd_metadata_row(
        make_metadata_row(),
        parquet_path=parquet_path,
        row_index=7,
        dataset_root=dataset_root,
        source_split="train",
    )

    assert record is not None
    assert record.dataset == "vimd"
    assert record.source_split == "train"
    assert record.split is Split.TRAIN
    assert record.speaker_id == "vimd:spk_11_0001"
    assert record.utterance_id == (
        "vimd:train:spk_11_0001:11_0001"
    )
    assert record.recording_id == record.utterance_id
    assert record.audio_storage is AudioStorage.PARQUET
    assert record.audio_path == "data/train-00000.parquet"
    assert record.audio_row_index == 7
    assert record.region == "North"
    assert record.province == "CaoBang"
    assert record.gender == "1"
    assert record.transcript == "Representative Vietnamese speech."


def test_exclude_contaminated_validation_speaker(tmp_path: Path) -> None:
    """A Validation row whose speaker occurs in Test should be excluded."""
    dataset_root = tmp_path / "vimd-dataset"

    record = parse_vimd_metadata_row(
        make_metadata_row(
            speaker_id="spk_73_0186",
            filename="73_0309.wav",
        ),
        parquet_path=dataset_root / "data" / "valid-00008.parquet",
        row_index=89,
        dataset_root=dataset_root,
        source_split="valid",
    )

    assert record is None


def test_retain_overlapping_speaker_in_test(tmp_path: Path) -> None:
    """The official Test rows must remain unchanged."""
    dataset_root = tmp_path / "vimd-dataset"

    record = parse_vimd_metadata_row(
        make_metadata_row(
            speaker_id="spk_73_0186",
            filename="73_0307.wav",
        ),
        parquet_path=dataset_root / "data" / "test-00008.parquet",
        row_index=134,
        dataset_root=dataset_root,
        source_split="test",
    )

    assert record is not None
    assert record.split is Split.TEST
    assert record.speaker_id == "vimd:spk_73_0186"


def test_reject_missing_required_metadata(tmp_path: Path) -> None:
    """Missing identity metadata must fail instead of producing guessed IDs."""
    dataset_root = tmp_path / "vimd-dataset"
    row = make_metadata_row()
    del row["filename"]

    with pytest.raises(ViMDMetadataError, match="filename"):
        parse_vimd_metadata_row(
            row,
            parquet_path=dataset_root / "data" / "train.parquet",
            row_index=0,
            dataset_root=dataset_root,
            source_split="train",
        )


def test_reject_unknown_source_split(tmp_path: Path) -> None:
    """Only the three documented ViMD source partitions are valid."""
    dataset_root = tmp_path / "vimd-dataset"

    with pytest.raises(ViMDMetadataError, match="source split"):
        parse_vimd_metadata_row(
            make_metadata_row(),
            parquet_path=dataset_root / "data" / "unknown.parquet",
            row_index=0,
            dataset_root=dataset_root,
            source_split="development",
        )


def test_reject_parquet_path_outside_dataset_root(
    tmp_path: Path,
) -> None:
    """Parquet references must remain inside the configured dataset root."""
    dataset_root = tmp_path / "vimd-dataset"
    outside_path = tmp_path / "outside" / "train.parquet"

    with pytest.raises(ViMDMetadataError, match="outside dataset root"):
        parse_vimd_metadata_row(
            make_metadata_row(),
            parquet_path=outside_path,
            row_index=0,
            dataset_root=dataset_root,
            source_split="train",
        )