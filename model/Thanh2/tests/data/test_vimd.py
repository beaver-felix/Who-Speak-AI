"""Tests for converting ViMD Parquet metadata into manifest records."""

from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from speaker_recognition.data.manifest import AudioStorage, Split
from speaker_recognition.data.vimd import (
    ViMDMetadataError,
    parse_vimd_metadata_row,
    iter_vimd_source_records,

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

def write_vimd_shard(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write a small metadata-only Parquet fixture with multiple row groups."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, row_group_size=1)


def test_stream_source_records_and_preserve_global_row_indexes(
    tmp_path: Path,
) -> None:
    """Streaming must preserve row positions while applying exclusions."""
    dataset_root = tmp_path / "vimd-dataset"
    shard_path = dataset_root / "data" / "valid-00000.parquet"

    write_vimd_shard(
        shard_path,
        [
            make_metadata_row(
                speaker_id="spk_01_0001",
                filename="01_0001.wav",
            ),
            make_metadata_row(
                speaker_id="spk_73_0186",
                filename="73_0309.wav",
            ),
            make_metadata_row(
                speaker_id="spk_01_0002",
                filename="01_0002.wav",
            ),
        ],
    )

    records = tuple(
        iter_vimd_source_records(
            dataset_root,
            source_split="valid",
            batch_size=2,
        )
    )

    assert len(records) == 2
    assert [record.audio_row_index for record in records] == [0, 2]
    assert all(record.split is Split.VALIDATION for record in records)


def test_stream_shards_in_deterministic_filename_order(
    tmp_path: Path,
) -> None:
    """Manifest order must not depend on filesystem enumeration order."""
    dataset_root = tmp_path / "vimd-dataset"

    write_vimd_shard(
        dataset_root / "data" / "train-00001.parquet",
        [
            make_metadata_row(
                speaker_id="spk_01_0002",
                filename="second.wav",
            )
        ],
    )
    write_vimd_shard(
        dataset_root / "data" / "train-00000.parquet",
        [
            make_metadata_row(
                speaker_id="spk_01_0001",
                filename="first.wav",
            )
        ],
    )

    records = tuple(
        iter_vimd_source_records(
            dataset_root,
            source_split="train",
        )
    )

    assert [record.audio_path for record in records] == [
        "data/train-00000.parquet",
        "data/train-00001.parquet",
    ]


def test_stream_rejects_missing_source_shards(tmp_path: Path) -> None:
    """A missing dataset mount must not become an empty experiment."""
    dataset_root = tmp_path / "vimd-dataset"

    with pytest.raises(ViMDMetadataError, match="No ViMD Parquet shards"):
        tuple(
            iter_vimd_source_records(
                dataset_root,
                source_split="test",
            )
        )