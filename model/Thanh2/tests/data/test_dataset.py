"""Tests for the shared lazy dataset and NumPy batch interface."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from speaker_recognition.data.dataset import (
    EvaluationSpeakerDataset,
    SpeakerDatasetError,
    TrainingSpeakerDataset,
    collate_evaluation_samples,
    collate_training_samples,
)
from speaker_recognition.data.manifest import (
    AudioStorage,
    ManifestRecord,
    Split,
)


def test_training_dataset_shares_file_and_parquet_loading(tmp_path: Path) -> None:
    """Both physical formats should produce the same training interface."""
    tidy_root = tmp_path / "tidyvoice"
    vimd_root = tmp_path / "vimd"
    tidy_path = tidy_root / "audio" / "sample.wav"
    tidy_path.parent.mkdir(parents=True)
    waveform = np.linspace(-0.8, 0.8, 100, dtype=np.float32)
    sf.write(tidy_path, waveform, 16000, subtype="PCM_16")

    shard_path = vimd_root / "data" / "train-00000.parquet"
    shard_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{"audio": {"bytes": _wav_bytes(waveform), "path": "sample.wav"}}]
        ),
        shard_path,
    )
    records = (
        _record(
            dataset="tidyvoice",
            speaker="speaker-b",
            utterance="file",
            storage=AudioStorage.FILE,
            audio_path="audio/sample.wav",
        ),
        _record(
            dataset="vimd",
            speaker="speaker-a",
            utterance="parquet",
            storage=AudioStorage.PARQUET,
            audio_path="data/train-00000.parquet",
            row_index=0,
        ),
    )
    dataset = TrainingSpeakerDataset(
        records,
        dataset_roots={"tidyvoice": tidy_root, "vimd": vimd_root},
        segment_samples=32,
        seed=42,
    )

    samples = [dataset[index] for index in range(len(dataset))]
    batch = collate_training_samples(samples)

    assert batch.waveforms.shape == (2, 32)
    assert batch.waveforms.dtype == np.float32
    assert batch.speaker_indices.dtype == np.int64
    assert dataset.speaker_to_index == {"speaker-a": 0, "speaker-b": 1}
    assert set(batch.datasets) == {"tidyvoice", "vimd"}


def test_training_crops_repeat_within_epoch_and_change_across_epochs(
    tmp_path: Path,
) -> None:
    """Dataset crops should implement the accepted epoch-specific seed rule."""
    root, record = _standalone_fixture(tmp_path, sample_count=100)
    dataset = TrainingSpeakerDataset(
        (record,),
        dataset_roots={"tidyvoice": root},
        segment_samples=20,
        seed=42,
    )

    first = dataset[0].waveform.copy()
    repeated = dataset[0].waveform.copy()
    dataset.set_epoch(1)
    next_epoch = dataset[0].waveform.copy()

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, next_epoch)


def test_evaluation_collator_flattens_crops_and_records_offsets(
    tmp_path: Path,
) -> None:
    """Variable crop counts should retain utterance aggregation boundaries."""
    root = tmp_path / "tidyvoice"
    records = []
    for name, sample_count in (("long", 40), ("short", 8)):
        path = root / "audio" / f"{name}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            path,
            np.linspace(-0.5, 0.5, sample_count, dtype=np.float32),
            16000,
            subtype="PCM_16",
        )
        records.append(
            _record(
                dataset="tidyvoice",
                speaker=f"speaker-{name}",
                utterance=name,
                storage=AudioStorage.FILE,
                audio_path=f"audio/{name}.wav",
                split=Split.TEST,
            )
        )
    dataset = EvaluationSpeakerDataset(
        records,
        split=Split.TEST,
        dataset_roots={"tidyvoice": root},
        segment_samples=16,
        segment_count=3,
    )

    batch = collate_evaluation_samples([dataset[0], dataset[1]])

    assert batch.waveforms.shape == (4, 16)
    np.testing.assert_array_equal(
        batch.segment_offsets,
        np.array([0, 3, 4], dtype=np.int64),
    )
    assert len(batch.utterance_ids) == 2


def test_dataset_record_order_is_input_order_independent(tmp_path: Path) -> None:
    """Stable indexing should not depend on manifest construction order."""
    root = tmp_path / "tidyvoice"
    records = []
    for name in ("z", "a"):
        path = root / f"{name}.wav"
        root.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.ones(20, dtype=np.float32), 16000)
        records.append(
            _record(
                dataset="tidyvoice",
                speaker=f"speaker-{name}",
                utterance=name,
                storage=AudioStorage.FILE,
                audio_path=f"{name}.wav",
            )
        )

    dataset = TrainingSpeakerDataset(
        tuple(reversed(records)),
        dataset_roots={"tidyvoice": root},
        segment_samples=10,
        seed=42,
    )

    assert [record.utterance_id for record in dataset.records] == [
        "tidyvoice:train:speaker-a:a",
        "tidyvoice:train:speaker-z:z",
    ]


def test_reject_file_path_escaping_dataset_root(tmp_path: Path) -> None:
    """Portable manifest paths must not read files outside their mount."""
    root = tmp_path / "tidyvoice"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    sf.write(outside, np.ones(20, dtype=np.float32), 16000)
    record = _record(
        dataset="tidyvoice",
        speaker="speaker",
        utterance="escape",
        storage=AudioStorage.FILE,
        audio_path="../outside.wav",
    )
    dataset = TrainingSpeakerDataset(
        (record,),
        dataset_roots={"tidyvoice": root},
        segment_samples=10,
        seed=42,
    )

    with pytest.raises(SpeakerDatasetError, match="escapes"):
        dataset[0]


def test_reject_non_contiguous_external_speaker_mapping(tmp_path: Path) -> None:
    """Classification head indexes must be stable, unique, and contiguous."""
    root, record = _standalone_fixture(tmp_path, sample_count=20)

    with pytest.raises(SpeakerDatasetError, match="contiguous"):
        TrainingSpeakerDataset(
            (record,),
            dataset_roots={"tidyvoice": root},
            segment_samples=10,
            seed=42,
            speaker_to_index={record.speaker_id: 2},
        )


def test_reject_wrong_split_and_empty_batches(tmp_path: Path) -> None:
    """Mode errors should fail before a training or evaluation loop starts."""
    root, record = _standalone_fixture(tmp_path, sample_count=20)

    with pytest.raises(SpeakerDatasetError, match="validation or test"):
        EvaluationSpeakerDataset(
            (record,),
            split=Split.TRAIN,
            dataset_roots={"tidyvoice": root},
            segment_samples=10,
            segment_count=2,
        )
    with pytest.raises(SpeakerDatasetError, match="empty training"):
        collate_training_samples([])
    with pytest.raises(SpeakerDatasetError, match="empty evaluation"):
        collate_evaluation_samples([])


def test_reject_noncanonical_dataset_sample_rate(tmp_path: Path) -> None:
    """The shared dataset boundary must not bypass accepted 16 kHz input."""
    root, record = _standalone_fixture(tmp_path, sample_count=20)

    with pytest.raises(SpeakerDatasetError, match="canonical 16000"):
        TrainingSpeakerDataset(
            (record,),
            dataset_roots={"tidyvoice": root},
            segment_samples=10,
            seed=42,
            target_sample_rate=8000,
        )


def _standalone_fixture(
    tmp_path: Path,
    *,
    sample_count: int,
) -> tuple[Path, ManifestRecord]:
    """Write one deterministic standalone training fixture."""
    root = tmp_path / "tidyvoice"
    path = root / "sample.wav"
    root.mkdir(parents=True)
    sf.write(
        path,
        np.linspace(-0.8, 0.8, sample_count, dtype=np.float32),
        16000,
        subtype="PCM_16",
    )
    return root, _record(
        dataset="tidyvoice",
        speaker="speaker-a",
        utterance="sample",
        storage=AudioStorage.FILE,
        audio_path="sample.wav",
    )


def _record(
    *,
    dataset: str,
    speaker: str,
    utterance: str,
    storage: AudioStorage,
    audio_path: str,
    split: Split = Split.TRAIN,
    row_index: int | None = None,
) -> ManifestRecord:
    """Create one canonical record fixture."""
    source_split = "valid" if split is Split.VALIDATION else split.value
    utterance_id = f"{dataset}:{source_split}:{speaker}:{utterance}"
    return ManifestRecord(
        dataset=dataset,
        source_split=source_split,
        split=split,
        utterance_id=utterance_id,
        speaker_id=speaker,
        recording_id=utterance_id,
        audio_storage=storage,
        audio_path=audio_path,
        audio_row_index=row_index,
    )


def _wav_bytes(waveform: np.ndarray) -> bytes:
    """Encode one PCM-16 waveform for a Parquet fixture."""
    buffer = BytesIO()
    sf.write(buffer, waveform, 16000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
