"""Tests for model-independent audio standardization."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from speaker_recognition.data.audio import (
    AudioDataError,
    ParquetAudioReader,
    canonicalize_audio,
    load_audio_file,
)
from speaker_recognition.data.manifest import (
    AudioStorage,
    ManifestRecord,
    Split,
)


def test_canonicalize_mono_16khz_without_amplitude_normalization() -> None:
    """Already compliant audio should retain samples and amplitude."""
    waveform = np.linspace(-0.25, 0.25, 1600, dtype=np.float32)

    canonical = canonicalize_audio(
        waveform,
        sample_rate=16000,
        target_sample_rate=16000,
    )

    assert canonical.dtype == np.float32
    assert canonical.shape == (1600,)
    np.testing.assert_allclose(canonical, waveform)


def test_downmix_stereo_and_resample_44100_to_16000() -> None:
    """ViMD stereo audio should become deterministic mono 16 kHz audio."""
    frame_count = 4410
    left = np.full(frame_count, 0.25, dtype=np.float32)
    right = np.full(frame_count, -0.25, dtype=np.float32)
    stereo = np.column_stack((left, right))

    canonical = canonicalize_audio(
        stereo,
        sample_rate=44100,
        target_sample_rate=16000,
    )

    assert canonical.dtype == np.float32
    assert canonical.shape == (1600,)
    np.testing.assert_allclose(canonical, 0.0, atol=1e-6)


def test_load_audio_file_returns_source_metadata(tmp_path: Path) -> None:
    """Standalone WAV loading should report original format information."""
    audio_path = tmp_path / "sample.wav"
    waveform = np.full(800, 0.25, dtype=np.float32)
    sf.write(
        audio_path,
        waveform,
        samplerate=16000,
        subtype="PCM_16",
    )

    audio = load_audio_file(
        audio_path,
        target_sample_rate=16000,
    )

    assert audio.sample_rate == 16000
    assert audio.original_sample_rate == 16000
    assert audio.original_channels == 1
    assert audio.waveform.shape == (800,)
    assert audio.duration_seconds == pytest.approx(0.05)
    np.testing.assert_allclose(
        audio.waveform,
        waveform,
        atol=4e-5,
    )


@pytest.mark.parametrize(
    ("waveform", "sample_rate"),
    [
        (np.array([], dtype=np.float32), 16000),
        (np.array([0.0, np.nan], dtype=np.float32), 16000),
        (np.zeros(100, dtype=np.float32), 0),
    ],
)
def test_reject_invalid_audio(
    waveform: np.ndarray,
    sample_rate: int,
) -> None:
    """Empty, non-finite, or invalid-rate audio must fail before modeling."""
    with pytest.raises(AudioDataError):
        canonicalize_audio(
            waveform,
            sample_rate=sample_rate,
            target_sample_rate=16000,
        )


def test_parquet_reader_decodes_and_resamples_embedded_audio(
    tmp_path: Path,
) -> None:
    """Embedded stereo ViMD audio should become canonical mono audio."""
    dataset_root = tmp_path / "vimd-dataset"
    shard_path = dataset_root / "data" / "test-00000.parquet"
    shard_path.parent.mkdir(parents=True)
    waveform = np.column_stack(
        (
            np.full(4410, 0.2, dtype=np.float32),
            np.full(4410, 0.4, dtype=np.float32),
        )
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"audio": {"bytes": _wav_bytes(waveform, 44100), "path": "x"}}]
        ),
        shard_path,
    )
    record = _parquet_record(row_index=0)

    audio = ParquetAudioReader(dataset_root).load(record)

    assert audio.waveform.shape == (1600,)
    assert audio.sample_rate == 16000
    assert audio.original_sample_rate == 44100
    assert audio.original_channels == 2
    assert float(np.median(audio.waveform)) == pytest.approx(0.3, abs=1e-3)


def test_parquet_reader_reuses_cached_row_group(tmp_path: Path) -> None:
    """Nearby reads should decode their shared Parquet row group once."""
    dataset_root = tmp_path / "vimd-dataset"
    shard_path = dataset_root / "data" / "test-00000.parquet"
    shard_path.parent.mkdir(parents=True)
    rows = [
        {
            "audio": {
                "bytes": _wav_bytes(
                    np.full(160, amplitude, dtype=np.float32),
                    16000,
                ),
                "path": f"{index}.wav",
            }
        }
        for index, amplitude in enumerate((0.1, 0.2))
    ]
    pq.write_table(pa.Table.from_pylist(rows), shard_path, row_group_size=2)
    reader = ParquetAudioReader(dataset_root)

    first = reader.load(_parquet_record(row_index=0))
    second = reader.load(_parquet_record(row_index=1))

    assert reader.row_group_reads == 1
    np.testing.assert_allclose(first.waveform, 0.1, atol=4e-5)
    np.testing.assert_allclose(second.waveform, 0.2, atol=4e-5)


def _wav_bytes(
    waveform: np.ndarray,
    sample_rate: int,
) -> bytes:
    """Encode an in-memory PCM-16 WAV fixture."""
    buffer = BytesIO()
    sf.write(
        buffer,
        waveform,
        samplerate=sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


def _parquet_record(*, row_index: int) -> ManifestRecord:
    """Create a canonical Parquet locator for audio-reader tests."""
    return ManifestRecord(
        dataset="vimd",
        source_split="test",
        split=Split.TEST,
        utterance_id=f"vimd:test:speaker:utterance-{row_index}",
        speaker_id="vimd:speaker",
        recording_id=f"vimd:test:speaker:utterance-{row_index}",
        audio_storage=AudioStorage.PARQUET,
        audio_path="data/test-00000.parquet",
        audio_row_index=row_index,
    )
