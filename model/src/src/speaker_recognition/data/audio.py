"""Model-independent audio loading and canonical signal conversion."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import gcd
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

from speaker_recognition.data.manifest import AudioStorage, ManifestRecord


class AudioDataError(ValueError):
    """Raised when audio cannot be converted into a valid model input."""


@dataclass(frozen=True, slots=True)
class CanonicalAudio:
    """Store a canonical waveform and its original format metadata.

    Attributes
    ----------
    waveform:
        Contiguous mono ``float32`` waveform at ``sample_rate``.
    sample_rate:
        Canonical sample rate used by every model adapter.
    original_sample_rate:
        Sample rate reported by the source audio container.
    original_channels:
        Channel count reported by the source audio container.
    """

    waveform: NDArray[np.float32]
    sample_rate: int
    original_sample_rate: int
    original_channels: int

    @property
    def duration_seconds(self) -> float:
        """Return canonical waveform duration in seconds."""
        return self.waveform.size / self.sample_rate


class ParquetAudioReader:
    """Decode ViMD audio bytes with a one-row-group in-memory cache.

    Parquet cannot read an arbitrary row without decoding its containing row
    group. Reusing that group avoids repeated large reads when a batch or
    sampler accesses nearby rows. Create one reader per data-loader worker;
    instances intentionally do not share mutable cache state across workers.

    Parameters
    ----------
    dataset_root:
        Root against which manifest Parquet paths are resolved.
    target_sample_rate:
        Canonical output sample rate in hertz.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        target_sample_rate: int = 16000,
    ) -> None:
        """Initialize an empty Parquet row-group cache."""
        _validate_sample_rate(
            target_sample_rate,
            field_name="target_sample_rate",
        )
        self._dataset_root = Path(dataset_root).expanduser().resolve()
        self._target_sample_rate = target_sample_rate
        self._cached_key: tuple[Path, int] | None = None
        self._cached_rows: list[object] = []
        self._row_group_reads = 0

    @property
    def row_group_reads(self) -> int:
        """Return the number of physical Parquet row-group reads."""
        return self._row_group_reads

    def load(self, record: ManifestRecord) -> CanonicalAudio:
        """Load one embedded-audio manifest record.

        Parameters
        ----------
        record:
            Manifest row with Parquet storage and a shard-global row index.

        Returns
        -------
        CanonicalAudio
            Decoded, downmixed, and resampled audio with source metadata.

        Raises
        ------
        AudioDataError
            If the locator, Parquet schema, embedded bytes, or audio is
            invalid.
        """
        if record.audio_storage is not AudioStorage.PARQUET:
            raise AudioDataError(
                "ParquetAudioReader requires a Parquet manifest record."
            )
        if record.audio_row_index is None:
            raise AudioDataError("Parquet record has no audio_row_index.")

        shard_path = (
            self._dataset_root / Path(record.audio_path)
        ).resolve()
        try:
            shard_path.relative_to(self._dataset_root)
        except ValueError as error:
            raise AudioDataError(
                f"Parquet shard is outside dataset root: {shard_path}"
            ) from error
        if not shard_path.is_file():
            raise AudioDataError(
                f"Parquet audio shard does not exist: {shard_path}"
            )

        parquet_file = pq.ParquetFile(shard_path)
        if "audio" not in parquet_file.schema_arrow.names:
            raise AudioDataError(
                f"Parquet shard has no audio column: {shard_path.name}"
            )

        row_group_index, group_start = _locate_row_group(
            parquet_file,
            record.audio_row_index,
        )
        cache_key = (shard_path, row_group_index)
        if cache_key != self._cached_key:
            table = parquet_file.read_row_group(
                row_group_index,
                columns=["audio"],
                use_threads=False,
            )
            self._cached_rows = table.column("audio").to_pylist()
            self._cached_key = cache_key
            self._row_group_reads += 1

        local_row_index = record.audio_row_index - group_start
        try:
            audio_value = self._cached_rows[local_row_index]
        except IndexError as error:
            raise AudioDataError(
                "Parquet audio row index is inconsistent with row-group "
                "metadata."
            ) from error

        if not isinstance(audio_value, dict):
            raise AudioDataError("ViMD audio value must be a struct mapping.")
        audio_bytes = audio_value.get("bytes")
        if not isinstance(audio_bytes, bytes) or not audio_bytes:
            raise AudioDataError("ViMD audio struct contains no audio bytes.")

        try:
            waveform, original_sample_rate = sf.read(
                BytesIO(audio_bytes),
                dtype="float32",
                always_2d=True,
            )
        except (OSError, RuntimeError) as error:
            raise AudioDataError(
                "Unable to decode embedded ViMD audio bytes."
            ) from error

        original_channels = int(waveform.shape[1])
        canonical_waveform = canonicalize_audio(
            waveform,
            sample_rate=int(original_sample_rate),
            target_sample_rate=self._target_sample_rate,
        )
        return CanonicalAudio(
            waveform=canonical_waveform,
            sample_rate=self._target_sample_rate,
            original_sample_rate=int(original_sample_rate),
            original_channels=original_channels,
        )


def canonicalize_audio(
    waveform: NDArray[np.generic],
    *,
    sample_rate: int,
    target_sample_rate: int = 16000,
) -> NDArray[np.float32]:
    """Convert a waveform to deterministic mono ``float32`` audio.

    Downmixing uses the arithmetic mean across channels. Resampling uses a
    polyphase anti-aliasing filter and then enforces the nearest duration-
    preserving target length. Amplitude normalization is intentionally not
    applied because the pretrained encoders expect natural waveform scale.

    Parameters
    ----------
    waveform:
        One-dimensional mono samples or a ``(frames, channels)`` array.
    sample_rate:
        Positive source sample rate in hertz.
    target_sample_rate:
        Positive canonical sample rate in hertz.

    Returns
    -------
    numpy.ndarray
        Contiguous one-dimensional ``float32`` waveform.

    Raises
    ------
    AudioDataError
        If shape, rate, length, or sample values are invalid.
    """
    _validate_sample_rate(sample_rate, field_name="sample_rate")
    _validate_sample_rate(
        target_sample_rate,
        field_name="target_sample_rate",
    )

    samples = np.asarray(waveform)
    if samples.ndim not in {1, 2}:
        raise AudioDataError(
            "waveform must have shape (frames,) or (frames, channels)."
        )
    if samples.shape[0] == 0:
        raise AudioDataError("waveform must contain at least one frame.")
    if samples.ndim == 2 and samples.shape[1] == 0:
        raise AudioDataError("waveform must contain at least one channel.")
    if not np.issubdtype(samples.dtype, np.number):
        raise AudioDataError("waveform samples must be numeric.")
    if not np.isfinite(samples).all():
        raise AudioDataError("waveform samples must all be finite.")

    if samples.ndim == 2:
        # Accumulating in float64 avoids avoidable rounding bias when stereo
        # or multichannel sources are downmixed.
        mono = samples.astype(np.float64, copy=False).mean(axis=1)
    else:
        mono = samples.astype(np.float64, copy=False)

    if sample_rate != target_sample_rate:
        common_divisor = gcd(sample_rate, target_sample_rate)
        upsample_factor = target_sample_rate // common_divisor
        downsample_factor = sample_rate // common_divisor
        canonical = resample_poly(
            mono,
            upsample_factor,
            downsample_factor,
        )

        # SciPy returns a ceiling-based length. Enforcing nearest duration
        # prevents one-sample inconsistencies across source sample rates.
        expected_length = (
            mono.size * target_sample_rate + sample_rate // 2
        ) // sample_rate
        canonical = _fit_resampled_length(canonical, expected_length)
    else:
        canonical = mono

    canonical = np.ascontiguousarray(canonical, dtype=np.float32)
    if canonical.size == 0 or not np.isfinite(canonical).all():
        raise AudioDataError("canonical waveform is empty or non-finite.")

    return canonical


def load_audio_file(
    audio_path: str | Path,
    *,
    target_sample_rate: int = 16000,
) -> CanonicalAudio:
    """Load a standalone audio file and convert it to canonical form.

    Parameters
    ----------
    audio_path:
        Path to a source audio file supported by libsndfile.
    target_sample_rate:
        Canonical output sample rate in hertz.

    Returns
    -------
    CanonicalAudio
        Canonical waveform plus original sample-rate and channel metadata.

    Raises
    ------
    AudioDataError
        If the file is missing, unreadable, or contains invalid audio.
    """
    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        raise AudioDataError(f"Audio file does not exist: {path}")

    try:
        waveform, original_sample_rate = sf.read(
            path,
            dtype="float32",
            always_2d=True,
        )
    except (OSError, RuntimeError) as error:
        raise AudioDataError(f"Unable to decode audio file: {path}") from error

    original_channels = int(waveform.shape[1])
    canonical_waveform = canonicalize_audio(
        waveform,
        sample_rate=int(original_sample_rate),
        target_sample_rate=target_sample_rate,
    )

    return CanonicalAudio(
        waveform=canonical_waveform,
        sample_rate=target_sample_rate,
        original_sample_rate=int(original_sample_rate),
        original_channels=original_channels,
    )


def _validate_sample_rate(value: int, *, field_name: str) -> None:
    """Require a positive integer sample rate."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AudioDataError(f"{field_name} must be a positive integer.")


def _fit_resampled_length(
    waveform: NDArray[np.float64],
    expected_length: int,
) -> NDArray[np.float64]:
    """Crop or right-pad a resampled signal to its nearest target length."""
    if expected_length <= 0:
        raise AudioDataError("resampling produced an invalid target length.")
    if waveform.size > expected_length:
        return waveform[:expected_length]
    if waveform.size < expected_length:
        return np.pad(
            waveform,
            (0, expected_length - waveform.size),
            mode="constant",
        )
    return waveform


def _locate_row_group(
    parquet_file: pq.ParquetFile,
    row_index: int,
) -> tuple[int, int]:
    """Return the row-group index and its shard-global starting offset."""
    if row_index < 0 or row_index >= parquet_file.metadata.num_rows:
        raise AudioDataError(
            f"Parquet row index {row_index} is outside shard bounds."
        )

    group_start = 0
    for group_index in range(parquet_file.metadata.num_row_groups):
        group_size = parquet_file.metadata.row_group(group_index).num_rows
        if row_index < group_start + group_size:
            return group_index, group_start
        group_start += group_size

    raise AudioDataError("Unable to locate Parquet row group.")
