"""Shared lazy dataset and batching boundary for every speaker model.

The classes intentionally return NumPy arrays. They satisfy PyTorch's map-style
dataset protocol through ``__len__`` and ``__getitem__``, while avoiding a
package dependency on PyTorch that could replace Kaggle's CUDA-matched build.
PyTorch DataLoader can use these datasets with the provided collators and the
training layer can convert each contiguous NumPy batch with ``torch.from_numpy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from speaker_recognition.data.audio import (
    AudioDataError,
    CanonicalAudio,
    ParquetAudioReader,
    load_audio_file,
)
from speaker_recognition.data.manifest import (
    AudioStorage,
    ManifestRecord,
    Split,
    validate_manifest,
)
from speaker_recognition.data.segments import (
    evenly_spaced_segments,
    random_fixed_segment,
    stable_segment_seed,
)


class SpeakerDatasetError(ValueError):
    """Raised when a shared dataset or batch violates runtime invariants."""


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One fixed-length training waveform and its classification identity."""

    waveform: NDArray[np.float32]
    speaker_index: int
    speaker_id: str
    utterance_id: str
    dataset: str


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    """All deterministic evaluation crops for one canonical utterance."""

    waveforms: NDArray[np.float32]
    speaker_id: str
    utterance_id: str
    dataset: str


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Contiguous fixed-shape arrays ready for zero-copy tensor conversion."""

    waveforms: NDArray[np.float32]
    speaker_indices: NDArray[np.int64]
    speaker_ids: tuple[str, ...]
    utterance_ids: tuple[str, ...]
    datasets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationBatch:
    """Flattened crop arrays plus offsets for utterance-level aggregation.

    If ``segment_offsets`` is ``[0, 3, 4]``, crops ``0:3`` belong to the first
    utterance and crop ``3:4`` belongs to the second. This supports short
    utterances producing one crop while longer utterances produce several.
    """

    waveforms: NDArray[np.float32]
    segment_offsets: NDArray[np.int64]
    speaker_ids: tuple[str, ...]
    utterance_ids: tuple[str, ...]
    datasets: tuple[str, ...]


class CanonicalAudioLoader:
    """Resolve dataset-relative locators and load either supported storage type.

    A separate instance is copied into each DataLoader worker. Parquet readers
    are created lazily so every worker owns its own one-row-group cache.
    """

    def __init__(
        self,
        dataset_roots: Mapping[str, str | Path],
        *,
        target_sample_rate: int = 16000,
    ) -> None:
        """Validate dataset roots without touching any audio payload."""
        if not dataset_roots:
            raise SpeakerDatasetError("At least one dataset root is required.")
        if (
            isinstance(target_sample_rate, bool)
            or not isinstance(target_sample_rate, int)
            or target_sample_rate <= 0
        ):
            raise SpeakerDatasetError(
                "target_sample_rate must be a positive integer."
            )
        if target_sample_rate != 16000:
            raise SpeakerDatasetError(
                "target_sample_rate must equal the accepted canonical 16000 Hz."
            )

        roots: dict[str, Path] = {}
        for dataset, raw_root in dataset_roots.items():
            if not isinstance(dataset, str) or not dataset.strip():
                raise SpeakerDatasetError(
                    "Dataset-root keys must be non-empty strings."
                )
            root = Path(raw_root).expanduser().resolve()
            if not root.is_dir():
                raise SpeakerDatasetError(
                    f"Dataset root does not exist for {dataset!r}: {root}"
                )
            roots[dataset] = root

        self._roots = roots
        self._target_sample_rate = target_sample_rate
        self._parquet_readers: dict[str, ParquetAudioReader] = {}

    @property
    def row_group_reads(self) -> int:
        """Return physical Parquet row-group reads by this worker instance."""
        return sum(
            reader.row_group_reads
            for reader in self._parquet_readers.values()
        )

    def load(self, record: ManifestRecord) -> CanonicalAudio:
        """Load one record through its canonical file or Parquet path."""
        try:
            root = self._roots[record.dataset]
        except KeyError as error:
            raise SpeakerDatasetError(
                f"No dataset root configured for {record.dataset!r}."
            ) from error

        try:
            if record.audio_storage is AudioStorage.FILE:
                audio_path = (root / Path(record.audio_path)).resolve()
                try:
                    audio_path.relative_to(root)
                except ValueError as error:
                    raise SpeakerDatasetError(
                        "Standalone audio path escapes its dataset root: "
                        f"{record.audio_path!r}."
                    ) from error
                return load_audio_file(
                    audio_path,
                    target_sample_rate=self._target_sample_rate,
                )

            if record.audio_storage is AudioStorage.PARQUET:
                reader = self._parquet_readers.get(record.dataset)
                if reader is None:
                    reader = ParquetAudioReader(
                        root,
                        target_sample_rate=self._target_sample_rate,
                    )
                    self._parquet_readers[record.dataset] = reader
                return reader.load(record)
        except AudioDataError as error:
            raise SpeakerDatasetError(
                f"Unable to load utterance {record.utterance_id!r}: {error}"
            ) from error

        raise SpeakerDatasetError(
            f"Unsupported storage for {record.utterance_id!r}: "
            f"{record.audio_storage!r}."
        )


class TrainingSpeakerDataset:
    """Lazy fixed-crop dataset for speaker-classification training.

    Call ``set_epoch`` before constructing each epoch's DataLoader iterator so
    crop seeds change deterministically. Until an epoch-aware sampler is added,
    use ``persistent_workers=False``; otherwise persistent worker copies would
    not observe a changed epoch value.
    """

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        *,
        dataset_roots: Mapping[str, str | Path],
        segment_samples: int,
        seed: int,
        target_sample_rate: int = 16000,
        speaker_to_index: Mapping[str, int] | None = None,
    ) -> None:
        """Validate records and build a stable speaker classification map."""
        _require_positive_integer(segment_samples, "segment_samples")
        _require_non_negative_integer(seed, "seed")
        snapshot = validate_manifest(records)
        selected = tuple(
            sorted(
                (record for record in snapshot if record.split is Split.TRAIN),
                key=lambda record: record.utterance_id,
            )
        )
        if not selected:
            raise SpeakerDatasetError("No canonical training records were found.")

        speakers = tuple(sorted({record.speaker_id for record in selected}))
        self._speaker_to_index = _validated_speaker_mapping(
            speakers,
            speaker_to_index,
        )
        self._records = selected
        self._audio_loader = CanonicalAudioLoader(
            dataset_roots,
            target_sample_rate=target_sample_rate,
        )
        self._segment_samples = segment_samples
        self._seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        """Return canonical training utterance count."""
        return len(self._records)

    def __getitem__(self, index: int) -> TrainingSample:
        """Load and crop one utterance deterministically for the current epoch."""
        record = self._record_at(index)
        audio = self._audio_loader.load(record)
        crop_seed = stable_segment_seed(
            global_seed=self._seed,
            epoch=self._epoch,
            utterance_id=record.utterance_id,
        )
        waveform = random_fixed_segment(
            audio.waveform,
            num_samples=self._segment_samples,
            seed=crop_seed,
        )
        return TrainingSample(
            waveform=waveform,
            speaker_index=self._speaker_to_index[record.speaker_id],
            speaker_id=record.speaker_id,
            utterance_id=record.utterance_id,
            dataset=record.dataset,
        )

    @property
    def speaker_to_index(self) -> dict[str, int]:
        """Return a detached stable classification-label mapping."""
        return dict(self._speaker_to_index)

    @property
    def records(self) -> tuple[ManifestRecord, ...]:
        """Return the deterministic canonical record order."""
        return self._records

    @property
    def epoch(self) -> int:
        """Return the epoch currently used for crop derivation."""
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        """Select the non-negative epoch used by deterministic crop seeds."""
        _require_non_negative_integer(epoch, "epoch")
        self._epoch = epoch

    def _record_at(self, index: int) -> ManifestRecord:
        """Read a record using strict integer indexing."""
        _validate_index(index, len(self._records))
        return self._records[index]


class EvaluationSpeakerDataset:
    """Lazy deterministic multi-crop dataset for Validation or Test."""

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        *,
        split: Split,
        dataset_roots: Mapping[str, str | Path],
        segment_samples: int,
        segment_count: int,
        target_sample_rate: int = 16000,
    ) -> None:
        """Validate and select one canonical evaluation partition."""
        if split not in {Split.VALIDATION, Split.TEST}:
            raise SpeakerDatasetError(
                "Evaluation dataset requires validation or test split."
            )
        _require_positive_integer(segment_samples, "segment_samples")
        _require_positive_integer(segment_count, "segment_count")
        snapshot = validate_manifest(records)
        selected = tuple(
            sorted(
                (record for record in snapshot if record.split is split),
                key=lambda record: record.utterance_id,
            )
        )
        if not selected:
            raise SpeakerDatasetError(
                f"No canonical {split.value} records were found."
            )

        self._records = selected
        self._audio_loader = CanonicalAudioLoader(
            dataset_roots,
            target_sample_rate=target_sample_rate,
        )
        self._segment_samples = segment_samples
        self._segment_count = segment_count

    def __len__(self) -> int:
        """Return utterance count in the selected evaluation split."""
        return len(self._records)

    def __getitem__(self, index: int) -> EvaluationSample:
        """Load one utterance and return deterministic timeline crops."""
        _validate_index(index, len(self._records))
        record = self._records[index]
        audio = self._audio_loader.load(record)
        waveforms = evenly_spaced_segments(
            audio.waveform,
            num_samples=self._segment_samples,
            segment_count=self._segment_count,
        )
        return EvaluationSample(
            waveforms=waveforms,
            speaker_id=record.speaker_id,
            utterance_id=record.utterance_id,
            dataset=record.dataset,
        )

    @property
    def records(self) -> tuple[ManifestRecord, ...]:
        """Return the deterministic canonical record order."""
        return self._records


def collate_training_samples(samples: Sequence[TrainingSample]) -> TrainingBatch:
    """Stack equal-length training samples into contiguous NumPy arrays."""
    if not samples:
        raise SpeakerDatasetError("Cannot collate an empty training batch.")
    try:
        waveforms = np.stack(
            [sample.waveform for sample in samples],
            axis=0,
        )
    except ValueError as error:
        raise SpeakerDatasetError(
            "Training waveforms must share one fixed segment length."
        ) from error

    return TrainingBatch(
        waveforms=np.ascontiguousarray(waveforms, dtype=np.float32),
        speaker_indices=np.asarray(
            [sample.speaker_index for sample in samples],
            dtype=np.int64,
        ),
        speaker_ids=tuple(sample.speaker_id for sample in samples),
        utterance_ids=tuple(sample.utterance_id for sample in samples),
        datasets=tuple(sample.dataset for sample in samples),
    )


def collate_evaluation_samples(
    samples: Sequence[EvaluationSample],
) -> EvaluationBatch:
    """Flatten variable crop counts and retain utterance boundary offsets."""
    if not samples:
        raise SpeakerDatasetError("Cannot collate an empty evaluation batch.")
    if any(
        sample.waveforms.ndim != 2 or sample.waveforms.shape[0] == 0
        for sample in samples
    ):
        raise SpeakerDatasetError(
            "Evaluation samples must contain non-empty, equal-length 2D crops."
        )
    segment_lengths = {
        sample.waveforms.shape[1]
        for sample in samples
    }
    if len(segment_lengths) != 1:
        raise SpeakerDatasetError(
            "Evaluation samples must contain non-empty, equal-length 2D crops."
        )

    crop_counts = [sample.waveforms.shape[0] for sample in samples]
    offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(crop_counts, dtype=np.int64),
        )
    )
    waveforms = np.concatenate(
        [sample.waveforms for sample in samples],
        axis=0,
    )
    return EvaluationBatch(
        waveforms=np.ascontiguousarray(waveforms, dtype=np.float32),
        segment_offsets=offsets,
        speaker_ids=tuple(sample.speaker_id for sample in samples),
        utterance_ids=tuple(sample.utterance_id for sample in samples),
        datasets=tuple(sample.dataset for sample in samples),
    )


def _validated_speaker_mapping(
    speakers: tuple[str, ...],
    provided: Mapping[str, int] | None,
) -> dict[str, int]:
    """Build or validate an exact contiguous classification-label mapping."""
    if provided is None:
        return {speaker: index for index, speaker in enumerate(speakers)}

    mapping = dict(provided)
    if set(mapping) != set(speakers):
        raise SpeakerDatasetError(
            "speaker_to_index keys must exactly match training speakers."
        )
    indices = tuple(mapping.values())
    if any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in indices
    ):
        raise SpeakerDatasetError("Speaker indexes must be integers.")
    if set(indices) != set(range(len(speakers))):
        raise SpeakerDatasetError(
            "Speaker indexes must be unique and contiguous from zero."
        )
    return mapping


def _validate_index(index: int, length: int) -> None:
    """Require a conventional non-negative map-style dataset index."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise SpeakerDatasetError("Dataset index must be an integer.")
    if index < 0 or index >= length:
        raise IndexError(f"Dataset index {index} is outside [0, {length}).")


def _require_positive_integer(value: int, field_name: str) -> None:
    """Validate a positive integer runtime setting."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpeakerDatasetError(f"{field_name} must be a positive integer.")


def _require_non_negative_integer(value: int, field_name: str) -> None:
    """Validate a non-negative integer runtime setting."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpeakerDatasetError(
            f"{field_name} must be a non-negative integer."
        )
