"""Canonical manifest types and dataset-level integrity validation.

The two required datasets expose incompatible storage layouts: TidyVoice uses
standalone WAV files, while ViMD stores audio bytes inside Parquet rows.  This
module defines one model-independent record format that preserves both layouts
without extracting or duplicating the underlying audio.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Callable


class Split(StrEnum):
    """Canonical experiment partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class AudioStorage(StrEnum):
    """Supported physical audio-storage layouts."""

    FILE = "file"
    PARQUET = "parquet"


class ManifestValidationError(ValueError):
    """Raised when canonical manifest integrity requirements are violated."""


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """Describe one utterance without loading its waveform.

    Parameters
    ----------
    dataset:
        Stable dataset identifier, such as ``tidyvoice`` or ``vimd``.
    source_split:
        Partition name supplied by the source dataset. This is retained even
        when the canonical protocol creates a different validation/test split.
    split:
        Canonical experiment partition.
    utterance_id:
        Dataset-scoped unique utterance identifier.
    speaker_id:
        Dataset-scoped speaker identity used for grouping and leakage checks.
    recording_id:
        Source-recording identifier used to prevent recording-level leakage.
    audio_storage:
        Whether audio is a standalone file or an embedded Parquet value.
    audio_path:
        POSIX-style path to the WAV file or containing Parquet shard. Paths are
        stored without assuming that a later runtime has the same OS separator.
    audio_row_index:
        Zero-based row index for Parquet audio. It must be absent for files.
    sample_rate, channels, num_frames, duration_seconds:
        Optional header metadata. These remain optional until audio validation
        has inspected the utterance.
    language, region, province, gender, transcript:
        Optional source metadata retained for analysis and stratification, not
        automatically treated as trusted prediction targets.
    """

    dataset: str
    source_split: str
    split: Split
    utterance_id: str
    speaker_id: str
    recording_id: str
    audio_storage: AudioStorage
    audio_path: str
    audio_row_index: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    num_frames: int | None = None
    duration_seconds: float | None = None
    language: str | None = None
    region: str | None = None
    province: str | None = None
    gender: str | None = None
    transcript: str | None = None

    def __post_init__(self) -> None:
        """Validate invariants that apply to a single manifest row."""
        required_text = {
            "dataset": self.dataset,
            "source_split": self.source_split,
            "utterance_id": self.utterance_id,
            "speaker_id": self.speaker_id,
            "recording_id": self.recording_id,
            "audio_path": self.audio_path,
        }

        for field_name, value in required_text.items():
            if not value or not value.strip():
                raise ManifestValidationError(
                    f"{field_name} must be a non-empty string."
                )

        # A normalized separator makes manifests portable between Windows and
        # Kaggle Linux without altering absolute/relative path semantics.
        normalized_path = str(PurePosixPath(self.audio_path.replace("\\", "/")))
        object.__setattr__(self, "audio_path", normalized_path)

        if self.audio_storage is AudioStorage.FILE:
            if self.audio_row_index is not None:
                raise ManifestValidationError(
                    "audio_row_index must be absent for standalone files."
                )
        elif self.audio_storage is AudioStorage.PARQUET:
            if self.audio_row_index is None or self.audio_row_index < 0:
                raise ManifestValidationError(
                    "Parquet audio requires a non-negative audio_row_index."
                )
        else:
            raise ManifestValidationError(
                f"Unsupported audio storage: {self.audio_storage!r}."
            )

        positive_values = {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
        }
        for field_name, value in positive_values.items():
            if value is not None and value <= 0:
                raise ManifestValidationError(
                    f"{field_name} must be positive when provided."
                )

        if self.num_frames is not None and self.num_frames < 0:
            raise ManifestValidationError(
                "num_frames must be non-negative when provided."
            )

    @property
    def audio_reference(self) -> tuple[AudioStorage, str, int | None]:
        """Return the physical locator used for duplicate detection."""
        return self.audio_storage, self.audio_path, self.audio_row_index


def validate_manifest(
    records: Iterable[ManifestRecord],
    *,
    require_speaker_disjoint: bool = True,
    require_recording_disjoint: bool = True,
) -> tuple[ManifestRecord, ...]:
    """Validate a complete canonical manifest and return an immutable snapshot.

    Validation intentionally happens before model training. Duplicate IDs,
    duplicate physical audio, or group leakage would otherwise inflate every
    model's results in the same misleading way.

    Parameters
    ----------
    records:
        Manifest rows to validate.
    require_speaker_disjoint:
        Require every speaker to occur in exactly one canonical split.
    require_recording_disjoint:
        Require every source recording to occur in exactly one split.

    Returns
    -------
    tuple[ManifestRecord, ...]
        Immutable materialization of the validated rows.

    Raises
    ------
    ManifestValidationError
        If the manifest is empty or violates an integrity constraint.
    """
    snapshot = tuple(records)
    if not snapshot:
        raise ManifestValidationError("Manifest must contain at least one row.")

    utterance_counts = Counter(record.utterance_id for record in snapshot)
    duplicate_utterance_ids = sorted(
        identifier
        for identifier, count in utterance_counts.items()
        if count > 1
    )
    if duplicate_utterance_ids:
        raise ManifestValidationError(
            "Duplicate utterance IDs: "
            f"{_format_examples(duplicate_utterance_ids)}"
        )

    audio_counts = Counter(record.audio_reference for record in snapshot)
    duplicate_audio = [
        reference
        for reference, count in audio_counts.items()
        if count > 1
    ]
    if duplicate_audio:
        raise ManifestValidationError(
            "Duplicate physical audio references: "
            f"{_format_examples(sorted(duplicate_audio, key=str))}"
        )

    if require_speaker_disjoint:
        _validate_group_disjointness(
            snapshot,
            group_name="speaker",
            group_value=lambda record: record.speaker_id,
        )

    if require_recording_disjoint:
        _validate_group_disjointness(
            snapshot,
            group_name="recording",
            group_value=lambda record: record.recording_id,
        )

    return snapshot


def _validate_group_disjointness(
    records: tuple[ManifestRecord, ...],
    *,
    group_name: str,
    group_value: Callable[[ManifestRecord], str],
) -> None:
    """Ensure a selected group identifier never crosses canonical splits."""
    splits_by_group: dict[str, set[Split]] = defaultdict(set)

    for record in records:
        identifier = group_value(record)
        splits_by_group[identifier].add(record.split)

    leaking_groups = sorted(
        identifier
        for identifier, splits in splits_by_group.items()
        if len(splits) > 1
    )
    if leaking_groups:
        raise ManifestValidationError(
            f"Cross-split {group_name} leakage: "
            f"{_format_examples(leaking_groups)}"
        )


def _format_examples(values: list[object], limit: int = 10) -> str:
    """Format a bounded list of validation examples for readable errors."""
    examples = values[:limit]
    suffix = "" if len(values) <= limit else f" (+{len(values) - limit} more)"
    return f"{examples}{suffix}"
