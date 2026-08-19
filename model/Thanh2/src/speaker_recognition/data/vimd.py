"""ViMD metadata conversion and canonical split policy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from speaker_recognition.data.manifest import (
    AudioStorage,
    ManifestRecord,
    Split,
)


class ViMDMetadataError(ValueError):
    """Raised when ViMD metadata violates the canonical protocol."""


VIMD_EXCLUDED_VALIDATION_SPEAKERS = frozenset(
    {
        "spk_73_0186",
        "spk_76_0219",
    }
)

_SOURCE_TO_CANONICAL_SPLIT = {
    "train": Split.TRAIN,
    "valid": Split.VALIDATION,
    "test": Split.TEST,
}


def parse_vimd_metadata_row(
    row: Mapping[str, object],
    *,
    parquet_path: str | Path,
    row_index: int,
    dataset_root: str | Path,
    source_split: str,
) -> ManifestRecord | None:
    """Convert one ViMD metadata row into a canonical manifest record.

    Audio bytes are intentionally not accessed. The returned Parquet locator
    allows waveform loading to remain lazy during training and evaluation.

    Parameters
    ----------
    row:
        Non-audio metadata selected from one Parquet row.
    parquet_path:
        Path to the containing Parquet shard.
    row_index:
        Zero-based row position within the shard.
    dataset_root:
        ViMD root containing the ``data`` directory.
    source_split:
        Source partition: ``train``, ``valid``, or ``test``.

    Returns
    -------
    ManifestRecord or None
        A canonical row, or ``None`` for one of the two explicitly excluded
        contaminated Validation speakers.

    Raises
    ------
    ViMDMetadataError
        If source metadata or physical location is invalid.
    """
    try:
        canonical_split = _SOURCE_TO_CANONICAL_SPLIT[source_split]
    except KeyError as error:
        raise ViMDMetadataError(
            f"Unsupported ViMD source split: {source_split!r}"
        ) from error

    if isinstance(row_index, bool) or not isinstance(row_index, int):
        raise ViMDMetadataError("row_index must be a non-negative integer.")
    if row_index < 0:
        raise ViMDMetadataError("row_index must be a non-negative integer.")

    root = Path(dataset_root).expanduser().resolve()
    shard_path = Path(parquet_path).expanduser().resolve()

    try:
        relative_shard_path = shard_path.relative_to(root)
    except ValueError as error:
        raise ViMDMetadataError(
            f"Parquet path is outside dataset root: {parquet_path}"
        ) from error

    if shard_path.suffix.lower() != ".parquet":
        raise ViMDMetadataError(
            f"ViMD storage must use Parquet shards: {shard_path.name}"
        )

    speaker_name = _required_text(row, "speakerID")
    filename = _required_text(row, "filename")

    if Path(filename).suffix.lower() != ".wav":
        raise ViMDMetadataError(
            f"ViMD filename must identify WAV audio: {filename!r}"
        )

    # Preserve official Test unchanged. These two source-Validation rows are
    # excluded because their identities already occur in Test.
    if (
        source_split == "valid"
        and speaker_name in VIMD_EXCLUDED_VALIDATION_SPEAKERS
    ):
        return None

    utterance_stem = Path(filename).stem
    utterance_id = ":".join(
        (
            "vimd",
            source_split,
            speaker_name,
            utterance_stem,
        )
    )

    return ManifestRecord(
        dataset="vimd",
        source_split=source_split,
        split=canonical_split,
        utterance_id=utterance_id,
        speaker_id=f"vimd:{speaker_name}",
        recording_id=utterance_id,
        audio_storage=AudioStorage.PARQUET,
        audio_path=relative_shard_path.as_posix(),
        audio_row_index=row_index,
        region=_optional_text(row.get("region")),
        province=_optional_text(row.get("province_name")),
        gender=_optional_text(row.get("gender")),
        transcript=_optional_text(row.get("text")),
    )


def _required_text(
    row: Mapping[str, object],
    field_name: str,
) -> str:
    """Read one required non-empty string field."""
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ViMDMetadataError(
            f"ViMD field {field_name!r} must be a non-empty string."
        )

    return value.strip()


def _optional_text(value: object) -> str | None:
    """Convert optional scalar metadata into a normalized string."""
    if value is None:
        return None

    normalized = str(value).strip()
    return normalized or None