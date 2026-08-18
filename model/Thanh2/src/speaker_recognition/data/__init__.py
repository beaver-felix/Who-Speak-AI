"""Dataset normalization and integrity utilities."""

from speaker_recognition.data.manifest import (
    AudioStorage,
    ManifestRecord,
    ManifestValidationError,
    Split,
    validate_manifest,
)

__all__ = [
    "AudioStorage",
    "ManifestRecord",
    "ManifestValidationError",
    "Split",
    "validate_manifest",
]