"""Shared speaker-recognition research package."""

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

__version__ = "0.1.0"