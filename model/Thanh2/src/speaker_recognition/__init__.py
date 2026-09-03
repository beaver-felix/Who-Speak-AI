"""Shared speaker-recognition research package."""

from speaker_recognition.configuration import (
    ConfigurationError,
    ResolvedConfig,
    resolve_layered_config,
    write_resolved_config,
)
from speaker_recognition.data.manifest import (
    AudioStorage,
    ManifestRecord,
    ManifestValidationError,
    Split,
    validate_manifest,
)

__all__ = [
    "AudioStorage",
    "ConfigurationError",
    "ManifestRecord",
    "ManifestValidationError",
    "ResolvedConfig",
    "Split",
    "resolve_layered_config",
    "validate_manifest",
    "write_resolved_config",
]

__version__ = "0.1.0"
