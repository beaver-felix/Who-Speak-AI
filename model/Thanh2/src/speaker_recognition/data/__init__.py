"""Dataset normalization and integrity utilities."""

from speaker_recognition.data.audio import (
    AudioDataError,
    CanonicalAudio,
    ParquetAudioReader,
    canonicalize_audio,
    load_audio_file,
)
from speaker_recognition.data.manifest import (
    AudioStorage,
    ManifestRecord,
    ManifestValidationError,
    Split,
    validate_manifest,
)

__all__ = [
    "AudioDataError",
    "AudioStorage",
    "CanonicalAudio",
    "ManifestRecord",
    "ManifestValidationError",
    "ParquetAudioReader",
    "Split",
    "canonicalize_audio",
    "load_audio_file",
    "validate_manifest",
]
