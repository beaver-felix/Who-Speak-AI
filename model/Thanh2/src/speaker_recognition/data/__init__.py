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
from speaker_recognition.data.segments import (
    SegmentError,
    evenly_spaced_segments,
    random_fixed_segment,
    stable_segment_seed,
)

__all__ = [
    "AudioDataError",
    "AudioStorage",
    "CanonicalAudio",
    "ManifestRecord",
    "ManifestValidationError",
    "ParquetAudioReader",
    "SegmentError",
    "Split",
    "canonicalize_audio",
    "evenly_spaced_segments",
    "load_audio_file",
    "random_fixed_segment",
    "stable_segment_seed",
    "validate_manifest",
]
