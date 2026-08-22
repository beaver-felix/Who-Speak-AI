"""Dataset normalization and integrity utilities."""

from speaker_recognition.data.audio import (
    AudioDataError,
    CanonicalAudio,
    ParquetAudioReader,
    canonicalize_audio,
    load_audio_file,
)
from speaker_recognition.data.dataset import (
    CanonicalAudioLoader,
    EvaluationBatch,
    EvaluationSample,
    EvaluationSpeakerDataset,
    SpeakerDatasetError,
    TrainingBatch,
    TrainingSample,
    TrainingSpeakerDataset,
    collate_evaluation_samples,
    collate_training_samples,
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
    "CanonicalAudioLoader",
    "EvaluationBatch",
    "EvaluationSample",
    "EvaluationSpeakerDataset",
    "ManifestRecord",
    "ManifestValidationError",
    "ParquetAudioReader",
    "SegmentError",
    "SpeakerDatasetError",
    "Split",
    "TrainingBatch",
    "TrainingSample",
    "TrainingSpeakerDataset",
    "canonicalize_audio",
    "collate_evaluation_samples",
    "collate_training_samples",
    "evenly_spaced_segments",
    "load_audio_file",
    "random_fixed_segment",
    "stable_segment_seed",
    "validate_manifest",
]
