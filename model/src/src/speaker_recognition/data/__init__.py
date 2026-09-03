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
    manifest_sha256,
    validate_manifest,
)
from speaker_recognition.data.sampling import (
    EpochSamplingError,
    select_speaker_capped_epoch,
    utterance_id_sha256,
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
    "EpochSamplingError",
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
    "manifest_sha256",
    "random_fixed_segment",
    "stable_segment_seed",
    "select_speaker_capped_epoch",
    "utterance_id_sha256",
    "validate_manifest",
]
